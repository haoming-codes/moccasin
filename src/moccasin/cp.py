from ortools.sat.python import cp_model
import networkx as nx
import sys
from .util import maybe_dump, get_mem_from_schedule, gcd_process, flatten
import math
import numpy as np
import time
import copy
import uuid
import os

class ObjectiveSolutionSaver(cp_model.CpSolverSolutionCallback):
    """Store the objective value and time of intermediate solutions."""

    def __init__(self, file_name):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0
        self.__start_time = time.time()
        self.file_name = file_name
        with open(self.file_name, "w") as f:
            pass

    def on_solution_callback(self):
        """Called on each new solution."""
        current_time = time.time()
        obj = self.ObjectiveValue()
        dic = {'Solution': self.__solution_count, 'time': current_time - self.__start_time, 'objective': obj}
        with open(self.file_name, "a") as f:
            f.write(str(dic)+'\n')
        self.__solution_count += 1

    def solution_count(self):
        """Returns the number of solutions found."""
        return self.__solution_count

def addAllHints(model, solver, vars):
    for var in vars:
        model.AddHint(var, solver.Value(var))

def get_event_from_stage(order, stage):
    if stage < order: assert False
    prev_stage_event = int(stage*(stage+1)/2)
    return prev_stage_event + order
def get_event_from_order(order, n):
    events = []
    for stage in range(n):
        if stage < order: continue
        events.append(get_event_from_stage(order, stage))
    return events


def remat(G, B=None, L=None, R=None, C=2, 
        timeout=3600, 
        discretized=True, 
        reservoir=True, 
        use_end_global=False, 
        use_stage=True, 
        topo_order=None,
        phase1=True,
        use_interval_length_domain=True,
        use_gcd=True,
        reservoir_option=2,
        paging=False,
        objective="min_runtime",
        log_dir="output"):
    assert objective in ("min_footprint", "min_runtime", "min_communication")
    phase1 = False if B is None else phase1
    topo_order = list(nx.topological_sort(G)) if topo_order is None else topo_order
    label_mapping = {old: topo_order.index(old) for old in G.nodes}
    G = nx.relabel_nodes(G, label_mapping)
    topo_order = list(nx.topological_sort(G))
    print(G.nodes, topo_order)
    params = copy.copy(locals())
    sys.stdout.flush()

    if C == "max": C = G.number_of_nodes()
    if use_gcd: 
        G, mems_gcd, cpus_gcd = gcd_process(G)
    else:
        mems_gcd, cpus_gcd = 1, 1
    if B is not None: B = math.ceil(B/mems_gcd)
    if L is not None: L = math.ceil(L/mems_gcd)
    if R is not None: R = math.ceil(R/cpus_gcd)
    log_file = str(uuid.uuid4())

    nodes = G.nodes
    edges = G.edges
    N = G.number_of_nodes()
    node_duration = nx.get_node_attributes(G, "cost_cpu")
    node_size = nx.get_node_attributes(G, "cost_ram")
    horizon_start = 0
    if use_stage:
        horizon_end = int(N*(N+1)/2*1.1) if discretized else print("???")
    else:
        horizon_end = int(N*N) if discretized else int(sum(node_duration.values())*N)
    T = horizon_end - horizon_start
    if use_interval_length_domain:
        interval_length_domain = sorted(list(set(np.logspace(np.log10(1), np.log10(T), num=5*N, endpoint=True, base=10, dtype=int).tolist())))

    # === Solver ===
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True
    solver.parameters.max_time_in_seconds = timeout
    solver.parameters.num_search_workers = 16

    # === CP Model ===
    CP_name = f"remat: G={G.graph['name']}, B={B}, C={C}"
    model = cp_model.CpModel()
    model.SetName(CP_name)

    # === Vars ===
    start = [[model.NewIntVarFromDomain(cp_model.Domain.FromValues(get_event_from_order(topo_order.index(v), N)), f'start_{v}^{i}') for i in range(C)] for v in nodes] if use_stage else \
        [[model.NewIntVar(horizon_start, horizon_end, f'start_{v}^{i}') for i in range(C)] for v in nodes]
    end = [[model.NewIntVar(horizon_start, horizon_end, f'end_{v}^{i}') for i in range(C)] for v in nodes]
    length = [[model.NewIntVarFromDomain(cp_model.Domain.FromValues(interval_length_domain), f'length_{v}^{i}') for i in range(C)] for v in nodes] if use_interval_length_domain else \
        [[model.NewIntVar(1 if discretized else node_duration[v], T, f'length_{v}^{i}') for i in range(C)] for v in nodes]
    interval_presence = [[model.NewBoolVar(f'presence_{v}^{i}') for i in range(C)] for v in nodes]
    interval = [[model.NewOptionalIntervalVar(start[v][i], length[v][i], end[v][i], interval_presence[v][i], f'interval_{v}^{i}') for i in range(C)] for v in nodes]
    schedule_duration = model.NewIntVar(sum(node_duration.values()), sum(node_duration.values())*C, 'schedule_duration')
    mem_footprint = model.NewIntVar(0, sum(node_size.values())*C, 'mem_footprint')
    comm_load = model.NewIntVar(0, sum(node_size.values())*C, 'comm_load')
    if not discretized: compute_intervals = [[model.NewOptionalFixedSizeIntervalVar(start[v][i], node_duration[v], interval_presence[v][i], f'compute_interval_{v}^{i}') for i in range(C)] for v in nodes]
    if not reservoir: x = [[[[model.NewBoolVar(f'x_{u}{v}^{i}{j}') for j in range(C)] for i in range(C)] for v in nodes] for u in nodes]
    if phase1:
        tau_cpu = model.NewIntVar(0, sum(node_duration.values())*C, 'tau_cpu')
        tau_mem = model.NewIntVar(0, sum(node_size.values())*C, 'tau_mem')
    if paging:
        page_in = [[model.NewBoolVar(f'page_in_{v}^{i}') for i in range(C)] for v in nodes]
        page_out = [[model.NewBoolVar(f'page_out_{v}^{i}') for i in range(C)] for v in nodes]

    # === Constr ===
    for v in nodes:
        # every node is computed at least once
        model.Add(interval_presence[v][0] == 1)
        if use_stage:
            # the first computation of v is its first appearance, i.e. stage v's last event
            model.Add(start[v][0] == get_event_from_order(topo_order.index(v), N)[0])
        for i in range(C):
            if i > 0:
                # valid interval
                model.Add(end[v][i-1] <= start[v][i]).OnlyEnforceIf([interval_presence[v][i], interval_presence[v][i-1]])
                model.AddImplication(interval_presence[v][i], interval_presence[v][i-1]) # tighten
                # valid paging
                if paging: model.Add(page_in[v][i] <= sum(page_out[v][:i]))
            # no need to page if no interval
            if paging: model.AddImplication(page_in[v][i], interval_presence[v][i])
            if paging: model.AddImplication(page_out[v][i], interval_presence[v][i])
		
    # definition of end of schedule
    if use_end_global:
        # the end of schedule is the last of end
        model.AddMaxEquality(schedule_duration, flatten(end))
    else:
        if paging:
            computed = [[model.NewBoolVar(f'computed_{v}^{i}') for i in range(C)] for v in nodes]
            for v in nodes: 
                for i in range(C): 
                    model.AddMinEquality(computed[v][i], (interval_presence[v][i], page_in[v][i].Not()))
        # the length of schedule is the sum of active node_duration
        model.Add(
            cp_model.LinearExpr.WeightedSum(
                list(flatten(interval_presence)) if not paging else list(flatten(computed)),
                list(flatten([[node_duration[v]]*C for v in nodes]))
            ) <= schedule_duration)
        
    # definition of communication load
    if paging:
        communicated = [[model.NewIntVar(0, 2, f'communicated_{v}^{i}') for i in range(C)] for v in nodes]
        for v in nodes: 
            for i in range(C): 
                model.Add(communicated[v][i] == page_in[v][i] + page_out[v][i])
        model.Add(
            cp_model.LinearExpr.WeightedSum(
                list(flatten(communicated)),
                list(flatten([[node_size[v]]*C for v in nodes]))
            ) <= comm_load)
    
    # memory budget
    model.AddCumulative(
        flatten(interval), 
        flatten([[node_size[v]]*C for v in nodes]), 
        mem_footprint)

    # one node computed at a time
    if not use_stage:
        if discretized:
            model.AddAllDifferent(flatten(start))
        else:
            model.AddNoOverlap(flatten(compute_intervals))
    
    # precedence constraint by reservoir constraint
    if reservoir:
        if reservoir_option == 1:
            for u, v in edges:
                times = flatten([
                    [start[u][i]+(1 if discretized else node_duration[u]), start[v][j], start[v][j]+(1 if discretized else node_duration[v]), end[u][i]]# if not paging else 
                    # [start[u][i]*page_in[v][i].Not()+(1 if discretized else node_duration[u]), start[v][j].Not(), start[v][j].Not()+(1 if discretized else node_duration[v]), end[u][i].Not()]
                    for i in range(C) for j in range(C)])
                changes = [1,-1,1,-1]*(C**2)
                actives = flatten([
                    [interval_presence[u][i], interval_presence[v][j] if not paging else computed[v][j], interval_presence[v][j] if not paging else computed[v][j], interval_presence[u][i]]
                    for i in range(C) for j in range(C)])
                model.AddReservoirConstraintWithActive(times, changes, actives, 0, 2*(C**2))
        if reservoir_option == 2:
            for u, v in edges:
                for j in range(C):
                    times = list(flatten([[
                        start[u][i]+(1 if discretized else node_duration[u]), end[u][i]] 
                        for i in range(C)])) + [start[v][j], start[v][j]+(1 if discretized else node_duration[v])]
                    changes = [1,-1]*(C) + [-1,1]
                    actives = list(flatten([[interval_presence[u][i], interval_presence[u][i]] for i in range(C)])) + [interval_presence[v][j] if not paging else computed[v][j], interval_presence[v][j] if not paging else computed[v][j]]
                    model.AddReservoirConstraintWithActive(times, changes, actives, 0, C+1)
        if reservoir_option == 3:
            for u, v in edges:
                for i in range(C):
                    for j in range(C):
                        times = [start[u][i]+(1 if discretized else node_duration[u]), start[v][j], start[v][j]+(1 if discretized else node_duration[v]), end[u][i]]
                        changes = [1,-1,1,-1]
                        actives = [interval_presence[u][i], interval_presence[v][j] if not paging else computed[v][j], interval_presence[v][j] if not paging else computed[v][j], interval_presence[u][i]]
                        model.AddReservoirConstraintWithActive(times, changes, actives, 0, 2)
    else: # precedence constraint by x_uv^ij
        # if u_i serves v_j, then they should be present at the right time
        for u, v in edges:
            for i in range(C):
                for j in range(C):
                    model.AddImplication(x[u][v][i][j], interval_presence[u][i])
                    model.AddImplication(x[u][v][i][j], interval_presence[v][j])
                    model.Add(start[u][i] + (1 if discretized else node_duration[u]) <= start[v][j]).OnlyEnforceIf(x[u][v][i][j])
                    model.Add(start[v][j] + (1 if discretized else node_duration[v]) <= end[u][i]).OnlyEnforceIf(x[u][v][i][j])
        # if v_j is present, then it should be served by some u_i
        for u, v in edges:
            for j in range(C):
                model.AddBoolOr([x[u][v][i][j] for i in range(C)]).OnlyEnforceIf(interval_presence[v][j])
        # if u_i is present, then it should serve some v_j
        for u in nodes:
            children = [e[1] for e in edges if e[0] == u]
            if len(children) == 0: continue
            for i in range(C):
                model.AddBoolOr([x[u][v][i][j] for j in range(C) for v in children]).OnlyEnforceIf(interval_presence[u][i]) # tighten

    # === Solve ===
    solution_printer = ObjectiveSolutionSaver(log_file) # only logs phase 2 objective
    fname = f"{log_dir}/{CP_name}.pkl"
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    pkl_dict = {}
    if phase1:
        if objective != "min_runtime" and R is not None:
            model.Add(tau_cpu >= schedule_duration)
            model.Add(tau_cpu >= R)
        if objective != "min_footprint" and B is not None:
            model.Add(tau_mem >= mem_footprint)
            model.Add(tau_mem >= B)
        model.Minimize(tau_cpu+tau_mem)
        status = solver.Solve(model)
        sys.stdout.flush()
        print('\n===========================================')
        print('Phase 1 Status = %s' % solver.StatusName(status))
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print(f"schedule_duration = {solver.Value(schedule_duration)*cpus_gcd}")
            print(f"mem_footprint = {solver.Value(mem_footprint)*mems_gcd}")
            print(f"comm_load = {solver.Value(comm_load)*mems_gcd}")
            print(f"solver.WallTime = {solver.WallTime()}")
        else:
            pkl_dict["status"] = solver.StatusName(status)
            maybe_dump(pkl_dict, fname)
            os.remove(log_file)
            return pkl_dict
        addAllHints(model, solver, vars=flatten([start, end, length, interval_presence, schedule_duration]))
        model.Proto().ClearField('objective')

    # add budget constraints
    if objective != "min_footprint" and B is not None:
        model.Add(mem_footprint <= B)
    if objective != "min_communication" and L is not None:
        model.Add(comm_load <= L)
    if objective != "min_runtime" and R is not None:
        model.Add(schedule_duration <= R)
    
    # Phase 2 objectives
    if objective == "min_runtime":
        model.Minimize(schedule_duration)
    elif objective == "min_communication":
        model.Minimize(comm_load)
    elif objective == "min_footprint":
        model.Minimize(mem_footprint)
    status = solver.SolveWithSolutionCallback(model, solution_printer)
    
    sys.stdout.flush()
    print('\n===========================================')
    print('Phase 2 Status = %s' % solver.StatusName(status))
    # print(f'Number of solutions found: {solution_printer.solution_count()}, after {solver.WallTime()} seconds')

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"schedule_duration = {solver.Value(schedule_duration)*cpus_gcd}")
        print(f"mem_footprint = {solver.Value(mem_footprint)*mems_gcd}")
        print(f"comm_load = {solver.Value(comm_load)*mems_gcd}")
        print(f"solver.WallTime = {solver.WallTime()}")
        vc = [[v for i in range(C)] for v in nodes]
        sorted_vs = sorted(
            [(solver.Value(v), solver.Value(s)) for v, s, p in zip(flatten(vc), flatten(start), flatten(interval_presence)) if solver.BooleanValue(p)],
            key=lambda x: x[1])
        schedule = [v for v, s in sorted_vs]
        mem_schedule, cpu_schedule = get_mem_from_schedule(G, schedule)
        topo_mem, topo_cpu = get_mem_from_schedule(G, topo_order)
        
        with open(log_file, 'r') as f:
            lines = f.readlines()
        solns = {}
        for line in lines:
            d = eval(line)
            d['objective'] *= cpus_gcd
            solns[d['Solution']] = d

        # === make result dict ===
        del params['G']
        params['G_name'] = G.graph['name']
        pkl_dict["params"] = params
        pkl_dict["solns"] = solns
        pkl_dict["status"] = solver.StatusName(status)
        pkl_dict["obj_val"] = solver.Value(schedule_duration)*cpus_gcd
        pkl_dict["mem_footprint"] = solver.Value(mem_footprint)*mems_gcd

        pkl_dict["schedule"] = schedule
        pkl_dict["mem_schedule"] = mem_schedule*mems_gcd
        pkl_dict["cpu_schedule"] = cpu_schedule*cpus_gcd
        pkl_dict["topo_mem"] = topo_mem*mems_gcd
        pkl_dict["topo_cpu"] = topo_cpu*cpus_gcd

        maybe_dump(pkl_dict, fname)
    else:
        pkl_dict["status"] = solver.StatusName(status)
        maybe_dump(pkl_dict, fname)

    print('===========================================\n')
    sys.stdout.flush()
    os.remove(log_file)

    return pkl_dict
