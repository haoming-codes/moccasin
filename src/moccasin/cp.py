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
import itertools
from pprint import pprint
from rockmate.def_op import RunOp, DelOp, OpSchedule
from contextlib import contextmanager

KG_TIME_TAG = 'time'
NX_TIME_TAG = 'cost_cpu'
KG_SIZE_TAG = 'memsize'
KG_NAME_TAG = NX_NAME_TAG = 'name'
KG_TYPE_TAG = 'type'
KG_TYPE_DATA_TAG = 'kdn'
KG_TYPE_COMPUTE_TAG = 'kcn'
NX_SIZE_TAG = 'cost_ram'


@contextmanager
def stdout_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different


def kdn_with_mul_indegree(G, attr=False):
    # when several forward operations share the same input, 
    # the corresponding backward operations contribute to 
    # the same data (by summing all the contributions). 
    kdn_nodes = [n for n, ntype in G.nodes.data(KG_TYPE_TAG) if ntype == KG_TYPE_DATA_TAG]
    kdn_predecessors = {n: G.predecessors(n) for n in kdn_nodes}
    odd_kdn = {n: p for n,p in kdn_predecessors.items() if len(list(p)) > 1}
    if attr:
        odd_kdn = {G.nodes[n][attr]: [G.nodes[p][attr] for p in ps] for n,ps in odd_kdn.items()}
    return odd_kdn

def kcn_with_mul_outdegree(G, attr=False):
    # when several forward operations share the same input, 
    # the corresponding backward operations contribute to 
    # the same data (by summing all the contributions). 
    kcn_nodes = [n for n, ntype in G.nodes.data(KG_TYPE_TAG) if ntype == KG_TYPE_COMPUTE_TAG]
    kcn_successors = {n: G.predecessors(n) for n in kcn_nodes}
    odd_kcn = {n: s for n,s in kcn_successors.items() if len(list(s)) > 1}
    if attr:
        odd_kcn = {G.nodes[n][attr]: [G.nodes[s][attr] for s in ss] for n,ss in odd_kcn.items()}
    return odd_kcn

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

class Moccasin:
    def __init__(self,
        G, B=None, L=None, R=None, C=2, 
        timeout=3600, 
        discretized=True, 
        reservoir=True, 
        use_end_global=False, 
        use_stage=True, 
        topo_order=None,
        use_interval_length_domain=True,
        use_gcd=True,
        reservoir_option=2,
        paging=False,
        objective="min_runtime"):
        assert objective in ("min_footprint", "min_runtime", "min_communication")

        self.topo_order = topo_order if topo_order else list(nx.topological_sort(G))
        if max(G.nodes) >= G.number_of_nodes():
            label_mapping = {old: self.topo_order.index(old) for old in G.nodes}
            G = nx.relabel_nodes(G, label_mapping)
            self.topo_order = [label_mapping[v] for v in self.topo_order]
        self.params = copy.copy(locals())
        if C == "max": C = G.number_of_nodes()
        if use_gcd: 
            G, self.mems_gcd, self.cpus_gcd = gcd_process(G)
        else:
            self.mems_gcd, self.cpus_gcd = 1, 1
        if B: B = math.ceil(B/self.mems_gcd)
        if L: L = math.ceil(L/self.mems_gcd)
        if R: R = math.ceil(R/self.cpus_gcd)
        self.G = G
        self.B = B
        self.L = L
        self.R = R
        self.C = C
        self.timeout = timeout
        self.discretized = discretized
        self.reservoir = reservoir
        self.use_end_global = use_end_global
        self.use_stage = use_stage
        self.use_interval_length_domain = use_interval_length_domain
        self.use_gcd = use_gcd
        self.reservoir_option = reservoir_option
        self.paging = paging
        self.objective = objective

        self.make_cpsat_model()

    @classmethod
    def from_kG(cls, kG, *args, name=None, **kwargs):
        G = cls.convert_graph(kG, name=name)
        obj = cls(G, *args, **kwargs)
        obj.kG = kG
        return obj
    
    def solve(self, phase1=True, log_dir="output", verbose=False):
        # === Solver ===
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = True
        solver.parameters.max_time_in_seconds = self.timeout
        solver.parameters.num_search_workers = 16
        self.feasible = False
        
        phase1 = phase1 if self.B else False
        
        # === Solve ===
        log_file = str(uuid.uuid4())
        solution_printer = ObjectiveSolutionSaver(log_file) # only logs phase 2 objective
        fname = f"{log_dir}/{self.model.Name()}.pkl"
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        pkl_dict = {}
        if phase1:
            if self.objective != "min_runtime" and self.R:
                self.model.Add(self.tau_cpu >= self.schedule_duration)
                self.model.Add(self.tau_cpu >= self.R)
            if self.objective != "min_footprint" and self.B:
                self.model.Add(self.tau_mem >= self.mem_footprint)
                self.model.Add(self.tau_mem >= self.B)
            self.model.Minimize(self.tau_mem)
            with stdout_redirected():
                status = solver.Solve(self.model)
            sys.stdout.flush()
            if verbose:
                print('\n===========================================')
                print('Phase 1 Status = %s' % solver.StatusName(status))
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                self.feasible = True
                if verbose:
                    print(f"mem_budget = {self.B*self.mems_gcd if self.B else None}")
                    print(f"time_budget = {self.R*self.cpus_gcd if self.R else None}")
                    print(f"comm_budget = {self.L*self.mems_gcd if self.L else None}")
                    print(f"schedule_duration = {solver.Value(self.schedule_duration)*self.cpus_gcd}")
                    print(f"mem_footprint = {solver.Value(self.mem_footprint)*self.mems_gcd}")
                    print(f"comm_load = {solver.Value(self.comm_load)*self.mems_gcd}")
                    print(f"solver.WallTime = {solver.WallTime()}")
            else:
                pkl_dict["status"] = solver.StatusName(status)
                maybe_dump(pkl_dict, fname)
                os.remove(log_file)
                return pkl_dict
            addAllHints(self.model, solver, vars=flatten([self.start, self.end, self.length, self.interval_presence, self.schedule_duration]))
            self.model.Proto().ClearField('objective')

        # add budget constraints
        if self.objective != "min_footprint" and self.B:
            self.model.Add(self.mem_footprint <= self.B)
        if self.objective != "min_communication" and self.L:
            self.model.Add(self.comm_load <= self.L)
        if self.objective != "min_runtime" and self.R:
            self.model.Add(self.schedule_duration <= self.R)
        
        # Phase 2 objectives
        if self.objective == "min_runtime":
            self.model.Minimize(self.schedule_duration)
        elif self.objective == "min_communication":
            self.model.Minimize(self.comm_load)
        elif self.objective == "min_footprint":
            self.model.Minimize(self.mem_footprint)
        with stdout_redirected():
            status = solver.SolveWithSolutionCallback(self.model, solution_printer)
        
        sys.stdout.flush()
        if verbose:
            print('\n===========================================')
            print('Phase 2 Status = %s' % solver.StatusName(status))
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            self.feasible = True
            if verbose:
                print(f"mem_budget = {self.B*self.mems_gcd if self.B else None}")
                print(f"time_budget = {self.R*self.cpus_gcd if self.R else None}")
                print(f"comm_budget = {self.L*self.mems_gcd if self.L else None}")
                print(f"schedule_duration = {solver.Value(self.schedule_duration)*self.cpus_gcd}")
                print(f"mem_footprint = {solver.Value(self.mem_footprint)*self.mems_gcd}")
                print(f"comm_load = {solver.Value(self.comm_load)*self.mems_gcd}")
                print(f"solver.WallTime = {solver.WallTime()}")
            vc = [[v for i in range(self.C)] for v in sorted(self.G.nodes)]
            sorted_vs = sorted(
                [(solver.Value(v), solver.Value(s)) for v, s, p in zip(flatten(vc), flatten(self.start), flatten(self.interval_presence)) if solver.BooleanValue(p)],
                key=lambda x: x[1])
            schedule = [v for v, s in sorted_vs]
            mem_schedule, cpu_schedule = get_mem_from_schedule(self.G, schedule)
            topo_mem, topo_cpu = get_mem_from_schedule(self.G, self.topo_order)
            
            with open(log_file, 'r') as f:
                lines = f.readlines()
            solns = {}
            for line in lines:
                d = eval(line)
                d['objective'] *= self.cpus_gcd
                solns[d['Solution']] = d

            # === make result dict ===
            del self.params['G']
            self.params['G_name'] = self.G.graph['name']
            pkl_dict["params"] = self.params
            pkl_dict["solns"] = solns
            pkl_dict["status"] = solver.StatusName(status)
            pkl_dict["obj_val"] = solver.Value(self.schedule_duration)*self.cpus_gcd
            pkl_dict["mem_footprint"] = solver.Value(self.mem_footprint)*self.mems_gcd

            pkl_dict["schedule"] = schedule
            pkl_dict["mem_schedule"] = mem_schedule*self.mems_gcd
            pkl_dict["cpu_schedule"] = cpu_schedule*self.cpus_gcd
            pkl_dict["topo_mem"] = topo_mem*self.mems_gcd
            pkl_dict["topo_cpu"] = topo_cpu*self.cpus_gcd
            self.schedule = schedule
            maybe_dump(pkl_dict, fname)
        else:
            pkl_dict["status"] = solver.StatusName(status)
            maybe_dump(pkl_dict, fname)

        if verbose:
            print('===========================================\n')
        sys.stdout.flush()
        os.remove(log_file)

        return pkl_dict

    def _get_augmented_graph(self, G, schedule):
        G_aug = nx.DiGraph()
        for j, v in enumerate(schedule):
            G_aug.add_node((j, v), cost_cpu=G.nodes[v]["cost_cpu"], cost_ram=G.nodes[v]["cost_ram"])
            for u in G.predecessors(v):
                for i in reversed(range(0, j)):
                    if schedule[i] == u:
                        G_aug.add_edge((i, u), (j, v))
                        break
        return G_aug
    
    def _get_in_mem_list(self, G_aug):
        in_mem_list = []
        for j, v in G_aug.nodes:
            in_mem = []
            for i, u in list(G_aug.nodes)[:j]:
                for k, w in G_aug.successors((i, u)):
                    if k >= j:
                        in_mem.append((i, u))
                        break
            # print()
            assert (j, v) not in in_mem
            in_mem.append((j, v))
            in_mem_list.append(in_mem)
        return in_mem_list

    def schedule_to(self, kG=None, schedule=None):
        kG = kG if kG else self.kG
        schedule = schedule if schedule else self.schedule
        G_aug = self._get_augmented_graph(self.G, schedule)
        in_mem_list = self._get_in_mem_list(G_aug)
        
        T = len(self.kG.list_kcn)
        I = len(self.kG.list_kdn)
        op_list = []
        alive_list = []
        alive_status = np.zeros(I + 2, dtype=bool)
        alive_status[-1] = 1  # input_data_kdn
        for i, cn in G_aug:
            out_mem = set(in_mem_list[i-1] if i > 0 else []) - set(in_mem_list[i])
            kdns = [kdn for i, n in out_mem for kdn in self.G.nodes[n]['kdn']]
            op_list += [DelOp(kdn) for kdn in kdns]
            kcn = self.G.nodes[cn]['kcn']
            op_list.append(RunOp(kcn))
        
        for i, op in enumerate(op_list):
            if "loss" in op.name:
                loss_i = i
                break

        fwd_sched = OpSchedule(
            op_list[: loss_i + 1],
            alive_list[: loss_i + 1],
            self.kG.input_kdn_data,
            self.kG.input_kdn_grad,
            self.kG.output_kdn_data,
            self.kG.list_kdn,
        )
        bwd_sched = OpSchedule(
            op_list[loss_i + 1 :],
            alive_list[loss_i + 1 :],
            self.kG.input_kdn_data,
            self.kG.input_kdn_grad,
            self.kG.output_kdn_data,
            self.kG.list_kdn,
        )
        # fwd_sched.del_input(kg)
        return fwd_sched, bwd_sched



    def make_cpsat_model(self):
        
        G, B, L, R, C = self.G, self.B, self.L, self.R, self.C
        discretized = self.discretized
        use_stage = self.use_stage
        use_interval_length_domain = self.use_interval_length_domain
        paging = self.paging
        topo_order = self.topo_order
        use_end_global = self.use_end_global
        reservoir = self.reservoir
        reservoir_option = self.reservoir_option

        nodes = sorted(G.nodes)
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

        # === CP Model ===
        CP_name = f"remat: G={G.graph['name']}, B={B}, C={C}"
        self.model = cp_model.CpModel()
        self.model.SetName(CP_name)

        # === Vars ===
        self.start = [[self.model.NewIntVarFromDomain(cp_model.Domain.FromValues(get_event_from_order(topo_order.index(v), N)), f'start_{v}^{i}') for i in range(C)] for v in nodes] if use_stage else \
            [[self.model.NewIntVar(horizon_start, horizon_end, f'start_{v}^{i}') for i in range(C)] for v in nodes]
        self.end = [[self.model.NewIntVar(horizon_start, horizon_end, f'end_{v}^{i}') for i in range(C)] for v in nodes]
        self.length = [[self.model.NewIntVarFromDomain(cp_model.Domain.FromValues(interval_length_domain), f'length_{v}^{i}') for i in range(C)] for v in nodes] if use_interval_length_domain else \
            [[self.model.NewIntVar(1 if discretized else node_duration[v], T, f'length_{v}^{i}') for i in range(C)] for v in nodes]
        self.interval_presence = [[self.model.NewBoolVar(f'presence_{v}^{i}') for i in range(C)] for v in nodes]
        self.interval = [[self.model.NewOptionalIntervalVar(self.start[v][i], self.length[v][i], self.end[v][i], self.interval_presence[v][i], f'interval_{v}^{i}') for i in range(C)] for v in nodes]
        self.schedule_duration = self.model.NewIntVar(sum(node_duration.values()), sum(node_duration.values())*C, 'schedule_duration')
        self.mem_footprint = self.model.NewIntVar(0, sum(node_size.values())*C, 'mem_footprint')
        self.comm_load = self.model.NewIntVar(0, sum(node_size.values())*C, 'comm_load')
        self.tau_cpu = self.model.NewIntVar(0, max(sum(node_duration.values())*C, R if R else 0), 'tau_cpu')
        self.tau_mem = self.model.NewIntVar(0, max(sum(node_size.values())*C, B if B else 0), 'tau_mem')    
        if not self.discretized: 
            self.compute_intervals = [[self.model.NewOptionalFixedSizeIntervalVar(self.start[v][i], node_duration[v], self.interval_presence[v][i], f'compute_interval_{v}^{i}') for i in range(C)] for v in nodes]
        if not self.reservoir: 
            self.x = [[[[self.model.NewBoolVar(f'x_{u}{v}^{i}{j}') for j in range(C)] for i in range(C)] for v in nodes] for u in nodes]
        if paging:
            self.computed = [[self.model.NewBoolVar(f'computed_{v}^{i}') for i in range(C)] for v in nodes]    
            self.page_in = [[self.model.NewBoolVar(f'page_in_{v}^{i}') for i in range(C)] for v in nodes]
            self.page_out = [[self.model.NewBoolVar(f'page_out_{v}^{i}') for i in range(C)] for v in nodes]
            self.communicated = [[self.model.NewIntVar(0, 2, f'communicated_{v}^{i}') for i in range(C)] for v in nodes]

        # === Constr ===
        for v in nodes:
            # every node is computed at least once
            self.model.Add(self.interval_presence[v][0] == 1)
            if use_stage:
                # the first computation of v is its first appearance, i.e. stage v's last event
                self.model.Add(self.start[v][0] == get_event_from_order(topo_order.index(v), N)[0])
            for i in range(C):
                if i > 0:
                    # valid interval
                    self.model.Add(self.end[v][i-1] <= self.start[v][i]).OnlyEnforceIf([self.interval_presence[v][i], self.interval_presence[v][i-1]])
                    self.model.AddImplication(self.interval_presence[v][i], self.interval_presence[v][i-1]) # tighten
                    # valid paging
                    if paging: self.model.Add(self.page_in[v][i] <= sum(self.page_out[v][:i]))
                # no need to page if no interval
                if paging: self.model.AddImplication(self.page_in[v][i], self.interval_presence[v][i])
                if paging: self.model.AddImplication(self.page_out[v][i], self.interval_presence[v][i])
            
        # definition of end of schedule
        if use_end_global:
            # the end of schedule is the last of end
            self.model.AddMaxEquality(self.schedule_duration, flatten(self.end))
        else:
            if paging:
                for v in nodes: 
                    for i in range(C): 
                        self.model.AddMinEquality(self.computed[v][i], (self.interval_presence[v][i], self.page_in[v][i].Not()))
            # the length of schedule is the sum of active node_duration
            self.model.Add(
                cp_model.LinearExpr.WeightedSum(
                    list(flatten(self.interval_presence)) if not paging else list(flatten(self.computed)),
                    list(flatten([[node_duration[v]]*C for v in nodes]))
                ) <= self.schedule_duration)
            
        # definition of communication load
        if paging:
            for v in nodes: 
                for i in range(C): 
                    self.model.Add(self.communicated[v][i] == self.page_in[v][i] + self.page_out[v][i])
            self.model.Add(
                cp_model.LinearExpr.WeightedSum(
                    list(flatten(self.communicated)),
                    list(flatten([[node_size[v]]*C for v in nodes]))
                ) <= self.comm_load)
        
        # memory budget
        self.model.AddCumulative(
            flatten(self.interval), 
            flatten([[node_size[v]]*C for v in nodes]), 
            self.mem_footprint)

        # one node computed at a time
        if not use_stage:
            if discretized:
                self.model.AddAllDifferent(flatten(self.start))
            else:
                self.model.AddNoOverlap(flatten(self.compute_intervals))
        
        # precedence constraint by reservoir constraint
        if reservoir:
            if reservoir_option == 1:
                for u, v in edges:
                    times = flatten([
                        [self.start[u][i]+(1 if discretized else node_duration[u]), self.start[v][j], self.start[v][j]+(1 if discretized else node_duration[v]), self.end[u][i]]# if not paging else 
                        # [start[u][i]*page_in[v][i].Not()+(1 if discretized else node_duration[u]), start[v][j].Not(), start[v][j].Not()+(1 if discretized else node_duration[v]), end[u][i].Not()]
                        for i in range(C) for j in range(C)])
                    changes = [1,-1,1,-1]*(C**2)
                    actives = flatten([
                        [self.interval_presence[u][i], self.interval_presence[v][j] if not paging else self.computed[v][j], self.interval_presence[v][j] if not paging else self.computed[v][j], self.interval_presence[u][i]]
                        for i in range(C) for j in range(C)])
                    self.model.AddReservoirConstraintWithActive(times, changes, actives, 0, 2*(C**2))
            if reservoir_option == 2:
                for u, v in edges:
                    for j in range(C):
                        times = list(flatten([[
                            self.start[u][i]+(1 if discretized else node_duration[u]), self.end[u][i]] 
                            for i in range(C)])) + [self.start[v][j], self.start[v][j]+(1 if discretized else node_duration[v])]
                        changes = [1,-1]*(C) + [-1,1]
                        actives = list(flatten([[self.interval_presence[u][i], self.interval_presence[u][i]] for i in range(C)])) + [self.interval_presence[v][j] if not paging else self.computed[v][j], self.interval_presence[v][j] if not paging else self.computed[v][j]]
                        self.model.AddReservoirConstraintWithActive(times, changes, actives, 0, C+1)
            if reservoir_option == 3:
                for u, v in edges:
                    for i in range(C):
                        for j in range(C):
                            times = [self.start[u][i]+(1 if discretized else node_duration[u]), self.start[v][j], self.start[v][j]+(1 if discretized else node_duration[v]), self.end[u][i]]
                            changes = [1,-1,1,-1]
                            actives = [self.interval_presence[u][i], self.interval_presence[v][j] if not paging else self.computed[v][j], self.interval_presence[v][j] if not paging else self.computed[v][j], self.interval_presence[u][i]]
                            self.model.AddReservoirConstraintWithActive(times, changes, actives, 0, 2)
        else: # precedence constraint by x_uv^ij
            # if u_i serves v_j, then they should be present at the right time
            for u, v in edges:
                for i in range(C):
                    for j in range(C):
                        self.model.AddImplication(self.x[u][v][i][j], self.interval_presence[u][i])
                        self.model.AddImplication(self.x[u][v][i][j], self.interval_presence[v][j])
                        self.model.Add(self.start[u][i] + (1 if discretized else node_duration[u]) <= self.start[v][j]).OnlyEnforceIf(self.x[u][v][i][j])
                        self.model.Add(self.start[v][j] + (1 if discretized else node_duration[v]) <= self.end[u][i]).OnlyEnforceIf(self.x[u][v][i][j])
            # if v_j is present, then it should be served by some u_i
            for u, v in edges:
                for j in range(C):
                    self.model.AddBoolOr([self.x[u][v][i][j] for i in range(C)]).OnlyEnforceIf(self.interval_presence[v][j])
            # if u_i is present, then it should serve some v_j
            for u in nodes:
                children = [e[1] for e in edges if e[0] == u]
                if len(children) == 0: continue
                for i in range(C):
                    self.model.AddBoolOr([self.x[u][v][i][j] for j in range(C) for v in children]).OnlyEnforceIf(self.interval_presence[u][i]) # tighten

    @staticmethod   
    def convert_graph(kG, name=None):
        nxDCG = Moccasin.kG_to_nx(kG, name)
        nxCG = Moccasin.dc_nx_to_c_nx(nxDCG)
        return nxCG

    @staticmethod
    def kG_to_nx(kG, name=None, name_to_listid=None):
        G = nx.DiGraph()
        G.graph['name'] = name

        # pprint(vars(kG.list_kcn[0]))
        # pprint(vars(kG.list_kcn[1]))
        # pprint(vars(kG.list_kcn[2]))
        # pprint(vars(kG.list_kdn[0]))
        # pprint(vars(kG.list_kdn[1]))
        # pprint(vars(kG.list_kdn[2]))
        # assert False
        G.add_nodes_from([node.unique_id for node in kG.list_kcn])
        G.add_nodes_from([node.unique_id for node in kG.list_kdn])
        G.add_edges_from([(u.unique_id, v.unique_id) for u in kG.list_kcn for v in u.users])
        # G.add_edges_from([(u.unique_id, v.unique_id) for v in kG.list_kcn for u in v.deps_real])
        G.add_edges_from([(u.unique_id, v.unique_id) for u in kG.list_kdn for v in u.users_real if v in kG.list_kcn])
        G.add_edges_from([(u.unique_id, v.unique_id) for v in kG.list_kdn for u in v.deps])
        nx.set_node_attributes(G, {node.unique_id: node for node in kG.list_kcn}, 'kcn')
        nx.set_node_attributes(G, {node.unique_id: node for node in kG.list_kdn}, 'kdn')
        nx.set_node_attributes(G, {node.unique_id: KG_TYPE_COMPUTE_TAG for node in kG.list_kcn}, KG_TYPE_TAG)
        nx.set_node_attributes(G, {node.unique_id: KG_TYPE_DATA_TAG for node in kG.list_kdn}, KG_TYPE_TAG)
        nx.set_node_attributes(G, {node.unique_id: node.name for node in kG.list_kcn+kG.list_kdn}, KG_NAME_TAG)
        nx.set_node_attributes(G, {node.unique_id: node.is_fwd for node in kG.list_kcn}, "is_fwd")
        nx.set_node_attributes(G, {node.unique_id: False for node in kG.list_kdn}, "is_fwd")
        nx.set_node_attributes(G, {node.unique_id: "bwd" if not node.main_code else
                                                'loss' if node.main_code[0] == 'loss' else
                                                node.main_code[1].func.id for node in kG.list_kcn}, "func")
        nx.set_node_attributes(G, {node.unique_id: node.info.tsize for node in kG.list_kdn}, "tsize")
        nx.set_node_attributes(G, {node.unique_id: node.info.memsize for node in kG.list_kdn}, KG_SIZE_TAG)
        nx.set_node_attributes(G, {node.unique_id: 100 for node in kG.list_kcn}, KG_TIME_TAG)

        if name_to_listid:
            nx.set_node_attributes(G, {node.unique_id: name_to_listid[node.name] for node in kG.list_kcn+kG.list_kdn}, "list_id")
        try:
            print("cycles", nx.find_cycle(G, orientation="original"))
            assert False
        except nx.exception.NetworkXNoCycle as e:
            pass
        return G
    
    @staticmethod
    def dc_nx_to_c_nx(dcG):
        G = dcG.copy()
        
        # set size/time/kdn for cn
        size_dict = {cn: sum([G.nodes[dn][KG_SIZE_TAG] for dn in dcG.successors(cn)]) for cn in [n for n, ntype in dcG.nodes.data(KG_TYPE_TAG) if ntype==KG_TYPE_COMPUTE_TAG]}
        nx.set_node_attributes(G, size_dict, name=NX_SIZE_TAG)
        kdn_dict = {cn: [G.nodes[dn]['kdn'] for dn in dcG.successors(cn)] for cn in [n for n, ntype in dcG.nodes.data(KG_TYPE_TAG) if ntype==KG_TYPE_COMPUTE_TAG]}
        nx.set_node_attributes(G, kdn_dict, name='kdn')
        time_dict = nx.get_node_attributes(G, KG_TIME_TAG)
        nx.set_node_attributes(G, time_dict, name=NX_TIME_TAG)
        
        # convert data output of odd kcn to kcn
        odd_kcn = kcn_with_mul_outdegree(G)
        for cn, dns in odd_kcn.items():
            for dn in dns:
                G.nodes[dn][KG_TYPE_TAG] = KG_TYPE_COMPUTE_TAG
                G.nodes[dn][KG_NAME_TAG] += '_extraction'
                G.nodes[dn]['is_fwd'] = G.nodes[cn]['is_fwd']
                G.nodes[dn][NX_SIZE_TAG] = G.nodes[dn][KG_SIZE_TAG]
                G.nodes[dn][NX_TIME_TAG] = 100
                G.nodes[dn]['kcn'] = None
        
        # add cn-to-cn edges
        new_edges = flatten([
            list(itertools.product(dcG.predecessors(n), dcG.successors(n))) 
            for n, ntype in dcG.nodes.data(KG_TYPE_TAG) if ntype==KG_TYPE_DATA_TAG])
        G.add_edges_from(new_edges)

        # remove dn
        G.remove_nodes_from([n for n, ntype in dcG.nodes.data(KG_TYPE_TAG) if ntype==KG_TYPE_DATA_TAG])
        
        return G



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
    phase1 = phase1 if B else False
    topo_order = topo_order if topo_order else list(nx.topological_sort(G))
    if max(G.nodes) >= G.number_of_nodes():
        label_mapping = {old: topo_order.index(old) for old in G.nodes}
        G = nx.relabel_nodes(G, label_mapping)
        topo_order = [label_mapping[v] for v in topo_order]
    params = copy.copy(locals())
    if C == "max": C = G.number_of_nodes()
    log_file = str(uuid.uuid4())
    sys.stdout.flush()

    if use_gcd: 
        G, mems_gcd, cpus_gcd = gcd_process(G)
    else:
        mems_gcd, cpus_gcd = 1, 1
    if B: B = math.ceil(B/mems_gcd)
    if L: L = math.ceil(L/mems_gcd)
    if R: R = math.ceil(R/cpus_gcd)
    
    nodes = sorted(G.nodes)
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
        tau_cpu = model.NewIntVar(0, max(sum(node_duration.values())*C, R if R else 0), 'tau_cpu')
        tau_mem = model.NewIntVar(0, max(sum(node_size.values())*C, B if B else 0), 'tau_mem')
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
        if objective != "min_runtime" and R:
            print("phase 1 aadding R")
            model.Add(tau_cpu >= schedule_duration)
            model.Add(tau_cpu >= R)
        if objective != "min_footprint" and B:
            print("phase 1 aadding B")
            model.Add(tau_mem >= mem_footprint)
            model.Add(tau_mem >= B)
        model.Minimize(tau_mem)
        status = solver.Solve(model)
        sys.stdout.flush()
        print('\n===========================================')
        print('Phase 1 Status = %s' % solver.StatusName(status))
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print(f"mem_budget = {B*mems_gcd if B else None}")
            print(f"time_budget = {R*cpus_gcd if R else None}")
            print(f"comm_budget = {L*mems_gcd if L else None}")
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
    if objective != "min_footprint" and B:
        model.Add(mem_footprint <= B)
    if objective != "min_communication" and L:
        model.Add(comm_load <= L)
    if objective != "min_runtime" and R:
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
        print(f"mem_budget = {B*mems_gcd if B else None}")
        print(f"time_budget = {R*cpus_gcd if R else None}")
        print(f"comm_budget = {L*mems_gcd if L else None}")
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
