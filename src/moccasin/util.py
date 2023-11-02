import os
import pickle
import networkx as nx
import random
import json
import math
import copy

def gen_unet_graph(forward_node_count):
    """
    gen_linear_graph will generate linear-style graphs like VGG and AlexNet.
    Method returns forward and backward graphs. Pass cost_ram and cost_cpu as kwargs.
    :param forward_node_count: number of forward (not backward nodes)
    :return: Graph object containing linear graph
    """
    G = nx.DiGraph()
    for i in range(forward_node_count * 2 + 1):
        G.add_node("node{}".format(i), cost_cpu=3, cost_ram=1, backward=None)
        if i > 0:
            G.add_edge("node{}".format(i), "node{}".format(i - 1))

    for i in range(forward_node_count):
        corresponding_bwd = (forward_node_count * 2) - i
        G.add_edge("node{}".format(corresponding_bwd), "node{}".format(i))
    
    G = nx.convert_node_labels_to_integers(G, label_attribute="name")
    # nx.set_node_attributes(G, 3, "name")
    # nx.set_node_attributes(G, 3, "cost_cpu")
    # nx.set_node_attributes(G, 1, "cost_ram")
    # nx.set_node_attributes(G, None, "backward")
    G.graph['name'] = f"unet_{forward_node_count}"
    return G

def gen_star_graph(input_node_count):
    G = nx.DiGraph()
    G.add_node("node{}".format(input_node_count), cost_cpu=3, cost_ram=1, backward=None)
    for i in range(input_node_count):
        G.add_node("node{}".format(i), cost_cpu=3, cost_ram=1, backward=None)
        G.add_edge("node{}".format(i), "node{}".format(input_node_count))
    G = nx.convert_node_labels_to_integers(G, label_attribute="name")
    G.graph['name'] = f"star_{input_node_count}"
    return G

def get_target_node(G):
    zero_out_degrees = [node for (node, val) in G.out_degree() if val == 0]
    # assert len(zero_out_degrees) == 1, f"zero_out_degrees: {zero_out_degrees}"
    # target_node = zero_out_degrees[0]
    return zero_out_degrees

def maybe_dump(pkl, fname):
    if os.path.exists(fname): 
        with open(fname, 'rb') as handle:
            result = pickle.load(handle)
        if not isinstance(result, list):
            result = [result]
    else:
        result = []
    result.append(pkl)
    with open(fname, 'wb') as handle:
        pickle.dump(result, handle)

def max_degree_ram(G):
    """compute minimum memory needed for any single node (ie inputs and outputs)"""
    # vfwd = [v for v in self.v if v not in self.backward_nodes]
    vfwd = G.nodes
    return max([sum([G.nodes[u]["cost_ram"] for u in G.predecessors(v)]) + G.nodes[v]["cost_ram"] for v in vfwd])

def save_graph(G):
    j = json.dumps(nx.node_link_data(G))
    with open(f"output/graphs/{G.graph['name']}_nx.json", "w") as outfile:
        outfile.write(j)
    print("saved graph")

def get_mem_from_schedule(G, schedule):
    G_aug = nx.DiGraph()
    for j, v in enumerate(schedule):
        G_aug.add_node((j, v), cost_cpu=G.nodes[v]["cost_cpu"], cost_ram=G.nodes[v]["cost_ram"])
        for u in G.predecessors(v):
            for i in reversed(range(0, j)):
                if schedule[i] == u:
                    G_aug.add_edge((i, u), (j, v))
                    break
    
    mem_footprint = []
    cpu_footprint = []
    for j, v in enumerate(schedule):
        in_mem = []
        for i, u in enumerate(schedule[:j]):
            for k, w in G_aug.successors((i, u)):
                if k >= j:
                    in_mem.append((i, u))
                    break
        # print()
        assert (j, v) not in in_mem
        in_mem.append((j, v))
        mem = sum([G_aug.nodes[node]["cost_ram"] for node in in_mem])
        mem_footprint.append(mem)
        cpu_footprint.append(G_aug.nodes[(j, v)]["cost_cpu"])

    return max(mem_footprint), sum(cpu_footprint)

def gcd_process(G):
    G = copy.deepcopy(G)
    mems = dict(nx.get_node_attributes(G, "cost_ram")).values()
    cpus = dict(nx.get_node_attributes(G, "cost_cpu")).values()
    mems_gcd = math.gcd(*mems)
    cpus_gcd = math.gcd(*cpus)
    for v in G.nodes:
        G.nodes[v]["cost_ram"] //= mems_gcd
        G.nodes[v]["cost_cpu"] //= cpus_gcd
    return G, mems_gcd, cpus_gcd

def flatten(l):
    for i in l:
        if isinstance(i, list):
            yield from flatten(i)
        else:
            yield i