import os
import pickle
import networkx as nx
import random
import json
import math
import copy

    
def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
    Licensed under Creative Commons Attribution-Share Alike 
    
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    G: the graph (must be a tree)
    
    root: the root node of current branch 
    - if the tree is directed and this is not given, 
      the root will be found and used
    - if the tree is directed and this is given, then 
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given, 
      then a random choice will be used.
    
    width: horizontal space allocated for this branch - avoids overlap with other branches
    
    vert_gap: gap between levels of hierarchy
    
    vert_loc: vertical location of root
    
    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''
    
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

            
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

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
    print(f"mems_gcd = {mems_gcd}, cpus_gcd = {cpus_gcd}")
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