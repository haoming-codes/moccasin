import networkx as nx
import json
from util import get_mem_from_schedule
import math
from cp import remat
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--graph_path', type=str, required=True)
parser.add_argument('-m', '--mem_multiplier', type=float, required=True)
parser.add_argument('-o', '--output_dir', type=str, required=False, default="output/icml")
parser.add_argument('-t', '--solver_timeout', type=int, required=False, default=3600)
parser.add_argument('-c', '--num_remat', type=int, required=False, default=2)
args = parser.parse_args()

with open(args.graph_path, "r") as f:
    d = f.read()
    G = nx.node_link_graph(json.loads(d), multigraph=False, directed=True, attrs={"source":'0', "target":'1'} if 'random_layered' in args.graph_path else None)
    if G.graph['name'].startswith('random_layered'):
        for v, data in G.nodes(data=True):
            data["cost_ram"] = data.pop("out_cost")
            data["cost_cpu"] = data.pop("duration")
    
topo_order = G.graph['order'] if 'order' in G.graph else list(nx.topological_sort(G))
topo_mem, _ = get_mem_from_schedule(G, topo_order)

B = math.ceil(topo_mem*args.mem_multiplier)
result = remat(
    G=G, 
    B=B,
    C=args.num_remat,
    timeout=args.solver_timeout*(1. if G.number_of_nodes() >= 500 else .5), 
    discretized=True, 
    reservoir=True, 
    use_end_global=False,
    use_stage=True,
    topo_order=topo_order,
    phase1=True,
    use_interval_length_domain=True,
    use_gcd=True,
    reservoir_option=2,
    log_dir=args.output_dir)
