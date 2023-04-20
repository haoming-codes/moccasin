import networkx as nx
import json
from util import get_mem_from_schedule
import math
from cp import remat

G_paths = [
    "data/random_layered_n100_w0.27_nlv0.75_ed0.2_scd0.14.json",
    "data/random_layered_n250_w0.43_nlv0.75_ed0.2_scd0.14.json",
    "data/random_layered_n500_w0.36_nlv0.75_ed0.2_scd0.14.json",
    "data/random_layered_n1000_w0.31_nlv0.75_ed0.2_scd0.14.json",
    "data/ResNet50 (MLSys)_256_(224, 224, 3)_train_nx.json",
    "data/fcn_8_vgg (MLSys)_32_(416, 608, 3)_train_nx.json",
]
mem_mults = [.8, .9]

for G_path in G_paths:
    with open(G_path, "r") as f:
        d = f.read()
        G = nx.node_link_graph(json.loads(d), multigraph=False, directed=True, attrs={"source":'0', "target":'1'} if 'random_layered' in G_path else None)
        if G.graph['name'].startswith('random_layered'):
            for v, data in G.nodes(data=True):
                data["cost_ram"] = data.pop("out_cost")
                data["cost_cpu"] = data.pop("duration")
    
    topo_order = G.graph['order'] if 'order' in G.graph else range(G.number_of_nodes())#list(nx.topological_sort(G))
    topo_mem, _ = get_mem_from_schedule(G, topo_order)
    for mul in mem_mults:
        B = math.ceil(topo_mem*mul)
        result = remat(
            G=G, 
            B=B,
            C=2,
            timeout=3600*(1. if G.number_of_nodes() >= 500 else .5), 
            discretized=True, 
            reservoir=True, 
            use_end_global=False,
            use_stage=True,
            topo_order=topo_order,
            phase1=True,
            use_interval_length_domain=True,
            use_gcd=True,
            reservoir_option=2,
            log_dir="output/icml")
