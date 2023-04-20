import matplotlib.pyplot as plt
import pickle
import pandas as pd
from os import listdir
from os.path import isfile, join
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'

name_map = {
    "fcn_8_vgg (MLSys)_32_(416, 608, 3)_train": "VGG",
    "ResNet50 (MLSys)_256_(224, 224, 3)_train": "ResNet50",
    "random_layered_n100_w0.27_nlv0.75_ed0.2_scd0.14": "RL100",
    "random_layered_n250_w0.43_nlv0.75_ed0.2_scd0.14": "RL250",
    "random_layered_n500_w0.36_nlv0.75_ed0.2_scd0.14": "RL500",
    "random_layered_n1000_w0.31_nlv0.75_ed0.2_scd0.14": "RL1000",
}
settings = [
    "icml_original/",
]
graphs = set([])

path = "./output/"
dics = {setting: [] for setting in settings}
progressions = {setting: {} for setting in settings}
for setting in settings:
    # print(path+setting+"/"+f)
    onlyfiles = sorted([join(path, setting, f) for f in listdir(join(path, setting)) if isfile(join(path, setting, f))])
    print(onlyfiles)
    for file_name in onlyfiles:
        with open(file_name, "rb") as f:
            data = pickle.load(f)
        data = data[-1]
        for soln in data['solns'].values():
            if soln['objective'] == data['obj_val']:
                best_time = soln['time']
                break
        row = {
            'Graph': name_map[data['params']['G_name']], 
            'Budget': "{:.0%}".format(data['params']['B']/data['topo_mem']), 
            '$\Delta$ Makespan': (data['cpu_schedule'] - data['topo_cpu']) / data['topo_cpu'], 
            '$\Delta$ Footprint': (data['mem_schedule'] - data['topo_mem']) / data['topo_mem'], 
            'Time': best_time
        }
        dics[setting].append(row)
        # print("Graph: {:s}, Budget: {:.1f}, $\Delta$ Makespan: {:.2f}%, $\Delta$ Footprint: {:.2f}%, Time: {:.1f}".format(
        #     name_map[data['params']['G_name']], 
        #     data['params']['B']/data['topo_mem'], 
        #     (data['cpu_schedule'] - data['topo_cpu']) / data['topo_cpu'] * 100, 
        #     (data['mem_schedule'] - data['topo_mem']) / data['topo_mem'] * 100,
        #     best_time))
        progression = sorted([(soln['time'], ((soln['objective']  - data['topo_cpu']) / data['topo_cpu'])) for soln in data['solns'].values()],key=lambda x: x[0])
        if row['Graph'] not in progressions[setting]:
            graphs.add(row['Graph']) 
            progressions[setting][row['Graph']] = {}
        progressions[setting][row['Graph']][row['Budget']] = progression

for setting in settings:
    df_unindexed = pd.DataFrame(dics[setting])
    df = df_unindexed.set_index(['Graph', 'Budget'])
    print(df)
    formatters = {
        'Graph': lambda x : x,
        'Budget': lambda x : x,
        '$\Delta$ Makespan': lambda x : "{:.1%}".format(x),
        '$\Delta$ Footprint': lambda x : "{:.1%}".format(x),
        'Time': lambda x : "{:.1f}".format(x)
    }
    print(df.to_latex(
        formatters=formatters, 
        multirow=True,
        caption=f"Mocassin results ({setting})",
        label=f"tab:cp/{setting}"))


# budgets = ["80%", "90%"]
# fig, axs = plt.subplots(
#     len(name_map), len(budgets),
#     figsize=(1*len(name_map), 5*len(budgets)), 
#     # constrained_layout=True
# )

# for row, graph in enumerate(graphs):
#     for col, budget in enumerate(budgets):
#         for setting in settings:
#             if graph not in progressions[setting] or budget not in progressions[setting][graph]:
#                 continue
#             x = [x for x, y in progressions[setting][graph][budget]]
#             y = [y for x, y in progressions[setting][graph][budget]]
#             axs[row, col].plot(x, y, label=setting)
#         axs[row, col].set_title(f'G = {graph}, Budget = {budget}')

# for ax in axs.flat:
#     ax.set(xlabel='Time (s)', ylabel='$\Delta$ Makespan')

# handles, labels = axs[0,0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8)
# plt.tight_layout()
# # # Hide x labels and tick labels for top plots and y ticks for right plots.
# # for ax in axs.flat:
# #     ax.label_outer()

# plt.savefig("progression.pdf", format="pdf")
# plt.savefig("progression.png")

# print(data.keys())
