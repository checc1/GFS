import dimod
import networkx as nx
from qubo_utils import QuboProblem
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from measures import mutual_info, entropy
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from collections import Counter

matplotlib.use("TkAgg")


data = load_diabetes(as_frame=True)
x, y = data.data, data.target
b = 50
dataN = pd.DataFrame(x, columns=data.feature_names)
x_arr = x.to_numpy()


def plotGraph(A):
    G = nx.from_numpy_array(A, create_using=nx.Graph)
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='dodgerblue',
            font_size=10, font_color='black',
            edge_color='gray', node_size=500,
            font_weight='medium',
            edgecolors='black', linewidths=1)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plotGraphGradient(G):
    pos = nx.circular_layout(G)
    edges = G.edges(data=True)
    weights = [d['weight'] for (_, _, d) in edges]
    norm = mcolors.Normalize(vmin=min(weights), vmax=max(weights))
    cmap = cm.coolwarm
    edge_colors = [cmap(norm(w)) for w in weights]
    nx.draw_networkx_nodes(G, pos, node_color='dodgerblue', node_size=800,
                           edgecolors='black', linewidths=1)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=1.5)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')
    #sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    #sm.set_array([])
    #plt.colorbar(sm, label="Edge Weight Intensity")

    plt.axis('off')
    plt.tight_layout()
    plt.show()


def create_adjacency(data: pd.DataFrame, x_arr:np.ndarray, edge_weight: str, pruning: float) -> np.ndarray:
    list_value = []

    if edge_weight == "pearson":
        for i in range(len(data.columns)):
            for j in range(len(data.columns)):
                #pearson_dummy = np.round(pearson(x_arr[:, i], x_arr[:, j]), 3)
                pearson_dummy = np.corrcoef(x_arr[:, i], x_arr[:, j])
                list_value.append(pearson_dummy[0,1])
        list_value = np.array(list_value)
        min_val, max_val = list_value.min(), list_value.max()
        if pruning > max_val or pruning < min_val:
            raise ValueError("pruning must be between {} and {}".format(min_val, max_val))
        else:
            for i in range(len(list_value)):
                if list_value[i] < pruning:
                    list_value[i] = 0.
        adj = list_value.reshape((len(data.columns), len(data.columns))) - np.eye(N=len(data.columns), M=len(data.columns))
        for i in range(adj.shape[0]):
            adj[i, i] = 0
        return adj
    elif edge_weight == "mutual_info":
        for i in range(len(data.columns)):
            for j in range(len(data.columns)):
                mi_dummy = mutual_info(x_arr[:, i], x_arr[:, j], bins=b)
                mi_dummy_norm = 2 * mi_dummy / (entropy(x_arr[:, i], bins=b) + entropy(x_arr[:, j], bins=b))
                list_value.append(mi_dummy_norm)
        list_value = np.array(list_value)
        min_val, max_val = list_value.min(), list_value.max()
        if pruning > max_val or pruning < min_val:
            raise ValueError("pruning must be between {} and {}".format(min_val, max_val))
        else:
            for i in range(len(list_value)):
                if list_value[i] < pruning:
                    list_value[i] = 0.
            adj = list_value.reshape((len(data.columns), len(data.columns))) - np.eye(N=len(data.columns),M=len(data.columns))
            return adj
    else:
        raise ValueError("edge_weight must be either 'pearson' or 'mutual_info'")

new_adj = create_adjacency(dataN,
                           x_arr,
                           "mutual_info",
                           pruning=0.05)

#print(new_adj)
#plotGraph(new_adj)

G_weighted = nx.from_numpy_array(new_adj, create_using=nx.Graph)
#plotGraphGradient(G_weighted)
#qubo.draw_solution(G_weighted, sol)


class Solution:
    def __init__(self, sampleset: dimod.SampleSet):
        self.sampleset = sampleset

    def to_dict(self) -> dict:
        sol_set_df = self.sampleset.to_pandas_dataframe()
        grouped = sol_set_df.groupby(sol_set_df.columns[:-2].tolist()).agg({
            'energy': 'first',
            'num_occurrences': 'count'
        }).reset_index()

        solution_states = []
        for i in range(len(grouped)):
            state = grouped.iloc[i, :].tolist()[:10]
            solution_states.append(state)

        label_states = [''.join(str(int(x)) for x in state) for state in solution_states]

        infoDict = {"solution": label_states,
                    "energy": grouped['energy'].tolist(),
                    "num_occurrences": grouped['num_occurrences'].tolist()}
        return infoDict

    def plot(self):
        infoDict = self.to_dict()
        fig, ax = plt.subplots()
        ax.bar(infoDict["solution"], infoDict['num_occurrences'], color='dodgerblue',
                edgecolor='black', label="100 num_reads")

        #ax.yaxis.set_minor_locator(AutoMinorLocator())

        ax.tick_params(which='both', width=1)
        ax.tick_params(which='major', length=7)
        ax.legend()
        ax.set_xlabel('Sample Index')
        ax.set_xticks(range(len(infoDict["solution"])), infoDict["solution"], rotation=45, ha='right')
        ax.set_ylabel('Frequency')
        plt.show()


list_of_dict = []
dict_w_solution = {"n_read": [], "solution": [], "frequency": [], "energy": []}

for r in range(1, 1010, 10):
    qubo = QuboProblem(G_weighted)
    sampleset, sol = qubo.solve_qubo(T_range=(10, 0.4), reps=r, sweeps=100, T_scheduling='linear')
    sol_samples = Solution(sampleset).to_dict()
    list_of_dict.append(sol_samples)

    max_idx = np.argmax(sol_samples['num_occurrences'])
    most_freq_sol = sol_samples['solution'][max_idx]
    freq = sol_samples['num_occurrences'][max_idx]
    dict_w_solution['n_read'].append(r)
    dict_w_solution['solution'].append(most_freq_sol)
    dict_w_solution['frequency'].append(freq)
    dict_w_solution['energy'].append(sol_samples['energy'][max_idx])

energy_v, sol_v, freq_v = [], [], []
keys = list(list_of_dict[0].keys())

for i in range(len(list_of_dict)):
    sol_v.append(list_of_dict[i][keys[0]])
    energy_v.append(list_of_dict[i][keys[1]])
    freq_v.append(list_of_dict[i][keys[2]])


most_frequent_sol_freq, most_frequent_sol = [], []

for x, sol in zip(freq_v, sol_v):
    max_idx = np.argmax(x)
    most_frequent_sol_freq.append(x[max_idx])
    most_frequent_sol.append(sol[max_idx])


count_sol = Counter(most_frequent_sol)
#print(most_frequent_sol)


def plot() -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(range(len(most_frequent_sol_freq)), most_frequent_sol_freq, marker='o', color="dodgerblue",
                label="most frequent solution",edgecolor='black', linewidth=0.5, zorder=10,
                linewidths=0.8)
    ax[1].bar(count_sol.keys(), count_sol.values(), color="dodgerblue", edgecolor='black', linewidth=0.5, zorder=10,)
    ax[0].set_xlabel("Num reads")
    ax[1].set_xlabel("State solution")
    ax[1].set_xticks(range(len(count_sol.keys())), count_sol.keys(), rotation=45, ha='right')
    plt.show()

def plot_solutions(sols: list[str], freqs: list[int]):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(sols, freqs, color="dodgerblue", edgecolor='black')
    ax.set_xlabel("Solution State")
    ax.set_ylabel("Frequency")
    ax.set_xticks(range(len(sols)), sols, rotation=45, ha='right')
    plt.show()


all_sols_keys = list(count_sol.keys())

Sols = {sol: [] for sol in count_sol.keys()}

for i in range(len(list_of_dict)):
    for sol in Sols.keys():
        if sol in list_of_dict[i]['solution']:
            idx = list_of_dict[i]['solution'].index(sol)
            freq = list_of_dict[i]['num_occurrences'][idx]
        else:
            freq = 0
        Sols[sol].append(freq)


fig, ax = plt.subplots(1, 2)
for sol, freqs in Sols.items():
    ax[0].scatter(dict_w_solution['n_read'], freqs, marker='o', label=sol, edgecolor='black', s=50, alpha=0.7)
ax[0].set_xlabel(r"$N$")
ax[0].set_ylabel("Frequency")
ax[0].legend()
plt.show()