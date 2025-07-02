import networkx as nx
from qubo_utils import QuboProblem
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from measures import mutual_info, entropy, pearson
import matplotlib.cm as cm
import matplotlib.colors as mcolors


data = load_diabetes(as_frame=True)
x, y = data.data, data.target
b = 50
dataN = pd.DataFrame(x, columns=data.feature_names)
x_arr = x.to_numpy()
list_mi = []

for i in range(len(dataN.columns)):
    for j in range(len(dataN.columns)):
        mi_dummy = mutual_info(x_arr[:, i], x_arr[:, j], bins=b)
        mi_dummy_norm = 2*mi_dummy / (entropy(x_arr[:, i], bins=b) + entropy(x_arr[:, j], bins=b))
        list_mi.append(mi_dummy_norm)
        #list_mi.append(mi_dummy)
list_mi_arr = np.asarray(list_mi)

list_pearson = []
#list_pearsonNp = []
for i in range(len(dataN.columns)):
    for j in range(len(dataN.columns)):
        pearson_dummy = np.round(pearson(x_arr[:, i], x_arr[:, j]),3)
        #pearson_Np = np.corrcoef(x_arr[:, i], x_arr[:, j])
        list_pearson.append(pearson_dummy[0, 1])
        #list_pearsonNp.append(float(pearson_Np[0, 1]))

list_pearson_arr = np.asarray(list_pearson)

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


new_adj = list_mi_arr.reshape((10,10)) - np.eye(N=10, M=10)
for i in range(new_adj.shape[0]):
    new_adj[i, i] = 0

plotGraph(new_adj)

for i in range(new_adj.shape[0]):
    for j in range(i+1, new_adj.shape[1]):
        if new_adj[i, j] < 0.1:
            new_adj[i, j] = 0
            new_adj[j, i] = 0


G_weighted = nx.from_numpy_array(new_adj, create_using=nx.Graph)
plotGraphGradient(G_weighted)
qubo = QuboProblem(G_weighted)
sampleset, sol = qubo.solve_qubo(T_range=(10, 0.4), reps=100, sweeps=100, T_scheduling='linear')
qubo.draw_solution(G_weighted, sol)