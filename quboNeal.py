import neal
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes
from measures import mutual_info, entropy
from dwave.system import DWaveSampler, EmbeddingComposite
import dimod


data = load_diabetes(as_frame=True)
x, y = data.data, data.target
dataN = pd.DataFrame(x, columns=data.feature_names)

def create_graph(data: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for feature in data.columns:
        G.add_node(feature)
    for i in range(len(data.columns)):
        for j in range(i + 1, len(data.columns)):
            G.add_edge(data.columns[i], data.columns[j])

    return G


def plot_graph(G: nx.Graph) -> None:
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=900,
            node_color='dodgerblue', font_size=16, font_color='black',
            edgecolors='black', linewidths=1, edge_color='gray')
    plt.show()


def feature_related_network(mi_vector):
    G = nx.Graph()
    n = int(np.sqrt(len(mi_vector)))

    Adj = mi_vector.reshape((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if Adj[i, j] > 0.055:  ## nobody forbids to "prune" the graph based on MI threshold
                G.add_edge(i, j, weight=Adj[i, j])
    return G


def plot_mi_graph(G: nx.Graph, data: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 10))
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=900,
            node_color='dodgerblue', font_size=16, font_color='black',
            edgecolors='black', linewidths=1, edge_color='gray')

    #labels = {i: data.columns[i] for i in range(len(data.columns))}
    #nx.draw_networkx_labels(G, pos, labels=labels, font_size=16)
    plt.show()
    print("Graph with ", G.number_of_nodes(), "nodes and ", G.number_of_edges(), "edges")


##### THIS IS THE TRUE MAXIMAL CLIQUE QUBO FORMULATION #####
def qubomatrix(G, A=1, B=2):
    #: TODO: it is taken from "Finding Maximum Cliques on the D-Wave Quantum Annealer" -> link: https://d-nb.info/1162373091/34
    n = G.number_of_nodes()
    Q = np.zeros((n, n))

    G_bar = nx.complement(G)
    for i in range(n):
        Q[i, i] = -A
    for u, v in G_bar.edges():
        Q[u, v] += B
        Q[v, u] += B
    return Q

#print(qubomatrix(feature_related_network(mi_vector), mi_vector))
def solve_qubo(Q: np.ndarray,
               T_range: tuple[float, float],
               reps: int,
               sweeps: int) -> tuple[dimod.SampleSet, dict]:
    B_range = (1 / T_range[0], 1 / T_range[1])
    bqm_dict = {(i, j): float(Q[i, j]) for i in range(Q.shape[0]) for j in range(Q.shape[1])}
    bqm = dimod.BinaryQuadraticModel(bqm_dict, vartype="BINARY")
    sampler = neal.SimulatedAnnealingSampler()

    sampleset = sampler.sample(bqm,
                                         beta_range=B_range,
                                         num_reads=reps,
                                         num_sweeps=sweeps,
                                         beta_schedule_type='geometric')
    best_sample = sampleset.first.sample
    return sampleset, best_sample


def draw_solution(G: nx.Graph, solution):
    pos = nx.circular_layout(G)
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, with_labels=True, node_size=900,
            node_color='dodgerblue', font_size=16, font_color='black',
            edgecolors='black', linewidths=1, edge_color='gray')

    solution_nodes = [node for node, value in solution.items() if value == 1]
    nx.draw_networkx_nodes(G, pos, nodelist=solution_nodes, node_color='tab:orange', node_size=900,
                           edgecolors='black', linewidths=1,)

    plt.show()

#draw_solution(feature_related_network(mi_vector), sol)
#adj_matrix = np.array([
    #[0, 1, 0, 0, 1],
    #[0, 0, 1, 0, 1],
    #[1, 0, 0, 0, 0],
    #[0, 0, 1, 0, 1],
    #[1, 1, 0, 1, 0]
    #])

#G = nx.from_numpy_array(adj_matrix)
list_mi = []
x_arr = dataN.to_numpy()
for i in range(len(dataN.columns)):
    for j in range(len(dataN.columns)):
        mi_dummy = mutual_info(x_arr[:, i], x_arr[:, j], bins=30)
        mi_dummy_norm = 2*mi_dummy / (entropy(x_arr[:, i], bins=30) + entropy(x_arr[:, j], bins=30))
        list_mi.append(mi_dummy_norm)
list_mi_arr = np.asarray(list_mi)

G = feature_related_network(list_mi_arr)
G_bar = nx.complement(G)
#nx.draw(G, nx.circular_layout(G), with_labels=True, node_size=900,
#        node_color='dodgerblue', font_size=16, font_color='black',
#        edge_color='gray')

Q = qubomatrix(G)

def dwave_solve(Q: np.ndarray, TOKEN: str) -> tuple[dimod.SampleSet, dict]:

    bqm_dict = {(i, j): float(Q[i, j]) for i in range(Q.shape[0]) for j in range(Q.shape[1])}
    bqm = dimod.BinaryQuadraticModel(bqm_dict, vartype="SPIN")
    sampler = EmbeddingComposite(DWaveSampler())
    sampleset = sampler.sample(bqm)
    best_sample = sampleset.first.sample
    return sampleset, best_sample

sol_set, sol = solve_qubo(Q, T_range=(10, 0.20), reps=100, sweeps=20)
sol_set_df = sol_set.to_pandas_dataframe()
#print(sol_set_df.to_numpy())
grouped = sol_set_df.groupby(sol_set_df.columns[:-2].tolist()).agg({
    'energy': 'first',
    'num_occurrences': 'count'
}).reset_index()


#draw_solution(G, sol)
solution_states = []
for i in range(len(grouped)):
    state = grouped.iloc[i, :].tolist()[:10]
    solution_states.append(state)


label_states = [''.join(str(int(x)) for x in state) for state in solution_states]

infoDict = {"solution": label_states,
            "energy": grouped['energy'].tolist(),
            "num_occurrences": grouped['num_occurrences'].tolist()}


plt.bar(infoDict["solution"], infoDict['num_occurrences'], color='dodgerblue',
        edgecolor='black',)
plt.xlabel('Sample Index')
plt.xticks(infoDict["solution"], rotation=45, ha='right')
plt.ylabel('Frequency')
plt.show()

print(infoDict)
