import pandas as pd
from sklearn.datasets import load_diabetes
import numpy as np
import networkx as nx
import neal
import matplotlib.pyplot as plt
import dimod
import time
import os



def entropy(X: np.ndarray, bins: int) -> float:
    """
    Compute the Shannon entropy of a variable X.
    :param X: (np.ndarray) The input variable;
    :return: (float) The Shannon entropy of the input variable.
    """
    binned_dist = np.histogram(X, bins)[0]
    probs = binned_dist / np.sum(binned_dist)
    probs = probs[np.nonzero(probs)]
    entropy_ = - np.sum(probs * np.log(probs))
    return entropy_


def joint_entropy(X: np.ndarray, Y: np.ndarray, bins: int) -> float:
    """
    Compute the joint Shannon entropy between two variables.
    Each variable could be two features expressed as arrays or a feature and a label.
    :param X: (np.ndarray) The first input variable (usually feature);
    :param Y: (np.ndarray) The second input variable (feature or label);
    :return: (float) The joint Shannon entropy of the two variables.
    """
    binned_dist = np.histogram2d(X, Y, bins)[0]
    probs = binned_dist / np.sum(binned_dist)
    probs = probs[np.nonzero(probs)]
    joint_e = - np.sum(probs * np.log(probs))
    return joint_e


def conditional_joint_entropy(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, bins: int) -> float:
    """
    Compute the joint Shannon entropy between three variables.
    :param X: (np.ndarray) The first input variable;
    :param Y: (np.ndarray) The second input variable;
    :param Z: (np.ndarray) The third input variable;
    :param bins: (int) The number of bins for discretization.
    :return: (float) The joint Shannon entropy of the three variables.
    """
    binned_dist = np.histogramdd((X, Y, Z), bins=bins)[0]
    probs = binned_dist / np.sum(binned_dist)
    probs = probs[np.nonzero(probs)]
    joint_e = - np.sum(probs * np.log(probs))
    return joint_e


def conditional_mutual_inf(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, bins: int) -> float:
    """
    Compute the conditional mutual information I(X;Y | Z).
    :param X: (np.ndarray) The first input variable;
    :param Y: (np.ndarray) The second input variable;
    :param Z: (np.ndarray) The conditioning variable;
    :param bins: (int) The number of bins for discretization.
    :return: (float) The conditional mutual information I(X;Y | Z).
    """
    HXZ = joint_entropy(X, Z, bins)
    HYZ = joint_entropy(Y, Z, bins)
    HXYZ = conditional_joint_entropy(X, Y, Z, bins)

    return HXZ + HYZ - HXYZ


def mutual_info(X: np.ndarray, Y: np.ndarray, bins: int) -> float:
    """
    Compute the mutual information between two input variables.
    :param X: (np.ndarray) The first input variable (usually feature);
    :param Y: (np.ndarray) The second input variable (feature or label);
    :return: The mutual information between the two variables.
    """
    HX = entropy(X, bins)
    HY = entropy(Y, bins)
    HXHY = joint_entropy(X, Y, bins)
    H = HX + HY - HXHY
    return H


def plot_network():
    seed = 8888
    features = x.columns
    np.random.seed(seed)
    G = nx.Graph()
    G.add_nodes_from(range(len(features)))
    G.add_edges_from([(i, j) for i in range(len(features)) for j in range(i+1, len(features))])
    pos = nx.circular_layout(G)
    mapping = {i: features[i] for i in range(len(features))}
    nx.relabel_nodes(G, mapping, copy=False)
    nx.draw(G, with_labels=True)
    plt.show()


def gaussian_kernel(x, xi, h):
    """Gaussian kernel in 1D."""
    return (1 / (np.sqrt(2 * np.pi) * h)) * np.exp(-0.5 * ((x - xi) / h) ** 2)

def parzen_prob_1d(data: np.ndarray, query_x: float, h: float = 0.1):
    N = len(data)
    return (1 / N) * np.sum([gaussian_kernel(query_x, xi, h) for xi in data])

def estimate_row_densities(df: pd.DataFrame, row_idx: int, h: float = 0.1):
    row = df.iloc[row_idx].values.astype(float)
    return np.array([parzen_prob_1d(row, x, h=h) for x in row])


data = load_diabetes(as_frame=True)
x, y = data.data, data.target

dataN = pd.DataFrame(x, columns=data.feature_names)
x_arr = x.to_numpy()
list_mi = []
for i in range(len(dataN.columns)):
    for j in range(len(dataN.columns)):
        mi_dummy = mutual_info(x_arr[:, i], x_arr[:, j], bins=30)
        mi_dummy_norm = 2*mi_dummy / (entropy(x_arr[:, i], bins=30) + entropy(x_arr[:, j], bins=30))
        list_mi.append(mi_dummy_norm)
list_mi_arr = np.asarray(list_mi)
#print(list_mi_arr.reshape((len(data.columns), len(data.columns))))
#for i in range()
#plot_network()
#parzen_p1d = estimate_row_densities(x, row_idx=0, h=0.1)
#print(parzen_p1d)

new_adj = list_mi_arr.reshape((10,10)) - np.eye(N=10, M=10)


"""
G_weighted = nx.from_numpy_array(new_adj, create_using=nx.Graph)
pos = nx.spring_layout(G_weighted)
#nx.draw(G_weighted, pos, with_labels=True, node_color='lightblue', font_size=10, font_color='black')
plt.imshow(new_adj, cmap="viridis")
plt.colorbar(label='MI', ticks=list(data.feature_names))
plt.show()
"""
penalty_lambda = 0.25
num_reads = 100

class qubosolver:
    def __init__(self,
                 W: np.ndarray,
                 b: int,
                 penalty_lambda: float,
                 num_reads: int) -> None:
        """
        Class to solve the QUBO problem. It takes a graph and the parameters for the QUBO problem.
        :param W: The weighted graph matrix.
        :param b: The number of nodes in the clique.
        :param penalty_lambda: The penalty lambda to be used as the constraint.
        :param num_reads: The number of reads to be used in the QUBO solver.

        """
        self.W = W
        self.b = b
        self.penalty_lambda = penalty_lambda
        self.num_reads = num_reads

    def qubo_formulation(self) -> np.ndarray:
        """
        QUBO formulation of the problem.
        :param self:
        :return: Q: (np.ndarray) The QUBO matrix.
        """
        n = self.W.shape[0]
        Q = self.W

        # TODO: the formula for the WMC mapped as QUBO problem is (same last paper):
        # TODO: max ∑i∈V node_weight_i x_i + ∑i∈V ∑j∈E edge_weight_ij x_i*x_j + \penalty * ∑i∈V (x_i - b)**2

        for i in range(n):
            Q[i, i] = Q[i, i] + self.penalty_lambda * (1 - 2 * self.b)
        for i in range(n):
            for j in range(i + 1, n):
                Q[i, j] = - Q[i, j] + 2 * self.penalty_lambda
                Q[j, i] = Q[i, j]

        return Q

    def solve(self) -> dict:
        """
        Solve the QUBO problem using the dimod library.
        :return: solution: (dict) The solution of the QUBO problem.
        """
        qubo_matrix = self.qubo_formulation()
        qubo_dict = {(i, j): float(qubo_matrix[i, j]) for i in range(qubo_matrix.shape[0]) for j in
                     range(qubo_matrix.shape[1])}

        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo_dict)
        sampler = dimod.SimulatedAnnealingSampler()
        start = time.time()
        results = sampler.sample(bqm, num_reads=self.num_reads)
        best_sample = results.first
        solution = {k: int(v) for k, v in best_sample.sample.items()}
        end = time.time()
        print("Elapsed time to get the solution:", np.subtract(end, start))
        print("Best sample:", solution)
        print("Best energy:", best_sample.energy)

        return solution

    def solution_draw(self) -> None:
        """
        Draw the solution of the problem, i.e., the maximal clique.
        :param self;
        :return: None
        """
        G = nx.from_numpy_array(self.W, create_using=nx.Graph)
        path_to_save = os.path.join(os.path.dirname(__file__), "imgs")
        sol = self.solve()
        np.random.seed(3)
        pos = nx.circular_layout(G)
        edge_labels = nx.get_edge_attributes(G, "weight")
        node_labels = nx.get_node_attributes(G, 'weight')
        nx.draw(G, pos, with_labels=True, node_color="royalblue", alpha=0.75, labels=node_labels)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black')
        clique_nodes = [node for node, val in sol.items() if val == 1]
        other_nodes = [node for node in G.nodes if node not in clique_nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=clique_nodes, node_color="orangered", label="Clique")
        nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, node_color="royalblue", label="Other")

        #plt.savefig(path_to_save + "/" + fr"$GFS_WMC_lamb{self.penalty_lambda}_b{self.b}$.png")
        #features = x.columns
        #mapping = {i: features[i] for i in range(len(features))}
        #nx.relabel_nodes(G, mapping, copy=False)
        plt.show()

    def solution_drawClique(self) -> None:
        """
        Draw the solution of the problem, highlighting the clique nodes and edges.
        """
        G = nx.from_numpy_array(self.W, create_using=nx.Graph)
        sol = self.solve()

        np.random.seed(3)
        pos = nx.circular_layout(G)

        clique_nodes = [node for node, val in sol.items() if val == 1]
        other_nodes = [node for node in G.nodes if node not in clique_nodes]
        clique_edges = [(u, v) for u, v in G.edges if u in clique_nodes and v in clique_nodes]
        other_edges = [(u, v) for u, v in G.edges if (u, v) not in clique_edges and (v, u) not in clique_edges]

        nx.draw_networkx_nodes(G, pos, nodelist=clique_nodes, node_color="orangered", label="Clique")
        nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, node_color="royalblue", label="Other")
        nx.draw_networkx_edges(G, pos, edgelist=clique_edges, edge_color="orangered", width=2)
        nx.draw_networkx_edges(G, pos, edgelist=other_edges, edge_color="lightgray", alpha=0.75)
        nx.draw_networkx_labels(G, pos)

        plt.legend([fr"$\lambda$= {self.penalty_lambda}"])
        plt.show()


def upper_tri_indexing(A):
    m = A.shape[0]
    r,c = np.triu_indices(m,1)
    return A[r,c]

### Min val = 0.016, max = 0.395, mean = 0.184
"""print(upper_tri_indexing(new_adj).max(), upper_tri_indexing(new_adj).min(),
      upper_tri_indexing(new_adj).mean())"""

threshold = 0.21
for i in range(new_adj.shape[0]):
    for j in range(new_adj.shape[1]):
        if new_adj[i, j] < threshold:
            new_adj[i, j] = 0.0
        else:
            new_adj[i, j] = new_adj[i, j]


def plotGraph(A):
    G = nx.from_numpy_array(A, create_using=nx.Graph)
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='royalblue', font_size=10, font_color='black')
    plt.show()

#plotGraph(new_adj)

G_weighted = nx.from_numpy_array(new_adj, create_using=nx.Graph)

qubo = qubosolver(
    W=new_adj,
    b=5,
    penalty_lambda=penalty_lambda,
    num_reads=num_reads
)

qubo.solution_drawClique()