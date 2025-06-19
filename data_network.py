import pandas as pd
from sklearn.datasets import load_diabetes
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt


data = load_diabetes(as_frame=True)
x, y = data.data, data.target

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

if __name__ == "__main__":
    x_arr = x.to_numpy()

    parzen_p1d = estimate_row_densities(x, row_idx=0, h=0.1)
    print(parzen_p1d)
