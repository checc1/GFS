import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import argparse
from utils import qubosolver



## The weights I have chosen for the graph EDGES are taken from a paper
#:TODO link -> https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=dd9044b6d843f77abfd65d7aba17fe94829551bc


def weighted_graph(dim: int,
                   node_weights: np.ndarray,
                   edge_weights: np.ndarray,
                   draw: bool = False) -> plt.show() or nx.Graph:
    """
    Create a weighted graph with the given dimension and draw it following a circular layout.
    The same seed is used to generate the graph, so it is reproducible. The list of weights
    is taken from a paper to verify the correctness of the weighted maximal clique algorithm.

    :param edge_weights:
    :param node_weights:
    :param dim: (int) The number of nodes in the graph;
    :param draw: (bool) If True, the graph is drawn;
    :return: (plt.show() or nx.Graph) The graph.
    """

    #node_weights = np.array([0.5, 0.2, 0.3, 0.6, 0.25])  ## these weights are chosen randomly by me and correspond to the NODE weights

    np.random.seed(3)
    G = nx.Graph()
    G.add_nodes_from(range(dim))
    G.add_edges_from([(i, j) for i in range(dim) for j in range(i+1, dim)])
    #weights = [0.7, 0.9, 0.2, 0.25, 0.6, 0.2, 0.25, 0.25, 0.15, 0.05]  ## they come from the paper
    nx.set_edge_attributes(G, dict(zip(G.edges(), edge_weights)), "weight")
    nx.set_node_attributes(G, dict(zip(G.nodes(), node_weights)), "weight")
    node_labels = nx.get_node_attributes(G, 'weight')
    if draw:
        pos = nx.circular_layout(G)
        nx.draw(G, pos, labels=node_labels, node_color="royalblue", alpha=0.75)
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        return plt.show()
    else:
        return G


### Now, by reading this paper, I got this information:
# :TODO link paper -> https://optimization-online.org/wp-content/uploads/2014/12/4678.pdf
# :TODO info -> The Maximum Weighted Clique Problem is to find, among all complete subgraphs with at most b nodes (for
# :TODO some integer b ∈ {1, 2, . . . , n}), a subgraph (clique) for which the sum of the weights of
# :TODO all the nodes and edges in the subgraph is maximum.





#node_weights=np.array([0.5, 0.2, 0.3, 0.6, 0.25])
if __name__ == "__main__":
    dim = 5
    parser = argparse.ArgumentParser("GFS_WMC algorithm on a given graph.")
    parser.add_argument("penalty_lambda", type=float, help="Penalty lambda")
    parser.add_argument("num_reads", type=int, help="Number of reads")
    parser.add_argument("node_weights", type=str, choices=["random", "equal"], help="Choose 'random' or 'equal' node weights")
    args = parser.parse_args()

    if args.node_weights == "random":
        node_weights = np.random.rand(dim)
    else:
        node_weights = np.array([1.0] * dim)

    g = weighted_graph(
        dim=dim,
        node_weights=node_weights,
        edge_weights=np.array([0.25, 0.9, 0.2, 0.7, 0.15, 0.2, 0.25, 0.25, 0.6, 0.05]),
        draw=False
    )
    print(nx.get_node_attributes(g, 'weight'))

    qubo = qubosolver(
        G=g,
        b=4,
        penalty_lambda=args.penalty_lambda,
        num_reads=args.num_reads
    )

    qubo.solution_draw()