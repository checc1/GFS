import dimod
import time
import networkx as nx
import os
import numpy as np
import matplotlib.pyplot as plt


class qubosolver:
    def __init__(self,
                 G: nx.Graph,
                 b: int,
                 penalty_lambda: float,
                 num_reads: int) -> None:
        """
        Class to solve the QUBO problem. It takes a graph and the parameters for the QUBO problem.
        :param G: The graph.
        :param b: The number of nodes in the clique.
        :param penalty_lambda: The penalty lambda to be used as the constraint.
        :param num_reads: The number of reads to be used in the QUBO solver.

        """
        self.G = G
        self.b = b
        self.penalty_lambda = penalty_lambda
        self.num_reads = num_reads

    def qubo_formulation(self) -> np.ndarray:
        """
        QUBO formulation of the problem.
        :param self:
        :return: Q: (np.ndarray) The QUBO matrix.
        """
        n = self.G.number_of_nodes()
        Q = np.zeros((n, n))

        # TODO: the formula for the WMC mapped as QUBO problem is (same last paper):
        # TODO: max ∑i∈V node_weight_i x_i + ∑i∈V ∑j∈E edge_weight_ij x_i*x_j + \penalty * ∑i∈V (x_i - b)**2

        node_weights = nx.get_node_attributes(self.G, 'weight')
        for i in range(n):
            Q[i, i] = - node_weights[i] + self.penalty_lambda * (1 - 2 * self.b)
        for i, j in self.G.edges:
            edge_w = self.G[i][j].get("weight", 0)
            Q[i, j] = - edge_w + 2 * self.penalty_lambda
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
        path_to_save = os.path.join(os.path.dirname(__file__), "imgs")
        sol = self.solve()
        np.random.seed(3)
        pos = nx.circular_layout(self.G)
        edge_labels = nx.get_edge_attributes(self.G, "weight")
        node_labels = nx.get_node_attributes(self.G, 'weight')
        nx.draw(self.G, pos, with_labels=True, node_color="royalblue", alpha=0.75, labels=node_labels)
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels)
        clique_nodes = [node for node, val in sol.items() if val == 1]
        other_nodes = [node for node in self.G.nodes if node not in clique_nodes]
        nx.draw_networkx_nodes(self.G, pos, nodelist=clique_nodes, node_color="orangered", label="Clique")
        nx.draw_networkx_nodes(self.G, pos, nodelist=other_nodes, node_color="royalblue", label="Other")
        plt.legend([fr"$\lambda= {self.penalty_lambda}$"])
        plt.savefig(path_to_save + "/" + fr"$GFS_WMC_lamb{self.penalty_lambda}_b{self.b}$.png")
        plt.show()