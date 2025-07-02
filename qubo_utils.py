import neal
import numpy as np
import dimod
import networkx as nx
from matplotlib import pyplot as plt


class QuboProblem:
    def __init__(self, G: nx.Graph, A=1, B=2):
        self.G = G
        self.A = A
        self.B = B

    def qubo(self):
        n = self.G.number_of_nodes()
        Q = np.zeros((n, n))
        G_bar = nx.complement(self.G)
        for i in range(n):
            Q[i, i] = -self.A
        for u, v in G_bar.edges():
            Q[u, v] += self.B
            Q[v, u] += self.B
        return Q

    def solve_qubo(self,
                   T_range: tuple[float, float],
                   reps: int,
                   sweeps: int,
                   T_scheduling: str) -> tuple[dimod.SampleSet, dict]:

        if T_scheduling not in ['geometric', 'linear', 'logarithmic']:
            raise ValueError("T_scheduling must be one of 'geometric', 'linear', or 'logarithmic'.")

        B_range = (1 / T_range[0], 1 / T_range[1])
        Q = self.qubo()
        bqm_dict = {(i, j): float(Q[i, j]) for i in range(Q.shape[0]) for j in range(Q.shape[1])}
        bqm = dimod.BinaryQuadraticModel(bqm_dict, vartype="BINARY")
        sampler = neal.SimulatedAnnealingSampler()

        sampleset = sampler.sample(bqm,
                                   beta_range=B_range,
                                   num_reads=reps,
                                   num_sweeps=sweeps,
                                   beta_schedule_type=T_scheduling)
        best_sample = sampleset.first.sample
        return sampleset, best_sample

    @staticmethod
    def draw_solution(G: nx.Graph, solution: dict):

        pos = nx.circular_layout(G)
        #plt.figure(figsize=(10, 10))
        nx.draw(G, pos, with_labels=True, node_size=900,
                node_color='dodgerblue', font_size=16, font_color='black',
                edgecolors='black', linewidths=1, edge_color='gray')

        solution_nodes = [node for node, value in solution.items() if value == 1]
        nx.draw_networkx_nodes(G, pos, nodelist=solution_nodes, node_color='tab:orange', node_size=900,
                               edgecolors='black', linewidths=1)

        plt.show()

