"""
Util functions for Max-Clique.
"""

from collections import defaultdict
from typing import Union

import networkx as nx
import numpy as np
from networkx import Graph


def create_qubo_max_clique(G: Graph) -> defaultdict(int):
    """
    Create a QUBO formulation of a random Max-Clique instance of an Erdös-Renyí graph.

    Args:
        G: Erdös-Renyí graph problem instance.

    Returns:
        QUBO of a random Max-Clique instance given a problem size.
    """
    G_C = nx.complement(G)
    Q = defaultdict(int)
    for i in G.nodes:
        Q[(i, i)] -= 1
    for i, j in G_C.edges:
        Q[(i, j)] += 2

    return Q


def random_clique_size(size: int) -> float:
    """Approximate average clique size found with a random search

    Args:
        size: Graph size

    Returns:
        Average clique size using random search algorithm
    """
    cliques = []
    for _ in range(1000):
        len_clique = 1
        for _ in range(size - 1):
            if np.random.random() < (1 / 2) ** len_clique:
                len_clique += 1
        cliques.append(len_clique)
    return sum(cliques) / len(cliques)


def naive_clique_size(G: Graph) -> float:
    """Calculate clique size found by naive search.

    Args:
        size: Graph size.

    Returns:
        Average clique size using naive search algorithm.
    """
    nodes = list(G.nodes())
    random_nodes = []
    while len(nodes) > 0:
        random_node = nodes.pop(np.random.randint(len(nodes)))
        random_nodes.append(random_node)
        H = G.subgraph(random_nodes)
        n = len(random_nodes)
        if H.size() != n * (n - 1) / 2:
            return n - 1
    return len(random_nodes)


def calculate_beta_max_clique(
    graph: Union[Graph, int],
    max_clique_result: float,
) -> float:
    """
    Calculate beta value for a Max-Clique optimization problem.
    If a graph is provided, beta is calculated based on exact result.

    Args:
        graph: Problem instance graph or graph size.
        max_clique_result: Found objective value.

    Returns:
        beta value for specific problem instance or problem size.

    """
    if isinstance(graph, Graph):  # only suitable for small graph sizes.
        random_score = np.average([naive_clique_size(graph) for _ in range(1000)])
        exact_score = nx.max_weight_clique(graph, weight=None)[1]
        if exact_score == random_score:
            return 1
        beta = (max_clique_result - random_score) / (exact_score - random_score)
    else:
        random_score = 1.6416325  # Approximation of sum_i^N i*(1-p^i)*p^(1/2 i (i-1))
        asymptote = (
            2 * np.log2(graph) - 2 * np.log2(np.log2(graph)) + 2 * np.log2(np.e / 2) + 1
        )
        beta = (max_clique_result - random_score) / (asymptote - random_score)
    return beta
