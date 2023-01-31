"""
Util functions for Max-Cut.
"""

from collections import defaultdict
from typing import Union

from networkx import Graph
from networkx.algorithms.approximation.maxcut import one_exchange


def create_qubo_max_cut(G: Graph) -> defaultdict(int):
    """
    Create a QUBO formulation of a random Max-Cut instance of an Erdös-Renyí graph.

    Args:
        G: Erdös-Renyí graph problem instance.

    Returns:
        QUBO of a random Max-Cut instance given a problem size.
    """
    Q = defaultdict(int)
    for i, j in G.edges:
        Q[(i, i)] += -1
        Q[(j, j)] += -1
        Q[(i, j)] += 2

    return Q


def calculate_beta_max_cut(graph: Union[Graph, int], max_cut_result: float) -> float:
    """
    Calculate beta value for a Max-Cut optimization problem.
    If a graph is provided, beta is calculated based on exact result.

    Args:
        graph: Problem instance graph or graph size.
        max_cut_result: Found objective value.

    Returns:
        beta value for specific problem instance or problem size.
    """
    if isinstance(graph, Graph):  # only suitable for small graph sizes.
        n = len(graph)
        random_score = n * (n - 1) / 8
        exact_score = one_exchange(graph)[0]
        beta = (max_cut_result - random_score) / (exact_score - random_score)
    else:
        random_score = graph**2 / 8
        beta = (max_cut_result - random_score) / (0.178 * pow(graph, 3 / 2))
    return beta
