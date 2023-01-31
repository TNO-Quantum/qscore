"""
Run a Q-score instance on the photonic simulated solver.
"""

import time
from typing import Optional

import networkx as nx
import numpy as np
from networkx import Graph
from strawberryfields.apps import clique, sample


def run_photonic_simulated(
    G: Graph, size: int, n_samples: int, timeout: Optional[int] = None
):
    """
    Function that solves a Q-score instance on a photonic simulator.
    Can only be used for Max-Clique problem instances.

    Args:
        G: Erdös-Renyí graph problem instance.
        size: Problem size.
        n_samples: Number of samples.
        timeout: timeout parameter.

    Returns:
        The largest found objective value. If no solution is found within the provided
        timeout limit, np.nan is being returned.
    """
    if G.size() == 0:
        return 1, 0, 0

    # Extract the adjacency matrix
    adj = nx.to_numpy_array(G)

    s = sample.sample(adj, n_mean=50, n_samples=n_samples)

    # Create upper and lower bounds for max cliques to search for
    max_clicks = 2 * np.log2(size)
    min_clicks = max(1, 2 * np.log(size) - 3)
    s = sample.postselect(s, min_clicks, max_clicks)
    subgraphs = sample.to_subgraphs(s, G)

    # Find cliques
    start = time.time()  # We only consider classical runtime
    shrunk = [clique.shrink(sg, G) for sg in subgraphs]
    objective_result = max([len(s) for s in shrunk])

    end = time.time()
    if timeout is not None and end - start > timeout:
        print("Failed to find a solution within timeout limit.")
        objective_result = np.nan

    return objective_result, end, start
