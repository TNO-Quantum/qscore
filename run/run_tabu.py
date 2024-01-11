"""
Run a Q-score instance using the D-Wave Tabu solver.
"""
import time
from collections import defaultdict
from typing import Optional

import numpy as np
from tabu import TabuSampler


def run_tabu(Q: defaultdict(int), size: int, timeout: Optional[int] = None) -> float:
    """
    Function that solves a Q-score instance on the D-Wave tabu solver.

    Args:
        Q: QUBO-formulation of Q-score instance.
        size: Problem size.
        timeout: timeout parameter.

    Returns:
        The largest found objective value. If no solution is found within the provided
        timeout limit, np.nan is being returned.
    """
    start = time.time()

    sampler = TabuSampler()
    sampleset = sampler.sample_qubo(Q, label=f"Problem-{size:2d}")

    objective_result = -sampleset.first.energy

    time_taken = time.time() - start
    if timeout is not None and time_taken > timeout:
        print("Failed to find a solution within timeout limit.")
        objective_result = np.nan

    return objective_result
