"""
Run a Q-score instance on the D-Wave Simulated Annealing solver.
"""

import time
from collections import defaultdict
from functools import partial
from typing import Optional

import neal
import numpy as np
from dwave.embedding.chain_strength import uniform_torque_compensation


def run_SA(
    Q: defaultdict(int), size: int, num_reads: int, timeout: Optional[int] = None
) -> float:
    """
    Function that solves a Q-score instance on the D-Wave Simulated Annealing solver.

    Args:
        Q: QUBO-formulation of Q-score instance.
        size: Problem size.
        num_reads: Number of states to be read from solver.
        timeout: timeout parameter.

    Returns:
        The largest found objective value. If no solution is found within the provided
        timeout limit, np.nan is being returned.
    """
    start = time.time()

    sampler = neal.SimulatedAnnealingSampler()
    chain_strength = partial(uniform_torque_compensation, prefactor=2)
    sampleset = sampler.sample_qubo(
        Q,
        chain_strength=chain_strength,
        num_reads=num_reads,
        label=f"Problem-{size:2d}",
    )

    objective_result = -sampleset.first.energy

    time_taken = time.time() - start
    if timeout is not None and time_taken > timeout:
        print("Failed to find a solution within timeout limit.")
        objective_result = np.nan

    return objective_result
