"""
Run a Max-Cut instance on the D-Wave Simulated Annealing solver.
"""

import time
from collections import defaultdict
from functools import partial
from typing import Tuple

import neal
import numpy as np
from dwave.embedding.chain_strength import uniform_torque_compensation


def run_SA(
    Q: defaultdict(int), timeout: int, size: int, num_reads: int
) -> Tuple[float]:
    """
    Function that solves a Max-Cut instance on the D-Wave Simulated Annealing solvers

    :param Q: QUBO-formulation of Max-Cut instance.
    :param timeout: timeout parameter.
    :param size: Problem size.
    :param num_reads: Number of states to be read from solver.

    :return: The largest found cut and the corresponding value of beta.
    """
    start = time.time()

    sampler = neal.SimulatedAnnealingSampler()
    chain_strength = partial(uniform_torque_compensation, prefactor=2)
    sampleset = sampler.sample_qubo(
        Q,
        chain_strength=chain_strength,
        num_reads=num_reads,
        label=f"Maximum Cut {size:2d}",
    )

    max_cut_result = -sampleset.first.energy
    random_score = size ** 2 / 8
    beta = (max_cut_result - random_score) / (0.178 * pow(size, 3 / 2))

    time_taken = time.time() - start
    if time_taken > timeout:
        print("failed to find a cut within timeout limit")
        max_cut_result = np.nan
        beta = 0
    return max_cut_result, beta
