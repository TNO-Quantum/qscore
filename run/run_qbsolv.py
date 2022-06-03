"""
Run a Max-Cut instance on the D-Wave qbsolv solver.
"""

import time
from collections import defaultdict
from typing import Tuple

import numpy as np
from dwave_qbsolv import QBSolv


def run_qbsolv(Q: defaultdict(int), timeout: int, size: int) -> Tuple[float]:
    """
    Function that solves a Max-Cut instance on the D-Wave qbsolv solver.

    :param Q: QUBO-formulation of Max-Cut instance.
    :param timeout: timeout parameter.
    :param size: Problem size.

    :return: The largest found cut and the corresponding value of beta.
    """
    start = time.time()

    sampler = QBSolv()
    sampleset = sampler.sample_qubo(Q, label=f"Maximum Cut {size:2d}")

    max_cut_result = -sampleset.first.energy
    random_score = size**2 / 8
    beta = (max_cut_result - random_score) / (0.178 * pow(size, 3 / 2))

    time_taken = time.time() - start
    if time_taken > timeout:
        print("failed to find a cut within timeout limit")
        max_cut_result = np.nan
        beta = 0
    return max_cut_result, beta
