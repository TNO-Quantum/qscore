"""
Run a Max-Cut instance on the D-Wave hybrid solver.
"""

from collections import defaultdict
from typing import Tuple

import dimod
from dwave.system import LeapHybridSampler

sampler_LeapHybrid = LeapHybridSampler(solver={"category": "hybrid"})


def run_hybrid(Q: defaultdict(int), timeout: int, size: int) -> Tuple[float]:
    """
    Function that solves a Max-Cut instance on the D-Wave hybrid solver.

    :param Q: QUBO-formulation of Max-Cut instance.
    :param timeout: timeout parameter.
    :param size: Problem size.

    :return: The largest found cut and the corresponding value of beta.
    """
    bqm = dimod.BQM.from_qubo(Q)
    sampleset = sampler_LeapHybrid.sample(
        bqm, label=f"Maximum Cut {size:2d}", time_limit=timeout
    )
    max_cut_result = -sampleset.first.energy

    random_score = size**2 / 8
    beta = (max_cut_result - random_score) / (0.178 * pow(size, 3 / 2))

    return max_cut_result, beta
