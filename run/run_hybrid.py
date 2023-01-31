"""
Run a Q-score instance on the D-Wave hybrid solver.
"""

from collections import defaultdict

import dimod
from dwave.system import LeapHybridSampler

sampler_LeapHybrid = LeapHybridSampler(solver={"category": "hybrid"})


def run_hybrid(
    Q: defaultdict(int),
    size: int,
    timeout: int,
) -> float:
    """
    Function that solves a Q-score instance on the D-Wave hybrid solver.

    Args:
        Q: QUBO-formulation of Q-score instance.
        size: Problem size.
        timeout: timeout parameter.

    Returns:
        The largest found objective value. If no solution is found within the provided
        timeout limit, np.nan is being returned.
    """
    bqm = dimod.BQM.from_qubo(Q)
    sampleset = sampler_LeapHybrid.sample(
        bqm, label=f"Problem-{size:2d}", time_limit=timeout
    )
    objective_result = -sampleset.first.energy

    return objective_result
