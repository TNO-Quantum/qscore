"""
Run a Q-score instance on a D-Wave QPU solver.
"""

from collections import defaultdict
from functools import partial
from typing import Optional

import dimod
import numpy as np
from dwave.embedding.chain_strength import uniform_torque_compensation
from dwave.system.composites import FixedEmbeddingComposite
from dwave.system.samplers import DWaveSampler
from minorminer import find_embedding


def run_dwave_qpu(
    Q: defaultdict(int),
    size: int,
    solver: str,
    num_reads: int = 1000,
    timeout: Optional[int] = None,
) -> float:
    """
    Function that solves a Q-score instance on D-Wave QPU solvers.

    Args:
        Q: QUBO-formulation of Q-score instance.
        size: Problem size.
        solver: QPU solver to be used.
        num_reads: Number of states to be read from solver.
        timeout: timeout parameter.

    Returns:
        The largest found objective value. If no embedding can be found
          np.nan is being returned.
    """
    chain_strength = partial(uniform_torque_compensation, prefactor=2)
    sampler = DWaveSampler(solver=solver)
    try:
        # Find embedding
        bqm = dimod.BQM.from_qubo(Q)
        source_edgelist = list(bqm.quadratic) + [(v, v) for v in bqm.linear]
        target_graph = sampler.to_networkx_graph()
        if timeout is None:
            timeout = 1000  # use default minorminer value
        embedding = find_embedding(source_edgelist, target_graph, timeout=timeout)
    except:  # No embedding can be found
        print("Failed to find embedding.")
        return np.nan

    # Apply embedding and sample
    sampler_embedded = FixedEmbeddingComposite(DWaveSampler(solver=solver), embedding)

    sampleset = sampler_embedded.sample_qubo(
        Q,
        chain_strength=chain_strength,
        num_reads=num_reads,
        label=f"Problem-{size:2d}",
    )
    objective_result = -sampleset.first.energy

    return objective_result
