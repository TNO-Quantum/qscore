"""
Run a Max-Cut instance on a D-Wave QPU solver.
"""

from collections import defaultdict
from functools import partial
from typing import Tuple

import dimod
import numpy as np
from dwave.embedding.chain_strength import uniform_torque_compensation
from dwave.system.composites import FixedEmbeddingComposite
from dwave.system.samplers import DWaveSampler
from minorminer import find_embedding


def run_qpu(
    Q: defaultdict(int),
    timeout: int,
    size: int,
    solver: str,
    num_reads: int = 1000,
) -> Tuple[float]:
    """
    Function that solves a Max-Cut instance on D-Wave QPU solvers.

    :param Q: QUBO-formulation of Max-Cut instance.
    :param timeout: timeout parameter.
    :param size: Problem size.
    :param solver: QPU solver to be used.
    :param num_reads: Number of states to be read from solver.

    :return: The largest found cut and the corresponding value of beta.
    """
    chain_strength = partial(uniform_torque_compensation, prefactor=2)
    sampler = DWaveSampler(solver=solver)
    try:
        # Find embedding
        bqm = dimod.BQM.from_qubo(Q)
        source_edgelist = list(bqm.quadratic) + [(v, v) for v in bqm.linear]
        target_graph = sampler.to_networkx_graph()
        embedding = find_embedding(source_edgelist, target_graph, timeout=timeout)
    except:  # No embedding can be found
        print("Failed to find embedding.")
        return np.nan, np.nan

    # Apply embedding and sample
    sampler_embedded = FixedEmbeddingComposite(DWaveSampler(solver=solver), embedding)

    sampleset = sampler_embedded.sample_qubo(
        Q,
        chain_strength=chain_strength,
        num_reads=num_reads,
        label=f"Maximum Cut {size:2d}",
    )
    max_cut_result = -sampleset.first.energy

    random_score = size**2 / 8
    beta = (max_cut_result - random_score) / (0.178 * pow(size, 3 / 2))

    return max_cut_result, beta
