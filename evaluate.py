"""
Run one Max-Cut instance on one of the five solvers.
"""
import argparse
from collections import defaultdict

import networkx as nx
import numpy as np

from run.run_hybrid import run_hybrid
from run.run_qbsolv import run_qbsolv
from run.run_qpu import run_qpu
from run.run_SA import run_SA


def parse_args() -> argparse.Namespace:
    """
    Parser function.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--size",
        help="Problem size",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-seed",
        "--seed",
        help="Random seed",
        type=int,
        required=False,
    )
    parser.add_argument(
        "-t",
        "--timeout",
        help="Solver timeout",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-n",
        "--num_reads",
        help="Number of qpu reads in case of a QPU or Simulated Annealing solver",
        type=int,
        required=False,
    )
    parser.add_argument(
        "-solver",
        "--solver",
        help="String of the D-Wave solver.",
        choices=[
            "Advantage_system4.1",
            "DW_2000Q_6",
            "hybrid",
            "qbsolv",
            "Simulated_Annealing",
        ],
        type=str,
        required=True,
    )

    args = parser.parse_args()
    return args


def create_qubo(size: int, seed=None) -> defaultdict(int):
    """
    Create a QUBO formulation of a random Max-Cut instance of an Erdös-Renyí graph.

    :param size: Problem instance size.
    :param seed: Random seed.

    :return: QUBO of a random Max-Cut instance given size.
    """
    if seed is None:
        seed = np.random.randint(100000)
    G = nx.erdos_renyi_graph(size, 1 / 2, seed=seed)

    # Initialize our Q matrix
    Q = defaultdict(int)
    for i, j in G.edges:
        Q[(i, i)] += -1
        Q[(j, j)] += -1
        Q[(i, j)] += 2

    return Q


if __name__ == "__main__":
    args = parse_args()

    size = args.size
    seed = args.seed
    timeout = args.timeout
    num_reads = args.num_reads
    solver = args.solver
    if num_reads is None and solver in [
        "Advantage_system4.1",
        "DW_2000Q_6",
        "Simulated_Annealing",
    ]:
        raise ValueError("num_reads has not been submitted while required by solver.")
    else:
        # Create qubo:
        Q = create_qubo(size, seed)

        # Solve Max cut instance
        if solver in ["Advantage_system4.1", "DW_2000Q_6"]:
            max_cut_result, beta = run_qpu(Q, timeout, size, solver, num_reads)
        elif solver == "hybrid":
            max_cut_result, beta = run_hybrid(Q, timeout, size)
        elif solver == "Simulated_Annealing":
            max_cut_result, beta = run_SA(Q, timeout, size, num_reads)
        elif solver == "qbsolv":
            max_cut_result, beta = run_qbsolv(Q, timeout, size)
        else:
            raise NotImplementedError(f"Provided Solver {solver} is not implemented")

    print(f"Max cut result: {max_cut_result}")
    print(f"Beta: {beta}")
