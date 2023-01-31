"""
Test if all individual run functions work with correct input.
"""

import networkx as nx
from qiskit_optimization.applications import Clique, Maxcut

from run.run_dwave_qpu import run_dwave_qpu
from run.run_hybrid import run_hybrid
from run.run_photonic_simulated import run_photonic_simulated
from run.run_QAOA import run_QAOA
from run.run_qbsolv import run_qbsolv
from run.run_SA import run_SA
from utils.max_cut import create_qubo_max_cut

size, timeout = 10, None
G = nx.erdos_renyi_graph(size, 1 / 2, seed=101)
Q = create_qubo_max_cut(G)


def test_dwave_qpu_advantage():
    run_dwave_qpu(Q, size, solver="Advantage_system4.1", num_reads=1000, timeout=None)


def test_dwave_qpu_2000Q():
    run_dwave_qpu(Q, size, solver="DW_2000Q_6", num_reads=1000, timeout=None)


def test_qbsolv():
    run_qbsolv(Q, size, timeout=None)


def test_SA():
    run_SA(Q, size, num_reads=10, timeout=None)


def test_hybrid():
    run_hybrid(Q, size, timeout=None)


def test_photonic_simulated():
    run_photonic_simulated(G, size, n_samples=10, timeout=None)


def test_QAOA_max_cut():
    max_cut = Maxcut(G)
    qp = max_cut.to_quadratic_program()
    run_QAOA(qp, provider=None)


def test_QAOA_max_cut():
    max_cut = Clique(G)
    qp = max_cut.to_quadratic_program()
    run_QAOA(qp, provider=None)
