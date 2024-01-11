"""
Test if evaluate function work with correct input.
"""
import pytest

from evaluate import main


@pytest.mark.parametrize(
    "solver",
    [
        "Advantage_system4.1",
        "hybrid",
        "tabu",
        "Simulated_Annealing",
        "QAOA",
    ],
)
def test_evaluate_main_max_cut(solver):
    main(problem_type="max-cut", size=10, solver=solver, num_reads=100)


@pytest.mark.parametrize(
    "solver",
    [
        "Advantage_system4.1",
        "hybrid",
        "tabu",
        "Simulated_Annealing",
        "Photonic_Simulation",
        "Photonic_quandela",
        "QAOA",
    ],
)
def test_evaluate_main_max_clique(solver):
    main(problem_type="max-clique", size=4, solver=solver, num_reads=100)
