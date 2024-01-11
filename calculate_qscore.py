"""
Run the Q-score instances for various sizes and save data.
"""
import json
import os

import networkx as nx
import numpy as np
from networkx.algorithms.approximation.maxcut import one_exchange

from evaluate import main


def calculate_qscore(
    nb_instances_per_size: int,
    size_range: list,
    file_name: str,
    include_exact_results: bool,
    problem_type: str,
    timeout: int,
    solver: str,
    seed: int,
    num_reads: int,
    provider: str,
    backend: str,
):
    """
    Run multiple Q-score instances for various problem sizes.
    Results are written to a json file.

    Args:
        nb_instances_per_size: Number of instances per graph size.
        size_range: Different graph sizes.
        file_name: Path where results will be saved.
        include_exact_results: Whether to include exact results for computing beta.
            Only advisable for small problem sizes.

        problem_type: see parse_args in evaluate.py.
        timeout: see parse_args in evaluate.py.
        solver: see parse_args in evaluate.py.
        seed: see parse_args in evaluate.py.
        num_reads: see parse_args in evaluate.py.
        provider: see parse_args in evaluate.py.
        backend: see parse_args in evaluate.py.

    Raises:
        FileExistsError: When the provided path already exists.
    """
    # Check if file already exists
    if os.path.exists(f"data{os.sep}" + file_name):
        raise FileExistsError(
            "Path already exists. Aborted as data otherwise might be overwritten!"
        )
    else:
        # Create data template
        all_data = {str(size): None for size in size_range}
        all_data["settings"] = {
            "_NB_INSTANCES_PER_SIZE": nb_instances_per_size,
            "_SIZE_RANGE": size_range,
            "FILE_NAME": file_name,
            "INCLUDE_EXACT_RESULTS": include_exact_results,
            "PROBLEM_TYPE": problem_type,
            "TIMEOUT": timeout,
            "SOLVER": solver,
            "SEED": seed,
            "NUM_READS": num_reads,
            "PROVIDER": provider,
            "BACKEND": backend,
        }
    for size in size_range:
        result, times = [], []
        exact_results = []
        for i in range(nb_instances_per_size):
            objective_result, _, time, G = main(
                problem_type=problem_type,
                size=size,
                timeout=timeout,
                solver=solver,
                seed=seed,
                num_reads=num_reads,
                provider=provider,
                backend=backend,
            )
            result.append(objective_result)
            times.append(time)
            all_data[str(size)] = {"result": result, "times": times}

            if include_exact_results:
                if problem_type == "max-clique":
                    exact_result = nx.max_weight_clique(G, weight=None)[1]
                elif problem_type == "max-cut":
                    exact_result = one_exchange(G)[0]
                exact_results.append(exact_result)
                all_data[str(size)]["exact-result"] = exact_results

            # Write data to file:
            with open(f"data{os.sep}" + file_name, "w") as f:
                json.dump(all_data, f)

            # Increase seed
            seed += 1

        print(
            f"Finished problem size: {size}, "
            f"average objective: {np.array(result).mean()}, "
            f"average problem time: {np.array(times).mean():2f}."
        )


if __name__ == "__main__":
    # Input arguments
    _NB_INSTANCES_PER_SIZE = 1
    _SIZE_RANGE = list(range(90000,20000, 1000))
    FILE_NAME = "tabu.json"
    INCLUDE_EXACT_RESULTS = False
    PROBLEM_TYPE = "max-cut"
    TIMEOUT = 60
    SOLVER = "tabu"
    _SEED = 101200
    NUM_READS = None
    PROVIDER = None
    BACKEND = None

    calculate_qscore(
        nb_instances_per_size=_NB_INSTANCES_PER_SIZE,
        size_range=_SIZE_RANGE,
        file_name=FILE_NAME,
        include_exact_results=INCLUDE_EXACT_RESULTS,
        problem_type=PROBLEM_TYPE,
        timeout=TIMEOUT,
        solver=SOLVER,
        seed=_SEED,
        num_reads=NUM_READS,
        provider=PROVIDER,
        backend=BACKEND,
    )
