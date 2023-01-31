"""
Test if calculate qscore works with correct input.
"""

import json
import os

import pytest

from calculate_qscore import calculate_qscore

nb_instances_per_size = 5
size_range = list(range(10))
file_name = "test_calculate_qscore.json"
include_exact_results = True
problem_type = "max-cut"
timeout = None
solver = "Simulated_Annealing"
seed = 101
num_reads = 10
provider = None
backend = None


def test_calculate_qscore():
    calculate_qscore(
        nb_instances_per_size=nb_instances_per_size,
        size_range=size_range,
        file_name=file_name,
        include_exact_results=include_exact_results,
        problem_type=problem_type,
        timeout=timeout,
        solver=solver,
        seed=seed,
        num_reads=num_reads,
        provider=provider,
        backend=backend,
    )

    assert os.path.exists(f"data{os.sep}" + file_name)
    with open(f"data{os.sep}" + file_name) as json_file:
        data = json.load(json_file)

    assert data["settings"] == {
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
    for k, v in data.items():
        if k == "settings":
            continue
        for result, exact_result in zip(v["result"], v["exact-result"]):
            result == exact_result

    message = "Path already exists. Aborted as data otherwise might be overwritten!"
    with pytest.raises(FileExistsError, match=message):
        calculate_qscore(
            nb_instances_per_size=nb_instances_per_size,
            size_range=size_range,
            file_name=file_name,
            include_exact_results=include_exact_results,
            problem_type=problem_type,
            timeout=timeout,
            solver=solver,
            seed=seed,
            num_reads=num_reads,
            provider=provider,
            backend=backend,
        )

    os.remove(f"data{os.sep}" + file_name)
