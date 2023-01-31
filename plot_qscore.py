"""
Plot beta vs N and time vs N Q-score graphs.
"""
import argparse
import json
import os
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.algorithms.approximation.maxcut import one_exchange

from utils.max_clique import calculate_beta_max_clique, naive_clique_size
from utils.max_cut import calculate_beta_max_cut


def parse_args() -> argparse.Namespace:
    """
    Parser function.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--file",
        help="Name of data file. (within /data folder)",
        required=True,
    )
    parser.add_argument(
        "-e",
        "--exact",
        action="store_true",
        help="Use graph instances to calculate beta",
        required=False,
    )
    parser.add_argument(
        "-t",
        "--time_constraint",
        action="store_true",
        help="Ignore time constraint",
        required=False,
    )
    args = parser.parse_args()
    return args


def plot_graph(
    file: str, exact: Optional[bool] = False, time_constraint: Optional[bool] = False
) -> None:
    """
    Plot Q-score graphs. Both the beta vs N and time vs N graphs are plotted.

    Args:
        file: Path to data file (within /data folder).
        exact: Whether to include exact results in beta calculation.
            Only suitable for small problem sizes.
    """
    # Load data from json file.
    with open(file) as json_file:
        data = json.load(json_file)

    problem_range = np.array(
        [int(k) for k in data.keys() if k != "settings" and data[k]["result"] != []]
    )
    problem_type = data["settings"]["PROBLEM_TYPE"]
    solver = data["settings"]["SOLVER"]

    # Do beta-calculations
    mins_beta, maxes_beta, means_beta, stds_beta = [], [], [], []
    mins_time, maxes_time, means_time, stds_time = [], [], [], []

    if exact:
        seed = data["settings"]["SEED"]
        for size in problem_range:
            print(f"Starting exact calculation for size: {size}")
            results = np.array(data[str(size)]["result"])
            times = np.array(data[str(size)]["times"])

            graphs = []  # Calculate graphs from seed
            for _ in range(len(results)):
                G = nx.erdos_renyi_graph(size, 1 / 2, seed=seed)
                graphs.append(G)
                seed += 1

            if "exact-result" in data[str(size)]:
                exact_results = np.array(data[str(size)]["exact-result"])
            else:  # Calculate exact result from graph
                exact_results = np.array(
                    [
                        one_exchange(G)[0]
                        if problem_type == "max-cut"
                        else nx.max_weight_clique(G, weight=None)[1]
                        for G in graphs
                    ]
                )

            betas = []
            if problem_type == "max-cut":
                for result, exact_result in zip(results, exact_results):
                    random_score = size * (size - 1) / 8
                    if random_score == exact_result:
                        betas.append(1)
                    else:
                        beta = (result - random_score) / (exact_result - random_score)
                        betas.append(beta)

            elif problem_type == "max-clique":
                for result, exact_result, G in zip(results, exact_results, graphs):
                    random_score = np.average(
                        [naive_clique_size(G) for _ in range(1000)]
                    )
                    if random_score == exact_result:
                        betas.append(1)
                    else:
                        beta = (result - random_score) / (exact_result - random_score)
                        betas.append(beta)

            betas = np.array(betas)
            mins_beta.append(betas.min())
            maxes_beta.append(betas.max())
            means_beta.append(betas.mean())
            stds_beta.append(betas.std())

            mins_time.append(times.min())
            maxes_time.append(times.max())
            means_time.append(times.mean())
            stds_time.append(times.std())

    else:
        calculate_beta_function = (
            calculate_beta_max_cut
            if problem_type == "max-cut"
            else calculate_beta_max_clique
        )
        for size in problem_range:
            results = np.array(data[str(size)]["result"])
            times = np.array(data[str(size)]["times"])
            times = times[~np.isnan(results)]
            results = results[~np.isnan(results)]

            if len(results) == 0:
                break

            betas = np.array(
                [calculate_beta_function(size, result) for result in results]
            )
            mins_beta.append(betas.min())
            maxes_beta.append(betas.max())
            means_beta.append(betas.mean())
            stds_beta.append(betas.std())

            mins_time.append(times.min())
            maxes_time.append(times.max())
            means_time.append(times.mean())
            stds_time.append(times.std())

    mins_beta, maxes_beta, means_beta, stds_beta = (
        np.array(mins_beta),
        np.array(maxes_beta),
        np.array(means_beta),
        np.array(stds_beta),
    )
    mins_time, maxes_time, means_time, stds_time = (
        np.array(mins_time),
        np.array(maxes_time),
        np.array(means_time),
        np.array(stds_time),
    )

    if time_constraint:
        qscore = max(problem_range[np.where(means_beta > 0.2)])
    else:
        qscore = max(problem_range[np.where((means_beta > 0.2) & (means_time < 60))])

    # Create plots:
    problem_range = problem_range[: len(mins_beta)]
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    fig.suptitle(f"Q-score {problem_type} = {qscore} for solver: {solver}")

    # Beta-plot
    axs[0].errorbar(
        x=problem_range,
        y=means_beta,
        yerr=[means_beta - mins_beta, maxes_beta - means_beta],
        fmt=".k",
        ecolor="blue",
        lw=1,
    )

    if exact:  # Cap stdv at beta=1
        yerr = np.array(
            [
                stds_beta,
                np.array([min(i, 1 - j) for i, j in zip(stds_beta, means_beta)]),
            ]
        )
    else:
        yerr = stds_beta
    axs[0].errorbar(x=problem_range, y=means_beta, yerr=yerr, fmt="ok", lw=2)
    axs[0].set_title(f"Beta {'(exact)' if exact else ''}")
    axs[0].set(
        xlabel="Problem size N",
        ylabel="Beta",
        xlim=[0, problem_range[-1] + 5],
        ylim=[-0.3, 1.5],
    )
    axs[0].axhline(y=0.2, color="r", linestyle="--")

    # Time-plot
    axs[1].set_title("Elapsed time")
    axs[1].set(
        xlabel="Problem size N",
        ylabel="Time (in s)",
        xlim=[0, problem_range[-1] + 5],
        ylim=[0, min(90, max(maxes_time) * 1.2)],
    )
    axs[1].errorbar(
        problem_range,
        means_time,
        [means_time - mins_time, maxes_time - means_time],
        fmt=".k",
        ecolor="blue",
        lw=1,
    )
    axs[1].errorbar(problem_range, means_time, stds_time, fmt="ok", lw=2)
    axs[1].axhline(y=60, color="r", linestyle="--")

    plt.show()


if __name__ == "__main__":
    args = parse_args()
    file = f"data{os.sep}" + args.file
    exact = args.exact
    time_constraint = args.time_constraint

    plot_graph(file, exact, time_constraint)
