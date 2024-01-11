"""
Run a Q-score instance on the photonic simulated solver.
"""
import ast
import itertools
import time
from typing import List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import perceval as pcvl
from networkx import Graph
from numpy import linalg
from numpy.typing import NDArray
from perceval.components.abstract_processor import AProcessor
from scipy.linalg import sqrtm
from scipy.special import binom
from strawberryfields.apps import clique


def read_quandela_api_key() -> str:
    """Read Quandela API key from the configuration/_QUANDELA_API_KEY file

    Returns:
        Quandela API key or if File not found an empty string.
    """
    try:
        with open("configuration/_QUANDELA_API_KEY", "r") as file:
            api_key = file.read().strip()
            return api_key
    except FileNotFoundError as e:
        print(f"Error reading API key: {e}")
        return ""


def initialize_remote_backend(
    backend: Optional[str] = None,
) -> Union[AProcessor, None]:
    """Initialize a Quandela backend.

    Args:
        backend: Name of remote Quandela backend or simulator.

    Options for backend are:

      - `qpu:ascella` (hardware),
      - `sim:slos` (simulator),
      - `sim:ascella` (simulator),
      - `sim:clifford` (simulator)
      - None, initialize local simulator.

    Returns:
        Remote backend instance or simulator.
    """
    if backend is not None:
        token_qcloud = read_quandela_api_key()
        return pcvl.RemoteProcessor(backend, token_qcloud)
    return pcvl.BackendFactory().get_backend("CliffordClifford2017")


def run_photonic_quandela(
    G: Graph,
    size: int,
    n_samples: int,
    backend: Optional[str] = None,
    timeout: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Function that solves a Q-score instance on a the Ascella QPU
    Can only be used for Max-Clique problem instances.

    Args:
        G: Erdös-Renyí graph problem instance.
        size: Problem size.
        n_samples: Number of samples
        backend: Quandela remote backend or simulator.
        timeout: timeout parameter.

    Returns:
        The largest found objective value. If no solution is found within the provided
        timeout limit, np.nan is being returned.

    Raises:
        ValueError in case a too small problem instance is provided.
    """
    start = time.time()
    if list(G.edges) == []:
        print("Graph is empty, so no photonic sampling required.")
        objective_result = 1
        end = time.time()
        return objective_result, end, start
    max_clicks = 2 * int(np.log2(size)) + 2
    min_clicks = max(2, 2 * int(np.log2(size)) - 2)

    results = []
    for size_clique in range(min_clicks, max_clicks):
        result = densest_subgraphs(G, size_clique, n_samples, backend)
        if result is None:
            pass
        else:
            result = [x[1] for x in result if x[0] > 0]
            shrunk_result = [clique.shrink(x, G) for x in result]
            # enlarged_result = [clique.search(x, G, 10) for x in shrunk_result]
            if timeout is not None and time.time() - start <= timeout:
                results += shrunk_result
            elif timeout is None:
                results += shrunk_result
    end = time.time()
    if results != []:
        objective_result = max([len(result) for result in results])
    else:
        print("Failed to find a solution with given hardware within timeout limit")
        objective_result = np.nan

    return objective_result, end, start


def post_selectionDS(samples: List[Graph], k: int):
    """
    Function that filters out sampled subgraphs of incorrect form.

    Args:
        samples: List of subgraphs.
        k: Size of subgraphs.

    Returns:
        Subgraphs in list that are of the correct form.
    """
    n_subg = int(len(samples[0]) / 2 / k)
    accepted = []
    for sample in samples:
        for i in range(n_subg):
            if all(sample[k * i : k * i + k]) == 1:
                accepted.append(sample)
    return accepted


def input_DS(m: int, k: int):
    """
    Required input state for the photonic algorithm.

    Args:
        m: Number of subgraphs.
        k: Size of each subgraph

    Returns:
        |1,1,1,...,0,0,0> k ones and 2*k*m-k zeros
    """
    return np.append(
        np.append(np.ones(k), np.zeros(k * m - k)), np.zeros(k * m)
    ).astype(int)


def construct_B(G: Graph, subG: Graph, k: int):
    """
    Construct the unitary matrix required for the photonic algorithm.

    Args:
        G: Erdös-Renyí graph problem instance.
        subG: List of nodes that has to be in each of the subgraphs to be considered.
        k: Size of subgraph we wish to find

    Returns:
        Matrix B containing all possible subgraphs of size k.
    """

    G_n = len(G.nodes)
    subG_n = len(subG)
    num_subgraphs = int(binom(G_n - subG_n, k - subG_n))
    print("Number of combinations for subgraphs:", num_subgraphs)

    nodes = list(G.nodes)
    test_nodes = [node for node in nodes if node not in subG]
    test_list = list(itertools.combinations(test_nodes, k - subG_n))

    poss_subg = [subG + list(i) for i in test_list]

    # Construction of B
    sub_m = [
        nx.convert_matrix.to_numpy_array(G.subgraph(sub_nodes))
        for sub_nodes in poss_subg
    ]
    B = np.zeros((k * num_subgraphs, k * num_subgraphs))
    for i, j in enumerate(sub_m):
        B[k * i : k * i + k, 0:k] = j
    return (B, poss_subg)


def densest_subgraphs(G: Graph, k: int, Ns: int, backend: Optional[str] = None):
    """
    Sample k-dense subgraphs of a graph G.

    Args:
        G: Erdös-Renyí graph problem instance.
        k: Size of subgraphs that can be returned.
        Ns: Number of samples to generate with quantum device.
        backend: string of Quandela backend instance.

    Returns:
        Subgraphs by order of selection.
    """

    # Initialization and preparing the device
    B = construct_B(G, [], k)
    if int(len(B[0]) / k) == 0:
        print("No potential subgraphs")
        return None
    inputState = input_DS(int(len(B[0]) / k), k)
    U, _ = to_unitary(B[0])
    U = pcvl.Matrix(U)

    # generating samples
    samples = []
    machine = initialize_remote_backend(backend)

    if backend is not None:  # Case remote backend/remote simulator
        circuit = pcvl.Unitary(U)
        if circuit.m > 12:
            print("circuit too large for hardware")
            return None
        machine.set_circuit(circuit)
        machine.with_input(pcvl.BasicState(inputState))
        machine.set_parameters({"HOM": 0.95, "transmittance": 0.1, "g2": 0.01})
        machine.min_detected_photons_filter(1)
        sampler = pcvl.algorithm.Sampler(machine)
        remote_job = sampler.samples.execute_sync(Ns)
        results = remote_job["results"]
        results = [
            ast.literal_eval(str(x).replace("|", "[").replace(">", "]"))
            for x in results
        ]
        samples += results

    else:  # Case local simulator
        local_simulator = machine(U)
        for _ in range(Ns):
            samples.append(list(local_simulator.sample(pcvl.BasicState(inputState))))

    samples = post_selectionDS(samples, k)

    timesG = np.zeros(len(B[1]))
    print(
        "Number of correct samples: ",
        len(samples),
        "\nTotal samples generated:",
        Ns,
    )
    for i in samples:
        indexG = i.index(1)
        timesG[int(indexG / k)] = timesG[int(indexG / k)] + 1
    return sorted(zip(timesG, B[1]), reverse=True)


def to_unitary(A: Union[Graph, NDArray]):
    """
    Embed a matrix into a larger matrix to make it unitary.

    Args:
        A: Matrix to be embedded of size mxm, in matrix or graph form.

    Returns:
        2mx2m unitary matrix that contains A in the top left.
    """

    if type(A) == type(nx.Graph()):
        A = nx.convert_matrix.to_numpy_matrix(A)
    P, D, _ = linalg.svd(A)

    c = np.max(D)
    # if it is not complex, then np.sqrt will output nan in complex values
    An = np.matrix(A / c, dtype=complex)
    P = An
    m = len(An)
    Q = sqrtm(np.identity(m) - np.dot(An, An.conj().T))
    R = sqrtm(np.identity(m) - np.dot(An.conj().T, An))
    S = -An.conj().T
    Ubmat = np.bmat([[P, Q], [R, S]])
    Ubmat = Ubmat.real

    return (np.copy(Ubmat), c)
