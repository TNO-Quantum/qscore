"""
Run a Q-score instance using QAOA.
"""
import os
import time
from multiprocessing import AuthenticationError
from typing import Optional, Union

from qiskit import IBMQ, Aer
from qiskit.algorithms import QAOA
from qiskit.providers.backend import Backend
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.providers.ibmq import IBMQAccountCredentialsNotFound
from qiskit_optimization.algorithms import (
    MinimumEigenOptimizer,
    OptimizationResultStatus,
)
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.runtime import QAOAClient
from quantuminspire import credentials as qicredentials
from quantuminspire.qiskit import QI

HUB = "ibm-q"
GROUP = "open"
PROJECT = "main"


def initialize_backend(
    provider: Union[str, None],
    backend: str,
) -> Backend:
    """Initialize QAOA backend from either ibm or qi.

    Args:
        provider: Name of hardware provider.
        backend: Name of specific hardware.

    Returns:
        Backend instance and its provider.

    Raises:
        AuthenticationError: if authentication to ibm or qi failed.
        ValueError: if provider or backend is not among supported options.
    """
    if provider is None:
        return Aer.get_backend("qasm_simulator"), provider
    elif provider.lower() == "statevector simulator":
        return Aer.get_backend("aer_simulator_statevector"), provider
    elif provider.lower() == "qasm simulator":
        return Aer.get_backend("qasm_simulator"), provider

    elif provider.lower() == "ibm":
        try:
            IBMQ.load_account()
        except IBMQAccountCredentialsNotFound:
            raise AuthenticationError("Authentication with IBM failed.")
        provider = IBMQ.get_provider(hub=HUB, group=GROUP, project=PROJECT)
        try:
            backend = provider.get_backend(backend)
            return backend, provider
        except QiskitBackendNotFoundError:
            raise ValueError(f"Backend {backend} not among possible options.")

    elif provider.lower() == "qi":
        try:
            token = qicredentials.load_account()
            qi_authentication = qicredentials.get_token_authentication(token)
            QI_URL = os.getenv("API_URL", "https://api.quantum-inspire.com/")
            project_name = f"TNO Q-score {int(time.time())}"

            QI.set_authentication(qi_authentication, QI_URL, project_name=project_name)
            return QI.get_backend(backend), QI
        except:
            raise AuthenticationError("Authentication with QI failed.")

    raise ValueError(
        f"Provider {provider} and backend {backend} not among possible options."
    )


def run_QAOA(
    qp: QuadraticProgram,
    provider: Optional[str] = None,
    backend: Optional[str] = None,
    number_of_shots: Optional[int] = None,
    max_attempts: Optional[int] = 10,
) -> float:
    """
    Function that solves a Q-score instance using QAOA.

    Args:
        qp: Quadratic Problem of Q-score instance.
        backend: Name of the quantum instance backend.
        number_of_shots: Number of shots for hardware.

    Returns:
        The found objective value.

    Raises:
        ValueError: if no feasible solution was found in max_attempts attempts.
    """
    if number_of_shots is None:
        number_of_shots = 1024

    backend, provider_ = initialize_backend(provider, backend)
    backend.shots = number_of_shots
    if provider == "ibm":
        qaoa_mes = QAOAClient(
            provider=provider_,
            backend=backend,
            reps=1,
        )
    else:
        qaoa_mes = QAOA(reps=1, quantum_instance=backend)

    qaoa = MinimumEigenOptimizer(qaoa_mes)
    for _ in range(max_attempts):
        qaoa_result = qaoa.solve(qp)
        if qaoa_result.status == OptimizationResultStatus.SUCCESS:
            return qaoa_result.fval
    raise ValueError("Could not find feasible solution")


if __name__ == "__main__":
    b = initialize_backend("qi", "QX single-node simulator")
