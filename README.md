# TNO-Quantum / qscore

## Q-score evaluation

This repository contains python code to run the Q-score (Max-Cut and Max-Clique) benchmark on the following solver backends:

- D-Wave devices and solvers, precisely its `Advantage_system4.1` and `DW_2000Q_6` QPU solvers, its `Simulated Annealing` and `qbsolv` classical solver, and its `hybrid` solver.
- Gate-based hardware using QAOA on QuantumInspire and IBM hardware or simulators.
- Simulator for Gaussian Boson Sampling, a form of photonic quantum computing.   

For an introduction to the Q-score, see the reference below.
Q-score instances for Max-Cut or Max-Clique optimization problem can be run for different sizes and timeout limits. If no result is found within the allowed time limit, no objective result and a beta value of `0` is returned. Note that for the QPU solvers, the time limit considers embedding time only. The actual computation time will be slightly higher, but this difference will be in the order of milliseconds and will hence not influence the results. For similar reasons, for the photonic simulator, we only apply the time constraint to the classical runtime of the algorithm. To compute the Q-score, one runs for increasing graph size sufficiently many instances of the given code to check whether the average beta is larger than 0.2.

This code was used to obtain results for the following papers:

- ["Evaluating the Q-score of quantum annealers", Ward van der Schoot et al. (IEEE QSW 2022)](https://ieeexplore.ieee.org/document/9860191).
- ["Q-score Max-Clique: The first metric evaluation on multiple computational paradigms", Ward van der Schoot et al. (preprint)](https://arxiv.org/abs/2302.00639)

## Q-score introduction

In December 2020, Atos introduced a new quantum metric, called the Q-score, which is supposed to be a universal quantum metric applicable to all programmable quantum processors. The idea behind Q-score is to determine how well a quantum system can solve the Max-Cut problem, which is a real-life combinatorial problem. The Q-score is determined by the maximum number of variables within such a problem that the quantum system can optimize for. For a more elaborate description, see [this paper](https://arxiv.org/abs/2102.12973). 

The official announcement of Atos can be found [here](https://atos.net/en/2020/press-release_2020_12_04/atos-announces-q-score-the-only-universal-metrics-to-assess-quantum-performance-and-superiority) and the project with tools to calculate Q-score benchmarks on gate-based devices can be found in this [gitlab project](https://github.com/myQLM/qscore).

## Usage
A single Q-score instance can be run as follows:

```python
# Run a Max-Cut problem instance of size 10 on the Advantage QPU solver of D-Wave with a time limit of 60 seconds, returning 100 reads.
python evaluate.py -p "max-cut" -s 10 -t 60 -n 100 -solver "Advantage_system4.1"
```

Multiple Q-score instances for various sizes can be run as follows:

1. Modify the following parameters accordingly in `calculate_qscore.py`
    ```python
    # Input arguments
    _NB_INSTANCES_PER_SIZE = 10
    _SIZE_RANGE = list(range(2, 9, 2))
    FILE_NAME = "example.json"
    INCLUDE_EXACT_RESULTS = True
    PROBLEM_TYPE = "max-clique"
    TIMEOUT = 60
    SOLVER = "Simulated_Annealing"
    _SEED = 49430557
    NUM_READS = 1024
    PROVIDER = None
    BACKEND = None
    ```
2. Run the `calculate_qscore` script. A json file with results will be created inside the `data` folder.
    ```python
    python calculate_qscore.py
    ```

3. The beta vs N and time vs N Q-score graphs can be plotted for a given results file by running `plot_qscore.py`:

    ```python
    # Plot Q-score graph for results file
    python plot_qscore.py -f "example.json" -e
    ```

## Configuration

To use this code we assume that the reader has installed the requirements and set up access to the required solvers. 

Requirements can be installed using pip:
```terminal
python -m pip install -r requirements.txt
```

### Set up D-Wave configuration
To use the quantum annealing, simulated annealing, qbsolv or hybrid solver, we assume the reader has created a D-Wave Leap account and configured access to D-Wave's solvers correctly. To make an account, please visit the [website of D-wave](https://cloud.dwavesys.com/leap/login/?next=/leap/). To configure access to the solvers, please visit [these instructions](https://docs.ocean.dwavesys.com/en/stable/overview/sapi.html).

Example usage for the `Advantage_system4.1` QPU solver:

```python
python evaluate.py -p "max-cut" -s 10 -t 60 -n 100 -solver "Advantage_system4.1"
```
### Set up QuantumInspire configuration

To use the Quantum Inspire backend we assume the reader has created a QuantumInspire account and configured access correctly.

1. Create a Quantum Inspire account (https://www.quantum-inspire.com/)
2. Get an API token from the Quantum Inspire website.
3. With your API token run: 

```python
from quantuminspire.credentials import save_account
save_account('YOUR_API_TOKEN')
```

Example usage for the `Starmon-5` hardware backend:
```python
python evaluate.py -p "max-clique" -s 5 -t 60 -n 1024 -solver "QAOA" -provider "qi" -backend "Starmon-5" 
```

### Set up IBM configuration

To use IBM hardware backends, one need to create and IBM Quantum account, which can be done [here](https://quantum-computing.ibm.com/lab). 
After creating your account you can install your API key using:

```python
from qiskit import IBMQ		
IBMQ.save_account('MY_API_TOKEN')
```

Example usage for the IBM Lima device:
```python
python evaluate.py -p "max-clique" -s 5 -t 60 -n 1024 -solver "QAOA" -provider "ibm" -backend "ibmq_lima" 
```
