# TNO-Quantum / qscore_dwave

## Q-score evaluation on D-Wave solvers

This repository contains python code to run the Q-score benchmark on five different D-Wave devices and solvers, namely its Advantage and 2000-Q QPU solvers, its Simulated Annealing, its qbsolv classical solver, and its hybrid solver. For an introduction to the Q-score, see reference below. The code allows running a single Max-Cut instance on each of the five solvers with different sizes and timeout limits. The code returns both the Max-Cut result as the corresponding beta value. If no result is found within the allowed time limit, no Max-Cut result and a beta value of 0 are returned. Note that for the QPU solvers, the time limit considers embedding time only. The actual computation time will be slightly higher, but this difference will be in the order of milliseconds and will hence not influence the results. To compute the Q-score, one runs for increasing graph size sufficiently many instances of the given code to check whether the average beta is larger than 0.2.

This code was used to obtain results for the paper: "Evaluating the Q-score of quantum annealers", Ward van der Schoot et al. (IEEE QSW 2022).

## Set up D-Wave configuration
To use this code, we assume the reader has created a D-Wave Leap account and configured access to D-Wave's solvers correctly. To make an account, please visit https://cloud.dwavesys.com/leap/login/?next=/leap/ and to configure access to the solvers, please visit https://docs.ocean.dwavesys.com/en/stable/overview/sapi.html.

## Usage
The Q-score evaluation can be run as follows:

```python
# Run a Max-Cut problem instance of size 10 on the Advantage QPU solver of D-Wave with a time limit of 60 seconds, returning 100 reads.
python evaluation.py -s 10 -t 60 -n 100 -solver "Advantage_system4.1"
```

## Q-score introduction

In December 2020, Atos introduced a new quantum metric, called Q-score, that is supposed to be a universal quantum metric applicable to all programmable quantum processors. The idea behind Q-score is to determine how well a quantum system can solve the Max-Cut problem, a real-life combinatorial problem. The Q-score is determined by the maximum number of variables within such a problem that the quantum system can optimize for, see [this paper](https://arxiv.org/abs/2102.12973) for a more elaborate description. 

The official announcement of Atos can be found [here](https://atos.net/en/2020/press-release_2020_12_04/atos-announces-q-score-the-only-universal-metrics-to-assess-quantum-performance-and-superiority) and the project with tools to calculate Q-score benchmarks on gate-based devices can be found in this [gitlab project](https://github.com/myQLM/qscore).
