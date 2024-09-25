# SuperGrad: Differentiable Simulator for superconducting quantum processors
[![GitHub actions](https://img.shields.io/github/actions/workflow/status/iqubit-org/supergrad/python-publish.yml)](https://github.com/iqubit-org/supergrad/actions/workflows/python-publish.yml)
[![Docs](https://img.shields.io/badge/docs-link-green.svg?logo=read-the-docs)](https://supergrad.readthedocs.io/)
[![PyPI](https://img.shields.io/pypi/v/supergrad.svg?logo=pypi)](https://pypi.org/project/supergrad/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg?logo=apache)](https://github.com/iqubit-org/supergrad/blob/main/LICENSE)

SuperGrad is an open-source simulator designed to accelerate the development of superconducting quantum processors by incorporating gradient computation capabilities.

Notice: This package is currently in its early development stages. Please be aware that breaking changes to the API may occur.

## Why SuperGrad?

Superconducting processors offer significant design flexibility, including various types of qubits and interactions. With the large number of tunable parameters in a processor, gradient optimization becomes crucial. SuperGrad fills the gap in open-source software by providing a tightly integrated library capable of efficient backpropagation for gradient computation.

## Key Features

- Efficient backpropagation for gradient computation based on [JAX](https://github.com/google/jax/).
- User-friendly interface for constructing Hamiltonians
- Computation of both static and dynamic properties of composite systems, such as the unitary matrices of simultaneous gates

These features help us to speedup tasks including

- Optimal control
- Design optimization
- Experimental data fitting

## Installation

We suggest using python version >= 3.9.

```bash
pip install supergrad
```

## Examples

### Typical workflow

First, one need to define an interaction graph which describes qubits and their connectivity.
This is done with creating an instance of `Networkx.Graph` class.
There is a work-in-progress GUI for creating such graphs available at [https://github.com/iqubit-org/supergrad-gui](https://github.com/iqubit-org/supergrad-gui).

We consider Hamiltonians of the form

```math
H_{\mathrm{idle}}\left(\vec{h}\right)+\sum_{k=1}^{N_c}f_{k}\left(\vec{c},t\right)C_{k}\left(\vec{h}\right)
```

The parameters about only a single qubit are stored in the nodes of the graph.
These include parameters of superconducting qubits such as $E_C$, $E_J$, and parameters $\vec{c}$ of control pulses.
The couplings of qubits are stored in the edges between qubits, which is a subset of $\vec{h}$.
Then, some Helper classes will parse the graph, and create functions $f(\vec{h},\vec{c})$ which compute time-evolution unitary $U(\vec{h},\vec{c})$ or the energy spectrum of $H_{\mathrm{idle}}(\vec{h})$.
One can construct objective functions based on these results.
JAX can then be used to compute the gradient of an objective function and use it to run gradient optimization.

In general, we will use GHz and ns as units for energy and time parameters.

### Fluxonium with multi-path coupling (Main example)

[The jupyter notebook](https://github.com/iqubit-org/supergrad/blob/main/examples/fluxonium_multipath_coupling/fluxonium_mpcoupling_6q_cnot.ipynb)

This example is based on Nguyen, L. B. et al. Blueprint for a High-Performance Fluxonium Quantum Processor. PRX Quantum 3, 037001 (2022).
We simulate a 6 Fluxonium qubit system from an underlying periodic lattice.
Idling hamiltonian of the system is

```math
H(0) = \sum_i H_{\mathrm{f},i}  + \sum_{\langle i,j \rangle } H_{ij}
```

Hamiltonian of single Fluxonium is

```math
H_{\mathrm{f},i} = 4E_{\mathrm{C},i} n_i^2 + \frac{1}{2} E_{\mathrm{L},i} (\varphi_i +\varphi_{\mathrm{ext},i})^2
    -E_{\mathrm{J},i}\cos \left( \varphi_i \right)
```

The coupling terms have the form

```math
H_{ij} = J_{\mathrm{C}} n_i n_j - J_{\mathrm{L}} \varphi_i \varphi_j
```

The couplings are chosen in a way such that the idling $`ZZ`$-crosstalk is almost zero.
We compute the time-evolution and the Pauli error rates for
simultaneous single-qubit X gates and two-qubit CR gates.
More details can be found in [Ni, X. et al. Superconducting processor design optimization for quantum error correction performance. arXiv:2312.04186](https://arxiv.org/pdf/2312.04186).

### Transmon with tunable couplers

[The jupyter notebook](https://github.com/iqubit-org/supergrad/blob/main/examples/transmon_coupler/transmon_5q_x.ipynb)

This example is based on Xu, Y. et al. High-Fidelity, High-Scalability Two-Qubit Gate Scheme for Superconducting Qubits. Phys. Rev. Lett. 125, 240503 (2020).
We simulate a 5 transmon qubit system, where 3 of them are computational qubits and the other 2 are the couplers.
We compute the time-evolution and the Pauli error rates for
simultaneous single-qubit X gates.

### Fluxonium parameter fitting from experimental spectrum data

[The jupyter notebook](https://github.com/iqubit-org/supergrad/blob/main/examples/spectrum_fitting_fluxonium/spectrum_fitting.ipynb)

This is a quite different application compared to above ones.
Here we try to infer the parameters of the system from spectrum data from experiments.
We will consider the simplest case which is fitting the parameters of one Fluxonium.
But the procedure can be applied to more complex systems as well.

## Citation

If this project is helpful to you in your research, the use of SuperGrad in research publications is appropriately acknowledged by citing:

```
@misc{supergrad_2024,
      title={SuperGrad: a differentiable simulator for superconducting processors},
      author={Ziang Wang and Feng Wu and Hui-Hai Zhao and Xin Wan and Xiaotong Ni},
      year={2024},
      eprint={2406.18155},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```

which is also a good introduction to the simulator.
