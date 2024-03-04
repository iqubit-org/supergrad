# SuperGrad: Differentiable Simulator for superconducting quantum processors

This repository serves as a differentiable Hamiltonian simulator, primarily designed for superconducting quantum processors (hence the abbreviation <strong>SuperGrad</strong>). However, it has the potential to be expanded to accommodate other physical platforms. Its key feature lies in its ability to compute gradients with respect to objective functions, such as gate fidelity or functions that approximate logical error rates. The motivation behind gradient computation is rooted in the fact that superconducting processors often consist of numerous qubits, resulting in a rapid increase in the total number of optimizable parameters. Consequently, when optimizing the parameters of a quantum system or fitting them to experimental data, leveraging gradients can significantly accelerate these tasks.

## Installation

We suggest using python version >= 3.9.

```bash
pip install -e .
```

## Examples

### Typical workflow

First, one need to define an interaction graph which describes qubits and their connectivity.
This is done with creating an instance of Networkx.Graph class.
We consider Hamiltonians of the form

```math
H_{\mathrm{idle}}\left(\vec{h}\right)+\sum_{k=1}^{N_c}f_{k}\left(\vec{c},t\right)C_{k}\left(\vec{h}\right)
```

The parameters about only a single qubit are stored in the nodes of the graph.
These include parameters of superconducting qubits such as $`E_C`$, $`E_J`$, and parameters $`\vec{c}`$ of control pulses.
The couplings of qubits are stored in the edges between qubits, which is a subset of $`\vec{h}`$.
Then, some Helper classes will parse the graph, and create functions $`f(\vec{h},\vec{c})`$ which compute time-evolution unitary $`U(\vec{h},\vec{c})`$ or the energy spectrum of $`H_{\mathrm{idle}}(\vec{h})`$.
One can construct objective functions based on these results.
Jax can then be used to compute the gradient of an objective function and use it to run gradient optimization.

In general, we will use Ghz and ns as units for energy and time parameters.

### Fluxonium with multi-path coupling (Main example)

This example is based on Nguyen, L. B. et al. Blueprint for a High-Performance Fluxonium Quantum Processor. PRX Quantum 3, 037001 (2022).
We simulate a 6 fluxonium qubit system from an underlying periodic lattice.
Idling hamiltonian of the system is

```math
H(0) = \sum_i H_{\mathrm{f},i}  + \sum_{\langle i,j \rangle } H_{ij}
```

Hamiltonian of single fluxonium is

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
More details can be found in a future paper.

### Transmon with tunable couplers

This example is based on Xu, Y. et al. High-Fidelity, High-Scalability Two-Qubit Gate Scheme for Superconducting Qubits. Phys. Rev. Lett. 125, 240503 (2020).
We simulate a 5 transmon qubit system, where 3 of them are computational qubits and the other 2 are the couplers.
We compute the time-evolution and the Pauli error rates for
simultaneous single-qubit X gates.

### Fluxonium parameter fitting from experimental spectrum data

This is a quite different application compared to above ones.
Here we try to infer the parameters of the system from spectrum data from experiments.
We will consider the simplest case which is fitting the parameters of one fluxonium.
But the procedure can be applied to more complex systems as well.

## Design choices

### Usage of dm-haiku

dm-haiku provides a good way to organize the large number of device and control
parameters of the systems.
However, we have not implemented a user interface which efficiently utilize the
convenience provided by haiku.
Indeed, it seems that human need to manually input or adjust the majority of the
parameters.
For example, we often need the systems to be dispersively coupled so that the
experiments and numerical simulation can be performed unambiguously.
This might not be the case if we randomly pick parameters for qubits.
In contrary, neural networks are in general initialized with random weights.
Therefore, we think it is open to discussion that whether we should stop using
haiku in the future.
The decision will be easier to make once there are more use cases and examples.
Currently, the main advantage of stopping using haiku is that the code base can
be simpler to understand and the developers do not need to understand the usage
of haiku.

## To-do

### Better interface for describing parameters for optimization and shared parameters

For shared parameters, currently the options are very limited.
As examples, one might want to share a subset of qubit parameters, or share part
of the pulse parameters like the length of pulses.
It will be more clear how to make a better interface when there are more use cases.
