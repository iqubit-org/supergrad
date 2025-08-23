# SuperGrad: A Differentiable Simulator for Superconducting Processors

**Abstract**  
We present SuperGrad, a differentiable quantum circuit simulator tailored for superconducting quantum processors. It offers native support for parameterized quantum circuits expressed in terms of microwave control pulses, hardware-aware noise models, and efficient gradient computation via continuous adjoint methods. SuperGrad facilitates optimal control and parameter calibration for near-term superconducting quantum devices.

## 1. Introduction

As quantum processors scale up in size and complexity, designing and calibrating control protocols becomes increasingly challenging. Classical simulation, especially when differentiable, provides a powerful tool for developing and refining these protocols.

SuperGrad is a differentiable simulator focused on superconducting platforms. It models time evolution under microwave control pulses, supports various noise channels, and computes gradients using continuous adjoint methods. SuperGrad interfaces naturally with machine learning frameworks, enabling gradient-based optimization of control waveforms.

## 2. Background and Motivation

### 2.1. Superconducting Quantum Processors

Superconducting qubits such as transmons are controlled via shaped microwave pulses. The system dynamics are governed by a time-dependent Hamiltonian:

$$
H(t; \theta) = H_0 + \sum_i \theta_i(t) H_i
$$

where $\theta_i(t)$ are control amplitudes. Optimization tasks, such as pulse shaping and gate calibration, often involve gradients of an objective function with respect to $\theta_i(t)$.

### 2.2. Differentiable Simulation

Classical simulators typically compute quantum state evolution:

$$
|\psi(t)\rangle = \mathcal{T} \exp\left(-i \int_0^t H(s) ds\right) |\psi(0)\rangle
$$

SuperGrad differentiates this evolution via continuous adjoint sensitivity analysis, allowing efficient gradient computation even for long evolution times.

## 3. Architecture and Features

### 3.1. Modular Design

SuperGrad's core components:

- **Pulse Module**: defines pulse envelopes and parameterization
- **Quantum System Module**: specifies qubit and interaction Hamiltonians
- **Time Evolution Module**: integrates Schrödinger or Lindblad equations
- **Differentiation Engine**: implements the continuous adjoint method
- **Loss Functions**: fidelity, leakage, crosstalk metrics, etc.

### 3.2. Integration with ML Ecosystem

SuperGrad uses JAX as its backend. Users can define differentiable objectives and use any JAX-compatible optimizer (e.g., Adam, Optax) to optimize control pulses.

## 4. Gradient Computation

### 4.1. Continuous Adjoint Method

SuperGrad implements a continuous adjoint method to compute gradients of an objective $\mathcal{L}$ with respect to control parameters $\theta(t)$:

$$
\frac{d\mathcal{L}}{d\theta(t)} = \Re\left[\left\langle \lambda(t) \middle| \frac{\partial H(t)}{\partial \theta(t)} \middle| \psi(t) \right\rangle\right]
$$

where $\lambda(t)$ evolves backward in time via:

$$
\frac{d}{dt} \lambda(t) = i H(t) \lambda(t)
$$

This method is memory efficient and avoids explicit storage of intermediate states.

### 4.2. Memory Scaling

We analyze the memory cost of SuperGrad vs. traditional automatic differentiation. For circuits of depth $T$ and Hilbert space dimension $d$, traditional AD scales as $O(T d^2)$, while the adjoint method used in SuperGrad scales as $O(d^2)$, independent of $T$.

## 5. Benchmarks and Comparisons

### 5.1. Gate Calibration

We test SuperGrad on single- and two-qubit gate calibration tasks. Gradient-based optimization via SuperGrad converges significantly faster than finite-difference methods and achieves higher fidelity.

### 5.2. Performance on Large Systems

We benchmark SuperGrad on systems up to 16 qubits. SuperGrad's performance scales well with parallelization, and the adjoint method provides substantial memory savings.

### 5.3. Comparison with Other Simulators

We compare SuperGrad with:

- **QuTiP**: slower and memory intensive for large systems
- **Qiskit Dynamics**: lacks differentiable simulation support
- **PennyLane**: focuses on gate-level differentiation

SuperGrad offers unique advantages for pulse-level and hardware-aware differentiation.

## 6. Limitations and Future Work

Current limitations:

- Only supports basic noise models (e.g., amplitude damping)
- No support for real-time calibration data from hardware
- GPU scalability could be improved

Planned extensions:

- Add support for noise calibration via data-driven models
- Better GPU parallelization (e.g., NCCL communication)
- Extend to non-superconducting backends (e.g., ion traps)

## 7. Conclusion

SuperGrad is a differentiable simulator specialized for superconducting quantum systems. It supports pulse-level modeling, differentiable time evolution, and memory-efficient gradient computation. By enabling gradient-based control optimization, SuperGrad opens the door to more scalable and automated calibration of quantum devices.
