"""
Common dissipation channels for open quantum systems.

This module provides implementations of standard dissipation channels
"""

import jax.numpy as jnp
from supergrad.quantum_system import KronObj
from supergrad.utils.operators import sigmax, sigmay, sigmaz, destroy


def amplitude_damping_operator(qubit_index, rate, dims):
    """Create amplitude damping collapse operator for a specific qubit.

    Amplitude damping models energy relaxation from |1⟩ to |0⟩.
    The collapse operator is √γ σ₋ where σ₋ = |0⟩⟨1|.

    Args:
        qubit_index (int): Index of the qubit in the system
        rate (float): Damping rate γ
        dims (list): Dimensions of all subsystems

    Returns:
        KronObj: Collapse operator for amplitude damping
    """
    sigma_minus = destroy(2)  # |0⟩⟨1|
    return KronObj([jnp.sqrt(rate) * sigma_minus], dims, locs=[qubit_index])


def phase_damping_operator(qubit_index, rate, dims):
    """Create phase damping collapse operator for a specific qubit.

    Phase damping models pure dephasing without energy loss.
    The collapse operator is √γ σᵧ where σᵧ = |0⟩⟨0| - |1⟩⟨1|.

    Args:
        qubit_index (int): Index of the qubit in the system
        rate (float): Dephasing rate γ
        dims (list): Dimensions of all subsystems

    Returns:
        KronObj: Collapse operator for phase damping
    """
    sigma_z = sigmaz()  # |0⟩⟨0| - |1⟩⟨1|
    return KronObj([jnp.sqrt(rate) * sigma_z], dims, locs=[qubit_index])
