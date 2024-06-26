Quantum System
===============

.. currentmodule:: supergrad.quantum_system

.. automodule:: supergrad.quantum_system

Artificial Atom
------------------

.. autosummary::
    :toctree: _autosummary

    QuantumSystem
    CircuitLCJ
    CircuitLCJ.set_phi_basis
    CircuitLCJ.set_charge_basis
    CircuitLCJ.unify_state_phase
    Transmon
    Fluxonium
    Resonator

Interaction
---------------

.. autosummary::
    :toctree: _autosummary

    parse_interaction
    InteractionTerm
    InteractingSystem
    InteractingSystem.compute_energy_map
    InteractingSystem.transform_operator

Data Structure
---------------

.. autosummary::
    :toctree: _autosummary

    KronObj
    KronObj.compute_contraction_path
    KronObj.expm
    LindbladObj
    LindbladObj.add_lindblad_operator
    LindbladObj.compute_contraction_path
    LindbladObj.expm
