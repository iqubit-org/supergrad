__all__ = ['generate_x_data', 'generate_cnot_data']

import numpy as np

from qiskit_dynamics import Solver
from supergrad.utils.qiskit_interface import (to_qiskit_static_hamiltonian,
                                              to_qiskit_drive_hamiltonian)

from benchmark.utils.create_simultaneous_model import (create_simultaneous_x,
                                                       create_simultaneous_cnot)


class REFOptions(object):
    atol = 5e-14
    rtol = 5e-14
    max_qubit = 11


def compute_evolution_x(n_qubit):
    """Compute the evolution unitary by ode solver in rotating frame."""
    evo = create_simultaneous_x(n_qubit=n_qubit,
                                astep=None,
                                trotter_order=None,
                                diag_ops=None)
    # the args do not matter for derive Hamiltonian
    ham_static, hamiltonian_component_and_pulseshape, t_span = evo.construct_hamiltonian_and_pulseshape(
        evo.all_params)
    static_hamiltonian = to_qiskit_static_hamiltonian(ham_static)
    drive_hamiltonian, drive_signal = to_qiskit_drive_hamiltonian(
        hamiltonian_component_and_pulseshape)
    # Evolving by qiskit dynamics
    solver = Solver(
        static_hamiltonian,
        drive_hamiltonian,
        rotating_frame=static_hamiltonian,
    )
    u0 = np.eye(np.prod(evo.get_dims(evo.all_params)), dtype=complex)
    results = solver.solve(
        t_span=[0, t_span],
        y0=u0,
        signals=drive_signal,
        atol=REFOptions.atol,
        rtol=REFOptions.rtol,
    )
    return solver.model.rotating_frame.state_out_of_frame(t_span, results.y[-1])


def compute_evolution_cnot(n_qubit):
    """Compute the evolution unitary by ode solver in rotating frame."""
    evo = create_simultaneous_cnot(n_qubit=n_qubit,
                                   astep=None,
                                   trotter_order=None,
                                   diag_ops=None)
    # the args do not matter for derive Hamiltonian
    ham_static, hamiltonian_component_and_pulseshape, t_span = evo.construct_hamiltonian_and_pulseshape(
        evo.all_params)
    static_hamiltonian = to_qiskit_static_hamiltonian(ham_static)
    drive_hamiltonian, drive_signal = to_qiskit_drive_hamiltonian(
        hamiltonian_component_and_pulseshape)
    # Evolving by qiskit dynamics
    solver = Solver(
        static_hamiltonian,
        drive_hamiltonian,
        rotating_frame=static_hamiltonian,
    )
    u0 = np.eye(np.prod(evo.get_dims(evo.all_params)), dtype=complex)
    results = solver.solve(
        t_span=[0, t_span],
        y0=u0,
        signals=drive_signal,
        atol=REFOptions.atol,
        rtol=REFOptions.rtol,
    )
    return solver.model.rotating_frame.state_out_of_frame(t_span, results.y[-1])


def generate_x_data():
    n_qubit_range = range(1, REFOptions.max_qubit + 1)
    for n_qubit in n_qubit_range:
        u_qiskit = compute_evolution_x(n_qubit)
        np.save(f'data_simultaneous_x_{n_qubit}.npy', u_qiskit)
        print(
            f'Create data for simultaneous X gate in {n_qubit} qubit model is finished.'
        )
    return 0


def generate_cnot_data():
    n_qubit_range = range(2, REFOptions.max_qubit + 1, 2)
    for n_qubit in n_qubit_range:
        u_qiskit = compute_evolution_cnot(n_qubit)
        np.save(f'data_simultaneous_cnot_{n_qubit}.npy', u_qiskit)
        print(
            f'Create data for simultaneous CNOT gate in {n_qubit} qubit model is finished.'
        )
    return 0
