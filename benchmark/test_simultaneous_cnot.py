import os
import sys
import numpy as np
import jax
import pytest

from qiskit_dynamics import Solver
import qutip as qt
from supergrad.utils.qiskit_interface import (to_qiskit_static_hamiltonian,
                                              to_qiskit_drive_hamiltonian)
from supergrad.utils.qutip_interface import (to_qutip_operator,
                                             to_qutip_operator_function_pair)

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('/'.join(dir_path.split('/')[:-1]))

from benchmark.utils.create_simultaneous_model import create_simultaneous_cnot
from benchmark.utils.fidelity import fidelity

nqubit_list = range(2, 12, 2)
diag_ops_list = [True, False]
evaluation_mode_list = ['dense', 'sparse']


@pytest.mark.benchmark_gpu
@pytest.mark.benchmark_cpu
@pytest.mark.parametrize('n_qubit', nqubit_list)
@pytest.mark.parametrize('diag_ops', diag_ops_list)
def test_simultaneous_cnot_supergrad(benchmark, n_qubit, diag_ops):
    # bench supergrad
    evo = create_simultaneous_cnot(n_qubit=n_qubit,
                                   astep=3000,
                                   trotter_order=2,
                                   diag_ops=diag_ops)
    benchmark.group = f'simultaneous_cnot_{n_qubit}_qubits'
    # benchmark

    @benchmark
    def u_supergrad():
        return jax.jit(evo.product_basis)(evo.all_params).block_until_ready()

    u_ref = np.load(f'{dir_path}/ref_data/data_simultaneous_cnot_{n_qubit}.npy')
    assert abs(1 - fidelity(u_ref, u_supergrad)) < 1e-5


@pytest.mark.benchmark_cpu
@pytest.mark.parametrize('n_qubit', nqubit_list)
@pytest.mark.parametrize('evaluation_mode', evaluation_mode_list)
def test_simultaneous_cnot_qiskit(benchmark, n_qubit, evaluation_mode):
    # bench qiskit
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
    solver = Solver(static_hamiltonian,
                    drive_hamiltonian,
                    rotating_frame=static_hamiltonian,
                    evaluation_mode=evaluation_mode)
    u0 = np.eye(np.prod(evo.dims), dtype=complex)
    benchmark.group = f'simultaneous_cnot_{n_qubit}_qubits'

    @benchmark
    def u_qiskit():
        results = solver.solve(
            t_span=[0, t_span],
            y0=u0,
            signals=drive_signal,
            atol=1e-8,
            rtol=1e-8,
        )
        return solver.model.rotating_frame.state_out_of_frame(
            t_span, results.y[-1])

    u_ref = np.load(f'{dir_path}/ref_data/data_simultaneous_cnot_{n_qubit}.npy')
    assert abs(1 - fidelity(u_ref, u_qiskit)) < 1e-5


@pytest.mark.benchmark_cpu
@pytest.mark.parametrize('n_qubit', nqubit_list)
def test_simultaneous_cnot_qutip(benchmark, n_qubit):
    # bench qutip
    evo = create_simultaneous_cnot(n_qubit=n_qubit,
                                   astep=None,
                                   trotter_order=None,
                                   diag_ops=None)
    # the args do not matter for derive Hamiltonian
    ham_static, hamiltonian_component_and_pulseshape, t_span = evo.construct_hamiltonian_and_pulseshape(
        evo.all_params)
    ham_static_qutip = to_qutip_operator(ham_static)
    operator_function_pair_list = to_qutip_operator_function_pair(
        hamiltonian_component_and_pulseshape)

    u0 = qt.qeye(list(evo.dims))
    # config solver option
    options = qt.Options(nsteps=1000000, atol=1e-8, rtol=1e-8)
    benchmark.group = f'simultaneous_cnot_{n_qubit}_qubits'

    @benchmark
    def u_qutip():
        output = qt.sesolve([ham_static_qutip] + operator_function_pair_list,
                            u0, [0.0, t_span],
                            options=options)
        return output.states[-1].full()

    u_ref = np.load(f'{dir_path}/ref_data/data_simultaneous_cnot_{n_qubit}.npy')
    assert abs(1 - fidelity(u_ref, u_qutip)) < 1e-5
