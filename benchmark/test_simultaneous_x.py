import os
import sys
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from qiskit_dynamics import Solver
import qutip as qt
import supergrad
from supergrad.helper import Evolve
from supergrad.utils import basis
from supergrad.utils.fidelity import compute_fidelity_with_1q_rotation_axis
from supergrad.utils.memory_profiling import trace_max_memory_usage
from supergrad.utils.qiskit_interface import (to_qiskit_static_hamiltonian,
                                              to_qiskit_drive_hamiltonian)
from supergrad.utils.qutip_interface import (to_qutip_operator,
                                             to_qutip_operator_function_pair)
from supergrad.utils.sharding import distributed_state_fidelity

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('/'.join(dir_path.split('/')[:-1]))

from supergrad.scgraph.graph_mpc_fluxonium_1d import MPCFluxonium1D
from benchmark.utils.create_simultaneous_model import create_simultaneous_x
from benchmark.utils.fidelity import fidelity

nqubit_list = range(1, 12)
extreme_nqubit_list = range(2, 15)
diag_ops_list = [True, False]
evaluation_mode_list = ['dense', 'sparse']
astep_list = jnp.linspace(500, 5000, 21, dtype=int).tolist()
trotter_order_list = [1, 2, 4j, 4, None]


@pytest.mark.benchmark_grad
@pytest.mark.parametrize('n_qubit', nqubit_list)
def test_simultaneous_x_state_grad_lcam(benchmark, n_qubit):
    # bench supergrad
    evo = create_simultaneous_x(n_qubit=n_qubit,
                                astep=5000,
                                trotter_order=2,
                                diag_ops=True,
                                minimal_approach=True,
                                custom_vjp=True)
    benchmark.group = f'gradient_simultaneous_x_state_{n_qubit}_qubits'
    # benchmark
    target_state = basis(2**n_qubit, 2**n_qubit - 1)

    @jax.jit
    @jax.value_and_grad
    def infidelity(params):
        initial_state = basis(2**n_qubit)
        output = evo.product_basis(params, initial_state)
        return 1 - distributed_state_fidelity(target_state, output)

    @benchmark
    @partial(trace_max_memory_usage, pid=os.getpid())
    @jax.block_until_ready
    def vg_infidelity():
        return infidelity(evo.all_params)

    benchmark.extra_info.update({'memory': vg_infidelity[1]})
    # save the gradient
    jnp.save(
        f'{dir_path}/res_data/grad_lcam_simultaneous_x_state_{n_qubit}.npy',
        vg_infidelity[0])
    assert len(vg_infidelity[0]) == 2


@pytest.mark.benchmark_grad
@pytest.mark.parametrize('n_qubit', nqubit_list)
def test_simultaneous_x_state_grad_tad(benchmark, n_qubit):
    # bench supergrad
    evo = create_simultaneous_x(n_qubit=n_qubit,
                                astep=5000,
                                trotter_order=2,
                                diag_ops=True,
                                minimal_approach=True,
                                custom_vjp=None)
    benchmark.group = f'gradient_simultaneous_x_state_{n_qubit}_qubits'
    # benchmark
    target_state = basis(2**n_qubit, 2**n_qubit - 1)

    @jax.jit
    @jax.value_and_grad
    def infidelity(params):
        initial_state = basis(2**n_qubit)
        output = evo.product_basis(params, initial_state)
        return 1 - distributed_state_fidelity(target_state, output)

    @benchmark
    @partial(trace_max_memory_usage, pid=os.getpid())
    @jax.block_until_ready
    def vg_infidelity():
        return infidelity(evo.all_params)

    benchmark.extra_info.update({'memory': vg_infidelity[1]})
    # save the gradient
    jnp.save(f'{dir_path}/res_data/grad_tad_simultaneous_x_state_{n_qubit}.npy',
             vg_infidelity[0])
    assert len(vg_infidelity[0]) == 2


@pytest.mark.benchmark_grad
@pytest.mark.parametrize('n_qubit', nqubit_list)
def test_simultaneous_x_grad_lcam(benchmark, n_qubit):
    # bench supergrad
    evo = create_simultaneous_x(n_qubit=n_qubit,
                                astep=5000,
                                trotter_order=2,
                                diag_ops=True,
                                minimal_approach=True,
                                custom_vjp=True)
    benchmark.group = f'gradient_simultaneous_x_{n_qubit}_qubits'
    # benchmark
    u_ref = supergrad.tensor(*[np.array([[0, 1], [1, 0]])] * n_qubit)
    u_ref = jax.device_put(u_ref, jax.devices('cpu')[0])

    @jax.jit
    @jax.value_and_grad
    def infidelity(params):
        return 1 - distributed_state_fidelity(u_ref, evo.product_basis(params))

    @benchmark
    @partial(trace_max_memory_usage, pid=os.getpid())
    @jax.block_until_ready
    def vg_infidelity():
        return infidelity(evo.all_params)

    benchmark.extra_info.update({'memory': vg_infidelity[1]})
    # save the gradient
    jnp.save(f'{dir_path}/res_data/grad_lcam_simultaneous_x_{n_qubit}.npy',
             vg_infidelity[0])
    assert len(vg_infidelity[0]) == 2


@pytest.mark.benchmark_grad
@pytest.mark.parametrize('n_qubit', nqubit_list)
def test_simultaneous_x_grad_tad(benchmark, n_qubit):
    # bench supergrad
    evo = create_simultaneous_x(n_qubit=n_qubit,
                                astep=5000,
                                trotter_order=2,
                                diag_ops=True,
                                minimal_approach=True,
                                custom_vjp=None)
    benchmark.group = f'gradient_simultaneous_x_{n_qubit}_qubits'
    # benchmark
    u_ref = supergrad.tensor(*[np.array([[0, 1], [1, 0]])] * n_qubit)
    u_ref = jax.device_put(u_ref, jax.devices('cpu')[0])

    @jax.jit
    @jax.value_and_grad
    def infidelity(params):
        return 1 - distributed_state_fidelity(u_ref, evo.product_basis(params))

    @benchmark
    @partial(trace_max_memory_usage, pid=os.getpid())
    @jax.block_until_ready
    def vg_infidelity():
        return infidelity(evo.all_params)

    benchmark.extra_info.update({'memory': vg_infidelity[1]})
    # save the gradient
    jnp.save(f'{dir_path}/res_data/grad_tad_simultaneous_x_{n_qubit}.npy',
             vg_infidelity[0])
    assert len(vg_infidelity[0]) == 2


@pytest.mark.benchmark_grad
@pytest.mark.parametrize('n_qubit', nqubit_list)
def test_simultaneous_x_grad_continuous(benchmark, n_qubit):
    # bench supergrad
    evo = create_simultaneous_x(n_qubit=n_qubit,
                                astep=5000,
                                trotter_order=2,
                                diag_ops=True,
                                minimal_approach=True,
                                custom_vjp='CAM')
    benchmark.group = f'gradient_simultaneous_x_{n_qubit}_qubits'
    # benchmark
    u_ref = supergrad.tensor(*[np.array([[0, 1], [1, 0]])] * n_qubit)
    u_ref = jax.device_put(u_ref, jax.devices('cpu')[0])

    @jax.jit
    @jax.value_and_grad
    def infidelity(params):
        return 1 - distributed_state_fidelity(u_ref, evo.product_basis(params))

    @benchmark
    @partial(trace_max_memory_usage, pid=os.getpid())
    @jax.block_until_ready
    def vg_infidelity():
        return infidelity(evo.all_params)

    benchmark.extra_info.update({'memory': vg_infidelity[1]})
    # save the gradient
    jnp.save(
        f'{dir_path}/res_data/grad_continuous_simultaneous_x_{n_qubit}.npy',
        vg_infidelity[0])
    assert len(vg_infidelity[0]) == 2


@pytest.mark.benchmark_cpu
@pytest.mark.parametrize('n_qubit', nqubit_list)
def test_simultaneous_x_grad_odeint(benchmark, n_qubit):
    # bench supergrad
    chain = MPCFluxonium1D(n_qubit, periodic=False, seed=42)
    chain.create_single_qubit_pulse(range(n_qubit), [50.0] * n_qubit,
                                    True,
                                    minimal_approach=True)

    evo = Evolve(chain,
                 truncated_dim=2,
                 compensation_option='no_comp',
                 solver='odeint',
                 options={
                     'atol': 1e-10,
                     'rtol': 1e-10,
                 })

    benchmark.group = f'gradient_simultaneous_x_{n_qubit}_qubits'
    # benchmark
    u_ref = supergrad.tensor(*[np.array([[0, 1], [1, 0]])] * n_qubit)
    u_ref = jax.device_put(u_ref, jax.devices('cpu')[0])

    @jax.jit
    @jax.value_and_grad
    def infidelity(params):
        return 1 - distributed_state_fidelity(u_ref, evo.product_basis(params))

    @benchmark
    @partial(trace_max_memory_usage, pid=os.getpid())
    @jax.block_until_ready
    def vg_infidelity():
        return infidelity(evo.all_params)

    benchmark.extra_info.update({'memory': vg_infidelity[1]})
    # save the gradient
    jnp.save(f'{dir_path}/res_data/grad_odeint_simultaneous_x_{n_qubit}.npy',
             vg_infidelity[0])
    assert len(vg_infidelity[0]) == 2


@pytest.mark.benchmark_gpu
@pytest.mark.benchmark_cpu
@pytest.mark.parametrize('n_qubit', nqubit_list)
@pytest.mark.parametrize('diag_ops', diag_ops_list)
def test_simultaneous_x_supergrad(benchmark, n_qubit, diag_ops):
    # bench supergrad
    evo = create_simultaneous_x(n_qubit=n_qubit,
                                astep=500,
                                trotter_order=2,
                                diag_ops=diag_ops)
    benchmark.group = f'simultaneous_x_{n_qubit}_qubits'
    # benchmark

    @benchmark
    def u_supergrad():
        return jax.jit(evo.product_basis)(evo.all_params).block_until_ready()

    u_ref = np.load(f'{dir_path}/ref_data/data_simultaneous_x_{n_qubit}.npy')
    assert abs(1 - fidelity(u_ref, u_supergrad)) < 1e-5


@pytest.mark.benchmark_gpu
@pytest.mark.benchmark_cpu
@pytest.mark.parametrize('n_qubit', nqubit_list)
def test_diagonalization(benchmark, n_qubit):
    # bench supergrad
    evo = create_simultaneous_x(n_qubit=n_qubit,
                                astep=500,
                                trotter_order=2,
                                minimal_approach=True,
                                diag_ops=False)
    benchmark.group = f'diagonalization_{n_qubit}_qubits'
    # benchmark

    @benchmark
    def eigenvectors():
        return jax.jit(evo.eigensystem)(evo.all_params)[1].block_until_ready()


@pytest.mark.benchmark_cpu
@pytest.mark.parametrize('astep', astep_list)
@pytest.mark.parametrize('trotter_order', trotter_order_list)
def test_simultaneous_x_convergence(benchmark, astep, trotter_order):
    """Testing convergence of the numerical methods and Trotter decomposition.

    Please compute the reference data before run this test.
    """
    # bench supergrad
    evo = create_simultaneous_x(n_qubit=8,
                                astep=astep,
                                trotter_order=trotter_order,
                                diag_ops=False)
    benchmark.group = f'simultaneous_x_trotter_{str(trotter_order)}_8qubits_atep_{str(astep)}'
    # benchmark

    @benchmark
    def u_supergrad():
        return jax.jit(evo.product_basis)(evo.all_params).block_until_ready()

    u_ref = np.load(f'{dir_path}/ref_data/data_simultaneous_x_8.npy')
    fidelity = compute_fidelity_with_1q_rotation_axis(
        u_ref, u_supergrad, compensation_option='no_comp')[0]
    print(fidelity)
    benchmark.extra_info.update({'fidelity': float(fidelity)})


@pytest.mark.benchmark_extreme
@pytest.mark.parametrize('n_qubit', extreme_nqubit_list)
def test_simultaneous_x_supergrad_extreme(benchmark, n_qubit):
    # bench supergrad
    evo = create_simultaneous_x(n_qubit=n_qubit,
                                astep=5000,
                                trotter_order=2,
                                diag_ops=False,
                                minimal_approach=True)
    benchmark.group = f'simultaneous_x_{n_qubit}_qubits'

    @benchmark
    @jax.block_until_ready
    def u_supergrad():
        return evo.product_basis(evo.all_params)

    # return benchmark(evo.product_basis, evo.all_params)


@pytest.mark.benchmark_cpu
@pytest.mark.parametrize('n_qubit', nqubit_list)
@pytest.mark.parametrize('evaluation_mode', evaluation_mode_list)
def test_simultaneous_x_qiskit(benchmark, n_qubit, evaluation_mode):
    # bench qiskit
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
    solver = Solver(static_hamiltonian,
                    drive_hamiltonian,
                    rotating_frame=static_hamiltonian,
                    evaluation_mode=evaluation_mode)
    benchmark.group = f'simultaneous_x_{n_qubit}_qubits'
    u0 = np.eye(np.prod(evo.dims), dtype=complex)

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

    u_ref = np.load(f'{dir_path}/ref_data/data_simultaneous_x_{n_qubit}.npy')
    assert abs(1 - fidelity(u_ref, u_qiskit)) < 1e-5


@pytest.mark.benchmark_cpu
@pytest.mark.parametrize('n_qubit', nqubit_list)
def test_simultaneous_x_qutip(benchmark, n_qubit):
    # bench qutip
    evo = create_simultaneous_x(n_qubit=n_qubit,
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
    benchmark.group = f'simultaneous_x_{n_qubit}_qubits'

    @benchmark
    def u_qutip():
        output = qt.sesolve([ham_static_qutip] + operator_function_pair_list,
                            u0, [0.0, t_span],
                            options=options)
        return output.states[-1].full()

    u_ref = np.load(f'{dir_path}/ref_data/data_simultaneous_x_{n_qubit}.npy')
    assert abs(1 - fidelity(u_ref, u_qutip)) < 1e-5
