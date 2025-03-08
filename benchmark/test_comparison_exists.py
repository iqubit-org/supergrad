import os
import sys
import gc
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import haiku as hk
import pytest

from qiskit_dynamics import Solver
import qutip as qt
import scqubits as scq
import supergrad
from supergrad.utils.memory_profiling import trace_max_memory_usage
from supergrad.utils.qiskit_interface import to_qiskit_drive_hamiltonian
from supergrad.utils.qutip_interface import (to_qutip_operator_function_pair)
from supergrad.utils.sharding import distributed_state_fidelity

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('/'.join(dir_path.split('/')[:-1]))

from benchmark.utils.create_simultaneous_model import create_simultaneous_x
from benchmark.utils.create_multipath_chain_scqubits import create_qubit_chain

# sweep configurations
nqubit_list = range(6, 7)
# global configurations
astep = 5000
trotter_order = 2
diag_ops = True
minimal_approach = True
custom_vjp = True
add_random = False
drive_for_state_phase = 'phase'

# unify the state phase based on the phase operator (as the same as the scqubits)


# concatenate simulation function
def forward_simulation_scqubits_qiskit(params, n_qubit, drive_signal, t_span,
                                       rotating_frame):
    u_ref = supergrad.tensor(*[np.array([[0, 1], [1, 0]])] * n_qubit)
    hilbertspace, fm_list = create_qubit_chain(params, n_qubit, 2)
    # time evolution by qutip using the scqubits Hamiltonian
    ham_static_scq = hilbertspace.hamiltonian()
    # create the drive terms
    # using the qiskit dynamics solver
    static_hamiltonian = ham_static_scq.full()
    drive_hamiltonian = [
        scq.identity_wrap(fm.phi_operator(), fm,
                          hilbertspace.subsystem_list).full() for fm in fm_list
    ]

    if rotating_frame:
        # Evolving by qiskit dynamics in rotating frame
        solver = Solver(static_hamiltonian,
                        drive_hamiltonian,
                        rotating_frame=static_hamiltonian,
                        array_library="jax")
    else:
        solver = Solver(static_hamiltonian,
                        drive_hamiltonian,
                        array_library="jax")
    # initial states array
    u0 = np.eye(u_ref.shape[0], dtype=complex)

    results = solver.solve(
        t_span=[0, t_span],
        y0=u0,
        signals=drive_signal,
        method='jax_odeint',
        atol=1e-8,
        rtol=1e-8,
    )
    if rotating_frame:
        u_qiskit = solver.model.rotating_frame.state_out_of_frame(
            t_span, results.y[-1])
    else:
        u_qiskit = results.y[-1]

    return 1 - distributed_state_fidelity(u_ref, u_qiskit)


def forward_simulation_scqubits_qutip(params, n_qubit,
                                      operator_function_pair_list, t_span):
    u_ref = supergrad.tensor(*[np.array([[0, 1], [1, 0]])] * n_qubit)
    hilbertspace, fm_list = create_qubit_chain(params, n_qubit, 2)
    # time evolution by qutip using the scqubits Hamiltonian
    ham_static_scq = hilbertspace.hamiltonian()
    # create the drive terms
    drive_opt = [[
        scq.identity_wrap(fm.phi_operator(), fm, hilbertspace.subsystem_list),
        opt_pulse[1]
    ] for fm, opt_pulse in zip(fm_list, operator_function_pair_list)]
    # using the qutip solver
    u0 = qt.qeye([2] * n_qubit)
    # config solver option
    options = qt.Options(nsteps=1e6, atol=1e-8, rtol=1e-8)

    output = qt.sesolve([ham_static_scq] + drive_opt,
                        u0, [0.0, t_span],
                        options=options)
    u_qutip = output.states[-1].full()

    return 1 - distributed_state_fidelity(u_ref, u_qutip)


# benchmark functions


@pytest.mark.benchmark_cpu
@pytest.mark.parametrize('n_qubit', nqubit_list)
def test_overall_forward_simulation_supergrad(benchmark, n_qubit):
    # create 1d chain model, apply simultaneous X gates
    evo = create_simultaneous_x(n_qubit=n_qubit,
                                astep=astep,
                                trotter_order=trotter_order,
                                diag_ops=diag_ops,
                                minimal_approach=minimal_approach,
                                custom_vjp=custom_vjp,
                                add_random=add_random,
                                drive_for_state_phase=drive_for_state_phase)
    tag = f'overall_forward_simulation_supergrad_{n_qubit}_qubits'
    benchmark.group = tag

    # benchmark
    u_ref = supergrad.tensor(*[np.array([[0, 1], [1, 0]])] * n_qubit)
    u_ref = jax.device_put(u_ref, jax.devices('cpu')[0])

    @jax.jit
    def infidelity(params):
        return 1 - distributed_state_fidelity(u_ref, evo.product_basis(params))

    # compute the gradients by the supergrad
    @benchmark
    @partial(trace_max_memory_usage, pid=os.getpid())
    @jax.block_until_ready
    def v_infidelity():
        return infidelity(evo.all_params)

    benchmark.extra_info.update({'memory': v_infidelity[1]})
    # save the gradient
    jnp.save(f'{dir_path}/res_data/{tag}.npy', v_infidelity[0])
    assert v_infidelity[0] <= 1


@pytest.mark.benchmark_grad
@pytest.mark.parametrize('n_qubit', nqubit_list)
def test_time_evolution_differentiable_simulation_supergrad(benchmark, n_qubit):
    evo = create_simultaneous_x(n_qubit=n_qubit,
                                astep=astep,
                                trotter_order=trotter_order,
                                diag_ops=diag_ops,
                                minimal_approach=minimal_approach,
                                custom_vjp=custom_vjp,
                                add_random=add_random,
                                drive_for_state_phase=drive_for_state_phase)
    tag = f'time_evolution_differentiable_simulation_supergrad_{n_qubit}_qubits'
    benchmark.group = tag

    # benchmark
    u_ref = supergrad.tensor(*[np.array([[0, 1], [1, 0]])] * n_qubit)
    u_ref = jax.device_put(u_ref, jax.devices('cpu')[0])

    @jax.jit
    @jax.value_and_grad
    def infidelity(params):
        # only compute the gradients for pulse parameters
        params = hk.data_structures.merge(evo.all_params, params)
        return 1 - distributed_state_fidelity(u_ref, evo.product_basis(params))

    # compute the gradients by the supergrad
    @benchmark
    @partial(trace_max_memory_usage, pid=os.getpid())
    @jax.block_until_ready
    def vg_infidelity():
        return infidelity(evo.pulse_params)

    benchmark.extra_info.update({'memory': vg_infidelity[1]})
    # save the gradient
    jnp.save(f'{dir_path}/res_data/{tag}.npy', vg_infidelity[0])
    assert len(vg_infidelity[0]) == 2


@pytest.mark.benchmark_grad
@pytest.mark.parametrize('n_qubit', nqubit_list)
def test_overall_differentiable_simulation_supergrad(benchmark, n_qubit):
    evo = create_simultaneous_x(n_qubit=n_qubit,
                                astep=astep,
                                trotter_order=trotter_order,
                                diag_ops=diag_ops,
                                minimal_approach=minimal_approach,
                                custom_vjp=custom_vjp,
                                add_random=add_random,
                                drive_for_state_phase=drive_for_state_phase)
    tag = f'overall_differentiable_simulation_supergrad_{n_qubit}_qubits'
    benchmark.group = tag

    # benchmark
    u_ref = supergrad.tensor(*[np.array([[0, 1], [1, 0]])] * n_qubit)
    u_ref = jax.device_put(u_ref, jax.devices('cpu')[0])

    @jax.jit
    @jax.value_and_grad
    def infidelity(params):
        return 1 - distributed_state_fidelity(u_ref, evo.product_basis(params))

    # compute the gradients by the supergrad
    @benchmark
    @partial(trace_max_memory_usage, pid=os.getpid())
    @jax.block_until_ready
    def vg_infidelity():
        return infidelity(evo.all_params)

    benchmark.extra_info.update({'memory': vg_infidelity[1]})
    # save the gradient
    jnp.save(f'{dir_path}/res_data/{tag}.npy', vg_infidelity[0])
    assert len(vg_infidelity[0]) == 2


@pytest.mark.benchmark_cpu
@pytest.mark.parametrize('n_qubit', nqubit_list)
def test_overall_forward_simulation_scqubits_qiskit(benchmark,
                                                    n_qubit,
                                                    rotating_frame=False):
    """Benchmark 1
    Benchmark overall simulation (SCQubit + qiskit dynamics) between toolchain
    and supergrad.
    """
    evo = create_simultaneous_x(n_qubit=n_qubit,
                                astep=astep,
                                trotter_order=trotter_order,
                                diag_ops=diag_ops,
                                minimal_approach=minimal_approach,
                                custom_vjp=custom_vjp,
                                add_random=add_random,
                                drive_for_state_phase=drive_for_state_phase)
    tag = f'overall_forward_simulation_scqubits_qiskit_{n_qubit}_qubits'
    benchmark.group = tag

    _, hamiltonian_component_and_pulseshape, t_span = evo.construct_hamiltonian_and_pulseshape(
        evo.all_params)
    _, drive_signal = to_qiskit_drive_hamiltonian(
        hamiltonian_component_and_pulseshape)
    params = jax.tree.map(lambda x: np.array(x), evo.all_params)

    # benchmark
    @benchmark
    @partial(trace_max_memory_usage, pid=os.getpid())
    @jax.block_until_ready
    def v_infidelity():
        return forward_simulation_scqubits_qiskit(params, n_qubit, drive_signal,
                                                  t_span, rotating_frame)

    benchmark.extra_info.update({'memory': v_infidelity[1]})
    # save the gradient
    jnp.save(f'{dir_path}/res_data/{tag}.npy', v_infidelity[0])
    assert v_infidelity[0] <= 1


@pytest.mark.benchmark_cpu
@pytest.mark.parametrize('n_qubit', nqubit_list)
def test_overall_forward_simulation_scqubits_qutip(benchmark, n_qubit):
    """Benchmark 1
    Benchmark overall simulation (SCQubit + qutip) between toolchain
    and supergrad.
    """
    evo = create_simultaneous_x(n_qubit=n_qubit,
                                astep=astep,
                                trotter_order=trotter_order,
                                diag_ops=diag_ops,
                                minimal_approach=minimal_approach,
                                custom_vjp=custom_vjp,
                                add_random=add_random,
                                drive_for_state_phase=drive_for_state_phase)
    tag = f'overall_forward_simulation_scqubits_qutip_{n_qubit}_qubits'
    benchmark.group = tag

    _, hamiltonian_component_and_pulseshape, t_span = evo.construct_hamiltonian_and_pulseshape(
        evo.all_params)
    operator_function_pair_list = to_qutip_operator_function_pair(
        hamiltonian_component_and_pulseshape)
    params = jax.tree.map(lambda x: np.array(x), evo.all_params)

    # benchmark
    @benchmark
    @partial(trace_max_memory_usage, pid=os.getpid())
    def v_infidelity():
        return forward_simulation_scqubits_qutip(params, n_qubit,
                                                 operator_function_pair_list,
                                                 t_span)

    benchmark.extra_info.update({'memory': v_infidelity[1]})
    # save the gradient
    jnp.save(f'{dir_path}/res_data/{tag}.npy', v_infidelity[0])
    assert v_infidelity[0] <= 1


@pytest.mark.benchmark_grad
@pytest.mark.parametrize('n_qubit', nqubit_list)
def test_time_evolution_differentiable_simulation_qiskit(
        benchmark, n_qubit, rotating_frame=False):
    """Benchmark 2
    Benchmark differential simulation of time evolution between qiskit dynamics
    and supergrad.
    """
    evo = create_simultaneous_x(n_qubit=n_qubit,
                                astep=astep,
                                trotter_order=trotter_order,
                                diag_ops=diag_ops,
                                minimal_approach=minimal_approach,
                                custom_vjp=custom_vjp,
                                add_random=add_random,
                                drive_for_state_phase=drive_for_state_phase)
    tag = f'time_evolution_differentiable_simulation_qiskit_{n_qubit}_qubits'
    benchmark.group = tag

    @jax.value_and_grad
    def infidelity(params):
        params = hk.data_structures.merge(evo.all_params, params)
        u_ref = supergrad.tensor(*[np.array([[0, 1], [1, 0]])] * n_qubit)
        # using Hamiltonian that created by SCQubits
        _, hamiltonian_component_and_pulseshape, t_span = evo.construct_hamiltonian_and_pulseshape(
            params)
        _, drive_signal = to_qiskit_drive_hamiltonian(
            hamiltonian_component_and_pulseshape)
        params_no_grad = jax.tree.map(lambda x: np.array(x), evo.all_params)
        hilbertspace, fm_list = create_qubit_chain(params_no_grad, n_qubit, 2)
        # time evolution by qutip using the scqubits Hamiltonian
        ham_static_scq = hilbertspace.hamiltonian()
        # create the drive terms
        # using the qiskit dynamics solver
        static_hamiltonian = ham_static_scq.full()
        drive_hamiltonian = [
            scq.identity_wrap(fm.phi_operator(), fm,
                              hilbertspace.subsystem_list).full()
            for fm in fm_list
        ]
        if rotating_frame:
            # Evolving by qiskit dynamics in rotating frame
            solver = Solver(static_hamiltonian,
                            drive_hamiltonian,
                            rotating_frame=static_hamiltonian,
                            array_library="jax")
        else:
            solver = Solver(static_hamiltonian,
                            drive_hamiltonian,
                            array_library="jax")
        u0 = np.eye(u_ref.shape[0], dtype=complex)

        results = solver.solve(
            t_span=[0, t_span],
            y0=u0,
            signals=drive_signal,
            method='jax_odeint',
            atol=1e-8,
            rtol=1e-8,
        )
        if rotating_frame:
            u_qiskit = solver.model.rotating_frame.state_out_of_frame(
                t_span, results.y[-1])
        else:
            u_qiskit = results.y[-1]
        return 1 - distributed_state_fidelity(u_ref, u_qiskit)

    # benchmark
    @benchmark
    @partial(trace_max_memory_usage, pid=os.getpid())
    @jax.block_until_ready
    def vg_infidelity():
        return infidelity(evo.pulse_params)

    benchmark.extra_info.update({'memory': vg_infidelity[1]})
    benchmark.extra_info.update(
        {'num_params': len(ravel_pytree(evo.pulse_params)[0])})

    # save the gradient
    jnp.save(f'{dir_path}/res_data/{tag}.npy', vg_infidelity[0])
    assert len(vg_infidelity[0]) == 2


@pytest.mark.benchmark_grad
@pytest.mark.parametrize('n_qubit', nqubit_list)
def test_overall_differentiable_simulation_fdm_supergrad(benchmark, n_qubit):
    """Benchmark 3
    Benchmark gradient computation between supergrad and finite different method(FDM).
    Every forward simulation is performed using supergrad, compute the gradients
    using the forward difference.
    """
    evo = create_simultaneous_x(n_qubit=n_qubit,
                                astep=astep,
                                trotter_order=trotter_order,
                                diag_ops=diag_ops,
                                minimal_approach=minimal_approach,
                                custom_vjp=custom_vjp,
                                add_random=add_random,
                                drive_for_state_phase=drive_for_state_phase)
    tag = f'overall_differentiable_simulation_fdm_supergrad_{n_qubit}_qubits'
    benchmark.group = tag

    # benchmark
    u_ref = supergrad.tensor(*[np.array([[0, 1], [1, 0]])] * n_qubit)
    u_ref = jax.device_put(u_ref, jax.devices('cpu')[0])
    params, unflatten = ravel_pytree(evo.all_params)

    @jax.jit
    def infidelity(params):
        return 1 - distributed_state_fidelity(u_ref, evo.product_basis(params))

    # benchmark
    @benchmark
    @partial(trace_max_memory_usage, pid=os.getpid())
    @jax.block_until_ready
    def vg_infidelity():
        center_val = infidelity(evo.all_params)
        # Set a step size for finite differences calculations
        eps = 1e-4
        # sweep over all parameters
        grad_numerical = []
        for index in range(len(params)):
            # using forward diff
            params_var = params.at[index].add(eps)
            grad_numerical.append(
                (infidelity(unflatten(params_var)) - center_val) / eps)
            print(f'grad_numerical {index}: {grad_numerical[-1]}')
            gc.collect()
        # unflatten the numerical gradients
        grad_numerical = unflatten(jnp.array(grad_numerical))
        return center_val, grad_numerical

        # # sweep over all parameters
        # using jax.vmap to compute the gradients
        # @partial(jax.vmap, in_axes=(0, None))
        # def compute_finite_diff_at_coordinate(index, eps):
        #     params_var = params.at[index].add(eps)
        #     return (infidelity(unflatten(params_var)) - center_val) / eps

        # grad_numerical = compute_finite_diff_at_coordinate(
        #     jnp.arange(len(params)), eps)
        # grad_numerical = unflatten(grad_numerical)
        # return center_val, grad_numerical

    benchmark.extra_info.update({'memory': vg_infidelity[1]})
    benchmark.extra_info.update({'num_params': len(params)})
    # save the gradient
    jnp.save(f'{dir_path}/res_data/{tag}.npy', vg_infidelity[0])
    assert len(vg_infidelity[0]) == 2


@pytest.mark.benchmark_grad
@pytest.mark.parametrize('n_qubit', nqubit_list)
def test_overall_differentiable_simulation_fdm_scqubits_qiskit(
        benchmark, n_qubit, rotating_frame=False):
    """Benchmark 3
    Benchmark gradient computation between supergrad and finite different method(FDM).
    Every forward simulation is performed using SCQubits and qiskit dynamics.
    """
    evo = create_simultaneous_x(n_qubit=n_qubit,
                                astep=astep,
                                trotter_order=trotter_order,
                                diag_ops=diag_ops,
                                minimal_approach=minimal_approach,
                                custom_vjp=custom_vjp,
                                add_random=add_random,
                                drive_for_state_phase=drive_for_state_phase)
    tag = f'overall_differentiable_simulation_fdm_scqubits_qiskit_{n_qubit}_qubits'
    benchmark.group = tag

    # benchmark
    u_ref = supergrad.tensor(*[np.array([[0, 1], [1, 0]])] * n_qubit)
    u_ref = jax.device_put(u_ref, jax.devices('cpu')[0])
    params, unflatten = ravel_pytree(evo.all_params)

    def infidelity(params, rotating_frame=rotating_frame):
        _, hamiltonian_component_and_pulseshape, t_span = evo.construct_hamiltonian_and_pulseshape(
            params)

        _, drive_signal = to_qiskit_drive_hamiltonian(
            hamiltonian_component_and_pulseshape)
        params = jax.tree.map(lambda x: np.array(x), params)

        return forward_simulation_scqubits_qiskit(params, n_qubit, drive_signal,
                                                  t_span, rotating_frame)

    # benchmark
    @benchmark
    @partial(trace_max_memory_usage, pid=os.getpid())
    def vg_infidelity():
        center_val = infidelity(evo.all_params)
        # Set a step size for finite differences calculations
        eps = 1e-4
        # sweep over all parameters
        grad_numerical = []
        for index in range(len(params)):
            # using forward diff
            params_var = params.at[index].add(eps)
            grad_numerical.append(
                (infidelity(unflatten(params_var)) - center_val) / eps)
            print(f'grad_numerical {index}: {grad_numerical[-1]}')
            gc.collect()
        # unflatten the numerical gradients
        grad_numerical = unflatten(jnp.array(grad_numerical))
        return center_val, grad_numerical

    benchmark.extra_info.update({'memory': vg_infidelity[1]})
    benchmark.extra_info.update({'num_params': len(params)})
    # save the gradient
    jnp.save(f'{dir_path}/res_data/{tag}.npy', vg_infidelity[0])
    assert len(vg_infidelity[0]) == 2


@pytest.mark.benchmark_grad
@pytest.mark.parametrize('n_qubit', nqubit_list)
def test_overall_differentiable_simulation_fdm_scqubits_qutip(
        benchmark, n_qubit):
    """Benchmark 3
    Benchmark gradient computation between supergrad and finite different method(FDM).
    Every forward simulation is performed using SCQubits and QuTiP.
    """
    evo = create_simultaneous_x(n_qubit=n_qubit,
                                astep=astep,
                                trotter_order=trotter_order,
                                diag_ops=diag_ops,
                                minimal_approach=minimal_approach,
                                custom_vjp=custom_vjp,
                                add_random=add_random,
                                drive_for_state_phase=drive_for_state_phase)
    tag = f'overall_differentiable_simulation_fdm_scqubits_qutip_{n_qubit}_qubits'
    benchmark.group = tag

    # benchmark
    u_ref = supergrad.tensor(*[np.array([[0, 1], [1, 0]])] * n_qubit)
    u_ref = jax.device_put(u_ref, jax.devices('cpu')[0])
    params, unflatten = ravel_pytree(evo.all_params)

    def infidelity(params):
        _, hamiltonian_component_and_pulseshape, t_span = evo.construct_hamiltonian_and_pulseshape(
            params)

        operator_function_pair_list = to_qutip_operator_function_pair(
            hamiltonian_component_and_pulseshape)
        params = jax.tree.map(lambda x: np.array(x), params)

        return forward_simulation_scqubits_qutip(params, n_qubit,
                                                 operator_function_pair_list,
                                                 t_span)

    # benchmark
    @benchmark
    @partial(trace_max_memory_usage, pid=os.getpid())
    def vg_infidelity():
        center_val = infidelity(evo.all_params)
        # Set a step size for finite differences calculations
        eps = 1e-4
        # sweep over all parameters
        grad_numerical = []
        for index in range(len(params)):
            # using forward diff
            params_var = params.at[index].add(eps)
            grad_numerical.append(
                (infidelity(unflatten(params_var)) - center_val) / eps)
            print(f'grad_numerical {index}: {grad_numerical[-1]}')
            gc.collect()
        # unflatten the numerical gradients
        grad_numerical = unflatten(jnp.array(grad_numerical))
        return center_val, grad_numerical

    benchmark.extra_info.update({'memory': vg_infidelity[1]})
    benchmark.extra_info.update({'num_params': len(params)})
    # save the gradient
    jnp.save(f'{dir_path}/res_data/{tag}.npy', vg_infidelity[0])
    assert len(vg_infidelity[0]) == 2
