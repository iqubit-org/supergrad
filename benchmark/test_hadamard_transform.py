import os
import sys
from functools import partial
import jax.numpy as jnp
import jax

import pytest

from supergrad.utils import basis
from supergrad.utils.gates import hadamard_transform
from supergrad.utils.memory_profiling import trace_max_memory_usage

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('/'.join(dir_path.split('/')[:-1]))

from supergrad.utils.sharding import get_sharding, distributed_state_fidelity

from benchmark.utils.create_simultaneous_model import create_hadamard_transform

nqubit_list = range(1, 6)


@pytest.mark.benchmark_grad
@pytest.mark.parametrize('n_qubit', nqubit_list)
def test_hadamard_state_grad_lcam(benchmark, n_qubit):
    # bench supergrad
    evo = create_hadamard_transform(n_qubit=n_qubit,
                                    astep=5000,
                                    trotter_order=2,
                                    diag_ops=False,
                                    minimal_approach=True,
                                    custom_vjp=True)
    benchmark.group = f'gradient_hadamard_state_{n_qubit}_qubits'
    # benchmark
    target_state = jnp.ones([2**n_qubit, 1]) / jnp.sqrt(2**n_qubit)

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
    jnp.save(f'{dir_path}/res_data/grad_lcam_hadamard_state_{n_qubit}.npy',
             vg_infidelity[0])
    assert len(vg_infidelity[0]) == 2


@pytest.mark.benchmark_grad
@pytest.mark.parametrize('n_qubit', nqubit_list)
def test_hadamard_state_grad_tad(benchmark, n_qubit):
    # bench supergrad
    evo = create_hadamard_transform(n_qubit=n_qubit,
                                    astep=5000,
                                    trotter_order=2,
                                    diag_ops=False,
                                    minimal_approach=True,
                                    custom_vjp=None)
    benchmark.group = f'gradient_hadamard_state_{n_qubit}_qubits'
    # benchmark
    target_state = jnp.ones([2**n_qubit, 1]) / jnp.sqrt(2**n_qubit)

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
    jnp.save(f'{dir_path}/res_data/grad_tad_hadamard_state_{n_qubit}.npy',
             vg_infidelity[0])
    assert len(vg_infidelity[0]) == 2


@pytest.mark.benchmark_grad
@pytest.mark.parametrize('n_qubit', nqubit_list)
def test_hadamard_unitary_grad_lcam(benchmark, n_qubit):
    # bench supergrad
    evo = create_hadamard_transform(n_qubit=n_qubit,
                                    astep=5000,
                                    trotter_order=2,
                                    diag_ops=False,
                                    minimal_approach=True,
                                    custom_vjp=True)
    benchmark.group = f'gradient_hadamard_unitary_{n_qubit}_qubits'
    # benchmark
    u_ref = hadamard_transform(n_qubit)
    u_ref = jax.device_put(u_ref, get_sharding(None, 'p'))

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
    jnp.save(f'{dir_path}/res_data/grad_lcam_hadamard_unitary_{n_qubit}.npy',
             vg_infidelity[0])
    assert len(vg_infidelity[0]) == 2


@pytest.mark.benchmark_grad
@pytest.mark.parametrize('n_qubit', nqubit_list)
def test_hadamard_unitary_grad_tad(benchmark, n_qubit):
    # bench supergrad
    evo = create_hadamard_transform(n_qubit=n_qubit,
                                    astep=5000,
                                    trotter_order=2,
                                    diag_ops=False,
                                    minimal_approach=True,
                                    custom_vjp=None)
    benchmark.group = f'gradient_hadamard_unitary_{n_qubit}_qubits'
    # benchmark
    u_ref = hadamard_transform(n_qubit)
    u_ref = jax.device_put(u_ref, get_sharding(None, 'p'))

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
    jnp.save(f'{dir_path}/res_data/grad_tad_hadamard_unitary_{n_qubit}.npy',
             vg_infidelity[0])
    assert len(vg_infidelity[0]) == 2
