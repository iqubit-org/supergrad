import os
import sys
import jax
import jax.numpy as jnp
import pytest

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('/'.join(dir_path.split('/')[:-1]))

from benchmark.utils.create_simultaneous_model import (create_simultaneous_x,
                                                       create_simultaneous_cnot)

trotter_order_list = [2, 4j, 4, None]
diag_ops_list = [True, False]


@pytest.mark.parametrize('trotter_order', trotter_order_list)
@pytest.mark.parametrize('diag_ops', diag_ops_list)
def test_simultaneous_x_gradient(trotter_order, diag_ops):
    n_qubit = 2
    astep = 3000
    ref_evo = create_simultaneous_x(n_qubit=n_qubit,
                                    astep=astep,
                                    trotter_order=trotter_order,
                                    diag_ops=diag_ops)

    @jax.grad
    def cal_ref_grad(params):
        u_cal = ref_evo.product_basis(params)
        return jnp.sum(u_cal).real

    ref_grad = cal_ref_grad(ref_evo.all_params)

    evo = create_simultaneous_x(n_qubit=n_qubit,
                                astep=astep,
                                trotter_order=trotter_order,
                                diag_ops=diag_ops,
                                custom_vjp=True)

    @jax.grad
    def cal_grad(params):
        u_cal = evo.product_basis(params)
        return jnp.sum(u_cal).real

    grad = cal_grad(evo.all_params)

    assert all(
        jax.tree.leaves(
            jax.tree_util.tree_map(lambda a, b: jnp.allclose(a, b), ref_grad,
                                   grad)))


@pytest.mark.parametrize('trotter_order', trotter_order_list)
@pytest.mark.parametrize('diag_ops', diag_ops_list)
def test_simultaneous_cnot_gradient(trotter_order, diag_ops):
    n_qubit = 2
    astep = 5000
    ref_evo = create_simultaneous_cnot(n_qubit=n_qubit,
                                       astep=astep,
                                       trotter_order=trotter_order,
                                       diag_ops=diag_ops)

    @jax.grad
    def cal_ref_grad(params):
        u_cal = ref_evo.product_basis(params)
        return jnp.sum(u_cal).real

    ref_grad = cal_ref_grad(ref_evo.all_params)

    evo = create_simultaneous_cnot(n_qubit=n_qubit,
                                   astep=astep,
                                   trotter_order=trotter_order,
                                   diag_ops=diag_ops,
                                   custom_vjp=True)

    @jax.grad
    def cal_grad(params):
        u_cal = evo.product_basis(params)
        return jnp.sum(u_cal).real

    grad = cal_grad(evo.all_params)

    assert all(
        jax.tree.leaves(
            jax.tree_util.tree_map(lambda a, b: jnp.allclose(a, b, rtol=1e-4),
                                   ref_grad, grad)))
