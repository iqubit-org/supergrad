# %%
import os
import sys
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import supergrad
from supergrad.utils.sharding import distributed_state_fidelity

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('/'.join(dir_path.split('/')[:-2]))

from benchmark.utils.create_simultaneous_model import create_simultaneous_x
# %%
# create 1d chain model, apply simultaneous X gates
# as a baseline approach to compute gradients using differentiable simulation
# we using the supergrad with LCAM method
n_qubit = 4
# %%
# bench supergrad
evo = create_simultaneous_x(n_qubit=n_qubit,
                            astep=5000,
                            trotter_order=2,
                            diag_ops=True,
                            minimal_approach=True,
                            custom_vjp=True)
u_ref = supergrad.tensor(*[np.array([[0, 1], [1, 0]])] * n_qubit)


def infidelity(params):
    return 1 - distributed_state_fidelity(u_ref, evo.product_basis(params))


center_val = infidelity(evo.all_params)
# %%
# Set a step size for finite differences calculations
eps = 1e-4
params, unflatten = ravel_pytree(evo.all_params)


# sweep over all parameters
# %%
@partial(jax.vmap, in_axes=(0, None))
def compute_finite_diff_at_coordinate(index, eps=1e-4):
    params_var = params.at[index].add(eps)
    return (infidelity(unflatten(params_var)) - center_val) / eps


grad_numerical = compute_finite_diff_at_coordinate(jnp.arange(len(params)), eps)
grad_numerical = unflatten(grad_numerical)
# %%
# using the JAX auto-diff transformation
grad_ad = jax.value_and_grad(infidelity)(evo.all_params)
# %%
# compare the gradients
diff_grad = jax.tree.map(lambda x, y: jnp.linalg.norm(x - y), grad_ad[1],
                         grad_numerical)
diff_grad
# %%
