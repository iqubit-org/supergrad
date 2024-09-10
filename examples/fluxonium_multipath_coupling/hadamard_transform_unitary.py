# %%
import os

# Use 8 CPU devices to test the distributed computation.
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from supergrad.helper import Evolve
from supergrad.utils import tensor, compute_fidelity_with_1q_rotation_axis
from supergrad.utils.gates import x_gate, hadamard_transform
from supergrad.utils.fidelity import compute_average_fidelity_with_leakage

from supergrad.scgraph.graph_mpc_fluxonium_1d import MPCFluxonium1D
from supergrad.utils.sharding import sharding_put, distributed_state_fidelity, distributed_fidelity_with_auto_vz_compensation

# %%
n_qubit = 4
chain = MPCFluxonium1D(n_qubit, periodic=False, seed=42)
chain.create_single_qubit_pulse(range(n_qubit), [
    50.0,
] * n_qubit, True, factor=0.25, minimal_approach=True)

target_unitary = hadamard_transform(n_qubit)
# %%
# batch psi0 to multi-device
target_unitary = sharding_put(target_unitary, P(None, 'p'))
jax.debug.visualize_array_sharding(target_unitary)
# %%
evo = Evolve(chain,
             truncated_dim=2,
             share_params=True,
             unify_coupling=True,
             compensation_option='no_comp',
             solver='ode_expm',
             options={
                 'astep': 2000,
                 'trotter_order': 1,
                 'progress_bar': True,
                 'custom_vjp': True,
             })
output = evo.product_basis(evo.all_params)
jax.debug.visualize_array_sharding(output)
# %%
fidelity, comp_output = distributed_fidelity_with_auto_vz_compensation(
    target_unitary, output)


# %%
def debug_distributed_state_fidelity(target_states, computed_states):
    assert target_states.sharding == computed_states.sharding
    ar_state_fidelity = jnp.sum(jnp.conj(target_states) * computed_states,
                                axis=0)
    # return jnp.abs(ar_state_fidelity)**2
    return jnp.abs(ar_state_fidelity.mean())**2


debug_distributed_state_fidelity(output, target_unitary)


# state_out, jac = jax.value_and_grad(distributed_state_fidelity)(output, target_unitary)
# distributed_state_fidelity(target_unitary[:, 0, jnp.newaxis], comp_output[:, 0, jnp.newaxis])
# %%
def compute_fidelity(params):
    realized_unitary = evo.product_basis(params)
    return distributed_state_fidelity(realized_unitary, target_unitary)
    fidelity, res_unitary = compute_fidelity_with_1q_rotation_axis(
        target_unitary, realized_unitary, compensation_option='only_vz')
    return fidelity


v, g = jax.value_and_grad(compute_fidelity)(evo.all_params)  # 0.96108984
# %%
