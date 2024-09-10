# %%
import os

# Use 8 CPU devices to test the distributed computation.
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from supergrad.helper import Evolve
from supergrad.utils.gates import hadamard_transform
from supergrad.scgraph.graph_mpc_fluxonium_1d import MPCFluxonium1D
from supergrad.utils.sharding import sharding_put, distributed_state_fidelity, distributed_fidelity_with_auto_vz_compensation

# %%
n_qubit = 8
chain = MPCFluxonium1D(n_qubit, periodic=False, seed=42)
chain.create_single_qubit_pulse(range(n_qubit), [50.0] * n_qubit,
                                True,
                                factor=0.25,
                                minimal_approach=True)

target_unitary = hadamard_transform(n_qubit)
# %%
# batch psi0 to multi-device
target_unitary = sharding_put(target_unitary, P(None, 'p'))
jax.debug.visualize_array_sharding(target_unitary)
# %%
evo = Evolve(chain,
             truncated_dim=2,
             compensation_option='no_comp',
             solver='ode_expm',
             options={
                 'astep': 5000,
                 'trotter_order': 2,
                 'progress_bar': True,
                 'custom_vjp': True,
             })
output = evo.product_basis(evo.all_params)
jax.debug.visualize_array_sharding(output)
# %%
fidelity, comp_output = distributed_fidelity_with_auto_vz_compensation(
    target_unitary, output)
jax.debug.visualize_array_sharding(comp_output)

# %%
distributed_state_fidelity(output, target_unitary)


# %%
def compute_fidelity(params):
    realized_unitary = evo.product_basis(params)
    fidelity, res_unitary = distributed_fidelity_with_auto_vz_compensation(
        target_unitary, realized_unitary)
    return 1 - fidelity


v, g = jax.value_and_grad(compute_fidelity)(evo.all_params)  # 0.07157539
# %%
