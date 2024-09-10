# %%
import jax
import jax.numpy as jnp

from supergrad.helper import Evolve
from supergrad.utils import tensor, basis, compute_fidelity_with_1q_rotation_axis
from supergrad.utils.gates import x_gate, hadamard_transform
from supergrad.utils.fidelity import compute_average_fidelity_with_leakage
from supergrad.utils.sharding import distributed_state_fidelity, distributed_overlap_with_auto_vz_compensation

from supergrad.scgraph.graph_mpc_fluxonium_1d import MPCFluxonium1D
# %%
n_qubit = 22
chain = MPCFluxonium1D(n_qubit, periodic=False, seed=42)
chain.create_single_qubit_pulse(range(n_qubit), [
    50.0,
] * n_qubit, True, factor=0.25, minimal_approach=True)

target_state = jnp.ones([2**n_qubit, 1]) / jnp.sqrt(2**n_qubit)
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
# %%
initial_state = basis(2**n_qubit)
output = evo.product_basis(evo.all_params, initial_state)
# %%
fidelity, comp_output = distributed_overlap_with_auto_vz_compensation(
    target_state, output)
# %%
fid, jac = jax.value_and_grad(distributed_state_fidelity)(output, target_state)


# distributed_state_fidelity(target_unitary[:, 0, jnp.newaxis], comp_output[:, 0, jnp.newaxis])
# %%
def compute_fidelity(params):
    realized_state = evo.product_basis(params, initial_state)
    return distributed_state_fidelity(realized_state, target_state)
    # fidelity, res_unitary = compute_fidelity_with_1q_rotation_axis(
    #     target_unitary, realized_unitary, compensation_option='only_vz')
    # return fidelity


v, g = jax.value_and_grad(compute_fidelity)(evo.all_params)  # 0.96108984
# %%
