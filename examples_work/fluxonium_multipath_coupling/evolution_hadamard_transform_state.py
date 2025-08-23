# %%
import jax
import jax.numpy as jnp

from supergrad.helper import Evolve
from supergrad.utils import basis
from supergrad.utils.sharding import distributed_overlap_with_auto_vz_compensation

from supergrad.scgraph.graph_mpc_fluxonium_1d import MPCFluxonium1D
# %%
n_qubit = 4
chain = MPCFluxonium1D(n_qubit, periodic=False, seed=42)
chain.create_single_qubit_pulse(range(n_qubit), [50.0] * n_qubit,
                                True,
                                factor=0.25,
                                minimal_approach=True)

target_state = jnp.ones([2**n_qubit, 1]) / jnp.sqrt(2**n_qubit)
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


# %%
def compute_fidelity(params):
    initial_state = basis(2**n_qubit)
    realized_state = evo.product_basis(params, initial_state)
    fid, opt_state = distributed_overlap_with_auto_vz_compensation(
        target_state, realized_state)
    return 1 - fid


compute_fidelity(evo.all_params)
# %%
v, g = jax.value_and_grad(compute_fidelity)(evo.all_params)
print(v)  # 0.01142255
# %%
