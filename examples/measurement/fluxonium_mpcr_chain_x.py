# %%
import jax
import numpy as np

from supergrad.helper import Evolve, Spectrum
from supergrad.utils import tensor, compute_fidelity_with_1q_rotation_axis
from supergrad.utils.gates import x_gate, identity
from supergrad.utils.utility import create_state_init_with_idle

from supergrad.scgraph.graph_fluxonium_mpc_res import MPCFRes1D
# %%
xi = 10
n_qubit = 2
chain = MPCFRes1D(n_qubit, periodic=False, seed=42)
chain.create_idle_pulse(range(n_qubit), [np.pi / 2 / xi] * n_qubit)
minimal_model = chain.subgraph(['fm0', 'fm1', 'res0'])
spec = Spectrum(minimal_model, truncated_dim=2, share_params=True)
et = spec.energy_tensor(spec.all_params)
# %%
evo = Evolve(minimal_model,
             truncated_dim=5,
             share_params=True,
             idle_subsystem=chain.resonator_subsystem,
             compensation_option='no_comp',
             solver='ode_expm',
             options={
                 'astep': 2000,
                 'trotter_order': 2,
                 'progress_bar': True,
                 'custom_vjp': True,
             })
# %%
# initialize the cavity to the coherent state
psi_list, _ = create_state_init_with_idle(evo.graph.sorted_nodes,
                                          evo.idle_subsystem,
                                          evo.get_dims(evo.all_params),
                                          idle_state=2)
# cavity interacting
cr_unitary = evo.product_basis(evo.all_params, psi_list)
# %%
target_unitary = identity(2**n_qubit)


def compute_fidelity(params):
    cr_unitary = evo.eigen_basis(params)
    fidelity, res_unitary = compute_fidelity_with_1q_rotation_axis(
        target_unitary, cr_unitary, compensation_option='only_vz')
    return fidelity


compute_fidelity(evo.all_params)  # 0.96108984
# %%
