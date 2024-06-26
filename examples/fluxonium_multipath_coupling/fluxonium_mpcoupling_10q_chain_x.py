# %%
import jax

from supergrad.helper import Evolve
from supergrad.utils import tensor, compute_fidelity_with_1q_rotation_axis
from supergrad.utils.gates import x_gate

from supergrad.scgraph.graph_mpc_fluxonium_1d import MPCFluxonium1D
# %%
n_qubit = 10
chain = MPCFluxonium1D(n_qubit, periodic=False, seed=42)
chain.create_single_qubit_pulse(range(n_qubit), [
    50.0,
] * n_qubit, True)

target_unitary = tensor(*[
    x_gate(),
] * n_qubit)
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
def compute_fidelity(params):
    cr_unitary = evo.eigen_basis(params)
    fidelity, res_unitary = compute_fidelity_with_1q_rotation_axis(
        target_unitary, cr_unitary, compensation_option='only_vz')
    return fidelity


compute_fidelity(evo.all_params)  # 0.96108984
# %%
# compute the gradient with respect to the control and device parameters
jax.grad(compute_fidelity)(evo.all_params)

# %%
