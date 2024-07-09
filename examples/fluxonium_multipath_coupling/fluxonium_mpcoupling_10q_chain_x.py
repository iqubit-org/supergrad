# %%
import jax

from supergrad.helper import Evolve
from supergrad.utils import tensor, compute_fidelity_with_1q_rotation_axis
from supergrad.utils.gates import x_gate

from supergrad.scgraph.graph_mpc_fluxonium_1d import MPCFluxonium1D
# %%
n_qubit = 10
chain = MPCFluxonium1D(n_qubit, periodic=False, seed=42)
# Must set before any operations
chain.share_params = True
chain.unify_coupling = True
chain.set_all_node_attr(truncated_dim=2)

chain.create_single_qubit_pulse(range(n_qubit), [ 50.0, ] * n_qubit)

target_unitary = tensor(*[ x_gate(), ] * n_qubit)

# %%
evo = Evolve(chain,
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


print(compute_fidelity(evo.all_params))  # 0.96108984
# %%
# compute the gradient with respect to the control and device parameters
print(jax.grad(compute_fidelity)(evo.all_params))

# %%
