# %%
import jax

from supergrad.common_functions import Evolve
from supergrad.utils import tensor, compute_fidelity_with_1q_rotation_axis
from supergrad.utils.gates import x_gate

from graph_mpc_fluxonium_1d import MPCFluxonium1D
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
             options={
                 'astep': 2000,
                 'trotter_order': 1,
                 'progress_bar': True,
                 'fwd_ad': False
             })


# %%
def compute_fidelity(params):
    cr_unitary = evo.eigen_basis(params)
    fideity, res_unitary = compute_fidelity_with_1q_rotation_axis(
        target_unitary,
        cr_unitary,
        compensation_option='only_vz',
        n_qubit=n_qubit)
    return fideity


compute_fidelity(evo.static_params)  # 0.96108984
# %%
# compute the gradient with respect to the control and device parameters
jax.grad(compute_fidelity)(evo.static_params)

# %%
