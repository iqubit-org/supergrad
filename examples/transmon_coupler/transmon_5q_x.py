# %%
import jax.numpy as jnp
import haiku as hk

import supergrad
from supergrad.common_functions import Evolve
from supergrad.utils import compute_fidelity_with_1q_rotation_axis
from supergrad.utils.gates import sigmax
from supergrad.utils.optimize import scipy_minimize, adam_opt

from graph_transmon_1d import XGateTmon1D

truncated_dim = 5
enable_var = True
share_params = False
compensation_option = 'only_vz'
opt = 'adam'

gate_graph = XGateTmon1D(seed=1)
qubit_subset = gate_graph.subgraph(['q0', 'q1', 'q2', 'q3', 'q4'])
evo = Evolve(qubit_subset,
             truncated_dim,
             enable_var,
             share_params,
             coupler_subsystem=['q1', 'q3'],
             compensation_option=compensation_option,
             options={
                 'astep': 5000,
                 'trotter_order': 2
             })
unitary = supergrad.tensor(*([sigmax()] * 3))


# %%
def infidelity(params, unitary):
    params = hk.data_structures.merge(evo.static_params, params)

    sim_u = evo.eigen_basis(params)
    fidelity_vz, _ = compute_fidelity_with_1q_rotation_axis(unitary,
                                                            sim_u,
                                                            n_qubit=3,
                                                            opt_method=None)

    return jnp.log10(1 - fidelity_vz)  # , sim_u


params = {
    'q0_pulse_cos': {
        'amp': jnp.array(0.0922632),
        'length': jnp.array(39.99841052),
        'omega_d': jnp.array(31.89213402),
        'phase': jnp.array(-0.06459036)
    },
    'q2_pulse_cos': {
        'amp': jnp.array(0.10390872),
        'length': jnp.array(39.92211365),
        'omega_d': jnp.array(27.99554391),
        'phase': jnp.array(0.05805683)
    },
    'q4_pulse_cos': {
        'amp': jnp.array(0.09196213),
        'length': jnp.array(39.88357291),
        'omega_d': jnp.array(31.97277349),
        'phase': jnp.array(-0.07858071)
    },
}
# %%
if __name__ == '__main__':

    print(infidelity(params, unitary))  # -2.394032
    if opt == 'scipy':

        res = scipy_minimize(infidelity,
                             params,
                             args=(unitary),
                             method='l-bfgs-b',
                             logging=True,
                             options={'maxiter': 2000})
    elif opt == 'adam':
        adam_opt(infidelity, params, (unitary,), {'adam_lr': 0.01})
# %%
