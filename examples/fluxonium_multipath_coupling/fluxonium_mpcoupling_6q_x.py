# %%
import jax.numpy as jnp
import haiku as hk

import supergrad
from supergrad.common_functions import Evolve
from supergrad.utils import compute_fidelity_with_1q_rotation_axis
from supergrad.utils.gates import sigmax
from supergrad.utils.optimize import scipy_minimize, adam_opt

from graph_5x5_periodic import XGatePeriodicGraphOpt

truncated_dim = 3
enable_var = True
share_params = True
unify_coupling = True
compensation_option = 'only_vz'
opt = 'adam'
# instance the quantum processor graph, and choose a subset for time evolution
gate_graph = XGatePeriodicGraphOpt(1)
qubit_subset = gate_graph.subgraph(['q02', 'q03', 'q12', 'q13', 'q22', 'q23'])

evo = Evolve(qubit_subset, truncated_dim, enable_var, share_params,
             unify_coupling, compensation_option)
unitary = supergrad.tensor(*([sigmax()] * len(qubit_subset.nodes)))


# %%
def infidelity(params, unitary):
    params = hk.data_structures.merge(evo.static_params, params)
    # evolve system on the eigen basis.
    sim_u = evo.eigen_basis(params)
    # calculate fidelity
    fidelity_vz, _ = compute_fidelity_with_1q_rotation_axis(unitary,
                                                            sim_u,
                                                            n_qubit=6,
                                                            opt_method=None)

    return jnp.log10(1 - fidelity_vz)


# %%
# list the parameters that will be updated during the optimization
params = {
    'q02_pulse_cos': {
        'amp': jnp.array(0.07467672),
        'length': jnp.array(39.82459069),
        'omega_d': jnp.array(3.70771137),
        'phase': jnp.array(0.7558433)
    },
    'q03_pulse_cos': {
        'amp': jnp.array(0.06839676),
        'length': jnp.array(39.81302627),
        'omega_d': jnp.array(2.56897537),
        'phase': jnp.array(0.85390569)
    },
    'q12_pulse_cos': {
        'amp': jnp.array(0.07142831),
        'length': jnp.array(39.85070846),
        'omega_d': jnp.array(3.07586917),
        'phase': jnp.array(1.58941357)
    },
    'q13_pulse_cos': {
        'amp': jnp.array(0.07763859),
        'length': jnp.array(39.85217838),
        'omega_d': jnp.array(4.20299531),
        'phase': jnp.array(1.43135504)
    },
    'q22_pulse_cos': {
        'amp': jnp.array(0.08054434),
        'length': jnp.array(39.80346128),
        'omega_d': jnp.array(5.00262381),
        'phase': jnp.array(1.39100452)
    },
    'q23_pulse_cos': {
        'amp': jnp.array(0.07422721),
        'length': jnp.array(39.85217838),
        'omega_d': jnp.array(3.63596315),
        'phase': jnp.array(0.26554983)
    },
}
# %%
# show the optimization procedure
if __name__ == '__main__':
    print(infidelity(params, unitary))  # -3.2589998
    if opt == 'scipy':
        res = scipy_minimize(infidelity,
                             params,
                             args=(unitary),
                             method='l-bfgs-b',
                             logging=True,
                             options={'maxiter': 2000})
    elif opt == 'adam':
        adam_opt(infidelity, params, args=(unitary,), options={'adam_lr': 0.01})
# %%
