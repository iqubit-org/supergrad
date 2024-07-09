# %%
import jax.numpy as jnp
import haiku as hk

import supergrad
from supergrad.helper import Evolve
from supergrad.utils import compute_fidelity_with_1q_rotation_axis
from supergrad.utils.gates import sigmax
from supergrad.utils.optimize import scipy_minimize, adam_opt

from supergrad.scgraph.graph_mpc_fluxonium_5x5_periodic import XGatePeriodicGraphOpt

truncated_dim = 3
add_random = True
share_params = True
unify_coupling = True
compensation_option = 'only_vz'
opt = 'adam'

# instance the quantum processor graph, and choose a subset for time evolution
gate_graph = XGatePeriodicGraphOpt(1)
gate_graph.set_all_node_attr(truncated_dim=truncated_dim)
if add_random:
    gate_graph.add_lcj_params_variance_to_graph()


qubit_subset = gate_graph.subscgraph(['q02', 'q03', 'q12', 'q13', 'q22', 'q23'])
qubit_subset.share_params = share_params
qubit_subset.unify_coupling = unify_coupling

evo = Evolve(qubit_subset)
unitary = supergrad.tensor(*([sigmax()] * len(qubit_subset.nodes)))


# %%
def infidelity(params, unitary):
    # evolve system on the eigen basis.
    sim_u = evo.eigen_basis(params)
    # calculate fidelity
    fidelity_vz, _ = compute_fidelity_with_1q_rotation_axis(unitary,
                                                            sim_u,
                                                            opt_method=None)

    return jnp.log10(1 - fidelity_vz)


# %%
# list the parameters that will be updated during the optimization
params = {"nodes":{
    'q02':{'pulse':{'p1':{
        'amp': jnp.array(0.07467672),
        'length': jnp.array(39.82459069),
        'omega_d': jnp.array(3.70771137),
        'phase': jnp.array(0.7558433)
    }}},
    'q03':{'pulse':{'p1':{
        'amp': jnp.array(0.06839676),
        'length': jnp.array(39.81302627),
        'omega_d': jnp.array(2.56897537),
        'phase': jnp.array(0.85390569)
    }}},
    'q12':{'pulse':{'p1':{
        'amp': jnp.array(0.07142831),
        'length': jnp.array(39.85070846),
        'omega_d': jnp.array(3.07586917),
        'phase': jnp.array(1.58941357)
    }}},
    'q13':{'pulse':{'p1':{
        'amp': jnp.array(0.07763859),
        'length': jnp.array(39.85217838),
        'omega_d': jnp.array(4.20299531),
        'phase': jnp.array(1.43135504)
    }}},
    'q22':{'pulse':{'p1':{
        'amp': jnp.array(0.08054434),
        'length': jnp.array(39.80346128),
        'omega_d': jnp.array(5.00262381),
        'phase': jnp.array(1.39100452)
    }}},
    'q23':{'pulse':{'p1':{
        'amp': jnp.array(0.07422721),
        'length': jnp.array(39.85217838),
        'omega_d': jnp.array(3.63596315),
        'phase': jnp.array(0.26554983)
    }}},
}}
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
