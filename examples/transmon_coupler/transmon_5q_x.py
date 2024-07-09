import sys
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import qutip

#jax.config.update("jax_disable_jit", True)

import supergrad
from supergrad.helper import Evolve
from supergrad.utils import compute_fidelity_with_1q_rotation_axis
from supergrad.utils.gates import sigmax
from supergrad.utils.optimize import scipy_minimize, adam_opt

from supergrad.scgraph.graph_transmon_1d import XGateTmon1D

truncated_dim = 5
compensation_option = 'only_vz'
opt = 'adam'

gate_graph = XGateTmon1D(seed=1)
for name in ["q0", "q2", "q4"]:
    gate_graph.nodes[name]["truncated_dim"] = truncated_dim

gate_graph.add_lcj_params_variance_to_graph()

subsys = ['q0', 'q1', 'q2', 'q3', 'q4']
unitary = supergrad.tensor(*([sigmax()] * 3))
qubit_subset = gate_graph.subscgraph(subsys)
evo = Evolve(qubit_subset,
             solver="ode_expm",
             options={
                 'astep': 5000,
                 'trotter_order': 2
             })


def infidelity(params, unitary):
    sim_u = evo.eigen_basis(params)
    # sim_u = evo.product_basis(params)
    fidelity_vz, _ = compute_fidelity_with_1q_rotation_axis(unitary,
                                                            sim_u,
                                                            opt_method=None)

    return jnp.log10(1 - fidelity_vz)  # , sim_u


params = {"nodes": {
    "q0": {"pulse": {"p1": {
        'amp': jnp.array(0.0922632),
        'length': jnp.array(39.99841052),
        'omega_d': jnp.array(31.89213402),
        'phase': jnp.array(-0.06459036)
    }}},
    "q2": {"pulse": {"p1": {
        'amp': jnp.array(0.10390872),
        'length': jnp.array(39.92211365),
        'omega_d': jnp.array(27.99554391),
        'phase': jnp.array(0.05805683)
    }}},
    "q4": {"pulse": {"p1": {
        'amp': jnp.array(0.09196213),
        'length': jnp.array(39.88357291),
        'omega_d': jnp.array(31.97277349),
        'phase': jnp.array(-0.07858071)
    }}},
}}
params = {"nodes": dict([(key, val) for key, val in params["nodes"].items() if key in subsys])}


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
