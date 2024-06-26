# %%
import jax.numpy as jnp
import haiku as hk

from supergrad.helper import Evolve
from supergrad.utils import tensor, permute, compute_fidelity_with_1q_rotation_axis
from supergrad.utils.optimize import scipy_minimize, adam_opt
from supergrad.utils.gates import cnot

from supergrad.scgraph.graph_mpc_fluxonium_5x5_periodic import CNOTGatePeriodicGraphOpt

truncated_dim = 3  # how many levels we keep after diagonalization
add_random = True  # whether to add deviations to fluxonium parameters ec,ej,el
share_params = True  # whether we treat marked parameters as shared when computing gradients
unify_coupling = True  # set all couplings to be equal

compensation_option = 'arbit_single'  # allow arbitary single-qubit gate compensation
# This is useful for checking how many errors are fundamentally multi-qubit

# This specify one order of the qubits such that the simultaneous gates are [cnot()] * 3
unitary_order = ['q02', 'q03', 'q12', 'q13', 'q22', 'q23']

# instance the quantum processor graph, and choose a subset for time evolution
graph = CNOTGatePeriodicGraphOpt(seed=1)
qubit_subset = graph.subgraph(unitary_order)
opt = 'scipy'

evo = Evolve(qubit_subset, truncated_dim, add_random, share_params,
             unify_coupling, compensation_option)

# Compute the order of qubits in qubit_subset related to unitary_order
qubit_order = [unitary_order.index(key) for key in qubit_subset.sorted_nodes]
# Compute the corresponding target unitary based on the above order
unitary = permute(tensor(*([cnot()] * 3)), [2] * len(qubit_subset.nodes),
                  qubit_order)


# %%
def infidelity(params, unitary):
    params = hk.data_structures.merge(evo.all_params, params)
    # Compute the time evolution unitary in the eigenbasis.
    sim_u = evo.eigen_basis(params)
    # calculate fidelity
    fidelity_vz, _ = compute_fidelity_with_1q_rotation_axis(unitary,
                                                            sim_u,
                                                            opt_method=None)

    return jnp.log10(1 - fidelity_vz)


# %%
# list the parameters that will be updated during the optimization
params = {
    'q02_pulse_rampcos': {
        'amp': jnp.array(0.18128846),
        'omega_d': jnp.array(2.58934559),
        'phase': jnp.array(-0.24290228),
        't_plateau': jnp.array(69.93608145),
        't_ramp': jnp.array(29.92806488)
    },
    'q12_pulse_rampcos': {
        'amp': jnp.array(0.17872194),
        'omega_d': jnp.array(4.19989714),
        'phase': jnp.array(-0.01543561),
        't_plateau': jnp.array(69.95327862),
        't_ramp': jnp.array(29.94562879)
    },
    'q22_pulse_rampcos': {
        'amp': jnp.array(0.23370657),
        'omega_d': jnp.array(3.65018191),
        'phase': jnp.array(-0.50006247),
        't_plateau': jnp.array(69.96302405),
        't_ramp': jnp.array(29.98769304)
    },
}
# %%
if __name__ == '__main__':
    print(infidelity(params, unitary))  # -3.1848728
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
