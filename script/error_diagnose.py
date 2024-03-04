# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import jax.numpy as jnp
from jax.config import config
import haiku as hk

from supergrad.common_functions import Evolve
from supergrad.utils import tensor, permute, compute_fidelity_with_1q_rotation_axis
from supergrad.utils.gates import cnot
from supergrad.utils.fidelity import plot_pauli_diagnose_partition

from examples.fluxonium_multipath_coupling.graph_5x5_periodic import CNOTGatePeriodicGraph, CNOTGatePeriodicGraphOpt

config.update('jax_enable_x64', True)

truncated_dim = 3
enable_var = True
share_params = True
unify_coupling = True
compensation_option = 'arbit_single'

# This specify one order of the qubits such that the simultaneous gates are [cnot()] * 3
unitary_order = ['q02', 'q03', 'q12', 'q13', 'q22', 'q23']

# instance the quantum processor graph, and choose a subset for time evolution
graph = CNOTGatePeriodicGraph(seed=1)
qubit_subset = graph.subgraph(unitary_order)

evo = Evolve(qubit_subset, truncated_dim, enable_var, share_params, unify_coupling,
             compensation_option)
# load example after optimization
graph_after = CNOTGatePeriodicGraphOpt(seed=1)
qubit_subset_after = graph_after.subgraph(unitary_order)

evo_after = Evolve(qubit_subset_after, truncated_dim, enable_var, share_params,
                   unify_coupling, compensation_option)

# Compute the order of qubits in qubit_subset related to unitary_order
qubit_order = [unitary_order.index(key) for key in qubit_subset.sorted_nodes]
# Compute the corresponding target unitary based on the above order
unitary = permute(tensor(*([cnot()] * 3)), [2] * len(qubit_subset.nodes),
                  qubit_order)


# %%
def infidelity(params, unitary, evo: Evolve):
    params = hk.data_structures.merge(evo.static_params, params)
    # Compute the time evolution unitary in the eigenbasis.
    sim_u = evo.eigen_basis(params)
    # calculate fidelity
    fidelity_vz, _ = compute_fidelity_with_1q_rotation_axis(unitary,
                                                            sim_u,
                                                            n_qubit=6,
                                                            opt_method=None)

    return jnp.log10(1 - fidelity_vz), sim_u


# %%
# run time evolution
evo.static_params.update(graph.compensation)
infid, sim_u = infidelity(evo.static_params, unitary, evo)
print(infid)
evo_after.static_params.update(graph_after.compensation)
infid_after, ux_after = infidelity(evo_after.static_params, unitary, evo_after)
print(infid_after)
# %%
# set plot config
# Find the path of the New Times font file
font_path = font_manager.findfont(
    font_manager.FontProperties(family='Times New Roman'))

# Set the font properties
font_props = font_manager.FontProperties(fname=font_path)

# Use the font properties in Matplotlib
plt.rcParams['font.family'] = font_props.get_name()
plt.rcParams.update({'font.size': 16.5})
fig = plt.figure(figsize=(8, 5.5))
markers = ["o", "^"]
colors = ['teal', 'orange', 'purple', 'red']
f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]

handles = [f(markers[i], "k") for i in range(2)]
handles += [f("s", colors[i]) for i in range(4)]
labels = ["Before optimization", "After optimization"] + [
    '1-location error', '2-location error', '3-location error',
    '4-location error'
]
plot_pauli_diagnose_partition(unitary, ux_after, marker='^')
plot_pauli_diagnose_partition(unitary, sim_u, marker='o')
plt.legend(handles, labels)
plt.savefig('simul_two_gubit_gate_error_distribution.pdf')
# %%
