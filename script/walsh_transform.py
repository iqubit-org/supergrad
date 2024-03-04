# %%
from jax.config import config
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from supergrad.common_functions import Spectrum
from supergrad.utils.sys_diagnose import walsh_transform, plot_walsh_transform

from examples.fluxonium_multipath_coupling.graph_5x5_periodic import CNOTGatePeriodicGraph, CNOTGatePeriodicGraphOpt

config.update('jax_enable_x64', True)

truncated_dim = 4
enable_var = True
share_params = True
unify_coupling = True
compensation_option = 'arbit_single'

# This specify one order of the qubits such that the simultaneous gates are [cnot()] * 3
unitary_order = ['q02', 'q03', 'q12', 'q13', 'q22', 'q23']

# instance the quantum processor graph, and choose a subset
graph_before = CNOTGatePeriodicGraph(seed=1)
qubit_subset_before = graph_before.subgraph(unitary_order)
spec_before = Spectrum(qubit_subset_before, truncated_dim, enable_var)

graph_after = CNOTGatePeriodicGraphOpt(seed=1)
qubit_subset_after = graph_after.subgraph(unitary_order)
spec_after = Spectrum(qubit_subset_after, truncated_dim, enable_var)

# %%
# plt walsh transform before optimization
energy_nd = spec_before.energy_tensor(spec_before.static_params)
coeff, bitstring = walsh_transform(energy_nd)
# remove the lowest coefficient term
coeff = coeff[:-1]
bitstring = bitstring[:-1]
# %%
# Find the path of the New Times font file
font_path = font_manager.findfont(
    font_manager.FontProperties(family='Times New Roman'))

# Set the font properties
font_props = font_manager.FontProperties(fname=font_path)

# Use the font properties in Matplotlib
plt.rcParams['font.family'] = font_props.get_name()
plt.rcParams.update({'font.size': 16.5})
fig = plt.figure(figsize=(8, 5.5))
plot_walsh_transform(coeff, bitstring)
# save the figure as pdf
plt.savefig('walsh_before_optimization.pdf', format='pdf')
plt.show()
# %%
# plt walsh transform after optimization
energy_nd = spec_after.energy_tensor(spec_after.static_params)
coeff, bitstring = walsh_transform(energy_nd)
# remove the lowest coefficient term
coeff = coeff[:-1]
bitstring = bitstring[:-1]
fig = plt.figure(figsize=(8, 5.5))
plot_walsh_transform(coeff, bitstring)
# save the figure as pdf
plt.savefig('walsh_after_optimization.pdf', format='pdf')
plt.show()
# %%
