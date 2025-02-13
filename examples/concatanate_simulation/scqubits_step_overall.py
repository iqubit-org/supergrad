# %%
import os
import sys
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

from qiskit_dynamics import Solver
import qutip as qt
import scqubits as scq
import supergrad
from supergrad.helper import Evolve, Spectrum
from supergrad.utils import basis
from supergrad.utils.fidelity import compute_fidelity_with_1q_rotation_axis
from supergrad.utils.memory_profiling import trace_max_memory_usage
from supergrad.utils.qiskit_interface import (to_qiskit_static_hamiltonian,
                                              to_qiskit_drive_hamiltonian)
from supergrad.utils.qutip_interface import (to_qutip_operator,
                                             to_qutip_operator_function_pair)
from supergrad.utils.sharding import distributed_state_fidelity

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('/'.join(dir_path.split('/')[:-2]))

from supergrad.scgraph.graph_mpc_fluxonium_1d import MPCFluxonium1D

from benchmark.utils.create_simultaneous_model import create_simultaneous_x
from benchmark.utils.fidelity import fidelity
# %%
# create 1d chain model, apply simultaneous X gates
# as a baseline approach to compute gradients using differentiable simulation
n_qubit = 4
# %%
# we using the supergrad with LCAM method
# def test_simultaneous_x_grad_lcam(benchmark, n_qubit):
# bench supergrad
evo = create_simultaneous_x(n_qubit=n_qubit,
                            astep=5000,
                            trotter_order=2,
                            diag_ops=True,
                            minimal_approach=True,
                            custom_vjp=True)
spec = Spectrum(evo.graph, add_random=False)
# energy_tensor = spec.energy_tensor(spec.all_params)
# %%
# Hands-on create chain model by scqubits
cutoff = 120
truncated_dim = 5
params = jax.tree.map(lambda x: np.array(x), evo.all_params)


def create_fluxonium(label, params):
    return scq.Fluxonium(EC=params[label]['ec'],
                         EJ=params[label]['ej'],
                         EL=params[label]['el'],
                         flux=params[label]['phiext'] / 2 / np.pi,
                         cutoff=cutoff,
                         truncated_dim=truncated_dim,
                         evals_method='evals_jax_dense',
                         id_str=label)


# fm0 = create_fluxonium(label='fm0', params=params)
# fm0.eigenvals()
fm_list = [
    create_fluxonium(label=f'fm{i}', params=params) for i in range(n_qubit)
]
# %%
# add coupling terms and create the hilbertspace
hilbertspace = scq.HilbertSpace(fm_list)


def add_multi_coupling_term(fm_list, label1, label2,
                            hilbertspace: scq.HilbertSpace, params):
    hilbertspace.add_interaction(
        g=float(params[f'capacitive_coupling_{label1}_{label2}']['strength']),
        op1=fm_list[int(label1.lstrip('fm'))].n_operator,
        op2=fm_list[int(label2.lstrip('fm'))].n_operator,
        id_str=f'capacitive_coupling_{label1}_{label2}')
    hilbertspace.add_interaction(
        g=float(params[f'inductive_coupling_{label1}_{label2}']['strength']),
        op1=fm_list[int(label1.lstrip('fm'))].phi_operator,
        op2=fm_list[int(label2.lstrip('fm'))].phi_operator,
        id_str=f'inductive_coupling_{label1}_{label2}')


for i in range(n_qubit - 1):
    add_multi_coupling_term(fm_list,
                            label1=f'fm{i}',
                            label2=f'fm{i+1}',
                            hilbertspace=hilbertspace,
                            params=params)

hilbertspace.eigenvals()
# %%
energy_tensor = spec.energy_tensor(spec.all_params)
jnp.sort(energy_tensor.flatten())
# %%
# create the drive terms
# %%
# {
#     'capacitive_coupling_fm0_fm1': {
#         'strength': Array(0.12566371, dtype=float64, weak_type=True)
#     },
#     'capacitive_coupling_fm1_fm2': {
#         'strength': Array(0.12566371, dtype=float64, weak_type=True)
#     },
#     'capacitive_coupling_fm2_fm3': {
#         'strength': Array(0.12566371, dtype=float64, weak_type=True)
#     },
#     'fm0': {
#         'ec': Array(6.28318531, dtype=float64, weak_type=True),
#         'ej': Array(25.13274123, dtype=float64, weak_type=True),
#         'el': Array(5.65486678, dtype=float64, weak_type=True),
#         'phiext': Array(3.14159265, dtype=float64, weak_type=True)
#     },
#     'fm0_pulsesq_cos': {
#         'amp': Array(0.05651893, dtype=float64),
#         'length': Array(50., dtype=float64, weak_type=True),
#         'omega_d': Array(3.05289284, dtype=float64),
#         'phase': Array(0., dtype=float64, weak_type=True)
#     },
#     'fm1': {
#         'ec': Array(6.28318531, dtype=float64, weak_type=True),
#         'ej': Array(25.13274123, dtype=float64, weak_type=True),
#         'el': Array(6.28318531, dtype=float64, weak_type=True),
#         'phiext': Array(3.14159265, dtype=float64, weak_type=True)
#     },
#     'fm1_pulsesq_cos': {
#         'amp': Array(0.05926816, dtype=float64),
#         'length': Array(50., dtype=float64, weak_type=True),
#         'omega_d': Array(3.71523252, dtype=float64),
#         'phase': Array(0., dtype=float64, weak_type=True)
#     },
#     'fm2': {
#         'ec': Array(6.28318531, dtype=float64, weak_type=True),
#         'ej': Array(25.13274123, dtype=float64, weak_type=True),
#         'el': Array(6.91150384, dtype=float64, weak_type=True),
#         'phiext': Array(3.14159265, dtype=float64, weak_type=True)
#     },
#     'fm2_pulsesq_cos': {
#         'amp': Array(0.06122666, dtype=float64),
#         'length': Array(50., dtype=float64, weak_type=True),
#         'omega_d': Array(4.21605355, dtype=float64),
#         'phase': Array(0., dtype=float64, weak_type=True)
#     },
#     'fm3': {
#         'ec': Array(6.28318531, dtype=float64, weak_type=True),
#         'ej': Array(25.13274123, dtype=float64, weak_type=True),
#         'el': Array(5.65486678, dtype=float64, weak_type=True),
#         'phiext': Array(3.14159265, dtype=float64, weak_type=True)
#     },
#     'fm3_pulsesq_cos': {
#         'amp': Array(0.05729331, dtype=float64),
#         'length': Array(50., dtype=float64, weak_type=True),
#         'omega_d': Array(3.36019615, dtype=float64),
#         'phase': Array(0., dtype=float64, weak_type=True)
#     },
#     'inductive_coupling_fm0_fm1': {
#         'strength': Array(-0.01256637, dtype=float64, weak_type=True)
#     },
#     'inductive_coupling_fm1_fm2': {
#         'strength': Array(-0.01256637, dtype=float64, weak_type=True)
#     },
#     'inductive_coupling_fm2_fm3': {
#         'strength': Array(-0.01256637, dtype=float64, weak_type=True)
#     }
# }
