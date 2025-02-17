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
n_qubit = 2
truncated_dim = 2
add_random = False
# %%
# we using the supergrad with LCAM method
# def test_simultaneous_x_grad_lcam(benchmark, n_qubit):
# bench supergrad
evo = create_simultaneous_x(n_qubit=n_qubit,
                            astep=5000,
                            trotter_order=2,
                            diag_ops=True,
                            minimal_approach=True,
                            custom_vjp=True,
                            add_random=add_random,
                            drive_for_state_phase='phase')
# unify the state phase based on the phase operator (as the same as the scqubits)
spec = Spectrum(evo.graph, truncated_dim=2, add_random=add_random)
# energy_tensor = spec.energy_tensor(spec.all_params)
# %%
# Hands-on create chain model by scqubits
params = jax.tree.map(lambda x: np.array(x), evo.all_params)


def create_fluxonium(label, params):
    return scq.Fluxonium(EC=params[label]['ec'],
                         EJ=params[label]['ej'],
                         EL=params[label]['el'],
                         flux=params[label]['phiext'] / 2 / np.pi,
                         cutoff=120,
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

hilbertspace.eigenvals(2**n_qubit)
# %%
energy_tensor = spec.energy_tensor(spec.all_params)
jnp.sort(energy_tensor.flatten())
# %%
ham_static, hamiltonian_component_and_pulseshape, t_span = evo.construct_hamiltonian_and_pulseshape(
    evo.all_params)
ham_static_qutip = to_qutip_operator(ham_static)
operator_function_pair_list = to_qutip_operator_function_pair(
    hamiltonian_component_and_pulseshape)
u_ref = supergrad.tensor(*[np.array([[0, 1], [1, 0]])] * n_qubit)
# %%
# create the drive terms
ham_static_scq = hilbertspace.hamiltonian()
hilbertspace.generate_lookup()
drive_opt = [[
    scq.identity_wrap(fm.phi_operator(), fm, hilbertspace.subsystem_list),
    opt_pulse[1]
] for fm, opt_pulse in zip(fm_list, operator_function_pair_list)]

dressed_drive_opt = [[
    hilbertspace.op_in_dressed_eigenbasis(fm.phi_operator), opt_pulse[1]
] for fm, opt_pulse in zip(fm_list, operator_function_pair_list)]
np.allclose(ham_static_scq.full(), ham_static.full())
# %%
# time evolution by supergrad
u_supergrad = evo.product_basis(evo.all_params)
# fid, res_unitary = compute_fidelity_with_1q_rotation_axis(
#     u_ref, u_supergrad, compensation_option='only_vz')
# fid
# %%
# time evolution by qutip using the scqubits Hamiltonian
u0 = qt.qeye(list(evo.get_dims(evo.all_params)))
# config solver option
options = qt.Options(nsteps=1e6, atol=1e-8, rtol=1e-8)

output = qt.sesolve([ham_static_scq] + drive_opt,
                    u0, [0.0, t_span],
                    options=options)
# output = qt.sesolve([qt.qdiags(ham_static_scq.eigenenergies())] +
#                     dressed_drive_opt,
#                     qt.qeye([[4]]), [0.0, t_span],
#                     options=options)
# output = qt.sesolve([ham_static_qutip] + operator_function_pair_list,
#                     u0, [0.0, t_span],
#                     options=options)
u_scq = output.states[-1].full()
# fid, res_unitary = compute_fidelity_with_1q_rotation_axis(
#     u_ref, u_scq, compensation_option='only_vz')
fidelity(u_scq, u_supergrad)
# %%
# using the qiskit dynamics API
static_hamiltonian = ham_static_scq.full()
drive_hamiltonian = [
    scq.identity_wrap(fm.phi_operator(), fm,
                      hilbertspace.subsystem_list).full() for fm in fm_list
]
_, drive_signal = to_qiskit_drive_hamiltonian(
    hamiltonian_component_and_pulseshape)
# Evolving by qiskit dynamics in rotating frame
solver = Solver(static_hamiltonian,
                drive_hamiltonian,
                rotating_frame=static_hamiltonian)
u0 = np.eye(np.prod(evo.get_dims(evo.all_params)), dtype=complex)

results = solver.solve(
    t_span=[0, t_span],
    y0=u0,
    signals=drive_signal,
    atol=1e-8,
    rtol=1e-8,
)
u_qiskit = solver.model.rotating_frame.state_out_of_frame(t_span, results.y[-1])
fidelity(u_qiskit, u_supergrad)
# %%
# Evolving by qiskit dynamics
solver = Solver(static_hamiltonian, drive_hamiltonian)
u0 = np.eye(np.prod(evo.get_dims(evo.all_params)), dtype=complex)

results = solver.solve(
    t_span=[0, t_span],
    y0=u0,
    signals=drive_signal,
    atol=1e-8,
    rtol=1e-8,
)
fidelity(u_qiskit, u_supergrad)
# %%
