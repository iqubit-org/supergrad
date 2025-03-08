# %%
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp

from qiskit_dynamics import Solver
import qutip as qt
import scqubits as scq
import supergrad
from supergrad.helper import Spectrum
from supergrad.utils.fidelity import compute_fidelity_with_1q_rotation_axis
from supergrad.utils.qiskit_interface import to_qiskit_drive_hamiltonian
from supergrad.utils.qutip_interface import (to_qutip_operator,
                                             to_qutip_operator_function_pair)

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('/'.join(dir_path.split('/')[:-2]))

from benchmark.utils.create_simultaneous_model import create_simultaneous_x
from benchmark.utils.create_multipath_chain_scqubits import create_qubit_chain
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
# compute the energy spectrum by the supergrad
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
# time evolution by supergrad
u_supergrad = evo.product_basis(evo.all_params)
fid, res_unitary = compute_fidelity_with_1q_rotation_axis(
    u_ref, u_supergrad, compensation_option='only_vz')
fid
# %%
params = jax.tree.map(lambda x: np.array(x), evo.all_params)
hilbertspace, fm_list = create_qubit_chain(params, n_qubit, truncated_dim)
# compute the energy spectrum by the scqubits
hilbertspace.eigenvals(2**n_qubit)
# %%
# time evolution by qutip using the scqubits Hamiltonian
ham_static_scq = hilbertspace.hamiltonian()
# create the drive terms
drive_opt = [[
    scq.identity_wrap(fm.phi_operator(), fm, hilbertspace.subsystem_list),
    opt_pulse[1]
] for fm, opt_pulse in zip(fm_list, operator_function_pair_list)]
# check the Hamiltonian
np.allclose(ham_static_scq.full(), ham_static.full())
# %%
# using the qutip solver
u0 = qt.qeye(list(evo.get_dims(evo.all_params)))
# config solver option
options = qt.Options(nsteps=1e6, atol=1e-8, rtol=1e-8)

output = qt.sesolve([ham_static_scq] + drive_opt,
                    u0, [0.0, t_span],
                    options=options)
u_scq = output.states[-1].full()
fid, res_unitary = compute_fidelity_with_1q_rotation_axis(
    u_ref, u_scq, compensation_option='only_vz')
print(f'Compensated qutip fidelity: {fid}')
# check the results of time evolution
fidelity(u_scq, u_supergrad)
# %%
# using the qiskit dynamics solver
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
u_qiskit_rot = solver.model.rotating_frame.state_out_of_frame(
    t_span, results.y[-1])
fid_qiskit_rot, _ = compute_fidelity_with_1q_rotation_axis(
    u_ref, u_qiskit_rot, compensation_option='only_vz')
print(f'Compensated qutip fidelity: {fid_qiskit_rot}')
fidelity(u_qiskit_rot, u_supergrad)
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
u_qiskit = results.y[-1]
fid_qiskit, _ = compute_fidelity_with_1q_rotation_axis(
    u_ref, u_qiskit, compensation_option='only_vz')
print(f'Compensated qutip fidelity: {fid_qiskit}')
fidelity(u_qiskit, u_supergrad)
# %%
