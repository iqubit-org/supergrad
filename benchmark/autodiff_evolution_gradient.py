# %%
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

from qiskit_dynamics import Solver
import supergrad
from supergrad.utils.qiskit_interface import (to_qiskit_static_hamiltonian,
                                              to_qiskit_drive_hamiltonian)
from supergrad.utils.sharding import distributed_state_fidelity

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('/'.join(dir_path.split('/')[:-2]))

from benchmark.utils.create_simultaneous_model import create_simultaneous_x
# %%
# create 1d chain model, apply simultaneous X gates
# as a baseline approach to compute gradients using differentiable simulation
# we using the supergrad with LCAM method
n_qubit = 4
# %%
# bench supergrad
evo = create_simultaneous_x(n_qubit=n_qubit,
                            astep=5000,
                            trotter_order=2,
                            diag_ops=True,
                            minimal_approach=True,
                            custom_vjp=True)
u_ref = supergrad.tensor(*[np.array([[0, 1], [1, 0]])] * n_qubit)


@jax.jit
@jax.value_and_grad
def infidelity(params):
    params = hk.data_structures.merge(evo.all_params, params)
    return 1 - distributed_state_fidelity(u_ref, evo.product_basis(params))


# compute the gradients by the supergrad
v_supergrad, g_supergrad = infidelity(evo.pulse_params)


# %%
# using the qiskit dynamics solver
@jax.value_and_grad
def qiskit_pulse_obj(params):
    params = hk.data_structures.merge(evo.all_params, params)
    ham_static, hamiltonian_component_and_pulseshape, t_span = evo.construct_hamiltonian_and_pulseshape(
        params)
    # stop the gradient computation for the static hamiltonian
    static_hamiltonian = jax.lax.stop_gradient(
        to_qiskit_static_hamiltonian(ham_static))
    drive_hamiltonian, drive_signal = to_qiskit_drive_hamiltonian(
        hamiltonian_component_and_pulseshape)
    # Evolving by qiskit dynamics in rotating frame
    solver = Solver(static_hamiltonian,
                    jax.lax.stop_gradient(drive_hamiltonian),
                    rotating_frame=static_hamiltonian,
                    array_library="jax")
    u0 = np.eye(np.prod(evo.get_dims(evo.all_params)), dtype=complex)

    results = solver.solve(
        t_span=[0, t_span],
        y0=u0,
        signals=drive_signal,
        method='jax_odeint',
        atol=1e-8,
        rtol=1e-8,
    )
    u_qiskit = solver.model.rotating_frame.state_out_of_frame(
        t_span, results.y[-1])
    return 1 - distributed_state_fidelity(u_ref, u_qiskit)


v_qiskit, g_qiskit = qiskit_pulse_obj(evo.pulse_params)
# %%
# check the results of gradients
diff_grad = jax.tree.map(lambda x, y: jnp.linalg.norm(x - y), g_supergrad,
                         g_qiskit)
diff_grad
# %%
