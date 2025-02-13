# %%
import os
import sys
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

from qiskit_dynamics import Solver
import qutip as qt
import supergrad
from supergrad.helper import Evolve
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
# we using the supergrad with LCAM method
n_qubit = 4
# %%
# def test_simultaneous_x_grad_lcam(benchmark, n_qubit):
# bench supergrad
evo = create_simultaneous_x(n_qubit=n_qubit,
                            astep=5000,
                            trotter_order=2,
                            diag_ops=True,
                            minimal_approach=True,
                            custom_vjp=True)
# benchmark
u_ref = supergrad.tensor(*[np.array([[0, 1], [1, 0]])] * n_qubit)
# u_ref = jax.device_put(u_ref, jax.devices('cpu')[0])

@jax.jit
@jax.value_and_grad
def infidelity(params):
    return 1 - distributed_state_fidelity(u_ref, evo.product_basis(params))


infidelity(evo.all_params)
# %%
# output of the gradients computation
# (Array(0.69959341, dtype=float64), {
#     'capacitive_coupling_fm0_fm1': {
#         'strength': Array(-0.00383161, dtype=float64, weak_type=True)
#     },
#     'capacitive_coupling_fm1_fm2': {
#         'strength': Array(-0.00914214, dtype=float64, weak_type=True)
#     },
#     'capacitive_coupling_fm2_fm3': {
#         'strength': Array(-0.00437054, dtype=float64, weak_type=True)
#     },
#     'fm0': {
#         'ec': Array(-0.15246967, dtype=float64, weak_type=True),
#         'ej': Array(0.04749101, dtype=float64, weak_type=True),
#         'el': Array(-0.13651917, dtype=float64, weak_type=True),
#         'phiext': Array(0.0001105, dtype=float64, weak_type=True)
#     },
#     'fm0_pulsesq_cos': {
#         'amp': Array(-0.07253371, dtype=float64),
#         'length': Array(0.17250946, dtype=float64),
#         'omega_d': Array(-11.11458648, dtype=float64),
#         'phase': Array(-0.4521345, dtype=float64)
#     },
#     'fm1': {
#         'ec': Array(0.09756752, dtype=float64, weak_type=True),
#         'ej': Array(-0.03089809, dtype=float64, weak_type=True),
#         'el': Array(0.0856037, dtype=float64, weak_type=True),
#         'phiext': Array(2.87246361e-06, dtype=float64, weak_type=True)
#     },
#     'fm1_pulsesq_cos': {
#         'amp': Array(-0.22041418, dtype=float64),
#         'length': Array(0.17326897, dtype=float64),
#         'omega_d': Array(2.97855551, dtype=float64),
#         'phase': Array(0.12220165, dtype=float64)
#     },
#     'fm2': {
#         'ec': Array(0.56080074, dtype=float64, weak_type=True),
#         'ej': Array(-0.18273928, dtype=float64, weak_type=True),
#         'el': Array(0.48153142, dtype=float64, weak_type=True),
#         'phiext': Array(7.93590519e-06, dtype=float64, weak_type=True)
#     },
#     'fm2_pulsesq_cos': {
#         'amp': Array(-0.19067468, dtype=float64),
#         'length': Array(0.17291987, dtype=float64),
#         'omega_d': Array(1.90321611, dtype=float64),
#         'phase': Array(0.09944897, dtype=float64)
#     },
#     'fm3': {
#         'ec': Array(-0.09918134, dtype=float64, weak_type=True),
#         'ej': Array(0.03072069, dtype=float64, weak_type=True),
#         'el': Array(-0.0905288, dtype=float64, weak_type=True),
#         'phiext': Array(0.0001092, dtype=float64, weak_type=True)
#     },
#     'fm3_pulsesq_cos': {
#         'amp': Array(-0.04134376, dtype=float64),
#         'length': Array(0.17433787, dtype=float64),
#         'omega_d': Array(14.21093452, dtype=float64),
#         'phase': Array(0.56368919, dtype=float64)
#     },
#     'inductive_coupling_fm0_fm1': {
#         'strength': Array(-0.85234604, dtype=float64, weak_type=True)
#     },
#     'inductive_coupling_fm1_fm2': {
#         'strength': Array(-1.5685977, dtype=float64, weak_type=True)
#     },
#     'inductive_coupling_fm2_fm3': {
#         'strength': Array(-0.86631625, dtype=float64, weak_type=True)
#     }
# })
# %%
@partial(trace_max_memory_usage, pid=os.getpid())
@jax.block_until_ready
def vg_infidelity():
    return infidelity(evo.all_params)

# save the gradient
jnp.save(f'{dir_path}/res_data/grad_lcam_simultaneous_x_{n_qubit}.npy',
            vg_infidelity[0])
assert len(vg_infidelity[0]) == 2
