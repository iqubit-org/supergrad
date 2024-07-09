# %%
import jax
import jax.numpy as jnp
import haiku as hk

from supergrad.helper import Evolve
from supergrad.utils import tensor, compute_fidelity_with_1q_rotation_axis
from supergrad.utils.fidelity import estimate_vsq
from supergrad.utils.gates import cnot
from supergrad.utils.optimize import scipy_minimize, adam_opt

from supergrad.scgraph.graph_mpc_fluxonium_1d import MPCFluxonium1D

# %%
n_qubit = 10
astep = 2000
trotter_order_external = '4j'
if trotter_order_external == '4j':
    trotter_order = 4j
else:
    trotter_order = trotter_order_external
opt = 'scipy'

chain = MPCFluxonium1D(n_qubit, periodic=False, seed=0)
chain.share_params = True
chain.unify_coupling = True
chain.set_all_node_attr(truncated_dim=2)
chain.set_compensation(compensation_option="arbit_single", update_only=False)
transform_mat = chain.create_cr_pulse([0, 2, 4, 6, 8], [1, 3, 5, 7, 9],
                                      [100.0, 100.0, 100.0, 100.0, 100.0])
# Note the transform matrix could be reused if not optimizing the device parameters.
target_unitary = tensor(*[cnot(), ] * (n_qubit // 2))

evo = Evolve(chain,
             options={
                 'astep': astep,
                 'trotter_order': trotter_order,
                 'progress_bar': True,
                 'custom_vjp': True,
             })

# %%
if __name__ == '__main__':
    # generate initial guess for single qubit gate compensation.
    # Each parameter in the dictionary will be optimized.
    # if the key is commented out, the parameter will not be optimized.
    params = {"nodes":{
        'fm0': {'pulse': {'p1': {
            'amp': jnp.array(0.13574979),
            # 'length': jnp.array(100.),
            'omega_d': jnp.array(3.67504869),
            'phase': jnp.array(0.)
        }}, 'compensation': {
            "pre_comp": jnp.array([-0.00117551, -0.99847706, 0.09863956]),
            "post_comp": jnp.array([-1.03269930e-03, -1.09300029e+00, -4.26010233e-03], )
        }},
        'fm1': {'compensation': {
            'pre_comp':
                jnp.array([-0.33066666, -2.31136773, 0.80243353]),
            'post_comp':
                jnp.array([-0.72901447, -2.25297321, -0.74278428]),
        }},
        'fm2': {'pulse': {'p1': {
            'amp': jnp.array(0.29515405),
            # 'length': jnp.array(100.),
            'omega_d': jnp.array(3.04681771),
            'phase': jnp.array(0.)
        }}, 'compensation': {
            'pre_comp':
                jnp.array([0.0108088, -0.05646675, 0.03983036]),
            'post_comp':
                jnp.array([0.00013838, -0.09555538, -0.00020102]),
        }},
        'fm3': {'compensation': {
            'pre_comp':
                jnp.array([0.46449631, 0.88272318, 0.82729624]),
            'post_comp':
                jnp.array([0.58288783, 0.86442187, -0.77451809]),
        }},
        'fm4': {'pulse': {'p1': {
            'amp': jnp.array(0.13536027),
            # 'length': jnp.array(100.),
            'omega_d': jnp.array(4.26428773),
            'phase': jnp.array(0.)
        }}, 'compensation': {
            'pre_comp':
                jnp.array([-0.01253507, -0.1208205, 0.07362858]),
            'post_comp':
                jnp.array([0.00316941, -0.19009767, -0.00412084]),
        }},
        'fm5': {'compensation': {
            'pre_comp':
                jnp.array([-0.93284435, 1.07637714, 0.83512873]),
            'post_comp':
                jnp.array([-3.00819395e-05, -4.26033212e-01, 2.43266372e-01], ),
        }},
        'fm6': {'pulse': {'p1': {
            'amp': jnp.array(0.10956991),
            # 'length': jnp.array(100.),
            'omega_d': jnp.array(3.71165778),
            'phase': jnp.array(0.)
        }}, 'compensation': {
            'pre_comp':
                jnp.array([1.54524860e-03, -1.66215795e+00, -1.37914189e-02], ),
            'post_comp':
                jnp.array([0.00394743, -1.46213493, -0.18641146]),
        }},
        'fm7': {
            'compensation': {
                'pre_comp':
                    jnp.array([-1.20566752, -1.91060872, 0.79038521]),
                'post_comp':
                    jnp.array([0.32501355, -1.12228695, -2.04066705]),
            }},
        'fm8': {'pulse': {'p1': {
            'amp': jnp.array(0.23898645),
            # 'length': jnp.array(100.),
            'omega_d': jnp.array(3.12145721),
            'phase': jnp.array(0.)
        }}, 'compensation': {
            'pre_comp':
                jnp.array([0.00108516, -0.6865961, 0.01995555]),
            'post_comp':
                jnp.array([-0.0010828, -0.71285211, 0.00639239]),
        }},
        'fm9': {
            'compensation': {
                'pre_comp':
                    jnp.array([0.76766522, 0.98371765, 0.82451643]),
                'post_comp':
                    jnp.array([0.76091061, 0.24625422, -0.18006744])
            }
        }
    }}
    if params is None:
        params = estimate_vsq(evo, evo.pulse_params, target_unitary, n_qubit,
                              transform_mat, params)


    def infidelity(params, unitary):
        # Compute the time evolution unitary in the eigenbasis.
        sim_u = evo.eigen_basis(params, transform_matrix=transform_mat)
        # calculate fidelity
        fidelity_vz, _ = compute_fidelity_with_1q_rotation_axis(
            unitary, sim_u, compensation_option='no_comp', opt_method=None)

        return jnp.log10(1 - fidelity_vz)


    # sanity check
    print(infidelity(params, target_unitary))  # -0.9661292
    # %%
    jax.value_and_grad(infidelity)(params, target_unitary)
    # %%
    if opt == 'scipy':

        res = scipy_minimize(infidelity,
                             params,
                             args=(target_unitary),
                             method='l-bfgs-b',
                             logging=True,
                             options={'maxiter': 2000})
    elif opt == 'adam':

        adam_opt(infidelity,
                 params,
                 args=(target_unitary,),
                 options={'adam_lr': 0.01})
# %%
