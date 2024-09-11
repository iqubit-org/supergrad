import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.experimental import mesh_utils
import jaxopt

import supergrad.utils.fidelity as fidelity_lib


def get_sharding(*partition_spec):
    """Get multi-devices with sharding.

    Args:
        data: `jax.numpy.ndarray`
            The data to be distributed batched.
        partition_spec: `jax.sharding.PartitionSpec`
            The partition specification. The sharding is done along the "p"
            axis_name.
    """
    devices = mesh_utils.create_device_mesh((jax.local_device_count(),))
    mesh = Mesh(devices, 'p')
    return NamedSharding(mesh, PartitionSpec(*partition_spec))


def distributed_state_fidelity(target_states, computed_states):
    """Compute the state fidelity or the composite state (unitary) fidelity.
    The fidelity could be computed in a multi-devices cluster with the same sharding.

    Args:
        target_states: The target states.
        computed_states: The computed states.
    """
    # assert target_states.sharding == computed_states.sharding
    ar_state_fidelity = jnp.sum(jnp.conj(target_states) * computed_states,
                                axis=0)
    return jnp.abs(ar_state_fidelity.mean())**2


def distributed_overlap_with_auto_vz_compensation(psi_target,
                                                  psi_computed,
                                                  opt_method: str = 'gd',
                                                  options: dict = None):
    """
    Compute inner product of states, allows for vz compensation after.

    Args:
        psi_target (array): Target state
        psi_computed (array): State
        opt_method (string): optimizer, set `gd` for using the Gradient Descent
            solver; set `lbfgs` for using the LBFGS solver.
        options (dict): additional keyword arguments to be passed to the `jaxopt`
            optimizer.
    """
    n_qubit = round(np.log2(psi_target.shape[0]))
    if options is None:
        options = {}
    # initialize single qubit compensation
    initial_guess = jnp.zeros([n_qubit])

    def error(params, psi_target, psi_computed):
        psi2 = fidelity_lib.apply_nz(params, psi_computed)
        return jnp.log(1 - distributed_state_fidelity(psi_target, psi2))

    if opt_method == 'lbfgs':
        # call jaxopt differentiable LBFGS solver
        solver = jaxopt.LBFGS(error, **options)
    elif opt_method == 'gd':
        solver = jaxopt.GradientDescent(error, **options)
    else:
        raise ValueError('opt_method should be `gd` or `lbfgs`.')

    res = solver.run(initial_guess,
                     psi_target=psi_target,
                     psi_computed=psi_computed)
    # print(initial_guess)
    # print(res.state)
    # print(res.params)
    psi_optimal = fidelity_lib.apply_nz(res.params, psi_computed)

    return distributed_state_fidelity(psi_target, psi_optimal), psi_optimal


def distributed_fidelity_with_auto_vz_compensation(u_target,
                                                   u_computed,
                                                   opt_method: str = 'gd',
                                                   options: dict = None):
    """
    Compute fidelity, allows for vz compensation before/after.

    Args:
        u_target (array): Target unitary
        u_computed (array): Unitary
        opt_method (string): optimizer, set `gd` for using the Gradient Descent
            solver; set `lbfgs` for using the LBFGS solver.
        options (dict): additional keyword arguments to be passed to the `jaxopt`
            optimizer.
    """
    n_qubit = round(np.log2(u_target.shape[0]))
    if options is None:
        options = {}
    # initialize single qubit compensation
    initial_guess = jnp.zeros([2 * n_qubit])

    def error(params, u_target, u_computed):
        u2 = fidelity_lib.apply_2nz(params, u_computed)
        return jnp.log(1 - distributed_state_fidelity(u_target, u2))

    if opt_method == 'lbfgs':
        # call jaxopt differentiable LBFGS solver
        solver = jaxopt.LBFGS(error, **options)
    elif opt_method == 'gd':
        solver = jaxopt.GradientDescent(error, **options)
    else:
        raise ValueError('opt_method should be `gd` or `lbfgs`.')

    res = solver.run(initial_guess, u_target=u_target, u_computed=u_computed)
    # print(initial_guess)
    # print(res.state)
    # print(res.params)
    u_optimal = fidelity_lib.apply_2nz(res.params, u_computed)

    return distributed_state_fidelity(u_target, u_optimal), u_optimal
