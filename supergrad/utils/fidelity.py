from typing import Tuple, Optional
import itertools
import numpy as np
from numpy import pi
import jax
import jax.numpy as jnp
import jaxopt
#import haiku as hk
from numpy import ndarray
from rich import print as rprint
import matplotlib.pyplot as plt

import supergrad
from supergrad.scgraph.graph import (parse_pre_comp_name, parse_post_comp_name)
from .optimize import scipy_minimize

pauli_mats = []
pauli_mats.append(np.eye(2))
pauli_mats.append(np.array([[0., 1.], [1., 0.]]))
pauli_mats.append(np.array([[0., -1j], [1j, 0.]]))
pauli_mats.append(np.array(np.diag((1., -1.))))


def compute_average_fidelity_with_leakage(u_target, u_computed):
    """
    Compute fidelity with auto state average

    Args:
        u_target (array): Target unitary
        u_computed (array): Test unitary
    """

    n = u_computed.shape[0]
    res = (jnp.trace(jnp.conj(u_computed.T) @ u_computed) +
           abs(jnp.trace(jnp.conj(u_target.T) @ u_computed)) ** 2) / n / (n + 1)
    return jnp.abs(res)


def compute_state_fidelity(psi_target, psi_computed):
    """
    Compute state fidelity

    Args:
        psi_target (array): Target state
        psi_computed (array): Test state
    """
    return jnp.abs(jnp.vdot(psi_target, psi_computed)) ** 2


def apply_2nz(ps, u):
    """Applies n Z-gate before U and other n after U.
    Each single qubit rotation is reorganized as on axis theta.

    Args:
        ps: 2n z_j coefficients for single qubit rotation
        u (ndarray): test unitary

    Returns:
        ndarray: Unitary with single qubit rotations applied
    """
    # Single qubit gate in 1Q system
    n_qubit = round(len(ps) / 2)

    pre_unitaries = [
        jnp.cos(pre_params) * pauli_mats[0] +
        1j * jnp.sin(pre_params) * pauli_mats[3] for pre_params in ps[:n_qubit]
    ]
    post_unitaries = [
        jnp.cos(post_params) * pauli_mats[0] +
        1j * jnp.sin(post_params) * pauli_mats[3]
        for post_params in ps[n_qubit:]
    ]
    u2 = supergrad.tensor(*pre_unitaries)
    u3 = supergrad.tensor(*post_unitaries)
    return u3 @ u @ u2


def conv_sq_u_to_angles(u, canonical: bool = True):
    """Converts a 2x2 unitary matrix in the representation of rotation on axis theta, phi and
    angle gamma.

    The output order is consistent with :func:`apply_6nsq_axis` function in the order of theta, gamma, phi, global phase.

    To avoid numerical issues, some angles are fixed if the [0,0] or [0,1] are very close to zero.

    Args:
        u:  2x2 matrix

    Returns:
        theta, gamma, phi, global phase: 4 angles in radian
    """
    tol = 1e-8

    # Off-diagonal term ~ 0, skip phi
    if jnp.abs(u[0, 1]) < tol:
        phi = 0.0
    else:
        # Check extreme cases
        ratio = u[1, 0] / u[0, 1]
        phi = jnp.angle(ratio).real / 2

    expmphi = jnp.cos(phi) - 1j * jnp.sin(phi)

    # Note it is best to check if arguments of arc* is real
    # For ill input, the output is ill, which is OK anyway.
    # Diagonal term ~ 0, skip calculation of gamma
    if jnp.abs(u[0, 0]) < tol:
        gamma = pi / 2
        expphig_sin_theta = u[0, 1] / (1j * expmphi)
        phig = jnp.angle(expphig_sin_theta ** 2) / 2
        expphig = jnp.cos(phig) + 1j * jnp.sin(phig)
        sin_theta = (expphig_sin_theta / expphig).real
        if jnp.abs(sin_theta) > 1:
            sin_theta /= jnp.abs(sin_theta)
        theta = jnp.arcsin(sin_theta)
    else:
        expphig_cos_gamma = (u[1, 1] + u[0, 0]) / 2
        phig = jnp.angle(expphig_cos_gamma ** 2) / 2
        expphig = jnp.cos(phig) + 1j * jnp.sin(phig)
        cos_gamma = expphig_cos_gamma / expphig
        if jnp.abs(cos_gamma) > tol:
            cos_gamma /= jnp.abs(cos_gamma)
        gamma = jnp.arccos(cos_gamma.real)

        # Close to zero, cannot be used later in division
        # No theta information
        if jnp.sin(gamma) < tol:
            theta = 0.0
        else:
            cos_theta = ((u[0, 0] - u[1, 1]) / (2j * expphig * jnp.sin(gamma))).real
            if jnp.abs(cos_theta) > 1:
                cos_theta /= jnp.abs(cos_theta)
            if jnp.abs(u[0,1]) < tol:
                theta = jnp.arccos(cos_theta)
            else:
                sin_theta = (u[0, 1] / (expphig * 1j * jnp.sin(gamma) * expmphi)).real
                if jnp.abs(sin_theta) > 1:
                    sin_theta /= jnp.abs(sin_theta)
                theta = jnp.atan2(sin_theta.real, cos_theta.real)

    return jnp.array([theta, gamma, phi, phig])

def canonicalize_sq_angles(ang: jax.Array):
    r"""
    Canonicalizes 1Q gate angles (theta, gamma, phi) in :func:`conv_sq_u_to_angles` and :func:`apply_6nsq_axis`.

    The range follows :math:`$\gamma\in[-\pi, \pi],\theta\in[0, \pi],\phi\in[0,\pi]$, 
    which means the axis in only in half sphere.

    There are three equivalent combinations of angles: 
    :math:`$(\gamma,\theta,\phi),(-\gamma,\pi-\theta,\pi+\phi),(-\gamma,\theta+\pi,\phi)$`

    Args:
        ang: 3-elements angles

    Returns:
        3-elements angles canonicalized.
    """
    # Change into [-pi, pi]
    ang = jnp.remainder(ang + pi, 2 * pi) - pi
    theta = ang[0]
    gamma = ang[1]
    phi = ang[2]
    if theta < 0:
        gamma = -gamma
        if phi < 0:
            theta = - theta - pi
            phi = pi + phi
        else:
            theta = theta + pi
    else:
        if phi < 0:
            gamma = -gamma
            theta = pi - theta
            phi = pi + phi
    return jnp.array((theta, gamma, phi))


def conv_sq_angles_to_u(p: jax.Array):
    """Construct a single-qubit gate from 3 angles (axis theta/phi, angle gamma), global phase optionally.

    Args:
        p (array): size 3: array of theta, gamma and phi.
            size 4: final one is the global phase.

    Returns:
        u: 2x2 unitary matrix
    """
    u = jnp.cos(p[1]) * pauli_mats[0] + 1j * jnp.sin(p[1]) * (jnp.cos(p[0]) * pauli_mats[3] + jnp.sin(p[0]) * (
            jnp.cos(p[2]) * pauli_mats[1] + jnp.sin(p[2]) * pauli_mats[2]))
    if p.size == 4:
        u *= jnp.cos(p[3]) + 1j * jnp.sin(p[3])
    return u


def conv_sq_angles_to_6nsq(a1: jax.Array, a0: jax.Array, b1: jax.Array, b0: jax.Array) -> jax.Array:
    """Converts 4 1Q angles to a 2D array compatible 6N SQ functions.

    The output is compatible with :func`supergrad.utils.fidelity.apply_6nsq_axis`.

    Note the global phase is discarded.

    Args:
        a1: angles of U(2) after on Q1
        a0: angles of U(2) after on Q0
        b1: angles of U(2) before on Q1
        b0: angles of U(2) before on Q0

    Returns:
        4x3 array for 6n SQ function
    """
    return jnp.stack([b1, b0, a1, a0], axis=0)[..., :3]


def conv_6nsq_to_sq_angles(p: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Converts 6N SQ 2D parameters to 4 1Q angles.

    The input is compatible with :func`supergrad.utils.fidelity.apply_6nsq_axis`.

    Note the global phase is not included, so

    Args:
        p: 4x3 arrays for 6N SQ

    Returns:
        4 arrays for 1Q angles in a1, a0, b1, b0.
    """
    return p[2], p[3], p[0], p[1]


def apply_6nsq_axis(ps, u):
    """
    Apply 3n single qubit rotation before U and other 3n after U
    Each 3 single qubit rotation are reorganized as on axis theta, phi and
    angle gamma.

    Args:
        ps (array): 6n parameters for single qubit rotation
        u (array): unitary
    """

    # Single qubit gate in 1Q system
    n_qubit = round(len(ps) / 2)

    pre_unitaries = [
        jnp.cos(pre_params[1]) * pauli_mats[0] + 1j * jnp.sin(pre_params[1]) *
        (jnp.cos(pre_params[0]) * pauli_mats[3] + jnp.sin(pre_params[0]) *
         (jnp.cos(pre_params[2]) * pauli_mats[1] +
          jnp.sin(pre_params[2]) * pauli_mats[2]))
        for pre_params in ps[:n_qubit]
    ]
    post_unitaries = [
        jnp.cos(post_params[1]) * pauli_mats[0] + 1j * jnp.sin(post_params[1]) *
        (jnp.cos(post_params[0]) * pauli_mats[3] + jnp.sin(post_params[0]) *
         (jnp.cos(post_params[2]) * pauli_mats[1] +
          jnp.sin(post_params[2]) * pauli_mats[2]))
        for post_params in ps[n_qubit:]
    ]

    u2 = supergrad.tensor(*pre_unitaries)
    u3 = supergrad.tensor(*post_unitaries)
    return u3 @ u @ u2


def log_infidelity_with_vz(params, u_target, u_computed):
    """
    Compute infidelity loss with virtual Z gates.

    Args:
        params (array): phase compensation parameters
        u_target (array): Target unitary
        u_computed (array): Unitary
    """
    u_target = jnp.asarray(u_target)
    u_computed = jnp.asarray(u_computed)
    u2 = apply_2nz(params, u_computed)
    return jnp.log(
        (1 - compute_average_fidelity_with_leakage(u_target, u2).real))


def log_infidelity_with_vsq(params, u_target, u_computed):
    """
    Compute infidelity loss with virtual single qubit gates.

    Args:
        params (array): phase compensation parameters
        u_target (array): Target unitary
        u_computed (array): Unitary
    """
    u_target = jnp.asarray(u_target)
    u_computed = jnp.asarray(u_computed)
    u2 = apply_6nsq_axis(params, u_computed)
    return jnp.log(
        (1 - compute_average_fidelity_with_leakage(u_target, u2).real))


def compute_compensation_iswap_gate_vz(u_target, u_computed):
    """Iswap gate with phase compensation. Performing the compensation through
    virtual Z gates.

    Args:
        u_target (array): Target unitary
        u_computed (array): Unitary
    """
    # check u_target is iswap gate
    # assert jnp.sum((iswap().full() - u_target)**
    #                2) < 1e-4, 'The compensation only support iswap gate'
    # compensate U with a diagonal matrix D
    phase00 = u_computed[0, 0] / jnp.abs(u_computed[0, 0])
    phase12 = u_computed[1, 2] / jnp.abs(u_computed[1, 2])
    phase21 = u_computed[2, 1] / jnp.abs(u_computed[2, 1])
    u_computed = u_computed @ jnp.conj(
        jnp.diag(
            jnp.array([
                phase00, -phase21 * 1j, -phase12 * 1j,
                         -phase12 * phase21 / phase00
            ])))

    return u_computed


def simple_vz_compensation_angle(u_target,
                                 u_computed) -> Tuple[np.ndarray, jnp.ndarray]:
    """Gets an estimated single sided compensation VZ angle.
    This follows u_target = D @ u_computed,
    so should be applied to cases where u_computed is almost a monomial.
    The order D @ u_computed is chosen because for CR gate most of the
    single qubit phase is from the rotating frame.
    Also assume qubits.

    Args:
        u_target (array): Target unitary
        u_computed (array): Unitary

    Returns:
        an array of angle per qubit to rotate
        a unitary matrix used as (U*U_computed) to get the compensated unitary
    """

    diff = u_target @ u_computed.conj().T
    d = jnp.diag(diff)
    n = int(np.log2(d.size))
    single_q_phase_lst = []
    for i in range(n):
        single_q_phase_lst.append(d[0] / d[2 ** (n - 1 - i)])
        single_q_phase_lst[-1] = single_q_phase_lst[-1] / jnp.abs(
            single_q_phase_lst[-1])

    vz_diag = jnp.array([
        1,
    ])
    for i in range(n):
        vz_diag = jnp.kron(vz_diag, jnp.array([1, single_q_phase_lst[i] ** (-1)]))

    return np.angle(np.array(single_q_phase_lst)), jnp.diag(vz_diag)


def simple_vz_compensation(u_target, u_computed) -> jnp.ndarray:
    """This is currently a single sided compensation,
    i.e. u_target = D @ u_computed,
    so should be applied to cases where u_computed is almost a monomial.
    The order D @ u_computed is chosen because for CR gate most of the
    single qubit phase is from the rotating frame.
    Also assume qubits.

    Args:
        u_target (array): Target unitary
        u_computed (array): Unitary

    Returns:
        the compensated unitary
    """
    return simple_vz_compensation_angle(u_target, u_computed)[1] @ u_computed


def estimate_vz(
        evo,
        params,
        unitary,
        num_qubit,
        transform_matrix=None,
        options: dict = None,
):
    """Estimates the Virtual-Z gate to correct

    Args:
        evo: class `supergrad.Evolve`
        params: params without "single_q_compensation"
        unitary: target unitary
        num_qubit: number of qubits
        transform_matrix: transform matrix to eigen basis
        options: additional keyword arguments to be passed to the `jaxopt`
            optimizer.

    Returns:
        a dictionary of parameters with
    """
    u_computed = evo.eigen_basis(params, transform_matrix)

    if options is None:
        options = {}
    error = log_infidelity_with_vz
    initial_guess = jnp.zeros([2 * num_qubit])
    solver = jaxopt.GradientDescent(error, **options)
    res = solver.run(initial_guess, u_target=unitary, u_computed=u_computed)
    print(res.state)
    qubit_list = getattr(evo.graph, 'qubit_subsystem', evo.graph.sorted_nodes)
    list_key = [parse_pre_comp_name(node) for node in qubit_list
                ] + [parse_post_comp_name(node) for node in qubit_list]

    params["single_q_compensation"] = dict([
        (key, val) for key, val in zip(list_key, res.params)
    ])

    return params


def estimate_vsq(
        evo,
        params,
        unitary,
        num_qubit,
        transform_matrix=None,
        initial_guess=None,
        options: dict = None,
        use_scipy_optimizer=False,
):
    """Estimates the virtual single qubit gate to correct.

    Args:
        evo: class `supergrad.Evolve`
        params: params without "single_q_compensation"
        unitary: target unitary
        num_qubit: number of qubits
        transform_matrix: transform matrix to eigen basis
        initial_guess: initial guess for the optimizer
        options: additional keyword arguments to be passed to the `jaxopt`
            optimizer.
        use_scipy_optimizer: use scipy optimizer instead of jaxopt

    Returns:
        a dictionary of parameters with
    """
    u_computed = evo.eigen_basis(params, transform_matrix)

    if options is None:
        options = {}
    error = log_infidelity_with_vsq

    if initial_guess is None:
        initial_guess = jnp.zeros([2 * num_qubit, 3])
    elif isinstance(initial_guess.get('single_q_compensation', None), dict):
        initial_guess = jnp.array(
            [val for val in initial_guess['single_q_compensation'].values()])
    if not use_scipy_optimizer:
        solver = jaxopt.GradientDescent(error, **options)
        res = solver.run(initial_guess, u_target=unitary, u_computed=u_computed)
        print(res.state)
    else:
        # create inline class with attribute 'params'
        class res(object):
            pass

        res.params = scipy_minimize(error,
                                    initial_guess,
                                    args=(unitary, u_computed),
                                    logging=True,
                                    **options)['x']
    qubit_list = getattr(evo.graph, 'qubit_subsystem', evo.graph.sorted_nodes)
    list_key = [parse_pre_comp_name(node) for node in qubit_list
                ] + [parse_post_comp_name(node) for node in qubit_list]

    params["single_q_compensation"] = dict([
        (key, val) for key, val in zip(list_key, res.params)
    ])

    return params


def compute_overlap_with_1q_rotation_axis(psi_target,
                                          psi_computed,
                                          opt_method: str = 'gd',
                                          compensation_option='only_vz',
                                          options: dict = None):
    """
    Compute inner product of states, allows for any single qubit rotation after

    Args:
        psi_target (array): Target state
        psi_computed (array): State
        opt_method (string): optimizer, set `gd` for using the Gradient Descent
            solver; set `lbfgs` for using the LBFGS solver, set `iswap_vz` for
            performing the iswap gate compensation through virtual Z gates.
        compensation_option: (string)
            Set single qubit compensation strategy, should be in
            ['no_comp', 'only_vz', 'arbit_single']
        options (dict): additional keyword arguments to be passed to the `jaxopt`
            optimizer.
    """
    assert compensation_option in ['no_comp', 'only_vz', 'arbit_single']
    n_qubit = round(np.log2(psi_target.shape[0]))
    if options is None:
        options = {}

    def apply_nz(params, psi):
        post_diag = [
            jnp.array([
                jnp.cos(post_params) + 1j * jnp.sin(post_params),
                jnp.cos(post_params) - 1j * jnp.sin(post_params)
            ]) for post_params in params
        ]

        ud3 = supergrad.tensor(*post_diag)
        return ud3[:, jnp.newaxis] * psi

    def apply_3nsq_axis(params, psi):
        post_unitaries = [
            jnp.cos(post_params[1]) * pauli_mats[0] +
            1j * jnp.sin(post_params[1]) *
            (jnp.cos(post_params[0]) * pauli_mats[3] + jnp.sin(post_params[0]) *
             (jnp.cos(post_params[2]) * pauli_mats[1] +
              jnp.sin(post_params[2]) * pauli_mats[2]))
            for post_params in params
        ]

        u3 = supergrad.tensor(*post_unitaries)
        return u3 @ psi

    # initialize single qubit compensation
    if compensation_option == 'only_vz':

        def error(params, psi_target, psi_computed):
            psi_compensated = apply_nz(params, psi_computed)
            return 1 - compute_state_fidelity(psi_target, psi_compensated)

        initial_guess = jnp.zeros([n_qubit])
    elif compensation_option == 'arbit_single':

        def error(params, psi_target, psi_computed):
            psi_compensated = apply_3nsq_axis(params, psi_computed)
            return 1 - compute_state_fidelity(psi_target, psi_compensated)

        initial_guess = jnp.zeros([n_qubit, 3])
    elif compensation_option == 'no_comp':
        return compute_state_fidelity(psi_target, psi_computed), psi_computed

    if opt_method == 'lbfgs':
        # call jaxopt differentiable LBFGS solver
        solver = jaxopt.LBFGS(error, **options)
    elif opt_method == 'gd':
        solver = jaxopt.GradientDescent(error, **options)
    elif opt_method is None:
        return compute_state_fidelity(psi_target, psi_computed), psi_computed
    else:
        raise ValueError('opt_method should be `gd`, `lbfgs`.')

    res = solver.run(initial_guess,
                     psi_target=psi_target,
                     psi_computed=psi_computed)
    print(initial_guess)
    print(res.state)
    print(res.params)
    if compensation_option == 'only_vz':
        psi_compensated = apply_nz(res.params, psi_computed)
    else:
        psi_compensated = apply_3nsq_axis(res.params, psi_computed)

    return compute_state_fidelity(psi_target, psi_compensated), psi_compensated


def compute_fidelity_with_1q_rotation_axis(u_target,
                                           u_computed,
                                           opt_method: str = 'gd',
                                           compensation_option='only_vz',
                                           options: dict = None,
                                           multi_init: bool = False,
                                           return_1q=False,
                                           initial_guess: Optional[jnp.ndarray]=None):
    """
    Compute fidelity, allows for any single qubit rotation before/after

    Args:
        u_target (array): Target unitary
        u_computed (array): Unitary
        opt_method (string): optimizer, set `gd` for using the Gradient Descent
            solver; set `lbfgs` for using the LBFGS solver, set `iswap_vz` for
            performing the iswap gate compensation through virtual Z gates.
        compensation_option: (string)
            Set single qubit compensation strategy, should be in
            ['no_comp', 'only_vz', 'arbit_single']
        options (dict): additional keyword arguments to be passed to the `jaxopt`
            optimizer.
        multi_init: use multiple initial guess to avoid local minimum.
            Only useful for `lbfgs` and `gd`.
        return_1q: whether to return 1Q gate coefficients.
        initial_guess: specify an initial guess of the single qubit rotations. Overrides `multi_init`.
            Note this must be consistent with `compensation_option`.

    Returns:
        fidelity,
        rotated unitary matrices,
        1q parameters in 1-d array if `return_1q` is True

    """
    assert compensation_option in ['no_comp', 'only_vz', 'arbit_single']
    assert not (multi_init and initial_guess is not None)
    n_qubit = round(np.log2(u_target.shape[0]))
    if options is None:
        options = {}
    f = compute_average_fidelity_with_leakage

    # Multiple initial guess
    initial_guess_multi = jnp.zeros(0)
    # initialize single qubit compensation
    if compensation_option == 'only_vz':
        error = log_infidelity_with_vz
        if initial_guess is None:
            initial_guess = jnp.zeros([2 * n_qubit])
        if multi_init:
            initial_guess_multi = jnp.eye(2 * n_qubit)
    elif compensation_option == 'arbit_single':
        error = log_infidelity_with_vsq
        if initial_guess is None:
            initial_guess = jnp.zeros([2 * n_qubit, 3])
        if multi_init:
            # Setup several starting points per qubit (others zero)
            # 2x2x2 combination of three angles (duplicated removed) with randomness
            initial_guess_1q = np.array([
                [0 + 0.4, 0 + 0.2, 0 + 0.65],
                [0.2, pi / 2 + 0.3, 0.0 + 0.45],
                [pi / 2 - 0.3, pi / 2 - 0.4, 0.0 + 0.35],
                [pi / 2 - 0.5, pi / 2 - 0.6, pi / 2 + 0.35],
            ])
            n_guess = initial_guess_1q.shape[0]
            initial_guess_multi = np.zeros((n_guess * n_qubit, 2 * n_qubit, 3))
            for ix_q in range(n_qubit):
                initial_guess_multi[ix_q * n_guess: ix_q * n_guess + n_guess, ix_q, :] = initial_guess_1q
            initial_guess_multi = jnp.asarray(initial_guess_multi)
    elif compensation_option == 'no_comp':
        return f(u_target, u_computed).real, u_computed

    # Create initial guess from simple_vz
    if opt_method is not None and opt_method.endswith("init2"):
        ar_angle_svz, u_init = simple_vz_compensation_angle(
            u_target, u_computed)
        # Convert to the apply_x format
        # single matrix element phase angle to sigmaz angle (/2)
        # single side to pre and post (/2)
        ar_angle = ar_angle_svz / 4
        if compensation_option == "only_vz":
            initial_guess = jnp.array(np.concatenate([ar_angle, ar_angle]))
        elif compensation_option == 'arbit_single':
            guess_side = np.concatenate(
                [np.zeros(ar_angle.shape),
                 np.zeros(ar_angle.shape), ar_angle])
            initial_guess = jnp.array(np.concatenate([guess_side, guess_side]))

    if opt_method == 'lbfgs' or opt_method == "lbfgs_init2":
        # call jaxopt differentiable LBFGS solver
        solver = jaxopt.LBFGS(error, **options)
    elif opt_method == 'gd' or opt_method == 'gd_init2':
        solver = jaxopt.GradientDescent(error, **options)
    elif opt_method == 'iswap_vz':
        u_optimal = compute_compensation_iswap_gate_vz(u_target, u_computed)
        return f(u_target, u_optimal).real, u_optimal
    elif opt_method == 'simple_vz':
        u_optimal = simple_vz_compensation(u_target, u_computed)
        return f(u_target, u_optimal).real, u_optimal
    elif opt_method is None:
        return f(u_target, u_computed).real, u_computed
    else:
        raise ValueError(
            'opt_method should be `gd`, `gd_init2`, `lbfgs`, `simple_vz` or `iswap_vz`.'
        )

    if multi_init and opt_method in ['lbfgs', 'gd']:
        list_res = [solver.run(initial_guess_single, u_target=u_target, u_computed=u_computed) for initial_guess_single
                    in initial_guess_multi]
        ix_min = jnp.argmin(jnp.asarray([x.state.value for x in list_res]))
        res = list_res[ix_min]
    else:
        res = solver.run(initial_guess, u_target=u_target, u_computed=u_computed)
    # print(initial_guess)
    # print(res.state)
    # print(res.params)
    if compensation_option == 'only_vz':
        u_optimal = apply_2nz(res.params, u_computed)
    else:
        u_optimal = apply_6nsq_axis(res.params, u_computed)

    if return_1q:
        return f(u_target, u_optimal).real, u_optimal, res.params
    else:
        return f(u_target, u_optimal).real, u_optimal


def u_to_pauli(u: ndarray) -> jnp.ndarray:
    """Converts unitary to weights of pauli errors.

    Args:
        u (ndarray): A unitary.
            We expect we already multiply it with the inverse of target unitary.

    Returns:
        jnp.ndarray: a 2D array
    """

    num_qubits = int(np.round(np.log2(u.shape[0])))

    def scan_func(carry, target_idx):
        pauli_mats_array = jnp.array(pauli_mats)
        u_pauli = supergrad.tensor(*pauli_mats_array[target_idx])
        pauli_weights = jnp.abs(jnp.trace(u @ u_pauli)) ** 2
        return carry, pauli_weights

    idxs = jnp.array(
        [idx for idx in itertools.product(range(4), repeat=num_qubits)])
    _, pauli_weights_array = jax.lax.scan(scan_func, 0, idxs)

    pauli_weights_tensor = jnp.reshape(pauli_weights_array, (4,) * num_qubits)

    return pauli_weights_tensor / 4 ** num_qubits


def cphase_angle_opt(u):
    """Optimizing the cphase angle.

    Args:
        u (array): a unitary
    """

    def loss(angles):
        sq_angle1, sq_angle2, c_angle = angles
        sq_phase1 = jnp.exp(1j * sq_angle1)
        sq_phase2 = jnp.exp(1j * sq_angle2)
        u_compensated = u @ jnp.diag(
            jnp.array([1, sq_phase1, sq_phase2, sq_phase1 * sq_phase2]))
        u_target = jnp.diag(jnp.array([1, 1, 1, jnp.exp(1j * c_angle)]))
        return 1 - compute_average_fidelity_with_leakage(
            u_target, u_compensated)

    res = scipy_minimize(loss, (0., 0., 0.))

    return res


def pauli_diagnose(target_u, sim_u, num_print=10):
    """Pauli error diagnose.

    Args:
        target_u (array): target unitary
        u (array): the unitary from time evolution
        num_print (int): The top num_print pauli error operators will be printed
    """

    def pauli_error_string(error_list):
        return ''.join([pauli_dict[i] for i in error_list])

    pauli_dict = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z'}

    err = sim_u.conj().T @ target_u
    pe = u_to_pauli(err)
    ind = np.unravel_index(np.argsort(pe, axis=None), pe.shape)
    ind_array = np.array(ind)
    for i in range(1, num_print + 1):
        print(pauli_error_string(ind_array[:, -i]), f'{pe[ind][-i]:10.4e}')

    return pe, ind_array


def pauli_diagnose_partition(target_u,
                             sim_u,
                             num_print=10,
                             part_lst=None,
                             default_color='white'):
    """Pauli error diagnose. Color the print result by how many partitions have errors

    Args:
        target_u (array): target unitary
        u (array): the unitary from time evolution
        num_print (int): The top num_print pauli error operators will be printed
        part_lst (list): list of list of int
            each nested list contains all qubit index in special partition
        default_color (string): The default color of error string
    """

    def pauli_error_string(error_list, prob_str):
        count = 0
        for p in part_lst:
            if np.sum(error_list[p]) >= 1:
                count += 1
        err_string = ''.join([pauli_dict[i] for i in error_list])
        err_string = err_string + '   ' + prob_str
        if count < 2:
            err_string = '[' + default_color + ']' + err_string + '[' + default_color + ']'
        if count == 2:
            err_string = '[bright_magenta]' + err_string + '[/bright_magenta]'
        if count == 3:
            err_string = '[bright_red]' + err_string + '[/bright_red]'

        return err_string

    if part_lst is None:
        print('part_lst is None!')
        return pauli_diagnose(target_u, sim_u, num_print)
    pauli_dict = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z'}

    err = sim_u.conj().T @ target_u
    pe = u_to_pauli(err)
    ind = np.unravel_index(np.argsort(pe, axis=None), pe.shape)
    ind_array = np.array(ind)
    for i in range(1, num_print + 1):
        rprint(pauli_error_string(ind_array[:, -i], f'{pe[ind][-i]:10.4e}'))

    return pe, ind_array


def compare_unitary(u1, u2, norm_ord=None):
    """Compare two unitary.

    Args:
        u1 (array): unitary
        u2 (array): unitary
        norm_ord ({non-zero int, inf, -inf, 'fro', 'nuc'}, optional):
            Order of the norm. inf means numpy's `inf` object.
            The default is None.
    Return:
        The maximum value of |u1 - u2|

    """
    # unify matrix phase depend on the maximum of first row
    target_idx = np.argmax(np.abs(u1[0]))
    angle_1 = np.angle(u1[0, target_idx])
    angle_2 = np.angle(u2[0, target_idx])
    # Apply the angle to basis
    u1 = np.exp(-1j * angle_1) * u1
    u2 = np.exp(-1j * angle_2) * u2
    d_unitary = np.abs(u1 - u2)
    # normalize
    d_unitary /= np.linalg.norm(d_unitary, ord=norm_ord)

    return np.max(d_unitary)


def plot_pauli_diagnose(target_u, sim_u, num_print=30):
    """Plot the pauli error diagnose results.

    Args:
        target_u (array): target unitary
        u (array): the unitary from time evolution
        num_print (int): The top num_print pauli error operators will be printed
    """

    def pauli_error_string(error_list):
        return ''.join([pauli_dict[i] for i in error_list])

    pauli_dict = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z'}

    pe, ind_array = pauli_diagnose(target_u, sim_u, num_print)
    ind = np.unravel_index(np.argsort(pe, axis=None), pe.shape)
    pe_lst = [error for error in reversed(pe[ind])]
    ind_lst = [pauli_error_string(idx) for idx in reversed(ind_array.T)]

    plt.scatter(ind_lst[:num_print],
                pe_lst[:num_print],
                label='Standard circuit')
    plt.yscale('log')
    plt.xlabel('Pauli errors')
    plt.ylabel('Coefficients')
    plt.xticks(rotation=90, family='monospace')
    plt.legend()


def plot_pauli_diagnose_partition(target_u,
                                  sim_u,
                                  num_print=30,
                                  part_lst=[[0, 1], [2, 3], [4, 5]],
                                  marker='.',
                                  default_color='black'):
    """Plot the pauli error diagnose results.

    Args:
        target_u (array): target unitary
        u (array): the unitary from time evolution
        num_print (int): The top num_print pauli error operators will be printed
        marker: marker for scatter
        default_color: default color if the index is not assigned in `color_dict`
    """

    def compute_count(error_list):
        count = 0
        if part_lst is not None:
            for p in part_lst:
                if np.sum(error_list[p]) >= 1:
                    count += 1
        return count

    def pauli_error_string(error_list):

        err_string = []
        # construct error list pair
        for p in part_lst:
            str_part = [pauli_dict[key] for key in error_list[p]]
            err_string.append(''.join(str_part))

        return ''.join(err_string)

    pauli_dict = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z'}
    color_dict = {0: 'black', 1: 'teal', 2: 'orange', 3: 'purple', 4: 'red'}
    label_dict = {
        i: r'weight $w(\vec{b}) = $' + f'{i}'
        for i, _ in enumerate(part_lst + [None])
    }

    err = sim_u.conj().T @ target_u
    pe = u_to_pauli(err)
    ind = np.unravel_index(np.argsort(pe, axis=None), pe.shape)
    ind_array = np.array(ind)
    # sort pauli error list
    pe = pe[ind]
    label_flag = np.zeros(len(label_dict))
    for i in range(1, num_print + 1):
        count = compute_count(ind_array[:, -i])
        if count == 0:
            continue  # ignore no error case
        if not label_flag[count]:
            plt.scatter(
                i,  # pauli_error_string(ind_array[:, -i]),
                pe[-i],
                label=label_dict[count],
                color=color_dict.get(count, default_color),
                marker=marker,
                s=60)
            label_flag[count] = 1
        else:
            plt.scatter(
                i,  # pauli_error_string(ind_array[:, -i]),
                pe[-i],
                color=color_dict.get(count, default_color),
                marker=marker,
                s=60)

    plt.yscale('log')
    plt.xlabel('Pauli errors $E$')
    plt.ylabel('Error rates')
    plt.xticks([])
    # plt.legend()
