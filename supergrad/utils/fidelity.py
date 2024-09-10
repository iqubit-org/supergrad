from typing import Tuple
import itertools
import numpy as np
import jax
import jax.numpy as jnp
import jaxopt
import haiku as hk
from numpy import ndarray
from rich import print as rprint
import matplotlib.pyplot as plt

import supergrad
from supergrad.scgraph.graph import (parse_pre_comp_name, parse_post_comp_name)
from .optimize import scipy_minimize

pauli_mats = []
pauli_mats.append(np.eye(2))
pauli_mats.append(np.array([[0., 1.], [1., 0.]]))
pauli_mats.append(np.array([[0., 1j], [-1j, 0.]]))
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
           abs(jnp.trace(jnp.conj(u_target.T) @ u_computed))**2) / n / (n + 1)
    return jnp.abs(res)


def compute_state_fidelity(psi_target, psi_computed):
    """
    Compute state fidelity

    Args:
        psi_target (array): Target state
        psi_computed (array): Test state
    """
    return jnp.abs(jnp.vdot(psi_target, psi_computed))**2


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
    pauli_0 = np.ones(2)
    pauli_3 = np.array([1, -1])
    pre_unitaries = [
        jnp.cos(pre_params) * pauli_0 + 1j * jnp.sin(pre_params) * pauli_3
        for pre_params in ps[:n_qubit]
    ]
    post_unitaries = [
        jnp.cos(post_params) * pauli_0 + 1j * jnp.sin(post_params) * pauli_3
        for post_params in ps[n_qubit:]
    ]
    u2 = supergrad.tensor(*pre_unitaries)
    u3 = supergrad.tensor(*post_unitaries)
    return (u3 * u.T).T * u2


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


def apply_nz(params, psi):
    """
    Apply n Z-gate after finial state.

    Args:
        params (array): n parameters for single qubit rotation
        psi (array): state
    """
    post_diag = [
        jnp.array([
            jnp.cos(post_params) + 1j * jnp.sin(post_params),
            jnp.cos(post_params) - 1j * jnp.sin(post_params)
        ]) for post_params in params
    ]

    ud3 = supergrad.tensor(*post_diag)
    return ud3[:, jnp.newaxis] * psi


def apply_3nsq_axis(params, psi):
    """
    Apply 3n single qubit rotation after final state.

    Args:
        params (array): 3n parameters for single qubit rotation
        psi (array): state
    """
    post_unitaries = [
        jnp.cos(post_params[1]) * pauli_mats[0] + 1j * jnp.sin(post_params[1]) *
        (jnp.cos(post_params[0]) * pauli_mats[3] + jnp.sin(post_params[0]) *
         (jnp.cos(post_params[2]) * pauli_mats[1] +
          jnp.sin(post_params[2]) * pauli_mats[2])) for post_params in params
    ]

    u3 = supergrad.tensor(*post_unitaries)
    return u3 @ psi


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
        single_q_phase_lst.append(d[0] / d[2**(n - 1 - i)])
        single_q_phase_lst[-1] = single_q_phase_lst[-1] / jnp.abs(
            single_q_phase_lst[-1])

    vz_diag = jnp.array([
        1,
    ])
    for i in range(n):
        vz_diag = jnp.kron(vz_diag, jnp.array([1, single_q_phase_lst[i]**(-1)]))

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
    u_computed = evo.eigen_basis(
        hk.data_structures.merge(evo.all_params, params), transform_matrix)

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
    u_computed = evo.eigen_basis(
        hk.data_structures.merge(evo.all_params, params), transform_matrix)

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
    # print(initial_guess)
    # print(res.state)
    # print(res.params)
    if compensation_option == 'only_vz':
        psi_compensated = apply_nz(res.params, psi_computed)
    else:
        psi_compensated = apply_3nsq_axis(res.params, psi_computed)

    return compute_state_fidelity(psi_target, psi_compensated), psi_compensated


def compute_fidelity_with_1q_rotation_axis(u_target,
                                           u_computed,
                                           opt_method: str = 'gd',
                                           compensation_option='only_vz',
                                           options: dict = None):
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
    """
    assert compensation_option in ['no_comp', 'only_vz', 'arbit_single']
    n_qubit = round(np.log2(u_target.shape[0]))
    if options is None:
        options = {}
    f = compute_average_fidelity_with_leakage
    # initialize single qubit compensation
    if compensation_option == 'only_vz':
        error = log_infidelity_with_vz
        initial_guess = jnp.zeros([2 * n_qubit])
    elif compensation_option == 'arbit_single':
        error = log_infidelity_with_vsq
        initial_guess = jnp.zeros([2 * n_qubit, 3])
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

    res = solver.run(initial_guess, u_target=u_target, u_computed=u_computed)
    # print(initial_guess)
    # print(res.state)
    # print(res.params)
    if compensation_option == 'only_vz':
        u_optimal = apply_2nz(res.params, u_computed)
    else:
        u_optimal = apply_6nsq_axis(res.params, u_computed)

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
        pauli_weights = jnp.abs(jnp.trace(u @ u_pauli))**2
        return carry, pauli_weights

    idxs = jnp.array(
        [idx for idx in itertools.product(range(4), repeat=num_qubits)])
    _, pauli_weights_array = jax.lax.scan(scan_func, 0, idxs)

    pauli_weights_tensor = jnp.reshape(pauli_weights_array, (4,) * num_qubits)

    return pauli_weights_tensor / 4**num_qubits


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
