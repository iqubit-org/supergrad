import numpy as np
import jax.numpy as jnp

from supergrad.utils.fidelity import u_to_pauli


def single_qubit_gate_pauli_error(target_u,
                                  sim_u,
                                  depolarizing_rate=0.,
                                  target_count=None):
    """Diagnose the single qubit gate pauil error.

    Args:
        target_u: target unitary matrix
        sim_u: the unitary matrix from simulation
        depolarizing_rate (float): the depolarizing error rate (e.g. t_gate * r)

    Returns:
        jnp.ndarray: The cumulative weights of pauli errors depend on the number
            of gate error.
    """
    pe = u_to_pauli(jnp.conj(sim_u).T @ target_u)
    ind = np.unravel_index(np.arange(pe.size), pe.shape)
    pe = pe[ind]  # unravel index
    # check the number of gate error
    count = np.count_nonzero(np.array(ind), axis=0)
    # debug
    if target_count is not None:
        # debug port
        return pe[count == target_count], jnp.array(ind)[:,
                                                         count == target_count]
    crosstalk_error = []
    # weight-1 error
    crosstalk_error.append(jnp.sum(pe[count == 1]) / 6 + depolarizing_rate)
    # weight-2 error
    temp_ind = np.array(ind)[:, count == 2].T
    # pickup joint 2-gates error
    horizon = (np.sum(temp_ind[:, 2:], axis=1) == 0)
    horizon += (np.sum(temp_ind[:, (0, 1, 4, 5)], axis=1) == 0)
    horizon += (np.sum(temp_ind[:, :4], axis=1) == 0)
    vertical = (np.sum(temp_ind[:, (1, 3, 4, 5)], axis=1) == 0)
    vertical += (np.sum(temp_ind[:, (0, 2, 4, 5)], axis=1) == 0)
    vertical += (np.sum(temp_ind[:, (0, 1, 3, 5)], axis=1) == 0)
    vertical += (np.sum(temp_ind[:, (0, 1, 2, 4)], axis=1) == 0)
    idx = np.argwhere(horizon + vertical)
    error_list = pe[count == 2][idx]
    crosstalk_error.append(jnp.sum(error_list) / 7)
    # weight-3 error
    temp_ind = np.array(ind)[:, count == 3].T
    # pickup L-Type simultaneous 3-gates error
    upper_l = (np.sum(temp_ind[:, 4:], axis=1) == 0)
    lower_l = (np.sum(temp_ind[:, :2], axis=1) == 0)
    # pickup I-Type simultaneous 3-gates error
    left_i = (np.sum(temp_ind[:, (1, 3, 5)], axis=1) == 0)
    right_i = (np.sum(temp_ind[:, (0, 2, 4)], axis=1) == 0)
    idx = np.argwhere(upper_l + lower_l + left_i + right_i)
    error_list = pe[count == 3][idx]
    crosstalk_error.append(jnp.sum(error_list) / 10)

    return jnp.array(crosstalk_error)


def two_qubit_gate_pauli_error(target_u, sim_u, part_lst, depolarizing_rate=0.):
    """Diagnose the single qubit gate pauil error.

    Args:
        target_u: target unitary matrix
        sim_u: the unitary matrix from simulation
        part_list: list of tow qubit gate pair (renormalization part)
        depolarizing_rate (float): the depolarizing error rate (e.g. t_gate * r)

    Returns:
        jnp.ndarray: The cumulative weights of pauli errors depend on the number
            of gate error.
    """
    pe = u_to_pauli(jnp.conj(sim_u).T @ target_u)
    ind = np.unravel_index(np.arange(pe.size), pe.shape)
    pe = pe[ind]  # unravel index
    ind_array = np.array(ind)

    # Group the pauli error in two qubit pairs
    group_ind = []
    for part in part_lst:
        count = np.count_nonzero(ind_array[part, :], axis=0)
        new_ind = np.zeros_like(count)
        new_ind[np.nonzero(count)] = 1
        group_ind.append(new_ind)
    group_ind = np.array(group_ind)

    # check the number of gate error
    count = np.count_nonzero(group_ind, axis=0)
    crosstalk_error = []
    # weight-1 error
    crosstalk_error.append(jnp.sum(pe[count == 1]) / 3 + depolarizing_rate)
    # weight-2 error
    temp_ind = np.array(group_ind)[:, count == 2].T
    # pickup joint 2-gates error
    upper = (temp_ind[:, 2] == 0)
    lower = (temp_ind[:, 0] == 0)
    idx = np.argwhere(upper + lower)
    error_list = pe[count == 2][idx]
    crosstalk_error.append(jnp.sum(error_list) / 2)
    # weight-3 error
    crosstalk_error.append(jnp.sum(pe[count == 3]))

    return jnp.array(crosstalk_error)
