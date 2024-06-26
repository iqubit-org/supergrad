import itertools
import numpy as np
import jax.numpy as jnp
import haiku as hk
import matplotlib.pyplot as plt

from supergrad.quantum_system.artificial_atom import Fluxonium

from qutip import Qobj
from qutip.qobj import ptrace
from qutip.entropy import entropy_vn


def state_entropy_vn(psi, sel):

    num_qubit = int(np.log2(psi.size))
    psi_qobj = Qobj(np.array(psi), dims=[[2] * num_qubit, [1] * num_qubit])
    dm = psi_qobj * psi_qobj.dag()
    partial_dm = ptrace(dm, sel)

    return entropy_vn(partial_dm, 2)


def inverse_participation_ratio(psi):
    """The inverse participation ratio discussed in
    (Transmon platform for quantum computing challenged by chaotic fluctuations) """

    return jnp.sum(jnp.abs(psi)**4)


def walsh_transform(energy_nd):
    """The Walsh-transform analysis.

    Return:
        coefficients, bitstrings
    """

    # truncation bosonic occupations
    trunc_slice = tuple(slice(0, 2) for _ in range(energy_nd.ndim))
    trunc_energy_nd = energy_nd[trunc_slice]
    # generate bitstring label
    bitstrings = [
        bit for bit in itertools.product((0, 1), repeat=energy_nd.ndim)
    ]
    # calculate the coefficients of the tau-Hamiltonian
    coeff_tau = []
    for bitstring in bitstrings:
        temp_sum = 0
        # Walsh-Hadamard transform
        for transform_b in bitstrings:
            temp_sum += (-1)**(np.array(bitstring) @ np.array(transform_b)
                               ) * trunc_energy_nd[transform_b]
        coeff_tau.append(temp_sum / 2**energy_nd.ndim)
    coeff_tau = np.abs(np.array(coeff_tau))
    # sort the coefficient
    sort_idx = np.argsort(-coeff_tau)  # in descending order
    return coeff_tau[sort_idx], np.array(bitstrings)[sort_idx]


def plot_walsh_transform(coeff, bitstring):
    """Plot the Walsh-transform coefficients."""

    # coeff, bitstring = walsh_transform(energy_nd)
    # change unit of coefficient
    coeff = coeff.copy() * 1e6 / 2 / np.pi
    # Sort weight of bitstring
    weight_2_coeff_lst = []
    weight_2_idx_lst = []
    weight_3_coeff_lst = []
    weight_3_idx_lst = []
    weight_other_coeff_lst = []
    weight_other_idx_lst = []
    for i in range(len(coeff)):
        if int(bitstring[i].sum()) == 2:
            weight_2_coeff_lst.append(coeff[i])
            weight_2_idx_lst.append(i)
        elif int(bitstring[i].sum()) == 3:
            weight_3_coeff_lst.append(coeff[i])
            weight_3_idx_lst.append(i)
        elif int(bitstring[i].sum()) > 3:
            weight_other_coeff_lst.append(coeff[i])
            weight_other_idx_lst.append(i)

    plt.scatter(weight_2_idx_lst,
                weight_2_coeff_lst,
                label=r'weight $w(\vec{b}) = 2$', s=60)
    plt.scatter(weight_3_idx_lst,
                weight_3_coeff_lst,
                label=r'weight $w(\vec{b}) = 3$', s=60)
    plt.scatter(weight_other_idx_lst,
                weight_other_coeff_lst,
                label=r'weight $w(\vec{b}) > 3$', s=60)
    plt.yscale('log')
    plt.xlabel(r'Bitstring $\vec{b}$')
    plt.xticks([])
    plt.yticks()
    plt.ylabel(r'$|c_\vec{b}|$ [kHz]')
    plt.legend()


def plot_subsystem_eigenenergies(params):
    """
    Plot eigenenergies for individual subsystems.
    TODO: implement for transmon, etc
    """

    @hk.without_apply_rng
    @hk.transform
    def fluxonium_eigenenergies():
        fluxonium = Fluxonium(phiext=0.5 * 2 * jnp.pi,
                              phi_max=5 * np.pi,
                              truncated_dim=10,
                              name='fluxonium')

        return fluxonium.eigenenergies()

    for subsys in params:
        rename_dict = {}
        rename_dict['fluxonium'] = params[subsys]
        ee = fluxonium_eigenenergies.apply(rename_dict)
        ee = ee - ee[0]
        plt.scatter(np.arange(3), ee[1:4], label=subsys)
        plt.legend()
    plt.show()


def print_pulse_frequencies(params, dict_key='omega_d', divide_2pi=True):

    for k in params:
        if dict_key in params[k]:
            if divide_2pi:
                print(k, float(params[k]['omega_d']) / 2 / np.pi)
            else:
                print(k, float(params[k]['omega_d']))


if __name__ == '__main__':
    params = {
        'fluxonium0': {
            'ec': jnp.array(1.),
            'ej': jnp.array(4.),
            'el': jnp.array(1.)
        },
        'fluxonium1': {
            'ec': jnp.array(1.),
            'ej': jnp.array(4.),
            'el': jnp.array(0.9)
        },
        'fluxonium2': {
            'ec': jnp.array(1.),
            'ej': jnp.array(4.),
            'el': jnp.array(1.1)
        },
        'fluxonium3': {
            'ec': jnp.array(1.),
            'ej': jnp.array(4.),
            'el': jnp.array(1.05)
        },
        'fluxonium4': {
            'ec': jnp.array(1.),
            'ej': jnp.array(4.),
            'el': jnp.array(0.95)
        }
    }
    plot_subsystem_eigenenergies(params)
