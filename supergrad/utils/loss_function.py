from functools import reduce
import numpy as np
import jax.numpy as jnp
from jax.scipy.stats import cauchy
import haiku as hk
import matplotlib.pyplot as plt

from supergrad.quantum_system import (Fluxonium, Resonator, InteractingSystem,
                                      parse_interaction)


def wrapper_yaqcs_data(data):
    """A wrapper for data from yaqcs experiment.

    Args:
        data: `DataSet` from yaqcs
    """
    # Return to base unit
    data.remove_units()
    # Format data
    if not data.is_adaptive:
        regular_grid = [True] * len(data.axis_names)
    else:
        regular_grid = data['regular_grid']
    # Get the result of experiment in complex number
    if not all(regular_grid):
        data = data.to_nested_list()
        assert 'mag' in data and 'phase' in data
        exp_array = []
        for mag, phase in zip(data['mag'], data['phase']):
            exp_array.append(10**(mag / 10) * np.exp(1j * phase))
        grid = []
        for ii, regular in enumerate(regular_grid):
            if regular:
                grid.append(data[data.axis_names[ii]])
            else:
                # unify non-regular grid
                name: str = data.axis_names[ii]
                nrgrid = data[name]
                union_nrgrid = reduce(np.union1d, nrgrid)
                grid.append(union_nrgrid)
                # full the union grid
                new_exp_array = []
                for jj, exp in enumerate(exp_array):
                    mask = np.isin(union_nrgrid, nrgrid[jj])
                    template = np.full(union_nrgrid.shape,
                                       np.nan,
                                       dtype=complex)
                    template[mask] = exp
                    new_exp_array.append(template)
        exp_array = np.stack(new_exp_array)

    else:
        data.compute_complex_from_mag_phase()
        exp_array = data['complex']
        grid = []
        for name in data.axis_names:
            grid.append(data[name])

    return tuple(grid + [
        exp_array,
    ])


def wrapper_phi_freq(bias_list, freq_list, periodic=True):
    """A wrapper that format `bias_list` and `freq_list`.

    Args:
        bias_list: the bias to adjust phi external in Fluxonium.
        freq_list: the frequency list.
        periodic: True for using periodic boundary condition of phi in simulation.
    """
    bias_list = jnp.asarray(bias_list)
    freq_list = jnp.asarray(freq_list) / 1e9  # unit in GHz
    # convert bias to phi_ext/phi_0
    bias_phi0 = hk.get_parameter('bias_phi0', [], init=jnp.zeros)
    bias_phipi = hk.get_parameter('bias_phipi', [], init=jnp.zeros)

    phi_list = (bias_list - bias_phi0) / (bias_phipi - bias_phi0) / 2
    if periodic:
        # use periodic boundary condition of phi
        mod_phi = jnp.modf(2 * jnp.modf(phi_list)[0])
        phi_list = (mod_phi[0] - mod_phi[1]) / 2

    return phi_list, freq_list


def truncated_mse(a, b, trancated_percentile: int = 95):
    """Truncated mean squared error"""

    a = jnp.asarray(a)
    b = jnp.asarray(b)
    mse = (a - b)**2
    # truncated differences
    mse = jnp.minimum(mse, jnp.percentile(mse, trancated_percentile))
    return jnp.sum(mse)


def _cal_spectrum_vs_phi(phi, phi_max=5 * np.pi, num_basis=50, **kwargs):
    # initial quantum system
    fq = Fluxonium(phiext=0.0, phi_max=phi_max, num_basis=num_basis, **kwargs)

    # calculate spectrum
    def calc_spectrum_vs_phi(phi):
        fq.phiext = phi * 2 * np.pi
        return fq.eigenenergies()

    # vmap it
    if len(phi) > 1:
        return hk.vmap(calc_spectrum_vs_phi, split_rng=False)(phi)
    else:
        return calc_spectrum_vs_phi(phi)


def _cal_transmission_vs_phi(phi,
                             num_photon,
                             eval_count=3,
                             phi_max=5 * np.pi,
                             num_basis=50,
                             truncated_dim=12):

    def calc_transmission_vs_phi(phi):

        # Physical components
        fqubit = Fluxonium(phiext=0.0,
                           phi_max=phi_max,
                           num_basis=num_basis,
                           truncated_dim=truncated_dim)
        cavity = Resonator(truncated_dim=12)
        subsystem_list = [fqubit, cavity]
        # specifying interactions between subsystems
        # There are some operators(`Callable` function`) in the components.
        inter_term = parse_interaction(op0=fqubit.n_operator,
                                       op1=cavity.annihilation_operator,
                                       add_hc=True)
        # the option `add_hc=True` adds the hermitian conjugate to the Hamiltonian
        # now instance InteractingSystem to handle components and InteractionTerms
        # one can set module name here
        hilbertspace = InteractingSystem(subsystem_list,
                                         inter_term,
                                         name='fluxonium_cavity')
        # override phiext
        hilbertspace['fluxonium'].phiext = phi * 2 * np.pi
        energy_nd = hilbertspace.compute_energy_map()
        return energy_nd[:eval_count, num_photon] - energy_nd[:eval_count,
                                                              num_photon - 1]

    # vmap it
    if len(phi) > 1:
        return hk.vmap(calc_transmission_vs_phi, split_rng=False)(phi)
    else:
        return calc_transmission_vs_phi(phi)


@hk.without_apply_rng
@hk.transform
def loss_fit_spectrum(bias_list, freq_list, end_level, periodic=True):
    """Fits parameters from fluxonium spectrum vs bias data.

    Args:
        bias_list (list[~numpy.ndarray]): Bias data.
        freq_list (list[~numpy.ndarray]): Spectrum data.
        end_level (int): Ending level for spectrum.
        periodic (bool): Using periodic boundary condition of phi
            when the target phi is beyond the phi basis range.
    """
    phi_list, freq_list = wrapper_phi_freq(bias_list, freq_list, periodic)

    spec_out = _cal_spectrum_vs_phi(phi_list)
    sim_freq_list = spec_out[:, end_level] - spec_out[:, 0]

    return truncated_mse(sim_freq_list, freq_list, 100)


@hk.without_apply_rng
@hk.transform
def loss_kl_div(xs, ys, exp_array, plot=False):
    """Loss function for fit qubit spectra by the Kullback-Leibler divergence.

    Args:
        data: dateset from experiment.
    """
    # Normalize
    data_exp = jnp.abs(exp_array -
                       jnp.nanmean(exp_array, axis=1, keepdims=True))
    data_exp = data_exp / jnp.nansum(data_exp)
    data_exp = jnp.nan_to_num(data_exp, nan=1e-4)
    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].pcolormesh(xs, ys, data_exp.T)
        axs[0].set_title(r'P(x, $\varepsilon$)')
        axs[0].set_xlabel(r'x')
        axs[0].set_ylabel(r'$\varepsilon [\text{GHz}] $')

    # convert bias to phi_ext/phi_0
    phi_list, ys = wrapper_phi_freq(xs, ys)

    spec_out = _cal_spectrum_vs_phi(phi_list)
    peak_list = spec_out[:, 1] - spec_out[:, 0]

    # set a bandwidth
    bandwidth = hk.get_parameter('bandwidth', [], init=jnp.zeros)
    noise = hk.get_parameter('noise', [], init=jnp.zeros)
    sim_data = []
    for peak in peak_list:
        sim_data.append(cauchy.pdf(ys, loc=peak, scale=bandwidth))
    sim_data = jnp.asarray(sim_data)

    # add noise
    sim_data += noise
    sim_data = sim_data / jnp.sum(sim_data)

    # plot
    if plot:
        axs[1].pcolormesh(xs, ys, sim_data.T)
        axs[1].set_title(r'Q(x, $\varepsilon$)')
        axs[1].set_xlabel(r'x')
        axs[1].set_ylabel(r'$\varepsilon [\text{GHz}] $')
        plt.show()
    # calculate kl div
    return jnp.nansum(data_exp * jnp.log(data_exp / sim_data))


@hk.without_apply_rng
@hk.transform
def loss_kl_div_res_transmission(xs, ys, exp_array, plot=False):
    """Loss function for fit qubit spectra by the Kullback-Leibler divergence.

    Args:
        data: dateset from experiment.
    """
    # Normalize
    exp_array = jnp.abs(10 * jnp.log10(exp_array))  # unit in db
    data_exp = jnp.nan_to_num(exp_array, nan=jnp.nanmin(exp_array))
    data_exp = jnp.abs(data_exp - jnp.nanmean(data_exp, axis=1, keepdims=True))
    data_exp = data_exp / jnp.nansum(data_exp)
    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].pcolormesh(xs, ys, data_exp.T)
        axs[0].set_title(r'$\mathbf{P}(\varepsilon,b)$')

    # convert bias to phi_ext/phi_0
    phi_list, ys = wrapper_phi_freq(xs, ys)

    spec_out = _cal_transmission_vs_phi(phi_list, 1)
    peak_list = spec_out[:, 0]

    # set a bandwidth
    bandwidth = hk.get_parameter('bandwidth', [], init=jnp.zeros)
    noise = hk.get_parameter('noise', [], init=jnp.zeros)
    sim_data = []
    for peak in peak_list:
        sim_data.append(cauchy.pdf(ys, loc=peak, scale=bandwidth))
    sim_data = jnp.asarray(sim_data)

    # add noise
    sim_data += noise
    sim_data = sim_data / jnp.sum(sim_data)

    # plot
    if plot:
        axs[1].pcolormesh(xs, ys, sim_data.T)
        axs[1].set_title(r'$\mathbf{Q}(\varepsilon,\varphi)$')
        plt.show()
    # calculate kl div
    return jnp.nansum(data_exp * jnp.log(data_exp / sim_data))


@hk.without_apply_rng
@hk.transform
def loss_fit_transmission(bias_list,
                          freq_list,
                          qubit_level,
                          num_photon,
                          periodic=True):
    """Loss function for fit cavity-qubit transmission spectrum.

    Args:
        bias_list (list[~numpy.ndarray]): Bias data.
        freq_list (list[~numpy.ndarray]): Spectrum data.
        qubit_level (int): Qubit energy level for transmission spectrum.
        num_photon (int): The number of photon in cavity.
        periodic (bool): Using periodic boundary condition of phi
            when the target phi is beyond the phi basis range.
    """
    phi_list, freq_list = wrapper_phi_freq(bias_list, freq_list, periodic)

    spec_out = _cal_transmission_vs_phi(phi_list, num_photon)
    sim_freq_list = spec_out[:, qubit_level]
    # use mean square error with l2-norm
    # norm = jnp.linalg.norm(freq_list - jnp.mean(freq_list))

    return truncated_mse(sim_freq_list, freq_list) * 1e4
