import os
import numpy as np
import jax
import haiku as hk

import supergrad
from supergrad.quantum_system import Fluxonium, Transmon

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)


def test_transmon_phase_basis():
    # Get cached data
    data_0 = np.load('data/data_create_transmon.npy')

    # construct model
    class ExploreTransmon(supergrad.Helper):

        def _init_quantum_system(self):
            self.tmon = Transmon(basis='phase',
                                 phiext=0.,
                                 truncated_dim=6,
                                 drive_for_state_phase='charge',
                                 n_max=31)

        def energy_spectrum(self):
            return self.tmon.eigenenergies()

    tmon = ExploreTransmon()
    # create params template
    ej_list = np.linspace(28, 32, 3)
    ec_list = np.linspace(1.0, 1.4, 3)
    ng_list = np.linspace(0.2, 0.4, 3)
    params = {'transmon': {'ec': ec_list, 'ej': ej_list, 'ng': ng_list}}

    multi_spec = jax.vmap(
        jax.vmap(
            jax.vmap(tmon.energy_spectrum,
                     in_axes=({
                         'transmon': {
                             'ec': None,
                             'ej': None,
                             'ng': 0
                         },
                     },)), ({
                         'transmon': {
                             'ec': 0,
                             'ej': None,
                             'ng': None
                         },
                     },)), ({
                         'transmon': {
                             'ec': None,
                             'ej': 0,
                             'ng': None
                         },
                     },))
    data_1 = multi_spec(params)

    assert np.allclose(data_0.flatten(), data_1.flatten())


def test_tunable_transmon_phase_basis():
    # Get cached data
    data_0 = np.load('data/data_create_tunable_transmon.npy')

    # construct model
    class ExploreTransmon(supergrad.Helper):

        def _init_quantum_system(self):
            self.tmon = Transmon(basis='phase',
                                 truncated_dim=6,
                                 d=hk.get_parameter('d', [], init=np.zeros),
                                 drive_for_state_phase='charge',
                                 n_max=40)

        def energy_spectrum(self):
            return self.tmon.eigenenergies()

    tmon = ExploreTransmon()
    # create params template
    ej_list = np.linspace(38, 42, 3)
    ec_list = np.linspace(0.1, 0.3, 3)
    d_list = np.linspace(0.05, 0.2, 3)
    flux_list = np.linspace(0.1, 0.5, 3)
    ng_list = np.linspace(0.2, 0.4, 3)
    params = {
        '~': {
            'd': d_list
        },
        'transmon': {
            'ec': ec_list,
            'ej': ej_list,
            'ng': ng_list,
            'phiext': flux_list * 2 * np.pi
        }
    }

    multi_spec = jax.vmap(
        jax.vmap(
            jax.vmap(
                jax.vmap(
                    jax.vmap(tmon.energy_spectrum,
                             in_axes=({
                                 '~': {
                                     'd': None,
                                 },
                                 'transmon': {
                                     'ec': None,
                                     'ej': None,
                                     'ng': 0,
                                     'phiext': None
                                 }
                             },)), ({
                                 '~': {
                                     'd': None,
                                 },
                                 'transmon': {
                                     'ec': None,
                                     'ej': None,
                                     'ng': None,
                                     'phiext': 0
                                 }
                             },)), ({
                                 '~': {
                                     'd': 0,
                                 },
                                 'transmon': {
                                     'ec': None,
                                     'ej': None,
                                     'ng': None,
                                     'phiext': None
                                 }
                             },)), ({
                                 '~': {
                                     'd': None,
                                 },
                                 'transmon': {
                                     'ec': 0,
                                     'ej': None,
                                     'ng': None,
                                     'phiext': None
                                 }
                             },)), ({
                                 '~': {
                                     'd': None,
                                 },
                                 'transmon': {
                                     'ec': None,
                                     'ej': 0,
                                     'ng': None,
                                     'phiext': None
                                 }
                             },))
    data_1 = multi_spec(params)

    assert np.allclose(data_0.flatten(), data_1.flatten())


def test_fluxonium_phase_basis():
    # Get cached data
    data_0 = np.load('data/data_create_fluxonium.npy')

    # construct model
    class ExploreFluxonium(supergrad.Helper):

        def _init_quantum_system(self):
            self.fmon = Fluxonium(basis='phase',
                                  truncated_dim=6,
                                  drive_for_state_phase='charge',
                                  put_phiext_on_inductor=False,
                                  phi_max=5 * np.pi)

        def energy_spectrum(self):
            return self.fmon.eigenenergies()

    tmon = ExploreFluxonium()
    # create params template
    ej_list = np.linspace(7, 10, 3)
    ec_list = np.linspace(2.2, 2.7, 3)
    el_list = np.linspace(0.3, 0.7, 3)
    flux_list = np.linspace(0.1, 0.5, 3)
    params = {
        'fluxonium': {
            'ec': ec_list,
            'ej': ej_list,
            'el': el_list,
            'phiext': flux_list * 2 * np.pi
        }
    }

    multi_spec = jax.vmap(
        jax.vmap(
            jax.vmap(
                jax.vmap(tmon.energy_spectrum,
                         in_axes=({
                             'fluxonium': {
                                 'ec': None,
                                 'ej': None,
                                 'el': None,
                                 'phiext': 0,
                             }
                         },)), ({
                             'fluxonium': {
                                 'ec': None,
                                 'ej': None,
                                 'el': 0,
                                 'phiext': None
                             }
                         },)), ({
                             'fluxonium': {
                                 'ec': 0,
                                 'ej': None,
                                 'el': None,
                                 'phiext': None
                             }
                         },)), ({
                             'fluxonium': {
                                 'ec': None,
                                 'ej': 0,
                                 'el': None,
                                 'phiext': None
                             }
                         },))
    data_1 = multi_spec(params)
    assert np.allclose(data_0.flatten(), data_1.flatten())


def test_fluxonium_phase_basis_puton_inductor():
    # Get cached data
    data_0 = np.load('data/data_create_fluxonium.npy')

    # construct model
    class ExploreFluxonium(supergrad.Helper):

        def _init_quantum_system(self):
            self.fmon = Fluxonium(basis='phase',
                                  truncated_dim=6,
                                  drive_for_state_phase='charge',
                                  put_phiext_on_inductor=True,
                                  phi_max=5 * np.pi)

        def energy_spectrum(self):
            return self.fmon.eigenenergies()

    tmon = ExploreFluxonium()
    # create params template
    ej_list = np.linspace(7, 10, 3)
    ec_list = np.linspace(2.2, 2.7, 3)
    el_list = np.linspace(0.3, 0.7, 3)
    flux_list = np.linspace(0.1, 0.5, 3)
    params = {
        'fluxonium': {
            'ec': ec_list,
            'ej': ej_list,
            'el': el_list,
            'phiext': flux_list * 2 * np.pi
        }
    }

    multi_spec = jax.vmap(
        jax.vmap(
            jax.vmap(
                jax.vmap(tmon.energy_spectrum,
                         in_axes=({
                             'fluxonium': {
                                 'ec': None,
                                 'ej': None,
                                 'el': None,
                                 'phiext': 0
                             }
                         },)), ({
                             'fluxonium': {
                                 'ec': None,
                                 'ej': None,
                                 'el': 0,
                                 'phiext': None
                             }
                         },)), ({
                             'fluxonium': {
                                 'ec': 0,
                                 'ej': None,
                                 'el': None,
                                 'phiext': None
                             }
                         },)), ({
                             'fluxonium': {
                                 'ec': None,
                                 'ej': 0,
                                 'el': None,
                                 'phiext': None
                             }
                         },))
    data_1 = multi_spec(params)
    assert np.allclose(data_0.flatten(), data_1.flatten(), atol=2e-3)
