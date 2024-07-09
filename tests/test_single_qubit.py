import os
import numpy as np
import jax
import pytest

import supergrad
from supergrad.quantum_system import Fluxonium, Transmon
from supergrad import Helper

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)


def test_transmon_phase_basis():
    # Get cached data
    data_0 = np.load('data/data_create_transmon.npy')

    # construct model
    class ExploreTransmon(Helper):
        list_function_not_require_init = ["energy_spectrum"]

        def init_quantum_system(self, params):
            super().init_quantum_system(params)
            self.tmon = Transmon(basis='phase',
                                 d=1.,
                                 phiej=0.,
                                 truncated_dim=6,
                                 drive_for_state_phase='charge',
                                 n_max=31,
                                 **(params["transmon"]))

        @Helper.decorator_auto_init
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
    class ExploreTransmon(Helper):

        def init_quantum_system(self, params):
            super().init_quantum_system(params)
            self.tmon = Transmon(basis='phase',
                                 truncated_dim=6,
                                 drive_for_state_phase='charge',
                                 n_max=40,
                                 **(params["transmon"]))

        @Helper.decorator_auto_init
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
        'transmon': {
            'd': d_list,
            'ec': ec_list,
            'ej': ej_list,
            'ng': ng_list,
            'phiej': flux_list * 2 * np.pi
        }
    }

    multi_spec = jax.vmap(
        jax.vmap(
            jax.vmap(
                jax.vmap(
                    jax.vmap(tmon.energy_spectrum,
                             in_axes=({
                                          'transmon': {
                                              'd': None,
                                              'ec': None,
                                              'ej': None,
                                              'ng': 0,
                                              'phiej': None
                                          }
                                      },)), ({
                                                 'transmon': {
                                                     'd': None,
                                                     'ec': None,
                                                     'ej': None,
                                                     'ng': None,
                                                     'phiej': 0
                                                 }
                                             },)), ({
                                                        'transmon': {
                                                            'd': 0,
                                                            'ec': None,
                                                            'ej': None,
                                                            'ng': None,
                                                            'phiej': None
                                                        }
                                                    },)), ({
                                                               'transmon': {
                                                                   'd': None,
                                                                   'ec': 0,
                                                                   'ej': None,
                                                                   'ng': None,
                                                                   'phiej': None
                                                               }
                                                           },)), ({
                                                                      'transmon': {
                                                                          'd': None,
                                                                          'ec': None,
                                                                          'ej': 0,
                                                                          'ng': None,
                                                                          'phiej': None
                                                                      }
                                                                  },))
    data_1 = multi_spec(params)

    assert np.allclose(data_0.flatten(), data_1.flatten())


def test_fluxonium_phase_basis():
    # Get cached data
    data_0 = np.load('data/data_create_fluxonium.npy')

    # construct model
    class ExploreFluxonium(Helper):

        def init_quantum_system(self, params):
            super().init_quantum_system(params)
            self.fmon = Fluxonium(basis='phase',
                                  truncated_dim=6,
                                  drive_for_state_phase='charge',
                                  put_phiext_on_inductor=False,
                                  phi_max=5 * np.pi,
                                  **(params["fluxonium"]))

        @Helper.decorator_auto_init
        def energy_spectrum(self):
            return self.fmon.eigenenergies()

    tmon = ExploreFluxonium()
    # create params template
    ej_list = np.linspace(7, 10, 3)
    ec_list = np.linspace(2.2, 2.7, 3)
    el_list = np.linspace(0.3, 0.7, 3)
    flux_list = np.linspace(0.1, 0.5, 3)
    params = {
        "fluxonium": {
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
                                      "fluxonium": {
                                          'ec': None,
                                          'ej': None,
                                          'el': None,
                                          'phiext': 0,
                                      }
                                  },)), ({
                                             "fluxonium": {
                                                 'ec': None,
                                                 'ej': None,
                                                 'el': 0,
                                                 'phiext': None
                                             }
                                         },)), ({
                                                    "fluxonium": {
                                                        'ec': 0,
                                                        'ej': None,
                                                        'el': None,
                                                        'phiext': None
                                                    }
                                                },)), ({
                                                           "fluxonium": {
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
    class ExploreFluxonium(Helper):

        def init_quantum_system(self, params):
            self.fmon = Fluxonium(basis='phase',
                                  truncated_dim=6,
                                  drive_for_state_phase='charge',
                                  put_phiext_on_inductor=True,
                                  phi_max=5 * np.pi,
                                  **(params["fluxonium"]))

        @Helper.decorator_auto_init
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
