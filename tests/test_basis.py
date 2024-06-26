import os
import numpy as np

import supergrad
from supergrad.quantum_system import Fluxonium, Transmon

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)


def test_transmon_basis():
    # construct model
    class ExploreTransmon(supergrad.Helper):

        def __init__(self, basis) -> None:
            super().__init__(basis=basis)

        def _init_quantum_system(self):
            self.tmon = Transmon(ec=1.2,
                                 ej=30.02,
                                 ng=0.3,
                                 phiext=0.,
                                 basis=self.kwargs['basis'],
                                 truncated_dim=6,
                                 drive_for_state_phase='charge',
                                 n_max=31,
                                 constant=True)

        def energy_spectrum(self):
            return self.tmon.eigenenergies()

        def operator(self):
            """Calculate the operator in the raw basis."""
            return np.asarray(
                [self.tmon.n_operator(),
                 self.tmon.phi_operator()])

    tmon_phase = ExploreTransmon('phase')
    tmon_charge = ExploreTransmon('charge')
    # compare spectra
    assert np.allclose(tmon_charge.energy_spectrum({}),
                       tmon_phase.energy_spectrum({}))
    # compare operator
    assert np.allclose(tmon_charge.operator({}), tmon_phase.operator({}))


def test_tunable_transmon_basis():
    # construct model
    class ExploreTransmon(supergrad.Helper):

        def __init__(self, basis) -> None:
            super().__init__(basis=basis)

        def _init_quantum_system(self):
            self.tmon = Transmon(ec=0.5,
                                 ej=50.0,
                                 d=0.01,
                                 phiext=0.1 * 2 * np.pi,
                                 ng=0.3,
                                 basis=self.kwargs['basis'],
                                 truncated_dim=6,
                                 drive_for_state_phase='charge',
                                 n_max=31,
                                 constant=True)

        def energy_spectrum(self):
            return self.tmon.eigenenergies()

        def operator(self):
            """Calculate the operator in the raw basis."""
            return np.asarray(
                [self.tmon.n_operator(),
                 self.tmon.phi_operator()])

    tmon_phase = ExploreTransmon('phase')
    tmon_charge = ExploreTransmon('charge')
    # compare spectra
    assert np.allclose(tmon_charge.energy_spectrum({}),
                       tmon_phase.energy_spectrum({}))
    # compare operator
    assert np.allclose(tmon_charge.operator({}), tmon_phase.operator({}))


def test_fluxonium_basis():
    # construct model
    class ExploreFluxonium(supergrad.Helper):

        def __init__(self, basis) -> None:
            super().__init__(basis=basis)

        def _init_quantum_system(self):
            self.fmon = Fluxonium(ec=2.5,
                                  ej=8.9,
                                  el=0.5,
                                  phiext=0.33 * 2 * np.pi,
                                  basis=self.kwargs['basis'],
                                  truncated_dim=6,
                                  drive_for_state_phase='charge',
                                  put_phiext_on_inductor=True,
                                  phi_max=5 * np.pi,
                                  constant=True)

        def energy_spectrum(self):
            return self.fmon.eigenenergies()

        def operator(self):
            """Calculate the operator in the raw basis."""
            return np.asarray(
                [self.fmon.n_operator(),
                 self.fmon.phi_operator()])

    fmon_phase = ExploreFluxonium('phase')
    fmon_charge = ExploreFluxonium('charge')
    # compare spectra
    assert np.allclose(fmon_charge.energy_spectrum({}),
                       fmon_phase.energy_spectrum({}))
    # compare operator
    assert np.allclose(fmon_charge.operator({}), fmon_phase.operator({}))


# @pytest.mark.slow
# def test_hermite_basis():
#     # construct model
#     class ExploreFluxonium(supergrad.Helper):

#         def __init__(self, basis) -> None:
#             super().__init__(basis=basis)

#         def _init_quantum_system(self):
#             self.fmon = Fluxonium(ec=2.5,
#                                   ej=8.9,
#                                   el=0.5,
#                                   phiext=0.33 * 2 * np.pi,
#                                   basis=self.kwargs['basis'],
#                                   truncated_dim=6,
#                                   drive_for_state_phase='charge',
#                                   put_phiext_on_inductor=False,
#                                   phi_max=5 * np.pi,
#                                   constant=True)

#         def energy_spectrum(self):
#             return self.fmon.eigenenergies()

#         def operator(self):
#             """Calculate the operator in the raw basis."""
#             return np.asarray(
#                 [self.fmon.n_operator(),
#                  self.fmon.phi_operator()])

#     fmon_phase = ExploreFluxonium('phase')
#     fmon_hermite = ExploreFluxonium('hermite')
#     print(
#         np.max(
#             np.abs(
#                 fmon_hermite.energy_spectrum({}) -
#                 fmon_phase.energy_spectrum({}))))
#     # compare spectra
#     assert np.allclose(fmon_hermite.energy_spectrum({}),
#                        fmon_phase.energy_spectrum({}))
#     # compare operator
#     assert np.allclose(fmon_hermite.operator({}), fmon_phase.operator({}))
