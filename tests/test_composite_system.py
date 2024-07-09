# %%
import os
import numpy as np
import pytest

import supergrad
from supergrad.quantum_system import Fluxonium, Transmon, InteractingSystem, Resonator, parse_interaction

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)


# %%

def test_composite_transmon():
    # load cached data
    spec_0 = np.load('data/data_composite_transmon_spec.npy')
    map_0 = np.load('data/data_composite_transmon_map.npy')

    class ExploreTransmon(supergrad.Helper):

        def init_quantum_system(self,params):
            super().init_quantum_system(params)
            tm1 = Transmon(ej=40.,
                           ec=0.2,
                           d=0.1,
                           phiej=0.23 * 2 * np.pi,
                           ng=0.3,
                           n_max=40,
                           truncated_dim=3,
                           )
            tm2 = Transmon(ej=15.,
                           ec=0.15,
                           d=0.2,
                           phiej=0.,
                           ng=0.,
                           n_max=30,
                           truncated_dim=3,
                           )
            res = Resonator(f_res=4.5,
                            truncated_dim=4,
                            remove_zpe=True)

            coup1 = parse_interaction(g_strength=0.1,
                                      op1=tm1.n_operator,
                                      op2=res.creation_operator,
                                      add_hc=True,
                                      )
            coup2 = parse_interaction(g_strength=0.2,
                                      op1=tm2.n_operator,
                                      op2=res.creation_operator,
                                      add_hc=True,
                                      )
            self.hilbertspace = InteractingSystem([tm1, tm2, res],
                                                  [coup1, coup2])

        def energy_spectrum(self):
            return self.hilbertspace.eigenenergies(10)

        def energy_in_bare_indices(self):
            return self.hilbertspace.compute_energy_map()

    composite_tmon = ExploreTransmon()
    composite_tmon.init_quantum_system({})
    spec_1 = composite_tmon.energy_spectrum()
    map_1 = composite_tmon.energy_in_bare_indices()
    assert np.allclose(spec_0, spec_1)
    # remove NaN in the energy map
    idx = np.argwhere(~np.isnan(map_0))
    assert np.allclose(map_0[idx], map_1.flatten()[idx])


def test_composite_fluxonium():
    # load cached data
    spec_0 = np.load('data/data_composite_fluxonium_spec.npy')
    map_0 = np.load('data/data_composite_fluxonium_map.npy')

    class ExploreTransmon(supergrad.Helper):

        def init_quantum_system(self,params):
            super().init_quantum_system(params)
            fm1 = Fluxonium(ej=3.5,
                            ec=1.6,
                            el=0.5,
                            phiext=0.33 * 2 * np.pi,
                            phi_max=5 * np.pi,
                            truncated_dim=4,
                            )
            coupler1 = Fluxonium(ej=4.0,
                                 ec=2.0,
                                 el=0.5,
                                 phiext=0.0,
                                 phi_max=5 * np.pi,
                                 truncated_dim=6,
                                 )
            fm2 = Fluxonium(ej=3.5,
                            ec=1.2,
                            el=0.5,
                            phiext=0.5 * 2 * np.pi,
                            phi_max=5 * np.pi,
                            truncated_dim=4,
                            )

            coup1 = parse_interaction(g_strength=0.04,
                                      op1=fm1.n_operator,
                                      op2=coupler1.n_operator,
                                      )
            coup2 = parse_interaction(g_strength=-0.02,
                                      op1=fm1.phi_operator,
                                      op2=coupler1.phi_operator,
                                      )
            coup3 = parse_interaction(g_strength=0.04,
                                      op1=fm2.n_operator,
                                      op2=coupler1.n_operator,
                                      )
            coup4 = parse_interaction(g_strength=-0.02,
                                      op1=fm2.phi_operator,
                                      op2=coupler1.phi_operator,
                                      )
            self.hilbertspace = InteractingSystem([fm1, coupler1, fm2],
                                                  [coup1, coup2, coup3, coup4])

        def energy_spectrum(self):
            return self.hilbertspace.eigenenergies(10)

        def energy_in_bare_indices(self):
            return self.hilbertspace.compute_energy_map()

    composite_tmon = ExploreTransmon()
    composite_tmon.init_quantum_system({})
    spec_1 = composite_tmon.energy_spectrum()
    map_1 = composite_tmon.energy_in_bare_indices()
    assert np.allclose(spec_0, spec_1)
    # remove NaN in the energy map
    idx = np.argwhere(~np.isnan(map_0))
    assert np.allclose(map_0[idx], map_1.flatten()[idx])
