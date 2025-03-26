from functools import partial
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import pytest

from supergrad.scgraph.graph import SCGraph
from supergrad.helper import Spectrum
from supergrad.scgraph.graph_mpc_fluxonium_1d import MPCFluxonium1D

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('/'.join(dir_path.split('/')[:-1]))

n_qubit = 4
truncated_dim = 5
coupling_array = np.linspace(0e-3, 50e-3, 3)
enhanced_modes_list = [
    'greedy', 'standard', 'enhanced_eigvec', 'enhanced_continuum'
]


def create_graph_vs_coupling(coupling):
    scg = MPCFluxonium1D(n_qubit, periodic=False, seed=42)
    mp_coupling = {"capacitive_coupling": {"strength": coupling * 2 * np.pi}}
    for edge in scg.edges:
        scg.edges[edge].update(mp_coupling)
    return scg


@jax.vmap
def get_energies(coupling):
    assign_q = create_graph_vs_coupling(coupling)
    spec_assign_q = Spectrum(assign_q, truncated_dim=truncated_dim)
    return spec_assign_q.energy_tensor(spec_assign_q.all_params,
                                       greedy_assign=False,
                                       enhanced_assign='spin_projective')


@jax.vmap
def get_energies_greedy(coupling):
    assign_q = create_graph_vs_coupling(coupling)
    spec_assign_q = Spectrum(assign_q, truncated_dim=truncated_dim)
    return spec_assign_q.energy_tensor(spec_assign_q.all_params,
                                       greedy_assign=True)


@jax.vmap
def get_energies_eigvec(coupling):
    assign_q = create_graph_vs_coupling(coupling)
    spec_assign_q = Spectrum(assign_q, truncated_dim=truncated_dim)
    ref_q = create_graph_vs_coupling(0e-3)
    ref_spec_q = Spectrum(ref_q, truncated_dim=truncated_dim)
    ref_eigvec = ref_spec_q.energy_tensor(ref_spec_q.all_params,
                                          greedy_assign=False,
                                          return_eigvec=True)
    return spec_assign_q.energy_tensor(spec_assign_q.all_params,
                                       greedy_assign=False,
                                       enhanced_assign=ref_eigvec)


def get_energies_continuum(coupling_array):
    scgraph_sweep = [
        create_graph_vs_coupling(coupling) for coupling in coupling_array
    ]

    spec_assign_q = Spectrum(scgraph_sweep[-1], truncated_dim=truncated_dim)
    return spec_assign_q.energy_tensor(spec_assign_q.all_params,
                                       greedy_assign=False,
                                       enhanced_assign=scgraph_sweep[:-1],
                                       return_enhanced_aux_val=True)


def get_spectra_method(mode):
    match mode:
        case 'greedy':
            return get_energies_greedy
        case 'standard':
            return get_energies
        case 'enhanced_eigvec':
            return get_energies_eigvec
        case 'enhanced_continuum':
            return get_energies_continuum
        case _:
            raise ValueError(f"Invalid mode: {mode}")


@pytest.mark.parametrize('mode', enhanced_modes_list)
def test_jit_compability(mode):

    @partial(jax.jit, static_argnums=0)
    def _test_compability(mode):
        return get_spectra_method(mode)(coupling_array)

    # find nan in the eigenvalues
    assert not jnp.isnan(_test_compability(mode)).any()


def _test_grad_compability(mode):

    @jax.jit
    @jax.value_and_grad
    def _test_grad_compability(coupling_array):
        # return jnp.mean(get_energies_anharmonic(t_array, ej_array, ec))
        return jnp.mean(get_spectra_method(mode)(coupling_array))

    val = ravel_pytree(_test_grad_compability(coupling_array))[0]
    assert not jnp.isnan(val).any()
    return val


def test_non_reassignment():
    val_standard = _test_grad_compability('standard')
    _ = _test_grad_compability('greedy')
    val_eigvec = _test_grad_compability('enhanced_eigvec')
    val_continuum = _test_grad_compability('enhanced_continuum')
    # assert jnp.allclose(val_standard, val_greedy)
    assert jnp.allclose(val_standard, val_eigvec)
    assert jnp.allclose(val_standard, val_continuum)
