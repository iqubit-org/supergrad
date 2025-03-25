# %%
from functools import partial
import numpy as np
import networkx as nx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import mplcursors
import matplotlib.pyplot as plt

from supergrad.scgraph.graph import SCGraph
from supergrad.helper import Spectrum

# preload a random key, let the random number generator generate the same random number
key = jax.random.PRNGKey(1)
truncated_dim = 5
enhanced_assign = True


# %%
class Params:

    def __init__(self, ec, ej, ng=0.0):
        self.ec = ec
        self.ej = ej
        # self.shared_param_mark = shared_param_mark
        # remove shared_param_mark because it only works for the back propagation
        self.ng = ng  # The highest bound level sensitive to ng


def create_transmon(params):
    ec = params.ec
    ej = params.ej
    ng = params.ng

    transmon = {
        "ec": ec,
        "ej": ej,
        "system_type": "transmon",
        'arguments': {
            'ng': ng,
            'num_basis': 62,
            'n_max': 31,
            'phiext': 0.,
        }
    }

    return transmon


class SingleTransmon(SCGraph):

    def __init__(self, transmon_0):
        super().__init__()

        # nodes represent qubits
        self.add_node("q0", **transmon_0)

    def get_energies(self):
        spec_1q = Spectrum(self,
                           truncated_dim=truncated_dim,
                           share_params=False,
                           unify_coupling=False)
        # disable share_params because it only works for the back propagation
        # disable "unify_coupling" and direct assign the coupling strength by the edge
        params_1q = spec_1q.all_params
        return spec_1q.energy_tensor(params_1q, greedy_assign=False)


# %% [markdown]
# ## Multiple Transmon Qubits in a Chain


# %%
# Construct the graph of multiple transmons
class Params_VariableEJ:

    def __init__(self, ec, ej_array, coupling, ng=0.0):
        self.ec = ec
        self.ej_array = ej_array
        self.coupling = coupling
        self.ng = ng  # The highest bound level sensitive to ng


def create_graph_vs_coupling(params: Params_VariableEJ):
    scg = SCGraph()
    # nodes represent qubits
    for i in range(len(params.ej_array)):
        scg.add_node(
            f"q{i}",
            **create_transmon(Params(params.ec, params.ej_array[i], params.ng)))

    # edges represent two-qubit interactions
    for i in range(len(params.ej_array) - 1):
        scg.add_edge(f"q{i}", f"q{i+1}",
                     **{'capacitive_coupling': {
                         'strength': params.coupling
                     }})
    return scg


# @jax.jit
def test_jax_compability(ec):
    twoQ = create_graph_vs_coupling(
        Params_VariableEJ(ec, [12.5, 13.0], 20e-3, ng=0.01))
    refQ = create_graph_vs_coupling(
        Params_VariableEJ(ec, [12.5, 13.0], 0e-3, ng=0.01))
    spec_twoq = Spectrum(twoQ)
    val = spec_twoq.energy_tensor(spec_twoq.all_params,
                                  greedy_assign=False,
                                  enhanced_assign=[refQ, refQ],
                                  return_enhanced_aux_val=True)
    return val


# test_jax_compability(250e-3)
# %%
@partial(jax.vmap, in_axes=(0, None, None))
def get_energies(coupling, ej_array, ec):
    twoQ = create_graph_vs_coupling(
        Params_VariableEJ(ec, ej_array, coupling, ng=0.01))
    spec_twoq = Spectrum(twoQ, truncated_dim=truncated_dim)
    return spec_twoq.energy_tensor(spec_twoq.all_params,
                                   greedy_assign=False,
                                   enhanced_assign='spin_projective')


@partial(jax.vmap, in_axes=(0, None, None))
def get_energies_anharmonic(coupling_array, ej_array, ec):
    anharmonic_rate = 1.22
    scgraph_sweep = [
        create_graph_vs_coupling(
            Params_VariableEJ(ec * anharmonic_rate,
                              ej_array / anharmonic_rate,
                              coupling_array * anharmonic_rate,
                              ng=0.01)),
        create_graph_vs_coupling(
            Params_VariableEJ(ec, ej_array, coupling_array, ng=0.01),)
    ]

    spec_twoq = Spectrum(scgraph_sweep[-1], truncated_dim=truncated_dim)
    return spec_twoq.energy_tensor(spec_twoq.all_params,
                                   greedy_assign=False,
                                   enhanced_assign=scgraph_sweep[:-1],
                                   return_enhanced_aux_val=False)


def get_energies_continuum(coupling_array, ej_array, ec):
    scgraph_sweep = [
        create_graph_vs_coupling(
            Params_VariableEJ(ec, ej_array, coupling, ng=0.01))
        for coupling in coupling_array
    ]

    spec_twoq = Spectrum(scgraph_sweep[-1], truncated_dim=truncated_dim)
    return spec_twoq.energy_tensor(spec_twoq.all_params,
                                   greedy_assign=False,
                                   enhanced_assign=scgraph_sweep[:-1],
                                   return_enhanced_aux_val=True)


# %% [markdown]
# ### Excitation energy of four coupled transmon qubits

# %%
n_transmons = 4
ej_mean = 12.5
ej_var = 0.5
ej_array = ej_mean + ej_var * jax.random.normal(key, n_transmons)
ec = 250e-3

t_array = jnp.linspace(0e-3, 50e-3, 50)


# energy_4q_array = get_energies_anharmonic(t_array, ej_array, ec, 3)
# print(energy_4q_array.shape)
@jax.jit
@jax.value_and_grad
def test_grad_compability(ec):
    return jnp.mean(get_energies_anharmonic(t_array, ej_array, ec))


# test_grad_compability(250e-3)
# %%
# %matplotlib widget
import mplcursors


def plot_spectra(energy_array, t_array, show_computational_basis=True):
    # minus ground state energy
    origin_shape = energy_array.shape
    spectra_4q = (energy_array.reshape(origin_shape[0], -1) -
                  energy_array.reshape(origin_shape[0], -1)[:, 0, jnp.newaxis])
    state_labels = [
        '|' + ''.join(map(str, np.unravel_index(i, origin_shape[1:]))) +
        r'$\rangle$' for i in np.arange(np.prod(origin_shape[1:]))
    ]

    lines = plt.plot(1e3 * t_array, spectra_4q, '.-', label=state_labels)
    plt.xlabel('Coupling (MHz)')
    plt.ylabel('Excitation frequency (GHz)')
    # plt.legend()
    if show_computational_basis is None:
        pass
    elif show_computational_basis:
        plt.ylim(4.3, 5.0)
    else:
        plt.ylim(9.0, 10.0)

    # Create an interactive cursor
    cursor = mplcursors.cursor(lines)


def plot_spectra_vs_coupling(enhanced_mode=0, show_computational_basis=True):
    if enhanced_mode == 0:
        energy_4q_array = get_energies(t_array, ej_array, ec)
    elif enhanced_mode == 1:
        energy_4q_array = get_energies_anharmonic(t_array, ej_array, ec)
    else:
        energy_4q_array = get_energies_continuum(t_array, ej_array, ec)

    print(energy_4q_array.shape)
    plot_spectra(energy_4q_array, t_array, show_computational_basis)
    return energy_4q_array


# %%
energy_ar = plot_spectra_vs_coupling(0, None)
plt.title('Spin projective assignment')
plt.show()
# %%
energy_ar = plot_spectra_vs_coupling(1, None)
plt.title('Anharmonicity adaptive assignment')
plt.show()
# %%
energy_ar = plot_spectra_vs_coupling(2, None)
plt.title('Continuum adjust couping assignment')
plt.show()

# %%
