# %%
from functools import partial
import itertools
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from supergrad.utils.sys_diagnose import walsh_transform, plot_walsh_transform
from supergrad.helper import Spectrum
from supergrad.scgraph import SCGraph

key = jax.random.PRNGKey(42)
truncated_dim = 2
n_transmons = 8
ej_mean = 12.5
ej_var = 0.5
ej_array = ej_mean + ej_var * jax.random.normal(key, n_transmons)
ec = 250e-3

t_array = jnp.linspace(0e-3, 50e-3, 50)
t = 5e-3

# %%
class TransmonParams:

    def __init__(self, ec, ej, ng=0.0):
        self.ec = ec
        self.ej = ej
        self.ng = ng  # The highest bound level sensitive to ng

    def create_transmon_dict(self):

        transmon = {
            "ec": self.ec,
            "ej": self.ej,
            "system_type": "transmon",
            'arguments': {
                'ng': self.ng,
                'num_basis': 62,
                'n_max': 31,
                'phiext': 0.,
            }
        }

        return transmon


# %%
# Construct the graph of multiple transmons
class TransmonParams_VariableEJ(TransmonParams):

    def __init__(self, ec, ej_array, coupling, ng=0):
        super().__init__(ec, ej_array, ng)

        self.coupling = coupling

    def __iter__(self):
        return iter([TransmonParams(self.ec, ej, self.ng) for ej in self.ej])

    def __len__(self):
        return len(self.ej)

    def create_graph_vs_coupling(self):
        scg = SCGraph()
        # nodes represent qubits
        for i, tm in enumerate(self):
            scg.add_node(f"q{i}", **tm.create_transmon_dict())
            if i > 1:
                # edges represent two-qubit interactions
                scg.add_edge(
                    f"q{i - 1}", f"q{i}",
                    **{'capacitive_coupling': {
                        'strength': self.coupling
                    }})

        return scg


# %%
def get_energies(coupling, ej_array, ec):
    twoQ = TransmonParams_VariableEJ(ec, ej_array, coupling, ng=0.01)
    twoQ_graph = twoQ.create_graph_vs_coupling()
    spec_twoq = Spectrum(twoQ_graph, truncated_dim=truncated_dim)
    return spec_twoq.energy_tensor(spec_twoq.all_params)


e_val = get_energies(t, ej_array, ec)

# %%
coeff, bitstring = walsh_transform(e_val)
bitstring
# %%
plot_walsh_transform(coeff, bitstring)
coeff
# %%
