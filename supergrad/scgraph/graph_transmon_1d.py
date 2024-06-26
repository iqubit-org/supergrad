import jax.numpy as jnp
import networkx as nx

from supergrad.scgraph.graph import SCGraph

# Before we multiply 2pi, ec, ej are in units of Ghz
# We multiply 2pi to them so that we follow the same convention of setting
# hbar=1 as done in qutip's sesolve
transmon_1 = {
    'ec': jnp.array(0.105 * 2 * jnp.pi),
    'ej': jnp.array(33.063 * 2 * jnp.pi),
    'system_type': 'transmon',
    'shared_param_mark': 1,
    'arguments': {
        'ng': 0.5,
        'num_basis': 62,
        'n_max': 31,
        'phiext': 0.,
    }
}
transmon_2 = {
    'ec': jnp.array(0.120 * 2 * jnp.pi),
    'ej': jnp.array(22.234 * 2 * jnp.pi),
    'system_type': 'transmon',
    'shared_param_mark': 2,
    'arguments': {
        'ng': 0.5,
        'num_basis': 62,
        'n_max': 31,
        'phiext': 0.,
    }
}
coupler_1 = {
    'ec': jnp.array(0.185 * 2 * jnp.pi),
    'ej': jnp.array(30.694 * 2 * jnp.pi),
    'system_type': 'transmon',
    'shared_param_mark': 3,
    'arguments': {
        'ng': 0.5,
        'num_basis': 62,
        'n_max': 31,
        'truncated_dim': 3,
        'd': 0,
        'phiext': 0.,
    }
}
g_12 = {
    'capacitive_coupling': {
        'strength': jnp.array(12e-3 * 2 * jnp.pi)
    },
}
g_1c = {
    'capacitive_coupling': {
        'strength': jnp.array(122e-3 * 2 * jnp.pi)
    },
}
g_2c = {
    'capacitive_coupling': {
        'strength': jnp.array(105e-3 * 2 * jnp.pi)
    },
}


class Tmon1D(SCGraph):
    """Device parameters for 4 qubits-4 couplers loop quantum processor graph.

    Args:
        seed : (int) The seed for random device parameters variance.
            default seed is None, means no variance.
    """

    def __init__(self, seed=None):
        super().__init__(None)
        # initialize graph
        temp_graph = nx.Graph()
        # adding attributes to nodes
        params = [transmon_1, coupler_1, transmon_2, coupler_1]
        for i in range(8):
            temp_graph.add_node(i, **params[i % 4])

        # adding attributes to edges
        tc_couplings = {
            0: [g_1c, g_12],
            1: [g_2c, None],
            2: [g_2c, g_12],
            3: [g_1c, None]
        }
        for i in range(8):
            for j, coupling in enumerate(tc_couplings[i % 4], 1):
                if coupling is not None:
                    temp_graph.add_edge(i, (i + j) % 8, **coupling)

        # relabel nodes
        label_mapping = dict(
            (label, ''.join(['q', str(label)])) for label in temp_graph.nodes)
        temp_graph = nx.relabel_nodes(temp_graph, label_mapping)
        # save temp_graph
        self.add_nodes_from(temp_graph.nodes.data())
        self.add_edges_from(temp_graph.edges.data())

        if seed is not None:
            self.add_lcj_params_variance_to_graph(seed=seed)


class XGateTmon1D(Tmon1D):
    """Add pulse parameters for simultaneous X gates.
    In the example, we add 3 pulses on `q0`, `q2` and `q4`.

    Args:
        seed : (int) The seed for random device parameters variance.
            default seed is None, means no variance.
    """

    def __init__(self, seed=None):
        super().__init__(seed)
        self.add_node(
            'q0', **{
                'pulse': {
                    'amp': jnp.array(0.0922632),
                    'length': jnp.array(39.99841052),
                    'omega_d': jnp.array(31.89213402),
                    'phase': jnp.array(-0.06459036),
                    'pulse_type': 'cos',
                    'operator_type': 'n_operator',
                    'delay': 0.
                },
                'compensation': {
                    'pre_comp': jnp.array(0.0086247),
                    'post_comp': jnp.array(-0.0086247)
                }
            })
        self.add_node(
            'q2', **{
                'pulse': {
                    'amp': jnp.array(0.10390872),
                    'length': jnp.array(39.92211365),
                    'omega_d': jnp.array(27.99554391),
                    'phase': jnp.array(0.05805683),
                    'pulse_type': 'cos',
                    'operator_type': 'n_operator',
                    'delay': 0.
                },
                'compensation': {
                    'pre_comp': jnp.array(0.3714724),
                    'post_comp': jnp.array(-0.3714724)
                }
            })
        self.add_node(
            'q4', **{
                'pulse': {
                    'amp': jnp.array(0.09196213),
                    'length': jnp.array(39.88357291),
                    'omega_d': jnp.array(31.97277349),
                    'phase': jnp.array(-0.07858071),
                    'pulse_type': 'cos',
                    'operator_type': 'n_operator',
                    'delay': 0.
                },
                'compensation': {
                    'pre_comp': jnp.array(0.79958712),
                    'post_comp': jnp.array(-0.79958712)
                }
            })
