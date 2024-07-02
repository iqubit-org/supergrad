from collections import deque
import jax.numpy as jnp
import networkx as nx

from supergrad.scgraph.graph import SCGraph

params_1 = {
    'ec': jnp.array(1. * 2 * jnp.pi),
    'ej': jnp.array(4. * 2 * jnp.pi),
    'el': jnp.array(1.1 * 2 * jnp.pi),
    'system_type': 'fluxonium',
    'shared_param_mark': 1,
    'arguments': {
        'phiext': 0.5 * 2 * jnp.pi,
        'phi_max': 5 * jnp.pi
    }
}
params_2 = {
    'ec': jnp.array(1. * 2 * jnp.pi),
    'ej': jnp.array(4. * 2 * jnp.pi),
    'el': jnp.array(1.2 * 2 * jnp.pi),
    'system_type': 'fluxonium',
    'shared_param_mark': 2,
    'arguments': {
        'phiext': 0.5 * 2 * jnp.pi,
        'phi_max': 5 * jnp.pi
    }
}
params_3 = {
    'ec': jnp.array(1. * 2 * jnp.pi),
    'ej': jnp.array(4. * 2 * jnp.pi),
    'el': jnp.array(1.0 * 2 * jnp.pi),
    'system_type': 'fluxonium',
    'shared_param_mark': 3,
    'arguments': {
        'phiext': 0.5 * 2 * jnp.pi,
        'phi_max': 5 * jnp.pi
    }
}
params_4 = {
    'ec': jnp.array(1. * 2 * jnp.pi),
    'ej': jnp.array(4. * 2 * jnp.pi),
    'el': jnp.array(0.8 * 2 * jnp.pi),
    'system_type': 'fluxonium',
    'shared_param_mark': 4,
    'arguments': {
        'phiext': 0.5 * 2 * jnp.pi,
        'phi_max': 5 * jnp.pi
    }
}
params_5 = {
    'ec': jnp.array(1. * 2 * jnp.pi),
    'ej': jnp.array(4. * 2 * jnp.pi),
    'el': jnp.array(0.9 * 2 * jnp.pi),
    'system_type': 'fluxonium',
    'shared_param_mark': 5,
    'arguments': {
        'phiext': 0.5 * 2 * jnp.pi,
        'phi_max': 5 * jnp.pi
    }
}
coupling_params = {
    'capacitive_coupling': {
        'strength': jnp.array(11.5e-3 * 2 * jnp.pi)
    },
    'inductive_coupling': {
        'strength': jnp.array(-1.0 * 2e-3 * 2 * jnp.pi)
    },
}


class PeriodicGraph(SCGraph):
    """Device parameters for 5x5 periodic quantum processor graph.

    Args:
        seed : (int) The seed for random device parameters variance.
            default seed is None, means no variance.
    """

    def __init__(self, seed=None):
        super().__init__(None)
        # initialize graph
        temp_graph = nx.grid_2d_graph(5, 5, periodic=True)
        # adding attributes to nodes
        params = deque([params_1, params_2, params_3, params_4, params_5])
        for i in range(5):
            for j in range(5):
                temp_graph.nodes[(i, j)].update(params[j])
            # params list right shift
            params.rotate(3)
        # relabel nodes
        label_mapping = dict(
            (label,
             ''.join(['q', str(label[0]), str(label[1])]))
            for label in temp_graph.nodes)
        temp_graph = nx.relabel_nodes(temp_graph, label_mapping)
        # adding attributes to edges
        for edge in temp_graph.edges:
            temp_graph.edges[edge].update(coupling_params)
        # save temp_graph
        self.add_nodes_from(temp_graph.nodes.data())
        self.add_edges_from(temp_graph.edges.data())

        if seed is not None:
            self.add_lcj_params_variance_to_graph(seed=seed)


class CNOTGatePeriodicGraph(PeriodicGraph):
    """Add pulse parameters for simultaneous CNOT gates.
    In the example, we add 3 CR pulses on `q02`, `q12`
    and `q22`.

    Args:
        seed : (int) The seed for random device parameters variance.
            default seed is None, means no variance.
    """

    def __init__(self, seed=None):
        super().__init__(seed)
        # adding optimized pulse shape to edges
        self.add_node(
            'q02', **{
                'pulse': {
                    "p1": {
                        'amp': jnp.array(0.19362652),
                        'omega_d': jnp.array(2.63342338),
                        'phase': jnp.array(-0.56882493),
                        't_plateau': jnp.array(69.9963988),
                        't_ramp': jnp.array(30.20130596),
                        'pulse_type': 'rampcos',
                        'operator_type': 'phi_operator',
                        'delay': 0.
                    }},
            })

        self.add_node(
            'q12', **{
                'pulse': {
                    "p1": {
                        'amp': jnp.array(0.18108519),
                        'omega_d': jnp.array(4.18157275),
                        'phase': jnp.array(0.27647924),
                        't_plateau': jnp.array(70.00154013),
                        't_ramp': jnp.array(29.98031495),
                        'pulse_type': 'rampcos',
                        'operator_type': 'phi_operator',
                        'delay': 0.
                    }},
            })

        self.add_node(
            'q22', **{
                'pulse': {
                    "p1": {
                        'amp': jnp.array(0.21301889),
                        'omega_d': jnp.array(3.51378339),
                        'phase': jnp.array(1.08374892),
                        't_plateau': jnp.array(69.94150468),
                        't_ramp': jnp.array(30.06869994),
                        'pulse_type': 'rampcos',
                        'operator_type': 'phi_operator',
                        'delay': 0.
                    }},
            })
        # the compensation parameters in previous optimization
        self._compensation = {
            'single_q_compensation': {
                'post_comp_q02':
                    jnp.array([0.26824387, -0.00582382, -0.19798702]),
                'post_comp_q03':
                    jnp.array([-1.42126234, 0.68387943, -0.09867095]),
                'post_comp_q12':
                    jnp.array([0.00466943, 0.78741568, 0.0862072]),
                'post_comp_q13':
                    jnp.array([-1.09492337, -1.34049224, 1.11393925]),
                'post_comp_q22':
                    jnp.array([-0.00160062, -1.50823985, -0.16763715]),
                'post_comp_q23':
                    jnp.array([-0.01605929, -0.1141277, 0.69661291]),
                'pre_comp_q02':
                    jnp.array([0.01907886, -0.0714353, 0.27239727]),
                'pre_comp_q03':
                    jnp.array([-0.77910168, 0.16258883, 0.00519836]),
                'pre_comp_q12':
                    jnp.array([-0.00220794, 0.38970001, 1.51744927]),
                'pre_comp_q13':
                    jnp.array([-0.40020693, 0.7668882, 0.74054753]),
                'pre_comp_q22':
                    jnp.array(
                        [-1.03353877e-03, -1.75575364e+00, 1.10831887e-02]),
                'pre_comp_q23':
                    jnp.array([1.99985832, -0.92168355, 0.49487715])
            }
        }

    @property
    def compensation(self):
        """The single qubit gate compensation."""
        return self._compensation
