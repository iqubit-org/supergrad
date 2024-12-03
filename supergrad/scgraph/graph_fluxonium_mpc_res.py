from collections import deque
from typing import Tuple

import numpy as np
import networkx as nx

from supergrad.helper.compute_spectrum import Spectrum
from supergrad.scgraph.graph_mpc_fluxonium_1d import MPCFluxonium1D
# device parameters for frequency pattern
fluxonium_type_0 = {
    "ec": 1.0 * 2 * np.pi,
    "ej": 4.0 * 2 * np.pi,
    "el": 0.9 * 2 * np.pi,
    "shared_param_mark": "grey",
    "phiext": np.pi,
    "system_type": "fluxonium",
}

fluxonium_type_1 = {
    "ec": 1.0 * 2 * np.pi,
    "ej": 4.0 * 2 * np.pi,
    "el": 1.0 * 2 * np.pi,
    "shared_param_mark": 'blue',
    "phiext": np.pi,
    "system_type": "fluxonium",
}

fluxonium_type_2 = {
    "ec": 1.0 * 2 * np.pi,
    "ej": 4.0 * 2 * np.pi,
    "el": 1.1 * 2 * np.pi,
    "shared_param_mark": "green",
    "phiext": np.pi,
    "system_type": "fluxonium",
}
qubit_type = [fluxonium_type_0, fluxonium_type_1, fluxonium_type_2]

mp_coupling = {
    "capacitive_coupling": {
        "strength": 20.0e-3 * 2 * np.pi
    },
    "inductive_coupling": {
        "strength": -1.0 * 2e-3 * 2 * np.pi
    },
}
f_res = np.linspace(6.35, 6.70, 3)
res_type = [{
    "f_res": f * 2 * np.pi,
    "shared_param_mark": f"res_{f:.2f}",
    "system_type": "resonator"
} for f in f_res]
res_coupling = {"capacitive_resonator": {"strength": 0.025 * 2 * np.pi}}


class MPCFRes1D(MPCFluxonium1D):
    """A class for multipath coupling fluxonium chain with coupled resonators.

        Args:
        n_qubit (int): number of qubits in the chain
        seed (int, optional): random seed for the graph to add variation.
            Defaults to None.
    """

    qubit_subsystem = []
    resonator_subsystem = []

    def __init__(self, n_qubit: int = None, periodic=None, seed=None):
        super().__init__()

        # initialize graph
        if n_qubit is not None:
            # one qubit coupled to one resonator
            temp_graph = nx.grid_graph([n_qubit])
            # adding attributes to nodes
            qubit_params = deque(qubit_type)
            res_params = deque(res_type)
            label_mapping = dict((label, ''.join(['fm', str(label)]))
                                 for label in temp_graph.nodes)
            # adding attributes to edges
            for edge in temp_graph.edges:
                temp_graph.edges[edge].update(mp_coupling)
            for i in range(n_qubit):
                temp_graph.nodes[i].update(qubit_params[i % len(qubit_type)])
                # attach resonator to qubit
                temp_graph.add_node(f'res{i}', **res_params[i % len(res_type)])
                temp_graph.add_edge(i, f'res{i}', **res_coupling)
            # relabel nodes
            temp_graph = nx.relabel_nodes(temp_graph, label_mapping)
            # save temp_graph
            self.add_nodes_from(temp_graph.nodes.data())
            self.add_edges_from(temp_graph.edges.data())
            self.qubit_subsystem = [f'fm{i}' for i in range(n_qubit)]
            self.resonator_subsystem = [f'res{i}' for i in range(n_qubit)]
        if seed is not None:
            # add variance to el ec ej params
            self.add_lcj_params_variance_to_graph(multi_err=0.01,
                                                  seed=seed,
                                                  prefix='fm')

    @property
    def list_component_name(self):
        return self.qubit_subsystem + self.resonator_subsystem

    def get_neighbor(self, name: str) -> Tuple[int, int]:
        """Gets the neighbor of given qubit.

        Note only qubits are neighbors of a coupler,
        and vice versa. Other couplings are ignored.


        Args:
            name: the name of node to find the neighbor

        Returns:
            a list of component name
        """
        list_n = self.neighbors(name)
        st0 = self.nodes[name]["system_type"]
        l1 = [n for n in list_n if self.nodes[n]["system_type"] != st0]
        return l1


if __name__ == '__main__':
    mpr = MPCFRes1D(4, seed=0)
    spec = Spectrum(mpr, truncated_dim=2, share_params=True)
    spec.ls_params()
    spec.get_model_eigen_basis(spec.all_params, mpr.sorted_nodes, [['fm0']])
    mpr.compute_static_properties_minimal([['fm0']], False)
