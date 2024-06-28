import copy
import numpy as np
import json
from jax.tree_util import tree_leaves, tree_map

import supergrad
from supergrad.scgraph.graph import SCGraph
from supergrad.utils.utility import (convert_to_json_compatible,
                                     convert_to_haiku_dict)
from .data.data_graph import CNOTGatePeriodicGraph

graph = CNOTGatePeriodicGraph(seed=1)
qubit_subset = graph.subgraph(['q02', 'q03', 'q12', 'q13', 'q22', 'q23'])


def graph_to_quantum_system(graph: SCGraph = graph,
                            add_random=False,
                            share_params=False):

    class TestGraph(supergrad.Helper):

        def _init_quantum_system(self):
            self.hilbertspace = graph.convert_graph_to_quantum_system(
                truncated_dim=3,
                phiext=0.5 * 2 * np.pi,
                phi_max=5 * np.pi,
                add_random=add_random,
                share_params=share_params)
            self.hamiltonian_component_and_pulseshape, self.pulse_endtime = graph.convert_graph_to_pulse_lst(
                self.hilbertspace, modulate_wave=True)

        def get_device_parameters(self):
            data = []
            for subsystem in self.hilbertspace.subsystem_list:
                data.append({
                    subsystem.name: [subsystem.ec, subsystem.ej, subsystem.el]
                })
            for interaction in self.hilbertspace.interaction_list:
                data.append({interaction.name: interaction.strength})
            return data

    tg = TestGraph()
    params = graph.convert_graph_to_parameters_haiku(share_params)
    return params, tg.get_device_parameters(params)


def test_sharing_parameters():
    hk_0, params_0 = graph_to_quantum_system(share_params=False)
    hk_1, params_1 = graph_to_quantum_system(share_params=True)
    assert len(hk_0) - len(hk_1) == 100
    assert np.allclose(tree_leaves(params_0), tree_leaves(params_1))


def test_enable_deviations():
    hk_0, params_0 = graph_to_quantum_system(add_random=False)
    hk_1, params_1 = graph_to_quantum_system(add_random=True,
                                             share_params=False)
    hk_2, params_2 = graph_to_quantum_system(add_random=True, share_params=True)
    assert len(hk_1) - len(hk_2) == 100
    assert len(hk_0) == len(hk_1)
    assert np.allclose(tree_leaves(params_1), tree_leaves(params_2))
    assert np.allclose(tree_leaves(params_0), tree_leaves(params_2), rtol=0.05)


def test_update_params(share_params=False):
    qubit_subset_temp = copy.deepcopy(qubit_subset)
    all_params = qubit_subset_temp.convert_graph_to_parameters_haiku(
        share_params=share_params)
    # save parameters to json after optimizing
    save_params = json.dumps(convert_to_json_compatible(all_params))
    # one can analyze and edit parameters
    # load optimized parameters(in json)
    load_params = convert_to_haiku_dict(json.loads(save_params))
    # convert to imaginary number
    load_params = tree_map(lambda x: x * 1.0j, load_params)
    # update graph parameters
    old_qubit_subset = copy.deepcopy(qubit_subset_temp)
    qubit_subset_temp.update_params(load_params, share_params=share_params)
    hk_0, params_0 = graph_to_quantum_system(old_qubit_subset)
    hk_1, params_1 = graph_to_quantum_system(qubit_subset_temp)
    assert np.allclose(tree_leaves(hk_0),
                       tree_leaves(tree_map(lambda x: x * -1.0j, hk_1)))
    assert np.allclose(tree_leaves(params_0),
                       tree_map(lambda x: x * -1.0j, tree_leaves(params_1)))


def test_update_params_sharing_params():
    test_update_params(share_params=True)
