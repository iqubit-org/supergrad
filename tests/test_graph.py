import copy
import numpy as np
import json
from jax.tree_util import tree_leaves, tree_map, tree_leaves_with_path

import supergrad
from supergrad.scgraph.graph import SCGraph
from supergrad.utils.format_conv import (convert_to_json_compatible,
                                         convert_to_device_array_dict)
from .data.data_graph import CNOTGatePeriodicGraph

graph = CNOTGatePeriodicGraph(seed=1)
qubit_subset = graph.subscgraph(['q02', 'q03', 'q12', 'q13', 'q22', 'q23'])


def count_params(dic):
    return len(dic["nodes"]) + len(dic["edges"])


def graph_to_quantum_system(graph: SCGraph = graph,
                            add_random=False,
                            share_params=False):
    class TestGraph(supergrad.Helper):

        def init_quantum_system(self, params):
            super().init_quantum_system(params)
            self.hilbertspace = graph.convert_graph_to_quantum_system(
                truncated_dim=3,
                phiext=0.5 * 2 * np.pi,
                phi_max=5 * np.pi,
                )
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
    graph.share_params = share_params
    if add_random:
        graph.add_lcj_params_variance_to_graph(multi_err=0.01, seed=12381)
    tg.init_quantum_system({})
    params = convert_to_device_array_dict(graph.convert_graph_to_parameters(share_params))
    return params, tg.get_device_parameters()


def test_sharing_parameters():
    hk_0, params_0 = graph_to_quantum_system(share_params=False)
    hk_1, params_1 = graph_to_quantum_system(share_params=True)
    assert count_params(hk_0) - count_params(hk_1) == 60
    assert np.allclose(tree_leaves(params_0), tree_leaves(params_1))


def test_add_randomiable():
    hk_0, params_0 = graph_to_quantum_system(add_random=False)
    hk_1, params_1 = graph_to_quantum_system(add_random=True,
                                             share_params=False)
    hk_2, params_2 = graph_to_quantum_system(add_random=True, share_params=True)
    assert count_params(hk_1) - count_params(hk_2) == 60
    assert count_params(hk_0) == count_params(hk_1)
    assert np.allclose(tree_leaves(params_1), tree_leaves(params_2))
    assert np.allclose(tree_leaves(params_0), tree_leaves(params_2), rtol=0.05)


def test_update_params(share_params=False):
    qubit_subset_temp: CNOTGatePeriodicGraph = copy.deepcopy(qubit_subset)
    qubit_subset_temp.share_params = share_params
    all_params = qubit_subset_temp.convert_graph_to_parameters()
    # save parameters to json after optimizing
    json_params = convert_to_json_compatible(all_params)
    save_params = json.dumps(json_params)
    # one can analyze and edit parameters
    # load optimized parameters(in json)
    load_params = convert_to_device_array_dict(json.loads(save_params))
    # convert to imaginary number
    load_params = tree_map(lambda x: x * 1.0j, load_params)
    # update graph parameters
    old_qubit_subset = copy.deepcopy(qubit_subset_temp)
    qubit_subset_temp.update_parameters(load_params)
    hk_0, params_0 = graph_to_quantum_system(old_qubit_subset, share_params=share_params)
    hk_1, params_1 = graph_to_quantum_system(qubit_subset_temp, share_params=share_params)
    assert np.allclose(tree_leaves(hk_0),
                       tree_leaves(tree_map(lambda x: x * -1.0j, hk_1)))
    assert np.allclose(tree_leaves(params_0),
                       tree_map(lambda x: x * -1.0j, tree_leaves(params_1)))


def test_update_params_sharing_params():
    test_update_params(share_params=True)
