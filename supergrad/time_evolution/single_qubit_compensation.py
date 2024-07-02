from typing import List
import numpy as np
import jax
import jax.numpy as jnp

from supergrad.scgraph.graph import (SCGraph, parse_pre_comp_name,
                                     parse_post_comp_name)

pauli_mats = []
pauli_mats.append(np.eye(2))
pauli_mats.append(np.array([[0., 1.], [1., 0.]]))
pauli_mats.append(np.array([[0., 1j], [-1j, 0.]]))
pauli_mats.append(np.array(np.diag((1., -1.))))


class SingleQubitCompensation():
    """Class for single qubit gates compensation.

    Args:
        graph: SuperGrad graph
        compensation_option (string): Set single qubit compensation strategy,
            should be in ['only_vz', 'arbit_single']
        coupler_subsystem: the name of coupler subsystem should be keep in |0>
            during the evolution.
        name: module name

    Raises:
        ValueError: the compensation data in graph shape mismatch with the type
    """

    def __init__(self,
                 graph: SCGraph,
                 coupler_subsystem=[],
                 name: str = 'single_q_compensation'):
        self.name = name
        # construct activation function
        self.pre_comp_angles: List[jax.Array] = []
        self.post_comp_angles: List[jax.Array] = []
        shape = []

        for node_name in graph.sorted_nodes:
            if node_name not in coupler_subsystem:
                node_comp = graph.nodes[node_name].get("compensation", {})
                self.pre_comp_angles.append(node_comp.get("pre_comp", jnp.zeros(shape)))
                self.post_comp_angles.append(node_comp.get("post_comp", jnp.zeros(shape)))

    def create_unitaries(self):
        """Create unitaries describe the single-qubit rotation before and after
        the time evolution.
        """
        # 1 param: do single-qubit Z-rotation before and after the time evolution
        # 3 params: do arbitrary single-qubit rotation before and after the time evolution
        list_pre_post = []
        for angles in [self.pre_comp_angles, self.post_comp_angles]:
            list_unitary = []
            for params in angles:
                if params.size == 1:
                    unitary = jnp.cos(params) * pauli_mats[0] + 1j * jnp.sin(params) * pauli_mats[3]
                elif params.size == 3:
                    unitary = (jnp.cos(params[1]) * pauli_mats[0] +
                               1j * jnp.sin(params[1]) *
                               (jnp.cos(params[0]) * pauli_mats[3] +
                                jnp.sin(params[0]) *
                                (jnp.cos(params[2]) * pauli_mats[1] +
                                 jnp.sin(params[2]) * pauli_mats[2])))
                else:
                    raise ValueError(f"Incompatible array size {params.size}")
                list_unitary.append(unitary)
            list_pre_post.append(list_unitary)

        return tuple(list_pre_post)
