import numpy as np
import jax.numpy as jnp
import haiku as hk

from supergrad.scgraph.graph import (SCGraph, parse_pre_comp_name,
                                     parse_post_comp_name)

pauli_mats = []
pauli_mats.append(np.eye(2))
pauli_mats.append(np.array([[0., 1.], [1., 0.]]))
pauli_mats.append(np.array([[0., 1j], [-1j, 0.]]))
pauli_mats.append(np.array(np.diag((1., -1.))))


class SingleQubitCompensation(hk.Module):
    """Class for single qubit gates compensation.

    Args:
        graph: SuperGrad graph
        compensation_option (string): Set single qubit compensation strategy,
            should be in ['only_vz', 'arbit_single']
        coupler_subsystem: the name of coupler subsystem should be keep in |0>
            during the evolution.
        name: module name
    """

    def __init__(self,
                 graph: SCGraph,
                 compensation_option='only_vz',
                 coupler_subsystem=[],
                 name: str = 'single_q_compensation'):
        super().__init__(name=name)
        assert compensation_option in ['only_vz', 'arbit_single']
        # construct activation function
        self.pre_comp_angles = []
        self.post_comp_angles = []
        self.compensation_option = compensation_option
        shape = [] if self.compensation_option == 'only_vz' else [3]

        for node in graph.sorted_nodes:
            if node not in coupler_subsystem:
                self.pre_comp_angles.append(
                    hk.get_parameter(parse_pre_comp_name(node),
                                     shape,
                                     init=jnp.zeros))
                self.post_comp_angles.append(
                    hk.get_parameter(parse_post_comp_name(node),
                                     shape,
                                     init=jnp.zeros))

    def create_unitaries(self):
        """Create unitaries describe the single-qubit rotation before and after
        the time evolution.
        """
        if self.compensation_option == 'only_vz':
            # do single-qubit Z-rotation before and after the time evolution
            pauli_0 = np.ones(2)
            pauli_3 = np.array([1, -1])
            pre_unitaries = [
                jnp.cos(pre_params) * pauli_0 +
                1j * jnp.sin(pre_params) * pauli_3
                for pre_params in self.pre_comp_angles
            ]
            post_unitaries = [
                jnp.cos(post_params) * pauli_0 +
                1j * jnp.sin(post_params) * pauli_3
                for post_params in self.post_comp_angles
            ]
        else:
            # do arbitrary single-qubit rotation before and after the time evolution
            pre_unitaries = [
                jnp.cos(pre_params[1]) * pauli_mats[0] +
                1j * jnp.sin(pre_params[1]) *
                (jnp.cos(pre_params[0]) * pauli_mats[3] +
                 jnp.sin(pre_params[0]) *
                 (jnp.cos(pre_params[2]) * pauli_mats[1] +
                  jnp.sin(pre_params[2]) * pauli_mats[2]))
                for pre_params in self.pre_comp_angles
            ]
            post_unitaries = [
                jnp.cos(post_params[1]) * pauli_mats[0] +
                1j * jnp.sin(post_params[1]) *
                (jnp.cos(post_params[0]) * pauli_mats[3] +
                 jnp.sin(post_params[0]) *
                 (jnp.cos(post_params[2]) * pauli_mats[1] +
                  jnp.sin(post_params[2]) * pauli_mats[2]))
                for post_params in self.post_comp_angles
            ]

        return pre_unitaries, post_unitaries
