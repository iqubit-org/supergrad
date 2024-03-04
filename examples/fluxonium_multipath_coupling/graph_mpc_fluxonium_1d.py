from copy import deepcopy
from collections import deque

import numpy as np
import networkx as nx

from supergrad.quantum_system.graph import BaseGraph
from supergrad.common_functions.compute_spectrum import Spectrum


class MPCFluxonium1D(BaseGraph):
    """The general graph for 1d multipath coupling fluxonium chain. One could
    select a 3-frequency pattern and decide the number of qubits in the chain.
    The graph is then used to create a quantum processor, and one could attach
    control pulse to any qubit using class method.

    Args:
        n_qubit (int): number of qubits in the chain
        periodic (bool): whether the chain is periodic(the chain becomes a loop) or not
        seed (int, optional): random seed for the graph to add variation.
            Defaults to None.
    """

    def __init__(self, n_qubit: int, periodic, seed=None):
        super().__init__()

        # initialize graph
        temp_graph = nx.grid_graph([n_qubit], periodic=periodic)

        # device parameters for frequency pattern
        fluxonium_type_1 = {
            "ec": 1.0 * 2 * np.pi,
            "ej": 4.0 * 2 * np.pi,
            "el": 0.9 * 2 * np.pi,
            "phiext": np.pi,
            "system_type": "fluxonium",
        }

        fluxonium_type_2 = {
            "ec": 1.0 * 2 * np.pi,
            "ej": 4.0 * 2 * np.pi,
            "el": 1.0 * 2 * np.pi,
            "phiext": np.pi,
            "system_type": "fluxonium",
        }

        fluxonium_type_3 = {
            "ec": 1.0 * 2 * np.pi,
            "ej": 4.0 * 2 * np.pi,
            "el": 1.1 * 2 * np.pi,
            "phiext": np.pi,
            "system_type": "fluxonium",
        }

        mp_coupling = {
            "capacitive_coupling": {
                "strength": 20.0e-3 * 2 * np.pi
            },
            "inductive_coupling": {
                "strength": -1.0 * 2e-3 * 2 * np.pi
            },
        }

        # adding attributes to nodes
        params = deque([fluxonium_type_1, fluxonium_type_2, fluxonium_type_3])
        for i in range(n_qubit):
            temp_graph.nodes[i].update(params[i % 3])
        # relabel nodes
        label_mapping = dict(
            (label, ''.join(['fm', str(label)])) for label in temp_graph.nodes)
        temp_graph = nx.relabel_nodes(temp_graph, label_mapping)
        # adding attributes to edges
        for edge in temp_graph.edges:
            temp_graph.edges[edge].update(mp_coupling)
        # save temp_graph
        self.add_nodes_from(temp_graph.nodes.data())
        self.add_edges_from(temp_graph.edges.data())

        if seed is not None:
            # add variance to el ec ej params
            self.add_lcj_params_variance_to_graph(multi_err=0.01, seed=seed)

    def compute_static_properties(self, list_drive_subsys, enable_var,
                                  **kwargs):
        """Compute static properties of the model.

        Args:
            list_drive_subsys: list of driving subsystem, each element is a list
                of qubit names
            enable_var: If true, will add random variance to the device parameters
            kwargs: other parameters for the Spectrum class
        """
        # Compute the model once
        spec = Spectrum(self, truncated_dim=2, enable_var=enable_var, **kwargs)
        list_energy_nd, list_n_mat, list_phi_mat, transform_matrix = spec.get_model_eigen_basis(
            spec.static_params, self.sorted_nodes, list_drive_subsys)
        return list_energy_nd, list_n_mat, list_phi_mat, transform_matrix

    def _add_crosstalk_pulse(self, name_drive, ar_crosstalk, pulse_dict):
        """Add crosstalk pulse based on the crosstalk matrix.

        Args:
            name_drive: the name of the drive qubit
            ar_crosstalk: the crosstalk matrix.  `ar[j,i]` means
                the pulse `A` applied on `i` qubit leads to `ar[j,i]*A` on `j`
                qubit.
            pulse_dict: the pulse dictionary
        """
        # Add full matrix
        dic2 = deepcopy(pulse_dict)
        dic2['crosstalk'] = {}
        for ix_name1, name in enumerate(self.sorted_nodes):
            # Keep only the wanted drive
            if name != name_drive:
                continue
            dic_c = {}
            for ix_name2, name2 in enumerate(self.sorted_nodes):
                v = ar_crosstalk[ix_name2, ix_name1]
                if ix_name1 != ix_name2 and v != 0:
                    dic_c[name2] = v
            # Only add when non-zero
            if (np.array(dic_c.values()) != 0).any():
                dic2["crosstalk"].update(dic_c)

        return dic2

    def create_cr_pulse(self,
                        ix_control_list: int,
                        ix_target_list: int,
                        tg_list: int,
                        enable_var: bool,
                        ar_crosstalk=None,
                        pulse_type: str = "cos"):
        """Creates CR pulses in the quantum processor.

        This function supports any multipath coupling model implemented.

        `index` is a model-dependent parameter depends on `self.sorted_nodes`

        Args:
            ix_control_list: list of the control qubit index, indicates where to
                apply the drive.
            ix_target_list: list of the target qubit index, indicates the target
                qubit that matched the drive frequency.
            tg_list: list of the gate time
            enable_var: If true, will add random variance to the device parameters
            ar_crosstalk: the crosstalk matrix.  `ar[j,i]` means
                the pulse `A` applied on `i` qubit leads to `ar[j,i]*A` on `j`
                qubit.
            pulse_type: the pulse type, by default it is "cos"

        Returns:
            transform_matrix
        """
        # Get the initial guess from the Hamiltonian
        list_drive_subsys = []
        for control, target in zip(ix_control_list, ix_target_list):
            # check the target qubit in the neighbors of control qubit.
            if self.sorted_nodes[target] not in self.neighbors(
                    self.sorted_nodes[control]):
                raise ValueError(
                    f"Target qubit {target} is not a neighbor of control qubit {control}"
                )
            # the index of drive_subsys must match the intrinsic order
            list_drive_subsys.append(
                [self.sorted_nodes[ix] for ix in sorted([control, target])])
        list_energy_nd, list_n_mat, list_phi_mat, transform_matrix = self.compute_static_properties(
            list_drive_subsys, enable_var)

        for control, target, tg, drive_subsys, energy_nd, n_mat, phi_mat in zip(
                ix_control_list, ix_target_list, tg_list, list_drive_subsys,
                list_energy_nd, list_n_mat, list_phi_mat):

            s_start = (0, 0)
            name_drive = self.sorted_nodes[control]
            if name_drive == drive_subsys[0]:
                s_target = (0, 1)
                s_control = (1, 0)
            else:
                s_target = (1, 0)
                s_control = (0, 1)
            fd = abs(energy_nd[s_target] - energy_nd[s_start])
            f_control = abs(energy_nd[s_control] - energy_nd[s_start])
            detuning = abs(fd - f_control)
            # use the effective coupling strength
            j_eff = 0.01 * 2 * np.pi
            # TODO: compute the effective coupling by n matrix and phi matrix
            # ar_nd = np.array(
            #     [n_drive[(*s_start, *s_target)] for n_drive in n_mat])
            # ar_phi = np.array(
            #     [n_drive[(*s_start, *s_target)] for n_drive in phi_mat])
            tau_eps_drive = np.pi / 2.0 * detuning / j_eff

            cr_pulse = {
                "amp": tau_eps_drive / tg,
                "omega_d": fd,
                "phase": 0.0,
                "length": float(tg),
                "pulse_type": pulse_type,
                "operator_type": "phi_operator",
                "delay": 0.0,
            }

            if ar_crosstalk is None:
                dic1 = deepcopy(cr_pulse)
                self.add_node(name_drive, pulse=dic1)
            else:
                dic2 = self._add_crosstalk_pulse(name_drive, ar_crosstalk,
                                                 cr_pulse)
                self.add_node(name_drive, pulse=dic2)
        return transform_matrix

    def create_single_qubit_pulse(self,
                                  ix_qubit_list: int,
                                  tg_list: int,
                                  enable_var: bool,
                                  ar_crosstalk=None,
                                  factor=0.5,
                                  pulse_type: str = "cos"):
        """Creates single qubit pulses in the quantum processor.

        This function supports any multipath coupling model implemented.

        `index` is a model-dependent parameter depends on `self.sorted_nodes`

        Args:
            ix_qubit_list: target qubit index. `ix_pair` is a model-dependent
            parameter, is the qubit index, indicates where to apply the drive.
            tg_list: the gate time
            enable_var: If true, will add random variance to the device parameters
            ar_crosstalk: the crosstalk matrix.  `ar[j,i]` means
                the pulse `A` applied on `i` qubit leads to `ar[j,i]*A` on `j`
                qubit.
            factor: the amplitude factor, driving a X gate by default.
            pulse_type: the pulse type, by default it is "cos"

        Returns:
            transform_matrix
        """
        # Get the initial guess from the Hamiltonian
        list_drive_subsys = [[self.sorted_nodes[ix]] for ix in ix_qubit_list]
        list_energy_nd, list_n_mat, list_phi_mat, transform_matrix = self.compute_static_properties(
            list_drive_subsys, enable_var)

        for ix_qubit, tg, energy_nd, n_mat, phi_mat in zip(
                ix_qubit_list, tg_list, list_energy_nd, list_n_mat,
                list_phi_mat):
            name_drive = self.sorted_nodes[ix_qubit]
            s_start = (0,)
            s_end = (1,)

            # nd = abs(n_mat[0][(*s_start, *s_end)])
            phid = abs(phi_mat[0][(*s_start, *s_end)])
            fd = abs(energy_nd[s_start] - energy_nd[s_end])

            dic_pulse = {
                'amp': factor * 2 / tg / phid * 2 * np.pi,
                'omega_d': fd,
                "phase": 0.0,
                "length": float(tg),
                "pulse_type": pulse_type,
                "operator_type": "phi_operator",
                "delay": 0.0,
            }

            if ar_crosstalk is None:
                # update amplitude
                new_pulse = deepcopy(dic_pulse)
                self.add_node(name_drive, pulse=new_pulse)
            else:
                dic2 = self._add_crosstalk_pulse(name_drive, ar_crosstalk,
                                                 dic_pulse)
                self.add_node(ix_qubit, pulse=dic2)
        return transform_matrix
