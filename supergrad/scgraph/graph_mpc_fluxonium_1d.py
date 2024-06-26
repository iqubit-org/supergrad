from copy import deepcopy
from collections import deque

import numpy as np
import networkx as nx

from supergrad.scgraph.graph import SCGraph
from supergrad.helper.compute_spectrum import Spectrum


class MPCFluxonium1D(SCGraph):
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

    def __init__(self, n_qubit: int = None, periodic=None, seed=None):
        super().__init__()

        # device parameters for frequency pattern
        fluxonium_type_1 = {
            "ec": 1.0 * 2 * np.pi,
            "ej": 4.0 * 2 * np.pi,
            "el": 0.9 * 2 * np.pi,
            "shared_param_mark": "grey",
            "phiext": np.pi,
            "system_type": "fluxonium",
        }

        fluxonium_type_2 = {
            "ec": 1.0 * 2 * np.pi,
            "ej": 4.0 * 2 * np.pi,
            "el": 1.0 * 2 * np.pi,
            "shared_param_mark": 'blue',
            "phiext": np.pi,
            "system_type": "fluxonium",
        }

        fluxonium_type_3 = {
            "ec": 1.0 * 2 * np.pi,
            "ej": 4.0 * 2 * np.pi,
            "el": 1.1 * 2 * np.pi,
            "shared_param_mark": "green",
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

        # initialize graph
        if n_qubit is not None:
            temp_graph = nx.grid_graph([n_qubit], periodic=periodic)
            # adding attributes to nodes
            params = deque(
                [fluxonium_type_1, fluxonium_type_2, fluxonium_type_3])
            for i in range(n_qubit):
                temp_graph.nodes[i].update(params[i % 3])
            # relabel nodes
            label_mapping = dict((label, ''.join(['fm', str(label)]))
                                 for label in temp_graph.nodes)
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

    def compute_static_properties(self, list_drive_subsys, add_random,
                                  **kwargs):
        """Compute static properties of the model.

        Args:
            list_drive_subsys: list of driving subsystem, each element is a list
                of qubit names
            add_random: If true, will add random deviations to the device parameters
            kwargs: other parameters for the Spectrum class
        """
        # Compute the model once
        spec = Spectrum(self, truncated_dim=2, add_random=add_random, **kwargs)
        list_energy_nd, list_n_mat, list_phi_mat, transform_matrix = spec.get_model_eigen_basis(
            spec.all_params, self.sorted_nodes, list_drive_subsys)
        return list_energy_nd, list_n_mat, list_phi_mat, transform_matrix

    def compute_static_properties_minimal(self, list_drive_subsys, add_random,
                                          **kwargs):
        """Compute static properties of the minimal drive subsystem, as an
        approach to find quantum gate initial guesses.

        Args:
            list_drive_subsys: list of driving subsystem, each element is a list
                of qubit names
            add_random: If true, will add random deviations to the device parameters
            kwargs: other parameters for the Spectrum class
        """
        list_sub_energy_nd = []
        list_sub_n_mat = []
        list_sub_phi_mat = []
        for drive_subsys in list_drive_subsys:
            spec = Spectrum(self.subgraph(drive_subsys),
                            truncated_dim=2,
                            add_random=add_random,
                            **kwargs)
            list_energy_nd, list_n_mat, list_phi_mat, _ = spec.get_model_eigen_basis(
                spec.all_params, spec.graph.sorted_nodes, [drive_subsys])
            list_sub_energy_nd.append(list_energy_nd[0])
            list_sub_n_mat.append(list_n_mat[0])
            list_sub_phi_mat.append(list_phi_mat[0])

        return list_sub_energy_nd, list_sub_n_mat, list_sub_phi_mat, None

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
                        add_random: bool,
                        ar_crosstalk=None,
                        pulse_type: str = "cos",
                        minimal_approach=True):
        """Creates CR pulses in the quantum processor.

        This function supports any multipath coupling model implemented.

        `index` is a model-dependent parameter depends on `self.sorted_nodes`

        Args:
            ix_control_list: list of the control qubit index, indicates where to
                apply the drive.
            ix_target_list: list of the target qubit index, indicates the target
                qubit that matched the drive frequency.
            tg_list: list of the gate time
            add_random: If true, will add random deviations to the device parameters
            ar_crosstalk: the crosstalk matrix.  `ar[j,i]` means
                the pulse `A` applied on `i` qubit leads to `ar[j,i]*A` on `j`
                qubit.
            pulse_type: the pulse type, by default it is "cos"
            minimal_approach: if True, will use the minimal subsystem approach to
                find the initial guess for the CNOT gate. Will not compute the
                transform matrix in this case.

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
        if minimal_approach:
            list_energy_nd, list_n_mat, list_phi_mat, transform_matrix = \
                self.compute_static_properties_minimal(list_drive_subsys, add_random)
        else:
            list_energy_nd, list_n_mat, list_phi_mat, transform_matrix = \
                self.compute_static_properties(list_drive_subsys, add_random)

        for control, target, tg, drive_subsys, energy_nd, n_mat, phi_mat in zip(
                ix_control_list, ix_target_list, tg_list, list_drive_subsys,
                list_energy_nd, list_n_mat, list_phi_mat):

            s_start = (0, 0)
            name_drive = self.sorted_nodes[control]
            name_target = self.sorted_nodes[target]
            if name_drive == drive_subsys[0]:
                s_target = (0, 1)
                s_control = (1, 0)
            else:
                s_target = (1, 0)
                s_control = (0, 1)
            fd = abs(energy_nd[s_target] - energy_nd[s_start])
            f_control = abs(energy_nd[s_control] - energy_nd[s_start])
            detuning = abs(fd - f_control)
            # compute the effective coupling by n matrices and phi matrices
            ar_nd = np.array([
                n_drive[(*s_start, *s)]
                for n_drive, s in zip(n_mat, [s_control, s_target])
            ])
            ar_phi = np.array([
                phi_drive[(*s_start, *s)]
                for phi_drive, s in zip(phi_mat, [s_control, s_target])
            ])
            jc = self.get_edge_data(
                name_drive, name_target)['capacitive_coupling']['strength']
            jl = self.get_edge_data(
                name_drive, name_target)['inductive_coupling']['strength']
            j_eff = np.abs(np.prod(ar_nd) * jc + np.prod(ar_phi) * jl)
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
                self.add_node(name_drive, pulsecr=dic1)
            else:
                dic2 = self._add_crosstalk_pulse(name_drive, ar_crosstalk,
                                                 cr_pulse)
                self.add_node(name_drive, pulsecr=dic2)
        return transform_matrix

    def create_single_qubit_pulse(self,
                                  ix_qubit_list: int,
                                  tg_list: int,
                                  add_random: bool,
                                  ar_crosstalk=None,
                                  factor=0.5,
                                  pulse_type: str = "cos",
                                  minimal_approach=False):
        """Creates single qubit pulses in the quantum processor.

        This function supports any multipath coupling model implemented.

        `index` is a model-dependent parameter depends on `self.sorted_nodes`

        Args:
            ix_qubit_list: target qubit index. `ix_pair` is a model-dependent
            parameter, is the qubit index, indicates where to apply the drive.
            tg_list: the gate time
            add_random: If true, will add random deviations to the device parameters
            ar_crosstalk: the crosstalk matrix.  `ar[j,i]` means
                the pulse `A` applied on `i` qubit leads to `ar[j,i]*A` on `j`
                qubit.
            factor: the amplitude factor, driving a X gate by default.
            pulse_type: the pulse type, by default it is "cos"
            minimal_approach: if True, will use the minimal subsystem approach to
                find the initial guess for the single qubit gate. Will not compute
                the transform matrix in this case.

        Returns:
            transform_matrix
        """
        # Get the initial guess from the Hamiltonian
        list_drive_subsys = [[self.sorted_nodes[ix]] for ix in ix_qubit_list]
        if minimal_approach:
            list_energy_nd, list_n_mat, list_phi_mat, transform_matrix = \
                self.compute_static_properties_minimal(list_drive_subsys, add_random)
        else:
            list_energy_nd, list_n_mat, list_phi_mat, transform_matrix = \
                self.compute_static_properties(list_drive_subsys, add_random)

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
                self.add_node(name_drive, pulsesq=new_pulse)
            else:
                dic2 = self._add_crosstalk_pulse(name_drive, ar_crosstalk,
                                                 dic_pulse)
                self.add_node(ix_qubit, pulsesq=dic2)
        return transform_matrix
