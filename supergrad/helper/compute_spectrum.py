from supergrad.helper.helper import Helper
from supergrad.scgraph.graph import SCGraph


class Spectrum(Helper):
    """Helper for constructing spectrum computing function of the quantum system
    based on the graph which contain all information about qubits and pulse.
    The functions constructed by this way are pure and transformable by JAX.

    Args:
        graph (SCGraph): The graph containing all Hamiltonian parameters.
        truncated_dim (int): desired dimension of the truncated qubit subsystem
        add_random (bool): If true, will add random deviations to the device parameters
        share_params (bool): Share device parameters between the qubits that
            have the same shared_param_mark. This is used only for gradient computation.
            One must define `shared_param_mark` in the `graph.nodes['qubit']['shared_param_mark']`.
        unify_coupling (bool): Let all couplings in the quantum system be the same.
            TODO: if set to true, which coupling will be used to do the computation?
    """

    def __init__(self,
                 graph,
                 truncated_dim=5,
                 add_random=True,
                 share_params=False,
                 unify_coupling=False,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.graph: SCGraph = graph
        self.truncated_dim = truncated_dim
        self.add_random = add_random
        self.share_params = share_params
        self.unify_coupling = unify_coupling

    def _init_quantum_system(self):
        self.hilbertspace = self.graph.convert_graph_to_quantum_system(
            add_random=self.add_random,
            share_params=self.share_params,
            unify_coupling=self.unify_coupling,
            truncated_dim=self.truncated_dim,
        )

    def energy_tensor(self, greedy_assign=True):
        """Return the eigenenergy of quantum system in tensor form.

        Args:
            greedy_assign (bool): if True, use greedy assignment mode
                The greedy assignment mode ignores the issue "same state be assigned
                multiple times", due to the weak coupling assumption."""

        return self.hilbertspace.compute_energy_map(greedy_assign)

    def get_model_eigen_basis(self,
                              list_qubit_name,
                              list_drive_subsys,
                              greedy_assign=True):
        """Compute the quantum system properties in multi-qubit eigen basis.
        For example, energy tensor, n operator in eigen basis, phi operator and
        transform matrix will be computed in the same pure function.

        Args:
            list_qubit_name: qubit name list of string.
            list_drive_subsys: driving subsystem list of string.
            greedy_assign (bool): if True, use greedy assignment mode
                The greedy assignment mode ignores the issue "same state be assigned
                multiple times", due to the weak coupling assumption.
        """
        energy_nd = self.hilbertspace.compute_energy_map(greedy_assign)
        transform_matrix = self.hilbertspace.compute_transform_matrix()
        list_sub_energy_nd = []
        list_sub_n_mat = []
        list_sub_phi_mat = []
        for drive_subsys in list_drive_subsys:
            list_n_mat = [
                self.hilbertspace.transform_operator(
                    self.hilbertspace[name].n_operator).reshape(self.dims * 2)
                for name in drive_subsys
            ]
            list_phi_mat = [
                self.hilbertspace.transform_operator(
                    self.hilbertspace[name].phi_operator).reshape(
                        self.dims * 2) for name in drive_subsys
            ]
            # generate the slice index for drive_subsys, all the qubit not in the
            # drive_subsys will be set to 0
            ix = tuple([
                slice(None) if name in drive_subsys else 0
                for name in list_qubit_name
            ])
            if not any(ix):
                raise ValueError(
                    'The tensor slice index is all 0, please check the input.')
            sub_energy_nd = energy_nd[ix]
            list_sub_energy_nd.append(sub_energy_nd - sub_energy_nd.min())
            list_sub_n_mat.append([x[ix + ix] for x in list_n_mat])
            list_sub_phi_mat.append([x[ix + ix] for x in list_phi_mat])
        return list_sub_energy_nd, list_sub_n_mat, list_sub_phi_mat, transform_matrix
