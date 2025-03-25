import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

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
            **self.kwargs)

    def energy_tensor(self,
                      greedy_assign=True,
                      enhanced_assign=False,
                      return_enhanced_aux_val=False,
                      **kwargs):
        """Return the eigenenergy of quantum system in tensor form.

        Args:
            greedy_assign (bool): if True, use greedy assignment mode
                The greedy assignment mode ignores the issue "same state be assigned
                multiple times", due to the weak coupling assumption.
            enhanced_assign: use enhanced assignment
                if 'spin_projective', projective the qudit to the spin system,
                assign states by the overlap versus the computational basis,
                thus the energy map will be more accurate for the computational
                basis.
                if list of SCGraph, assign the states by the Continuum Adjust
                Coupling Tracking. Dependent assignments will be performed
                along the SCGraph list. Each assignment compute the overlap
                between the current state and the previous basis(using the same
                dimension of Hilbert space)
                if ndarray, we assume it's a set of eigenvector in the ndindex order.
                Ans directly assign the states by computing the overlap versus
                the eigenvector.
            return_enhanced_aux_val(bool): if True, return the auxiliary energy
                tensor value along the giving SCGraph list
        """
        if isinstance(enhanced_assign, list) and all(
                isinstance(item, SCGraph) for item in enhanced_assign):
            # assignment along the giving SCGraph list
            def body(carry, scgraph):
                sweep_self = Spectrum(unflatten(scgraph),
                                      truncated_dim=self.truncated_dim,
                                      add_random=self.add_random,
                                      share_params=self.share_params,
                                      unify_coupling=self.unify_coupling,
                                      *self.args,
                                      **self.kwargs)
                sweep_val, carry = sweep_self.energy_tensor(
                    sweep_self.all_params,
                    greedy_assign,
                    enhanced_assign=carry,
                    return_eigvec=True)
                return carry, sweep_val

            init_self = Spectrum(enhanced_assign[0],
                                 truncated_dim=self.truncated_dim,
                                 add_random=self.add_random,
                                 share_params=self.share_params,
                                 unify_coupling=self.unify_coupling,
                                 *self.args,
                                 **self.kwargs)
            aux_val, enhanced_assign_data = init_self.energy_tensor(
                init_self.all_params, greedy_assign, return_eigvec=True)
            if enhanced_assign[1:]:
                # ravel the SCGraph for lax.scan
                _, unflatten = ravel_pytree(enhanced_assign[0])
                graph_par = [
                    ravel_pytree(scg)[0] for scg in enhanced_assign[1:]
                ]
                enhanced_assign_data, sweep_aux_val = jax.lax.scan(
                    body, enhanced_assign_data, jnp.array(graph_par))
                aux_val = jnp.concatenate((aux_val[jnp.newaxis,
                                                   Ellipsis], sweep_aux_val))
            else:
                aux_val = aux_val[jnp.newaxis, Ellipsis]
        elif enhanced_assign == "spin_projective":
            # create a sub-Hilbertspace for the spin-chain
            new_self = Spectrum(self.graph,
                                truncated_dim=2,
                                add_random=self.add_random,
                                share_params=self.share_params,
                                unify_coupling=self.unify_coupling,
                                *self.args,
                                **self.kwargs)
            _, enhanced_assign_data = new_self.energy_tensor(
                new_self.all_params, greedy_assign, return_eigvec=True)
        elif isinstance(enhanced_assign, (jax.Array, np.ndarray)):
            enhanced_assign_data = enhanced_assign
        else:
            enhanced_assign_data = None

        output = self.hilbertspace.compute_energy_map(greedy_assign,
                                                      enhanced_assign_data,
                                                      **kwargs)
        if return_enhanced_aux_val:
            assert isinstance(enhanced_assign, list) and all(
                isinstance(item, SCGraph) for item in enhanced_assign
            ), "return_enhanced_aux_val only valid if enhanced_assign is a list of SCGraph"
            return jnp.concatenate((aux_val, output[jnp.newaxis, Ellipsis]))
        else:
            return output

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
                    self.hilbertspace[name].phi_operator).reshape(self.dims * 2)
                for name in drive_subsys
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
