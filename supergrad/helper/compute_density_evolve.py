import haiku as hk
import numpy as np
import jax.numpy as jnp

from supergrad.helper.helper import Helper
from supergrad.time_evolution import mesolve, mesolve_final_states_w_basis_trans
from supergrad.utils.utility import create_state_init, tensor
from supergrad.time_evolution.single_qubit_compensation import SingleQubitCompensation
from supergrad.scgraph.graph import SCGraph


class DensityEvolve(Helper):
    """Helper for constructing open quantum system time-evolution computing functions
    based on the graph which contains all information about qubits, pulses, and
    dissipation channels. Uses Lindblad master equation via mesolve.

    Args:
        graph (SCGraph): The graph containing both Hamiltonian and control
            parameters.
        truncated_dim (int):  Desired dimension of the truncated subsystem.
            Note that this applies to all qubits on the graph. One could
            set local configuration for each qubit in the `graph.nodes['qubit']['arguments']`
            to allow different dimension.
        add_random (bool): If true, will add random deviations to the device
            parameters according to the graph.
        share_params (bool): Share device parameters between the qubits that
            have the same shared_param_mark. This is used only for gradient
            computation. One must define `shared_param_mark` in the
            `graph.nodes['qubit']['shared_param_mark']`.
        unify_coupling (bool): Let all couplings in the quantum system be the same.
            TODO: if set to true, which coupling will be used to do the computation?
        coupler_subsystem: Qubits which we set to |0⟩ initially and at the end.
            TODO: make this more general.
        compensation_option: Set single qubit compensation strategy, should be in
            ['no_comp', 'only_vz', 'arbit_single']. 'no_comp' means we do no
            compensation. 'only_vz' means we will do single-qubit Z-rotation
            before and after the time evolution. 'arbit_single' means we will do
            arbitrary single-qubit rotation before and after the time evolution.
        solver: the type of time evolution solver, should be in ['ode_expm', 'odeint'].
        options: the arguments will be passed to solver.
            See `supergrad.time_evolution.mesolve`.
        c_ops: List of collapse operators for dissipation channels.
    """

    def __init__(self,
                 graph,
                 truncated_dim=5,
                 add_random=True,
                 share_params=False,
                 unify_coupling=False,
                 compensation_option='no_comp',
                 coupler_subsystem=[],
                 solver='ode_expm',
                 options={
                     'astep': 2000,
                     'trotter_order': 1
                 },
                 c_ops=None,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.graph: SCGraph = graph
        self.truncated_dim = truncated_dim
        self.add_random = add_random
        self.share_params = share_params
        self.unify_coupling = unify_coupling
        self.coupler_subsystem = coupler_subsystem
        if compensation_option not in ['no_comp', 'only_vz', 'arbit_single']:
            raise NotImplementedError(
                f'Strategy {compensation_option} has not been implemented.')
        self.compensation_option = compensation_option
        if solver not in ['ode_expm', 'odeint']:
            raise NotImplementedError(
                f'Solver {solver} has not been implemented.')
        self.solver = solver
        self.options = options
        self.c_ops = c_ops if c_ops is not None else []
        self.rho_list = None

    def _init_quantum_system(self):
        self.hilbertspace = self.graph.convert_graph_to_quantum_system(
            add_random=self.add_random,
            share_params=self.share_params,
            unify_coupling=self.unify_coupling,
            truncated_dim=self.truncated_dim,
            **self.kwargs)
        self.hamiltonian_component_and_pulseshape, self.pulse_endtime = self.graph.convert_graph_to_pulse_lst(
            self.hilbertspace, modulate_wave=True, **self.kwargs)
        if self.compensation_option in ['only_vz', 'arbit_single']:
            sqc = SingleQubitCompensation(self.graph, self.compensation_option,
                                          self.coupler_subsystem)
            self.pre_unitaries, self.post_unitaries = sqc.create_unitaries()

    @property
    def pulse_params(self):
        """The pulse parameters dictionary."""
        pulse_params = self.graph.convert_graph_to_pulse_parameters_haiku()
        if self.compensation_option != 'no_comp':
            # Add initial compensation if not provided in the graph
            initial_comp = self.graph.convert_graph_to_comp_initial_guess(
                self.compensation_option)
            pulse_params = hk.data_structures.merge(initial_comp, pulse_params)
        return pulse_params

    @property
    def all_params(self):
        """The static parameters dictionary."""
        all_params = self.graph.convert_graph_to_parameters_haiku(
            self.share_params, self.unify_coupling)
        if self.compensation_option != 'no_comp':
            # Add initial compensation if not provided in the graph
            initial_comp = self.graph.convert_graph_to_comp_initial_guess(
                self.compensation_option)
            all_params = hk.data_structures.merge(initial_comp, all_params)
        return all_params

    def _prepare_initial_density_matrices(self, psi_list=None, _remove_compensation=False):
        """Prepare the initial density matrices for time evolution.

        Converts state vectors |ψ⟩ to density matrices |ψ⟩⟨ψ| and applies pre-compensation.

        Args:
            psi_list (list, optional): The list of states for evolution. If None,
                all the states in computation basis will be used.
            _remove_compensation (bool): Whether to skip pre-compensation (for debug use).
        """
        if psi_list is None:
            states_config = []
            for i, node in enumerate(self.graph.sorted_nodes):
                if node in self.coupler_subsystem:
                    states_config.append([i, 1])
                else:
                    states_config.append([i, 2])
            psi_list, _ = create_state_init(self.dims, states_config)

        # Convert state vectors to density matrices: |ψ⟩ → |ψ⟩⟨ψ|
        if isinstance(psi_list, (list, tuple)):
            # Handle list of state vectors
            self.rho_list = jnp.array([jnp.outer(psi, jnp.conj(psi)) for psi in psi_list], dtype=jnp.complex128)
        elif isinstance(psi_list, np.ndarray):
            if psi_list.ndim == 2:  # Single state vector
                self.rho_list = jnp.outer(psi_list, jnp.conj(psi_list)).astype(jnp.complex128)
            elif psi_list.ndim == 3:  # Multiple state vectors
                self.rho_list = jnp.array([jnp.outer(psi, jnp.conj(psi)) for psi in psi_list], dtype=jnp.complex128)
            else:
                raise ValueError(f"Invalid psi_list shape: {psi_list.shape}")
        else:
            raise ValueError(f"Invalid psi_list type: {type(psi_list)}")

        # Apply pre-compensation if available
        if not _remove_compensation and self.compensation_option != 'no_comp':
            pre_u, _ = self._construct_compensation_unitaries()
            if pre_u is not None:
                # Apply pre-compensation to each density matrix: U_pre @ rho @ U_pre†
                self.rho_list = self._rho_similar_transform(self.rho_list, pre_u)

    def construct_hamiltonian_and_pulseshape(self):
        """Constructing the Hamiltonian and pulseshape for time evolution.

        Returns:
            (KronObj, list, float): The static Hamiltonian, the list of pair
                containing the time-dependent components of the Hamiltonian and
                corresponding pulse shape, and the maximum length of the pulse.
        """
        return self.hilbertspace.idling_hamiltonian_in_prod_basis(
        ), self.hamiltonian_component_and_pulseshape, self.pulse_endtime

    def _construct_compensation_unitaries(self):
        """Construct pre and post compensation unitaries.

        Returns:
            tuple: (pre_u, post_u) or (None, None) if no compensation
        """
        if self.compensation_option in ['only_vz', 'arbit_single']:
            pre_u = tensor(*self.pre_unitaries)
            post_u = tensor(*self.post_unitaries)
            return pre_u, post_u
        return None, None

    def _rho_similar_transform(self, rho, U):
        """Apply similarity transformation U† ρ U to density matrix.

        Supports multiple tensor shapes:
        - (N,N): Single density matrix
        - (B,N,N): Batch of density matrices
        - (T,B,N,N): Time series of batches

        Args:
            rho: Density matrix tensor
            U: Unitary transformation matrix

        Returns:
            Transformed density matrix tensor
        """
        if U is None:
            return rho

        if rho.ndim == 2:
            return U.conj().T @ rho @ U
        elif rho.ndim == 3:
            # (B,N,N)
            return jnp.einsum('ij,bjk,lk->bil', U.conj().T, rho, U)
        elif rho.ndim == 4:
            # (T,B,N,N)
            return jnp.einsum('ij,tbjk,lk->tbil', U.conj().T, rho, U)
        else:
            raise ValueError(f"Unsupported rho ndim: {rho.ndim}")

    def eigen_basis(self,
                    transform_matrix=None,
                    psi_list=None,
                    _remove_compensation=False,
                    **kwargs):
        """Running the time evolution in the eigenbasis.

        Args:
            transform_matrix: pre-computed transform matrix,
                using when design parameters is not optimized.
            psi_list: the list of states for evolution. If None, all the states
                in computation basis will be used.
            _remove_compensation: whether to remove the compensation(for debug use).
            kwargs: keyword arguments will be passed to `supergrad.time_evolution.mesolve`
        """
        if transform_matrix is None:
            u_to_eigen = self.hilbertspace.compute_transform_matrix()
        else:
            u_to_eigen = transform_matrix

        # Single qubit product basis
        # construct static hamiltonian
        ham_static = self.hilbertspace.idling_hamiltonian_in_prod_basis()

        ham = [ham_static, *self.hamiltonian_component_and_pulseshape]
        self._prepare_initial_density_matrices(psi_list, _remove_compensation)
        self.tlist = [0, self.pulse_endtime]

        # Use mesolve_final_states_w_basis_trans for density matrices
        sim_rho = mesolve_final_states_w_basis_trans(ham,
                                                     self.rho_list,
                                                     self.tlist,
                                                     transform_matrix=u_to_eigen,
                                                     c_ops=self.c_ops,
                                                     solver=self.solver,
                                                     options=self.options,
                                                     **kwargs)
        if not _remove_compensation:
            _, post_u = self._construct_compensation_unitaries()
            if post_u is not None:
                # Apply post-compensation: U_post† @ rho @ U_post
                sim_rho = self._rho_similar_transform(sim_rho, post_u)
        return sim_rho

    def product_basis(self,
                      psi_list=None,
                      _remove_compensation=False,
                      **kwargs):
        """Running the time evolution in the product basis.

        Args:
            psi_list: the list of states for evolution. If None, all the states
                in computation basis will be used.
            _remove_compensation: whether to remove the compensation(for debug use).
            kwargs: keyword arguments will be passed to `supergrad.time_evolution.mesolve`
        """
        # Single qubit product basis
        # construct static hamiltonian
        ham_static = self.hilbertspace.idling_hamiltonian_in_prod_basis()

        ham = [ham_static, *self.hamiltonian_component_and_pulseshape]
        self._prepare_initial_density_matrices(psi_list, _remove_compensation)
        self.tlist = [0, self.pulse_endtime]

        # Use mesolve_final_states_w_basis_trans for density matrices
        sim_rho = mesolve_final_states_w_basis_trans(ham,
                                                     self.rho_list,
                                                     self.tlist,
                                                     transform_matrix=None,
                                                     c_ops=self.c_ops,
                                                     solver=self.solver,
                                                     options=self.options,
                                                     **kwargs)
        if not _remove_compensation:
            _, post_u = self._construct_compensation_unitaries()
            if post_u is not None:
                # Apply post-compensation: U_post† @ rho @ U_post
                sim_rho = self._rho_similar_transform(sim_rho, post_u)
        return sim_rho

    def eigen_basis_trajectory(self,
                               tlist=None,
                               psi_list=None,
                               transform_matrix=None,
                               **kwargs):
        """Computing the time evolution trajectory in the eigen basis.

        Args:
            tlist: list of time steps.
            psi_list: the list of states for evolution. If None, all the states
                in computation basis will be used.
            transform_matrix: pre-computed transform matrix,
                using when design parameters is not optimized.
            kwargs: keyword arguments will be passed to `supergrad.time_evolution.mesolve`
        """
        if transform_matrix is None:
            u_to_eigen = self.hilbertspace.compute_transform_matrix()
        else:
            u_to_eigen = transform_matrix

        # Single qubit product basis
        # construct static hamiltonian
        ham_static = self.hilbertspace.idling_hamiltonian_in_prod_basis()

        ham = [ham_static, *self.hamiltonian_component_and_pulseshape]

        transform_matrix = u_to_eigen
        self._prepare_initial_density_matrices(psi_list, _remove_compensation=False)

        # Transform density matrices to eigenbasis: U @ rho @ U†
        rho_list_eigen = transform_matrix @ self.rho_list @ transform_matrix.conj().T

        if tlist is None:
            self.tlist = [0, self.pulse_endtime]
        else:
            self.tlist = tlist
        # Change astep as it is used between each steps
        options = self.options.copy()
        options["astep"] = options["astep"] // (len(self.tlist) - 1) + 1

        res = mesolve(ham,
                      rho_list_eigen,
                      self.tlist,
                      c_ops=self.c_ops,
                      solver=self.solver,
                      options=options,
                      **kwargs)

        # Transform back to product basis: U† @ rho @ U
        states = jnp.conj(transform_matrix).T @ res @ transform_matrix

        # Extract the states in the computational space
        # For density matrices, we need to extract diagonal elements
        pop = jnp.real(jnp.diagonal(states, axis1=-2, axis2=-1))
        tuple_ar_ix = tuple(np.argmax(pop, axis=1).flatten())

        return states, states[:, tuple_ar_ix, :]

    def product_basis_trajectory(self, tlist=None, psi_list=None, **kwargs):
        """Computing the time evolution trajectory in the product basis.

        Args:
            tlist: list of time steps.
            psi_list: the list of states for evolution. If None, all the states
                in computation basis will be used.
            kwargs: keyword arguments will be passed to `supergrad.time_evolution.mesolve`
        """
        # Single qubit product basis
        # construct static hamiltonian
        ham_static = self.hilbertspace.idling_hamiltonian_in_prod_basis()

        ham = [ham_static, *self.hamiltonian_component_and_pulseshape]
        self._prepare_initial_density_matrices(psi_list, _remove_compensation=False)
        rho_list = self.rho_list

        if tlist is None:
            self.tlist = [0, self.pulse_endtime]
        else:
            self.tlist = tlist
        # Change astep as it is used between each steps
        options = self.options.copy()
        options["astep"] = options["astep"] // (len(self.tlist) - 1) + 1

        res = mesolve(ham,
                      rho_list,
                      self.tlist,
                      c_ops=self.c_ops,
                      solver=self.solver,
                      options=options,
                      **kwargs)

        # Extract the states in the computational space
        # For density matrices, we need to extract diagonal elements
        pop = jnp.real(jnp.diagonal(res, axis1=-2, axis2=-1))
        tuple_ar_ix = tuple(np.argmax(pop, axis=1).flatten())

        return res, res[:, tuple_ar_ix, :]
