import haiku as hk
import numpy as np
import jax.numpy as jnp

from supergrad.helper.helper import Helper
from supergrad.time_evolution import mesolve, mesolve_final_states_w_basis_trans
from supergrad.utils.utility import create_density_init, tensor
from supergrad.time_evolution.single_qubit_compensation import SingleQubitCompensation
from supergrad.scgraph.graph import SCGraph


class DensityEvolve(Helper):
    r"""Helper for constructing open quantum system time-evolution computing functions
    based on the graph which contains all information about qubits, pulses, and
    dissipation channels. This class works directly with density matrices. The functions
    constructed this way are pure and can be transformed by Jax.

    Args:
        graph (SCGraph): The graph containing both Hamiltonian and control
            parameters.
        truncated_dim (int): Desired dimension of the truncated subsystem.
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

    def _create_default_density_matrices(self) -> jnp.ndarray:
        """Create default computational-basis density matrices as a batch (B, D, D)."""
        states_config = []
        for i, node in enumerate(self.graph.sorted_nodes):
            # Coupler qubits are fixed in |0⟩ state, regular qubits use |0⟩ and |1⟩
            states_config.append([i, 1] if node in self.coupler_subsystem else [i, 2])

        rho_list, _ = create_density_init(self.dims, states_config)  # (B, D, D)
        dtype = getattr(self.hilbertspace, "dtype", jnp.complex64)
        return jnp.asarray(rho_list, dtype=dtype)

    def _prepare_initial_density_matrices(self, rho_list=None, _remove_compensation=False):
        """Prepare the initial density matrices for time evolution.

        Args:
            rho_list (jnp.ndarray, optional): Custom density matrices for evolution.
                Shape should be (B, D, D) where B is batch size and D is Hilbert space dimension.
                If None, computational-basis density matrices will be created automatically.
            _remove_compensation (bool): Whether to skip pre-compensation (for debug use).
                Default is False. When True, compensation is not applied even if enabled.

        Returns:
            None: This method modifies `self.rho_list` in place.
        """
        if rho_list is None:
            self.rho_list = self._create_default_density_matrices()
        else:
            # Validate input shape
            if rho_list.ndim not in [2, 3]:
                raise ValueError(f"rho_list must be 2D (D,D) or 3D (B,D,D), got {rho_list.ndim}D with shape {rho_list.shape}")

            # Validate that it's square matrices
            if rho_list.ndim == 2:
                if rho_list.shape[0] != rho_list.shape[1]:
                    raise ValueError(f"2D rho_list must be square, got shape {rho_list.shape}")
            else:  # 3D
                if rho_list.shape[1] != rho_list.shape[2]:
                    raise ValueError(f"3D rho_list must have square matrices, got shape {rho_list.shape}")

            self.rho_list = rho_list

        # Apply compensation if requested and not removed
        if (not _remove_compensation) and self.compensation_option in ['only_vz', 'arbit_single']:
            pre_u, _ = self._construct_compensation_function()
            if pre_u is not None:
                self.rho_list = self._sim_pre(self.rho_list, pre_u)

        # Ensure physical properties
        self.rho_list = self._ensure_trace_preservation(self.rho_list)
        self.rho_list = self._ensure_hermiticity(self.rho_list)

    def construct_hamiltonian_and_pulseshape(self):
        """Constructing the Hamiltonian and pulseshape for time evolution.

        Returns:
            (KronObj, list, float): The static Hamiltonian, the list of pair
                containing the time-dependent components of the Hamiltonian and
                corresponding pulse shape, and the maximum length of the pulse.
        """
        return self.hilbertspace.idling_hamiltonian_in_prod_basis(
        ), self.hamiltonian_component_and_pulseshape, self.pulse_endtime

    def _construct_compensation_function(self):
        """Constructing the compensation matrix for virtual compensation.

        Returns:
            Callable: the function that add compensation before and after the
                time evolution unitary.
        """
        if self.compensation_option in ['only_vz', 'arbit_single']:
            pre_u = tensor(*self.pre_unitaries)
            post_u = tensor(*self.post_unitaries)
            return pre_u, post_u
        return None, None

    def eigen_basis(self,
                    tlist=None,
                    transform_matrix=None,
                    rho_list=None,
                    _remove_compensation=False,
                    **kwargs):
        """Running the time evolution in the eigenbasis.

        Args:
            tlist: list of time steps for evolution. Required parameter.
            transform_matrix: pre-computed transform matrix,
                using when design parameters is not optimized.
            rho_list: the list of density matrices for evolution. If None,
                computational basis density matrices will be used.
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
        self._prepare_initial_density_matrices(rho_list, _remove_compensation)

        if tlist is None:
            raise ValueError("tlist parameter is required for time evolution. "
                             "Please provide a list of time points (e.g., np.linspace(0, 10, 100)).")
        else:
            self.tlist = tlist

        # apply pre-compensation if requested
        pre_u, post_u = self._construct_compensation_function()
        rho0 = self.rho_list
        if (not _remove_compensation) and (pre_u is not None):
            rho0 = self._sim_pre(rho0, pre_u)

        # evolve in eigen basis
        sim_rho = mesolve_final_states_w_basis_trans(ham,
                                                     rho0,
                                                     self.tlist,
                                                     transform_matrix=u_to_eigen,
                                                     c_ops=self.c_ops,
                                                     solver=self.solver,
                                                     options=self.options,
                                                     **kwargs)

        # Ensure Hermiticity after evolution
        sim_rho = self._ensure_hermiticity(sim_rho)

        # apply post-compensation if requested
        if (not _remove_compensation) and (post_u is not None):
            sim_rho = self._sim_post(sim_rho, post_u)

        return sim_rho

    def product_basis(self,
                      tlist=None,
                      rho_list=None,
                      _remove_compensation=False,
                      **kwargs):
        """Running the time evolution in the product basis.

        Args:
            tlist: list of time steps for evolution. Required parameter.
            rho_list: the list of density matrices for evolution. If None,
                computational basis density matrices will be used.
            _remove_compensation: whether to remove the compensation(for debug use).
            kwargs: keyword arguments will be passed to `supergrad.time_evolution.mesolve`
        """
        # Single qubit product basis
        # construct static hamiltonian
        ham_static = self.hilbertspace.idling_hamiltonian_in_prod_basis()

        ham = [ham_static, *self.hamiltonian_component_and_pulseshape]
        self._prepare_initial_density_matrices(rho_list, _remove_compensation)

        if tlist is None:
            raise ValueError("tlist parameter is required for time evolution. "
                             "Please provide a list of time points (e.g., np.linspace(0, 10, 100)).")
        else:
            self.tlist = tlist

        # apply pre-compensation if requested
        pre_u, post_u = self._construct_compensation_function()
        rho0 = self.rho_list
        if (not _remove_compensation) and (pre_u is not None):
            rho0 = self._sim_pre(rho0, pre_u)

        # evolve in product basis
        sim_rho = mesolve_final_states_w_basis_trans(ham,
                                                     rho0,
                                                     self.tlist,
                                                     transform_matrix=None,
                                                     c_ops=self.c_ops,
                                                     solver=self.solver,
                                                     options=self.options,
                                                     **kwargs)

        # Ensure Hermiticity after evolution
        sim_rho = self._ensure_hermiticity(sim_rho)

        # apply post-compensation if requested
        if (not _remove_compensation) and (post_u is not None):
            sim_rho = self._sim_post(sim_rho, post_u)

        return sim_rho

    def eigen_basis_trajectory(self,
                               tlist=None,
                               rho_list=None,
                               transform_matrix=None,
                               _remove_compensation=False,
                               **kwargs):
        """Computing the time evolution trajectory in the eigen basis.

        Args:
            tlist: list of time steps.
            rho_list: the list of density matrices for evolution. If None,
                computational basis density matrices will be used.
            transform_matrix: pre-computed transform matrix,
                using when design parameters is not optimized.
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

        self._prepare_initial_density_matrices(rho_list, _remove_compensation)

        rho0 = self.rho_list  # (B,D,D)

        # Pre compensation
        pre_u, post_u = self._construct_compensation_function()
        if (not _remove_compensation) and (pre_u is not None):
            rho0 = self._sim_pre(rho0, pre_u)  # (B,D,D)

        # basis forward: product -> eigen
        rho0_e = self._sim_pre(rho0, u_to_eigen)  # (B,D,D)

        if tlist is None:
            raise ValueError("tlist parameter is required for time evolution. "
                             "Please provide a list of time points (e.g., np.linspace(0, 10, 100)).")
        else:
            self.tlist = tlist

        traj_e = mesolve(ham,
                         rho0_e,
                         self.tlist,
                         c_ops=self.c_ops,
                         solver=self.solver,
                         options=self._normalize_astep(self.options, self.tlist),
                         **kwargs)

        # basis back: eigen -> product
        traj_p = self._sim_post(traj_e, u_to_eigen)

        # Post compensation
        if (not _remove_compensation) and (post_u is not None):
            traj_p = self._sim_post(traj_p, post_u)

        # Handle diagonal extraction for both single and batch density matrices
        pop0 = jnp.real(jnp.diagonal(self.rho_list, axis1=-2, axis2=-1))
        if pop0.ndim == 1:  # single rho
            idx = int(jnp.argmax(pop0))
            traj_col = traj_p[:, idx, :]          # (T,D)
        else:               # batch rho
            idxs = jnp.argmax(pop0, axis=1)     # (B,)
            traj_col = traj_p[jnp.arange(traj_p.shape[0]), :, idxs, :]  # (B,T,D)

        return traj_p, traj_col

    def product_basis_trajectory(self,
                                 tlist=None,
                                 rho_list=None,
                                 _remove_compensation=False,
                                 **kwargs):
        """Computing the time evolution trajectory in the product basis.

        Args:
            tlist: list of time steps.
            rho_list: the list of density matrices for evolution. If None,
                computational basis density matrices will be used.
            _remove_compensation: whether to remove the compensation(for debug use).
            kwargs: keyword arguments will be passed to `supergrad.time_evolution.mesolve`
        """
        # Single qubit product basis
        # construct static hamiltonian
        ham_static = self.hilbertspace.idling_hamiltonian_in_prod_basis()

        ham = [ham_static, *self.hamiltonian_component_and_pulseshape]
        self._prepare_initial_density_matrices(rho_list, _remove_compensation)
        rho0 = self.rho_list  # (B,D,D)

        # Pre compensation
        pre_u, post_u = self._construct_compensation_function()
        if (not _remove_compensation) and (pre_u is not None):
            rho0 = self._sim_pre(rho0, pre_u)

        if tlist is None:
            raise ValueError("tlist parameter is required for time evolution. "
                             "Please provide a list of time points (e.g., np.linspace(0, 10, 100)).")
        else:
            self.tlist = tlist

        traj = mesolve(ham,
                       rho0,
                       self.tlist,
                       c_ops=self.c_ops,
                       solver=self.solver,
                       options=self._normalize_astep(self.options, self.tlist),
                       **kwargs)  # (T,D,D) or (B,T,D,D)

        # Post compensation
        if (not _remove_compensation) and (post_u is not None):
            traj = self._sim_post(traj, post_u)

        # Handle diagonal extraction for both single and batch density matrices
        pop0 = jnp.real(jnp.diagonal(self.rho_list, axis1=-2, axis2=-1))
        if pop0.ndim == 1:  # single rho
            idx = int(jnp.argmax(pop0))
            traj_col = traj[:, idx, :]          # (T,D)
        else:               # batch rho
            idxs = jnp.argmax(pop0, axis=1)     # (B,)
            traj_col = traj[jnp.arange(traj.shape[0]), :, idxs, :]  # (B,T,D)

        return traj, traj_col

    def _sim_pre(self, rho, U):
        """U ρ U†, works for rho with shape (..., D, D)
        In open system evolution, rho could be of shape (D,D), (B,D,D), (T,D,D) or (B,T,D,D).
        (D,D) for single density matrix, (B,D,D) for batch of density matrices,
        (T,D,D) for trajectory of single density matrix, (B,T,D,D) for trajectory of batch of density matrices.
        """
        # Validate input shape - rho should have at least 2 dimensions and last two should be square
        if rho.ndim < 2:
            raise ValueError(f"rho must have at least 2 dimensions, got {rho.ndim}D with shape {rho.shape}")

        if rho.shape[-1] != rho.shape[-2]:
            raise ValueError(f"Last two dimensions of rho must be square, got shape {rho.shape}")

        return jnp.einsum('ij,...jk,kl->...il', U, rho, U.conj().T)

    def _sim_post(self, rho, U):
        """U† ρ U, works for rho with shape (..., D, D)
        In open system evolution, rho could be of shape (D,D), (B,D,D), (T,D,D) or (B,T,D,D).
        (D,D) for single density matrix, (B,D,D) for batch of density matrices,
        (T,D,D) for trajectory of single density matrix, (B,T,D,D) for trajectory of batch of density matrices.
        """
        return jnp.einsum('ij,...jk,kl->...il', U.conj().T, rho, U)

    def _ensure_trace_preservation(self, rho):
        """Ensure density matrix has unit trace.
        """
        if rho.ndim == 2:
            trace = jnp.trace(rho)
            return rho / trace
        else:
            trace = jnp.trace(rho, axis1=-2, axis2=-1)
            return rho / trace[..., None, None]

    def _ensure_hermiticity(self, rho):
        """Ensure density matrix remains Hermitian.
        """
        if rho.ndim == 2:
            return (rho + jnp.conj(rho).T) / 2
        elif rho.ndim == 3:
            # For batch density matrices (B, D, D), transpose only the last two dimensions
            return (rho + jnp.conj(rho).transpose(0, 2, 1)) / 2
        else:
            # For higher dimensional arrays, transpose the last two dimensions
            return (rho + jnp.conj(rho).transpose(-1, -2)) / 2

    def _normalize_astep(self, options, tlist):
        """Normalize astep with safety checks.
        """
        if len(tlist) <= 1:
            return options
        options = options.copy()
        if options.get("astep", 0) > 0:
            options["astep"] = max(1, options["astep"] // (len(tlist) - 1))
        return options
