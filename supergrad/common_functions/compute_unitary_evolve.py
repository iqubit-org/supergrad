import haiku as hk
import numpy as np
import jax.numpy as jnp

import supergrad
from supergrad.time_evolution import sesolve, sesolve_final_states_w_basis_trans
from supergrad.utils import create_state_init
from supergrad.time_evolution.single_qubit_compensation import SingleQubitCompensation
from supergrad.quantum_system.graph import BaseGraph


class Evolve(supergrad.Helper):
    """Helper for constructing time-evolution computing functions based on the
    graph which contains all information about qubits and pulses. The functions
    constructed this way are pure and can be transformed by Jax.

    Args:
        graph (BaseGraph): The graph containing both Hamiltonian and control
            parameters.
        truncated_dim (int):  Desired dimension of the truncated subsystem.
            Note that this applies to all qubits on the graph. One could
            set local configuration for each qubit in the `graph.nodes['qubit']['arguments']`
            to allow different dimension.
        enable_var (bool): If true, will add random variance to the device
            parameters according to the graph.
        share_params (bool): Share device parameters between the qubits that
            have the same shared_param_mark. This is used only for gradient
            computation. One must define `shared_param_mark` in the
            `graph.nodes['qubit']['shared_param_mark']`.
        unify_coupling (bool): Let all couplings in the quantum system be the same.
            TODO: if set to true, which coupling will be used to do the computation?
        coupler_subsystem: Qubits which we set to |0> initially and at the end.
            TODO: make this more general.
        compensation_option: Set single qubit compensation strategy, should be in
            ['no_comp', 'only_vz', 'arbit_single']. 'no_comp' means we do no
            compenstaion. 'only_vz' means we will do single-qubit Z-rotation
            before and after the time evolution. 'arbit_single' means we will do
            arbitrary single-qubit rotation before and after the time evolution.
        solver: the type of time evolution solver, should be in ['ode_expm', 'odeint'].
        options: the arguments will be passed to solver.
            See `supergrad.time_evolution.sesolve`.
    """

    def __init__(self,
                 graph,
                 truncated_dim=5,
                 enable_var=True,
                 share_params=False,
                 unify_coupling=False,
                 compensation_option='no_comp',
                 coupler_subsystem=[],
                 solver='ode_expm',
                 options={
                     'astep': 2000,
                     'trotter_order': 1
                 },
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.graph: BaseGraph = graph
        self.truncated_dim = truncated_dim
        self.enable_var = enable_var
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
        self.static_params = self.graph.convert_graph_to_parameters_haiku(
            self.share_params, self.unify_coupling)
        self.pulse_params = self.graph.convert_graph_to_pulse_parameters_haiku()
        if self.compensation_option == 'no_comp':
            self.initial_comp = None
        else:
            self.initial_comp = self.graph.convert_graph_to_comp_initial_guess(
                self.compensation_option)
            self.static_params = hk.data_structures.merge(
                self.initial_comp, self.static_params)
            self.pulse_params = hk.data_structures.merge(
                self.initial_comp, self.pulse_params)

    def _init_quantum_system(self):
        self.hilbertspace = self.graph.convert_graph_to_quantum_system(
            enable_var=self.enable_var,
            share_params=self.share_params,
            unify_coupling=self.unify_coupling,
            truncated_dim=self.truncated_dim)
        self.dims = self.hilbertspace.dim
        states_config = []
        for i, node in enumerate(self.graph.sorted_nodes):
            if node in self.coupler_subsystem:
                states_config.append([i, 1])
            else:
                states_config.append([i, 2])
        self.psi_list, _ = create_state_init(self.dims, states_config)
        self.hamiltonian_component_and_pulseshape, self.pulse_endtime = self.graph.convert_graph_to_pulse_lst(
            self.hilbertspace, modulate_wave=True)
        if self.compensation_option in ['only_vz', 'arbit_single']:
            sqc = SingleQubitCompensation(self.graph, self.compensation_option,
                                          self.coupler_subsystem)
            self.pre_unitaries, self.post_unitaries = sqc.create_unitaries()

    def construct_hamiltonian_and_pulseshape(self):
        """Constructing the Hamiltonian and pulseshape for time evolution.
        """
        ham_static = self.hilbertspace.idling_hamiltonian_in_prod_basis()

        return ham_static, self.hamiltonian_component_and_pulseshape

    def eigen_basis(self, transform_matrix=None, **kwargs):
        """Running the time evolution in the eigenbasis.

        Args:
            transform_matrix: pre-computed transform matrix,
                using when design parameters is not optimized.
            kwargs: keyword arguments will be passed to `supergrad.time_evolution.sesolve`
        """
        if transform_matrix is None:
            u_to_eigen = self.hilbertspace.compute_transform_matrix()
        else:
            u_to_eigen = transform_matrix

        # Single qubit product basis
        # construct static hamiltonian
        ham_static = self.hilbertspace.idling_hamiltonian_in_prod_basis()

        ham = [ham_static, *self.hamiltonian_component_and_pulseshape]
        self.tlist = [0, self.pulse_endtime]
        sim_u = sesolve_final_states_w_basis_trans(ham,
                                                   self.psi_list,
                                                   self.tlist,
                                                   transform_matrix=u_to_eigen,
                                                   solver=self.solver,
                                                   options=self.options,
                                                   **kwargs)
        if self.compensation_option in ['only_vz', 'arbit_single']:
            if len(self.pre_unitaries) == 1:
                sim_u = self.pre_unitaries[0] @ sim_u @ self.post_unitaries[0]
            else:
                pre_u = supergrad.tensor(*self.pre_unitaries)
                post_u = supergrad.tensor(*self.post_unitaries)
                sim_u = post_u @ sim_u @ pre_u
        return sim_u

    def product_basis(self, **kwargs):
        """Running the time evolution in the product basis.

        Args:
            kwargs: keyword arguments will be passed to `supergrad.time_evolution.sesolve`
        """
        # Single qubit product basis
        # construct static hamiltonian
        ham_static = self.hilbertspace.idling_hamiltonian_in_prod_basis()

        ham = [ham_static, *self.hamiltonian_component_and_pulseshape]
        self.tlist = [0, self.pulse_endtime]
        sim_u = sesolve_final_states_w_basis_trans(ham,
                                                   self.psi_list,
                                                   self.tlist,
                                                   transform_matrix=None,
                                                   solver=self.solver,
                                                   options=self.options,
                                                   **kwargs)
        if self.compensation_option in ['only_vz', 'arbit_single']:
            pre_u = supergrad.tensor(*self.pre_unitaries)
            post_u = supergrad.tensor(*self.post_unitaries)
            sim_u = post_u @ sim_u @ pre_u
        return sim_u

    def eigen_basis_trajectory(self,
                               tlist=None,
                               transform_matrix=None,
                               **kwargs):
        """Computing the time evolution trajectory in the eigen basis.

        Args:
            tlist: list of time steps.
            transform_matrix: pre-computed transform matrix,
                using when design parameters is not optimized.
            kwargs: keyword arguments will be passed to `supergrad.time_evolution.sesolve`
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
        psi_list = transform_matrix @ self.psi_list

        if tlist is None:
            self.tlist = [0, self.pulse_endtime]
        else:
            self.tlist = tlist
        # Change astep as it is used between each steps
        options = self.options.copy()
        options["astep"] = options["astep"] // (len(self.tlist) - 1)
        res = sesolve(ham,
                      psi_list,
                      self.tlist,
                      solver=self.solver,
                      options=options)
        # Replace the last dimension with initial states
        # Now [time, comp, psi_init]
        res = jnp.swapaxes(res, 0, 3)[0]
        states = jnp.conj(transform_matrix).T @ res

        # Extract the states in the computational space
        pop = jnp.abs(psi_list).real**2
        tuple_ar_ix = tuple(np.argmax(pop, axis=1).flatten())

        return states, states[:, tuple_ar_ix, :]

    def product_basis_trajectory(self, tlist=None, **kwargs):
        """Computing the time evolution trajectory in the product basis.

        Args:
            tlist: list of time steps.
            kwargs: keyword arguments will be passed to `supergrad.time_evolution.sesolve`
        """
        # Single qubit product basis
        # construct static hamiltonian
        ham_static = self.hilbertspace.idling_hamiltonian_in_prod_basis()

        ham = [ham_static, *self.hamiltonian_component_and_pulseshape]

        psi_list = self.psi_list

        if tlist is None:
            self.tlist = [0, self.pulse_endtime]
        else:
            self.tlist = tlist
        # Change astep as it is used between each steps
        options = self.options.copy()
        options["astep"] = options["astep"] // (len(self.tlist) - 1)
        res = sesolve(ham,
                      psi_list,
                      self.tlist,
                      solver=self.solver,
                      options=options)
        # Replace the last dimension with initial states
        # Now [time, comp, psi_init]
        states = jnp.swapaxes(res, 0, 3)[0]

        # Extract the states in the computational space
        pop = jnp.abs(psi_list).real**2
        tuple_ar_ix = tuple(np.argmax(pop, axis=1).flatten())

        return states, states[:, tuple_ar_ix, :]
