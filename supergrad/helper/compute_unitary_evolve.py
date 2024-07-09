import copy
from typing import Optional, Literal
import jax
import numpy as np
import jax.numpy as jnp

from supergrad.helper.helper import Helper
from supergrad.time_evolution import sesolve, sesolve_final_states_w_basis_trans
from supergrad.utils.utility import create_state_init, tensor
from supergrad.time_evolution.single_qubit_compensation import SingleQubitCompensation
from supergrad.scgraph.graph import SCGraph

BasisType = Literal["eigen", "product"]


class Evolve(Helper):
    """Helper for constructing time-evolution computing functions based on the
    graph which contains all information about qubits and pulses. The functions
    constructed this way are pure and can be transformed by Jax.

    Args:
        graph (SCGraph): The graph containing both Hamiltonian and control
            parameters.
        solver: the type of time evolution solver, should be in ['ode_expm', 'odeint'].
        options: the arguments will be passed to solver.
            See `supergrad.time_evolution.sesolve`.
    """

    def __init__(self,
                 graph,
                 solver='ode_expm',
                 options=None,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if options is None:
            options = {'astep': 2000, 'trotter_order': 1}

        self.graph: SCGraph = graph
        self.coupler_subsystem = graph.get_subsystem_in_category("coupler")
        if solver not in ['ode_expm', 'odeint']:
            raise NotImplementedError(
                f'Solver {solver} has not been implemented.')
        self.solver = solver
        self.options = options
        self.psi_list = None
        self.hamiltonian_component_and_pulseshape = None
        self.pulse_endtime = 0
        self.pre_unitaries = None
        self.post_unitaries = None

    def init_quantum_system(self, params: dict):
        super().init_quantum_system(params)
        graph = copy.deepcopy(self.graph)
        graph.update_parameters(params)
        self.hilbertspace = graph.convert_graph_to_quantum_system()
        self.hamiltonian_component_and_pulseshape, self.pulse_endtime = graph.convert_graph_to_pulse_lst(
            self.hilbertspace)
        sqc = SingleQubitCompensation(graph, graph.get_subsystem_in_category("data"))
        self.pre_unitaries, self.post_unitaries = sqc.create_unitaries()

    def _prepare_initial_states(self, psi_list=None):
        """Prepare the initial states for time evolution.

        Args:
            psi_list (list, optional): The list of states for evolution. If None,
                all the states in computation basis will be used.
        """
        if psi_list is None:
            states_config = []
            for i, node in enumerate(self.graph.sorted_nodes):
                if node in self.coupler_subsystem:
                    states_config.append([i, 1])
                else:
                    states_config.append([i, 2])
            self.psi_list, _ = create_state_init(self.dims, states_config)
        else:
            self.psi_list = psi_list

    def construct_hamiltonian_and_pulseshape(self):
        """Constructing the Hamiltonian and pulseshape for time evolution.

        Returns:
            (KronObj, list, float): The static Hamiltonian, the list of pair
                containing the time-dependent components of the Hamiltonian and
                corresponding pulse shape, and the maximum length of the pulse.
        """
        return self.hilbertspace.idling_hamiltonian_in_prod_basis(
        ), self.hamiltonian_component_and_pulseshape, self.pulse_endtime

    def construct_compensation_function(self):
        """Constructing the compensation matrix for virtual compensation.

        Returns:
            Callable: the function that add compensation before and after the
                time evolution unitary.
        """
        if self.compensation_option in ['only_vz', 'arbit_single']:
            pre_u = tensor(*self.pre_unitaries)
            post_u = tensor(*self.post_unitaries)
            return lambda sim_u: post_u @ sim_u @ pre_u
        else:
            return None

    def get_basis_transform_matrix(self, basis: BasisType, transform_matrix=None) -> Optional[jax.Array]:
        """Gets the basis transform matrix before/after time evolution based on the basis.

        One can provide custom transform_matrix to replace the computed eigen basis.

        Args:
            basis: "product" for time evolution without basis transformation, "eigen" for in eigen basis.
            transform_matrix: pre-computed transform matrix, used when design parameters is not optimized.

        Returns:
            the transform_matrix
        """
        if basis == "eigen":
            if transform_matrix is None:
                u_to_eigen = self.hilbertspace.compute_transform_matrix()
            else:
                u_to_eigen = transform_matrix
        elif basis == "product":
            u_to_eigen = None
            if transform_matrix is not None:
                raise ValueError("Transform matrix cannot be used in the product basis")
        else:
            raise ValueError(f"Unknown basis type {basis}")

        return u_to_eigen

    def final_state(self,
                    basis: BasisType,
                    transform_matrix=None,
                    psi_list=None,
                    _remove_compensation=False,
                    **kwargs):
        """Running the time evolution in the eigen / product basis and get the final states.

        Args:
            basis: can be "eigen" or "product", indicate the basis of the evolution
            transform_matrix: pre-computed transform matrix,
                using when design parameters is not optimized.
            states: the list of states for evolution. If None, all the states
                in computation basis will be used.
            _remove_compensation: whether to remove the compensation(for debug use).
            kwargs: keyword arguments will be passed to `supergrad.time_evolution.sesolve`

        Returns:
            the final state after the time evolution
        """
        u_to_eigen = self.get_basis_transform_matrix(basis, transform_matrix)

        # Single qubit product basis
        # construct static hamiltonian
        ham_static = self.hilbertspace.idling_hamiltonian_in_prod_basis()

        ham = [ham_static, *self.hamiltonian_component_and_pulseshape]
        self._prepare_initial_states(psi_list)
        self.tlist = [0, self.pulse_endtime]
        sim_u = sesolve_final_states_w_basis_trans(ham,
                                                   self.psi_list,
                                                   self.tlist,
                                                   transform_matrix=u_to_eigen,
                                                   solver=self.solver,
                                                   options=self.options,
                                                   **kwargs)
        if not _remove_compensation:
            pre_u = tensor(*self.pre_unitaries)
            post_u = tensor(*self.post_unitaries)
            sim_u = post_u @ sim_u @ pre_u
        return sim_u

    def trajectory(self,
                   basis: str,
                   tlist=None,
                   psi_list=None,
                   transform_matrix=None,
                   **kwargs):
        """Computing the time evolution trajectory in the product / eigen basis.

        Args:
            tlist: list of time steps.
            psi_list: the list of states for evolution. If None, all the states
                in computation basis will be used.
            transform_matrix: pre-computed transform matrix,
                used when design parameters is not optimized.
            kwargs: keyword arguments will be passed to `supergrad.time_evolution.sesolve`
        """
        transform_matrix = self.get_basis_transform_matrix(basis, transform_matrix)

        # Single qubit product basis
        # construct static hamiltonian
        ham_static = self.hilbertspace.idling_hamiltonian_in_prod_basis()

        ham = [ham_static, *self.hamiltonian_component_and_pulseshape]

        self._prepare_initial_states(psi_list)
        if transform_matrix is not None:
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

        if transform_matrix is not None:
            states = jnp.conj(transform_matrix).T @ res

        # Extract the states in the computational space
        pop = jnp.abs(psi_list).real ** 2
        tuple_ar_ix = tuple(np.argmax(pop, axis=1).flatten())

        return states, states[:, tuple_ar_ix, :]

    @Helper.decorator_auto_init
    def product_basis(self, *args, **kwargs):
        return self.final_state(basis="product", *args, **kwargs)

    @Helper.decorator_auto_init
    def eigen_basis(self, *args, **kwargs):
        return self.final_state(basis="eigen", *args, **kwargs)