from typing import Callable, Union, List
import re
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

from supergrad.utils.utility import identity_wrap, const_init
from .base import QuantumSystem
from .kronobj import KronObj


def parse_interaction(**kwargs):
    """Parse subsystem_list and keyword arguments, specify the interaction
    between subsystems constructing the `InteractionTerm`.
    Here we try to provide a similar input format as the add_interaction function
    of scqubits.

    Examples:

        parse_interaction(strength=<float>,
                          op1=<Callable>,
                          op2=<Callable>,
                          …,
                          add_hc=<bool>,
                          name=<str>)

    Args:
        strength:
            coefficient parametrize the interaction strength,
            set `None` if you want to pass parameters by dm-haiku.
        op* (callable): subsystems operators
        add_hc: Default is False.
            If set to True, the Hermitian conjugate is added.
        name: module name

    """

    g = kwargs.pop("g", None)
    if g is None:
        g = kwargs.pop("strength", None)
    if g is None:
        g = kwargs.pop("g_strength", None)
    if g is None:
        g = kwargs.pop('J', None)

    add_hc = kwargs.pop("add_hc", False)
    constant = kwargs.pop('constant', False)
    name = kwargs.pop('name', 'interaction_term')

    operator_list = []
    for key, val in kwargs.items():
        if re.match(r"op\d+$", key) is None:
            raise TypeError(f"Unexpected keyword argument {key}.")
        operator_list.append(val)

    return InteractionTerm(strength=g,
                           operator_list=operator_list,
                           add_hc=add_hc,
                           constant=constant,
                           name=name)


class InteractionTerm(hk.Module):
    """Class for specifying a term in the interaction Hamiltonian of a composite
    Hilbert space.

    Args:
        operator_list: list of operator
        strength: coefficient for the interaction strength,
            set `None` if you want to pass parameters by dm-haiku.
        add_hc: If set to True, the Hermitian conjugate is added.
        constant (bool): True for manually setting parameters, False for using
            haiku's parameters management.
        name: module name
    """

    def __init__(
        self,
        operator_list: List[Callable],
        strength: float = None,
        add_hc: bool = False,
        constant: bool = False,
        name: str = 'interaction_term'
    ) -> None:
        super().__init__(name=name)
        if not constant:
            self.strength = hk.get_parameter('strength', [],
                                             init=const_init(strength))
        else:
            self.strength = strength
        self.operator_list = operator_list
        self.add_hc = add_hc

    def hamiltonian(
            self,
            subsystem_list: List) -> jnp.ndarray:
        """Returns the full Hamiltonian of the interacting quantum system
        described by the InteractingSystem object

        Args:
            subsystem_list:
                list of all quantum systems in InteractingSystem calling ``hamiltonian``,
                needed for identity wrapping
        """

        hamiltonian = self.id_wrap_all_ops(self.operator_list,
                                           subsystem_list)
        if self.add_hc:
            hamiltonian += hamiltonian.dag()
        return hamiltonian

    def id_wrap_all_ops(
            self,
            operator_list: List[Callable],
            subsystem_list: List[QuantumSystem]) -> KronObj:
        """Construct a KronObj which represents the interaction operator on the
        Hilbert space. Identity operators of other subsystems are tensored to the
        interaction operator
        """

        mat_list = []
        subsys_index = []
        dims = [subsystem.dim for subsystem in subsystem_list]
        for idx, operator in enumerate(operator_list):
            subsys_index.append(subsystem_list.index(operator.__self__))
            if idx == 0:
                coeff = self.strength
            else:
                coeff = 1
            # call operator
            mat_list.append(coeff * operator())

        return KronObj(mat_list, dims, subsys_index)


class InteractingSystem(QuantumSystem):
    """Class holding information about the whole system.

    Args:
        subsystem_list(list): list of `QuantumSystem`.
        interaction_list: list of `InteractionTerm`, one can use
            `parse_interaction` to construct `InteractionTerm` easily.
        name: module name
    """

    def __init__(self,
                 subsystem_list: List[QuantumSystem],
                 interaction_list: List[InteractionTerm] = None,
                 name='hilbert_space') -> None:
        super().__init__(name=name)

        # Initialize subsystem
        self._subsystems: List[QuantumSystem] = tuple(subsystem_list)
        self._subsystem_by_id_str = {
            subsystem.name: subsystem
            for subsystem in self._subsystems}
        # Initialize interaction
        if isinstance(interaction_list, InteractionTerm):
            interaction_list = [interaction_list]
        if interaction_list is None:
            interaction_list = []
        self.interaction_list = interaction_list
        self._interaction_term_by_id_str = {
            interaction_term.name: interaction_term
            for interaction_term in self.interaction_list}
        self._truncated_dim = self.dim

        # Cached variable
        self.eigval = None
        self.eigvec = None
        self.ar_index_map = None

    def __getitem__(self, key: Union[int, str]):

        if isinstance(key, int):
            return self._subsystems[key]
        if key in self._subsystem_by_id_str:
            return self._subsystem_by_id_str[key]

        raise KeyError(
            f"Unrecognized key: {key}. Key must be an integer index or a "
            "string specifying a subsystem of InteractingSystem.")

    def __iter__(self):
        return iter(self._subsystems)  # [qbt1,qbt2,in1]

    @property
    def subsystem_list(self) -> List[QuantumSystem]:
        return list(self._subsystems)

    @property
    def subsystem_count(self) -> int:
        """Returns number of subsys_list composing the joint Hilbert space"""
        return len(self._subsystems)

    def idling_hamiltonian_in_prod_basis(self) -> KronObj:
        """Generate hamiltonian in the product basis.

        Return:
            Hamiltonian of the composite system, including the interaction
            between components
        """
        hamiltonian = self.subsystem_hamiltonian()
        hamiltonian += self.coupling_hamiltonian()
        return hamiltonian

    def idling_hamiltonian(self) -> jnp.ndarray:
        """Generate hamiltonian in the product basis.

        Return:
            Hamiltonian of the composite system, including the interaction
            between components
        """
        return self.idling_hamiltonian_in_prod_basis().full()

    def subsystem_hamiltonian(self) -> KronObj:
        """Calculate bare hamiltonian.

        Return:
            composite Hamiltonian composed of bare Hamiltonians of subsys_list
            independent of the external parameter.
        """

        ham_list = []
        for subsys_index, subsys in enumerate(self):
            evals = subsys.eigenenergies()
            ham_list.append(
                KronObj([jnp.diag(evals)], self.dim, [subsys_index]))
        return sum(ham_list)

    def coupling_hamiltonian(self) -> jnp.ndarray:
        """Returns the interaction Hamiltonian, based on the interaction terms
        specified for the current InteractingSystem object.

        Return:
            interaction Hamiltonian
        """

        if not self.interaction_list:
            return 0

        operator_list = []
        for term in self.interaction_list:
            if isinstance(term, InteractionTerm):
                operator_list.append(
                    term.hamiltonian(self.subsystem_list))
            else:
                raise TypeError("Expected an instance of InteractionTerm, "
                                f"got {type(term)} instead.")
        hamiltonian = sum(operator_list)

        return hamiltonian

    def eigenenergies(self, evals_count: int = 6):
        """Returns array of eigenvalues.

        Args:
            evals_count:
                number of desired eigenvalues (default value = 6)
        """
        self.eigval, self.eigvec = self._calc_eigsys()

        return self.eigval[:evals_count]

    @property
    def dim(self):
        """Return Hilbert Space dimension."""
        return tuple([subsystem.dim for subsystem in self])

    @property
    def truncated_dim(self):
        """Return the dimension for truncated eigen basis."""
        return self._truncated_dim

    @truncated_dim.setter
    def truncated_dim(self, value):
        """Overwrite the dimension for truncated eigen basis."""
        if len(self.dim) != len(value):
            raise ValueError(
                'Truncated_dim must be tuple that have the same length'
                'as dim.'
            )
        self._truncated_dim = tuple(value)

    def compute_energy_map(self, greedy_assign=True) -> jnp.ndarray:
        """
        Compute the energy of the whole system in the dressed indices,
        enumerating eigenenergies and eigenstates as j=0,1,2,... starting from
        the ground state.
        Assign results to map back to bare product state indices, for a Hilbert
        space composed of three subsystems TODO, specify tuples of the form (l,m,n)
        where l, m, n denote the excitation levels of each bare subsystem.

        Noted we should calculate the matrices that permute the order of eigenvectors.
        The eigenproblem solver will sort the eigenvectors array by the
        eigenvalues, and one could get the "ascending" order. But we want to
        define a "tensor" order and make the eigenvectors array close to
        identity matrix(just like the coordinate array).

        Args:
            greedy_assign(bool): if True, use greedy assignment mode
                TODO The greedy assignment mode ignores the issue "same state be assigned
                multiple times", due to the weak coupling of subsystems.
        """

        self.eigval, self.eigvec = self._calc_eigsys()
        # Choose the row map to [i, j, ...] state, and find the maximum
        # amplitude the column contain the amplitude is the related eigenvector.
        # Now just use an aggressive greedy search
        amp = (jnp.conj(self.eigvec) * self.eigvec).real
        if not greedy_assign:
            # Initialize mapper matrix
            ar_index_map = jnp.zeros(self.truncated_dim, dtype=int)

            def scan_func(carry, target_idx):
                index_map, amp = carry
                ix = jnp.ravel_multi_index(target_idx, self.dim, 'clip')
                ix2 = jnp.argmax(amp[ix, :])
                # Set the weight of basis ix2 of all wave functions to be 0
                # to avoid assigning the same state for multiple times.
                # This will happen when, for instance:
                # |a> = |0> + |1>, |b> = |0> - |1>
                next_amp = amp.at[:, ix2].set(0)
                next_index_map = index_map.at[tuple(target_idx)].set(ix2)
                carry = [next_index_map, next_amp]
                return carry, None

            idxs = jnp.array([idx for idx in np.ndindex(self.truncated_dim)])
            carry = [ar_index_map, amp]
            (self.ar_index_map, _), _ = jax.lax.scan(scan_func, carry, idxs)

        else:
            # The greedy assignment mode ignores the issue "same state be assigned multiple
            # times", due to the weak coupling of subsystems.
            self.ar_index_map = jnp.argmax(amp, axis=1).reshape(self.truncated_dim)

        energy_nd = self.eigval[self.ar_index_map]

        return energy_nd

    def compute_transform_matrix(self):
        """Computes the matrices related to basis transformation.
        Require `compute_energy_map` to be run first to compute
        the transformation.

        Note all transformation keep the num_basis_resonator.

        Return:
            Unitary transform product basis to eigen basis
        """

        if self.ar_index_map is None:
            self.compute_energy_map()
        # Create quantities in the new Hamiltonian
        # Create a new unitary transform
        dim = np.prod(np.array(self.dim))  # use np for `jit`
        u_map = jnp.eye(dim)[self.ar_index_map.flatten()].T
        # rotate the peak of eigen states to diagonal
        u_product_to_eigen = self.eigvec @ u_map
        # here we must unify phase of eigen states, because eigh return states
        # with arbitrary phase.
        angle = jnp.angle(jnp.diag(u_product_to_eigen))
        # Apply the angle to basis
        ar_phase_sys = jnp.exp(-1j * angle)
        u_product_to_eigen *= ar_phase_sys  # numpy broadcast

        return u_product_to_eigen

    def transform_operator(self, operator, **kwargs):
        """Transform operator to current eigenbasis and truncated eigenbasis.

        Args:
            operator(callable): subsystem operator
            kwargs: keyword arguments pass to callable operator.
        Returns:
            TODO only return 1 thing; `operator in eigenbasis`, `operator in truncated eigenbasis`
        """

        opt_tensor = identity_wrap(operator, self.subsystem_list,
                                   **kwargs)
        opt_tensor = opt_tensor.full()
        u_product_to_eigen = self.compute_transform_matrix()
        opt_eig = jnp.conj(u_product_to_eigen).T @ opt_tensor @ u_product_to_eigen

        return opt_eig
