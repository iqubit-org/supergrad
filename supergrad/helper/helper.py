from typing import Callable
from abc import ABC, abstractmethod
import jax
import haiku as hk


class Helper(ABC):
    """Transform attribute to pure function.
    """

    def __init__(self, *args, **kwargs) -> None:

        self.args = args
        self.kwargs = kwargs
        self.graph = None
        self.hilbertspace = None

    @abstractmethod
    def _init_quantum_system(self):
        """Initialize quantum system"""

    def __getattribute__(self, attr, transform=True):
        """Use Haiku's decorator before get attribute."""

        origin_attr = super().__getattribute__(attr)
        if transform and isinstance(
                origin_attr,
                Callable) and not (attr.startswith('_') or attr == 'ls_params'):

            @hk.without_apply_rng
            @hk.transform
            def _transform_func(*args, **kwargs):
                self._init_quantum_system()
                return origin_attr(*args, **kwargs)

            return _transform_func.apply
        return origin_attr

    def ls_params(self, attr: str = None, *args, **kwargs):
        """Get all parameters for the selected attribute.

        Args:
            attr(str): Default `None`, get parameters for
            `self._init_quantum_system`.
        """

        @hk.transform
        def prepare_params():
            self._init_quantum_system()
            if attr is not None:
                self.__getattribute__(attr, transform=False)(*args, **kwargs)
            return

        rng = jax.random.PRNGKey(0)
        return prepare_params.init(rng)

    def _get_dims(self):
        """Get the dimension of the composited quantum system."""
        return self.hilbertspace.dim

    def get_dims(self):
        return self._get_dims()

    @property
    def dims(self):
        """The dimension of the composited quantum system.
        Only for internal use temporarily.
        """
        return self._get_dims()

    @property
    def device_params(self):
        """The device parameters dictionary."""
        return self.graph.convert_graph_to_parameters_haiku(
            self.share_params, self.unify_coupling, only_device_params=True)

    @property
    def all_params(self):
        """The static parameters dictionary."""
        all_params = self.graph.convert_graph_to_parameters_haiku(
            self.share_params, self.unify_coupling)
        return all_params

    def idling_hamiltonian_in_prod_basis(self):
        """Return the idling Hamiltonian in product basis."""
        return self.hilbertspace.idling_hamiltonian_in_prod_basis()

    def construct_transform_matrix(self):
        """Constructing the transform matrix from product basis to eigenbasis.
        """
        return self.hilbertspace.compute_transform_matrix()

    def eigensystem(self):
        """Calculate quantum system's eigenenergy and eigenvectors using
        `jax.scipy.linalg.eigh`.
        """
        return self.hilbertspace._calc_eigsys()

    def n_operator(self, node_name, transform=False):
        """Return the operator of the selected node.

        Args:
            node_name (str): name of the selected node.
            transform (bool): if True, return the transformed operator in eigen basis.
        """
        if transform:
            opt = self.hilbertspace.transform_operator(
                self.hilbertspace[node_name].n_operator).reshape(self.dims * 2)
            ix = tuple([
                slice(None) if name == node_name else 0
                for name in self.graph.sorted_nodes
            ])
            if not any(ix):
                raise ValueError(
                    'The tensor slice index is all 0, please check the input.')
            return opt[ix + ix]
        else:
            return self.hilbertspace[node_name].n_operator()

    def phi_operator(self, node_name, transform=False):
        """Return the operator of the selected node.

        Args:
            node_name (str): name of the selected node.
            transform (bool): if True, return the transformed operator in eigen basis.
        """
        if transform:
            opt = self.hilbertspace.transform_operator(
                self.hilbertspace[node_name].phi_operator).reshape(self.dims * 2)
            ix = tuple([
                slice(None) if name == node_name else 0
                for name in self.graph.sorted_nodes
            ])
            if not any(ix):
                raise ValueError(
                    'The tensor slice index is all 0, please check the input.')
            return opt[ix + ix]
        else:
            return self.hilbertspace[node_name].phi_operator()
