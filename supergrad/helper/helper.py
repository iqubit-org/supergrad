from typing import Callable, Literal
from abc import ABC, abstractmethod
import jax


InitQuantumSystemType = Literal["check", "bypass", "auto"]

class Helper(ABC):
    """Transform attribute to pure function.
    """
    # Indicate what functions in the class do not require "init_quantum_system"
    list_function_not_require_init = []

    @staticmethod
    def decorator_bypass_init(func):
        """This marks a function do not require run `init_quantum_system` before calling it.

        Returns:
            modified function with an attribute to mark this.
        """
        func.init_type = "bypass"
        return func

    @staticmethod
    def decorator_auto_init(func):
        """This creates a function with additional first parameters for JAX  to init the quantum system.

        It can be used to mimic the Haiku behavior (define a function and use it with the input parameters)

        Args:
            func: the original function

        Returns:
            A new function with additional first parameter as JAX to init the quantum system.

        """
        func.init_type = "auto"
        return func


    def __init__(self, *args, **kwargs) -> None:

        self.args = args
        self.kwargs = kwargs
        self.graph = None
        self.hilbertspace = None
        self._initialized = False

    def init_quantum_system(self, params: dict):
        """Initialize quantum system"""
        self._initialized = True

    def __getattribute__(self, attr, transform=True):
        """Use Haiku's decorator before get attribute."""

        origin_attr = super().__getattribute__(attr)
        if transform and isinstance(
                origin_attr,
                Callable) and not (attr.startswith('_') or attr == "init_quantum_system"):
            # Default: add initialization check
            init_type: InitQuantumSystemType = getattr(origin_attr, "init_type", "check")
            if init_type == "check":
                def _transform_func(*args, **kwargs):
                    if not self._initialized:
                        raise ValueError("This function call requires the initialization of the quantum system.")
                    return origin_attr(*args, **kwargs)

                return _transform_func
            # Automatically initialization
            elif init_type == "auto":
                def _transform_func(params, *args, **kwargs):
                    self.init_quantum_system(params)
                    return origin_attr(*args, **kwargs)

                return _transform_func
            # Do nothing
            elif init_type == "bypass":
                pass
            else:
                raise ValueError(f"Unknown function init_type {init_type}")
        return origin_attr

    def _get_dims(self):
        """Get the dimension of the composited quantum system."""
        return self.hilbertspace.dim

    @property
    def dims(self):
        """The dimension of the composited quantum system."""
        return self._get_dims()

    @property
    def device_params(self):
        """The device parameters dictionary."""
        return self.graph.convert_graph_to_parameters(only_device_params=True)

    @property
    def all_params(self):
        """The static parameters dictionary."""
        all_params = self.graph.convert_graph_to_parameters()
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
