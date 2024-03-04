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

    @abstractmethod
    def _init_quantum_system(self):
        """Initialize quantum system"""

    def __getattribute__(self, attr, transform=True):
        """Use Haiku's decorator before get attribute."""

        origin_attr = super().__getattribute__(attr)
        if transform and isinstance(origin_attr, Callable) and attr not in [
                '__getattribute__', '_init_quantum_system', 'ls_params'
        ]:

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
