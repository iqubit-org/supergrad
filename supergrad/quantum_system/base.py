from abc import ABC, abstractmethod
import jax.scipy as jsp
import haiku as hk


class QuantumSystem(ABC, hk.Module):
    """Generic quantum system class.
    """

    def __init__(self, name: str = 'quantum_system'):
        super().__init__(name=name)

        self._sys_type = type(self).__name__

    @property
    @abstractmethod
    def dim(self) -> int:
        """Return dimension of Hilbert space"""

    @abstractmethod
    def idling_hamiltonian(self):
        """Time independent part of Hamiltonian for QuantumSystem.
        """

    @abstractmethod
    def eigenenergies(self):
        """Eigenenergies of Quantum system.
        """

    def _calc_evals(self):
        """Calculate quantum system's eigenenergy using `jax.scipy.linalg.eigh`.

        """

        h = self.idling_hamiltonian()
        eigval = jsp.linalg.eigh(h, eigvals_only=True)
        return eigval

    def _calc_eigsys(self):
        """Calculate quantum system's eigenenergy and eigenvectors using
        `jax.scipy.linalg.eigh`.
        """

        h = self.idling_hamiltonian()
        eigval, eigvec = jsp.linalg.eigh(h, eigvals_only=False)
        return eigval, eigvec
