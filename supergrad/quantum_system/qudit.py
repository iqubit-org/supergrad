import numpy as np
import jax.numpy as jnp

from .base import QuantumSystem


class Qudit(QuantumSystem):
    """Class for a generic qudit system with N-levels and fixed matrix elements.
    """
    def __init__(self,
                 f,
                 n,
                 phi,
                 name: str = "qudit"):
        """Class for a general qudit.

        All frequencies and charge/phase matrix elements are fixed.
        The eigenvectors are I.

        Args:
            f: the frequency of each level, size N
            n: the charge matrix, size N*N
            phi: the phase matrix, size N*N
            name: the system name
        """
        QuantumSystem.__init__(self, name=name)
        self.truncated_dim = f.size
        self.f = f
        self.n = n
        self.phi = phi
        self.h = jnp.diag(f)

    @property
    def dim(self) -> int:
        return self.truncated_dim


    def idling_hamiltonian(self):
        return self.h

    def eigenenergies(self):
        return self.f

    def _calc_evals(self):
        return self.f


    def _calc_eigsys(self):
        return self.f, jnp.ones((self.truncated_dim, self.truncated_dim))


    def n_operator(self):
        return self.n


    def phi_operator(self):
        return self.phi


class StandardNonlinearOscillator(Qudit):
    """Class for a standard nonlinear osciallar system with N-levels and fixed matrix elements.
    """
    def __init__(self,
                 f,
                 n,
                 phi = None,
                 name: str = "qudit"):
        """Class for a standard non linear oscillator.

        All frequencies and charge/phase matrix elements are fixed.
        The eigenvectors are I.

        The charge/phase matrix are non-zero only between `i-i,i` levels.

        Args:
            f: the frequency of each level (except 0), size (N-1)
            n: the charge matrix, offset 1 from the main diagonal, size (N-1)
            phi: the phase matrix, offset 1 from the main diagonal, size (N-1)
                by default it is -i*n, so phi is Pauli Y when n is Pauli X (by default)
            name: the system name
        """
        assert f.size == n.size
        if phi is not None:
            assert f.size == phi.size

        QuantumSystem.__init__(self, name=name)
        self.truncated_dim = f.size + 1
        self.f = jnp.concatenate([jnp.array([0]), f])
        self.n = jnp.diag(n, 1)
        self.n = self.n + self.n.conj().T
        if phi is None:
            self.phi = -1j * self.n
        else:
            self.phi = jnp.diag(phi, 1)
            self.phi = self.phi + self.phi.conj().T
        self.h = jnp.diag(f)
