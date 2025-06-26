import jax
import numpy as np
import jax.numpy as jnp

from .base import QuantumSystem


class Qudit(QuantumSystem):
    """Class for a generic qudit system with N-levels and fixed matrix elements.
    """
    def __init__(self,
                 energy,
                 n,
                 phi,
                 name: str = "qudit"):
        """Class for a general qudit.

        All frequencies and charge/phase matrix elements are fixed.
        The eigenvectors are I.

        Args:
            energy: the energy of each level, size N
            n: the charge matrix, size N*N
            phi: the phase matrix, size N*N
            name: the system name
        """
        QuantumSystem.__init__(self, name=name)
        self.truncated_dim = f.size
        self.energy = energy
        self.n = n
        self.phi = phi
        self.h = jnp.diag(f)

    @property
    def dim(self) -> int:
        return self.truncated_dim


    def idling_hamiltonian(self):
        return self.h

    def eigenenergies(self):
        return self.energy

    def _calc_evals(self):
        return self.energy


    def _calc_eigsys(self):
        return self.energy, jnp.ones((self.truncated_dim, self.truncated_dim))


    def n_operator(self):
        return self.n


    def phi_operator(self):
        return self.phi


def conv_charge_phase_input(v) -> jax.Array:
    """Converts the input 1/2/3D data to charge or phase matrix.

        The n and phi input can be 1/2/3D, 

        * input 1D is real 1D
        * input 2D is complex 1D
        * input 3D is complex 2D.

        1D data is 1 off-diagonal terms in the matrix.

    Args:
        v: the 1/2/3D input

    Returns:
        jax Array 2D matrix
    """
    if isinstance(v, np.ndarray) or isinstance(v, jax.Array):
        return v
    n = jnp.array(v)
    if len(n.shape) >= 2:
        n = n[..., 0] + 1j * n[..., 1]

    if len(n.shape) == 1:
        n = jnp.diag(n, 1)
        n = n + n.conj().T
    return n


class StandardNonlinearOscillator(Qudit):
    """Class for a standard nonlinear osciallar system with N-levels and fixed matrix elements.
    """
    def __init__(self,
                 energy,
                 n,
                 phi = None,
                 name: str = "qudit"):
        """Class for a standard non linear oscillator.

        All frequencies and charge/phase matrix elements are fixed.
        The eigenvectors are I.

        The charge/phase matrix are non-zero only between `i-i,i` levels.

        The n and phi input can be 1/2/3D, 1D is real 1D, 2D is complex 1D, 3D is complex 2D.

        Args:
            energy: the energy of each level (except 0), size (N-1)
            n: the charge matrix,can be a complete matrix, or a 1D array as offset 1 from the main diagonal, size (N-1)
            phi: the phase matrix, can be a complete matrix, or a 1D array as offset 1 from the main diagonal, size (N-1)
                by default, phi is Pauli Y when n is Pauli X
            name: the system name
        """
        # Convert 2D list of 2-Tuples (complex) to array
        n = conv_charge_phase_input(n)

        if phi is not None:
            phi = conv_charge_phase_input(phi)

        dim = energy.size + 1

        assert dim == n.shape[0]

        if phi is not None:
            assert dim == phi.shape[0]

        QuantumSystem.__init__(self, name=name)
        self.truncated_dim = energy.size + 1
        self.energy = jnp.concatenate([jnp.array([0]), energy])
        self.n = n
        if phi is None:
            self.phi = jnp.tril(n) -1j * jnp.triu(n).T
        else:
            self.phi = jnp.array(phi)
        self.h = jnp.diag(energy)
