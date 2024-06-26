from abc import abstractmethod
from typing import Optional
import numpy as np
import jax.numpy as jnp

from .base import QuantumSystem


class CircuitLCJ(QuantumSystem):
    """Class for a generic L-C-J circuit.
    """

    def __init__(self,
                 b_transmon: bool = True,
                 basis: str = "phase",
                 num_basis: int = 0,
                 truncated_dim: int = 0,
                 n_max: int = 0,
                 phi_max: float = 0,
                 is_basis_sym: bool = False,
                 drive_for_state_phase: str = "charge",
                 name: str = 'circuit_lcj'):
        QuantumSystem.__init__(self, name=name)
        # Save basis related parameters
        self.basis = basis

        # Initialize basis related variables
        self.phi_max_int: int = 0  # phi_max / pi, must be integer
        self.n_max: int = 0  # n_max, must be integer
        self.phi_max: float = 0  # phi_max

        # Setup basis set
        self.b_transmon = b_transmon

        # Deal with number of basis
        self.is_basis_sym = is_basis_sym
        if is_basis_sym:
            if num_basis % 2 == 0:
                num_basis += 1

        # Sync parameter between two basis sets
        # To work with two basis sets together,
        # the n range  * phi range must be 2pi * points
        if self.basis in ("phase", "charge", "phase_only"):
            # n_max is always used first and override others if necessary
            if n_max == 0:
                # Check if phi_max is valid
                phi_max_int = phi_max / np.pi
                if np.abs(np.rint(phi_max_int) - phi_max_int) > 1e-4:
                    raise ValueError("phi_max must be a multiple of pi")
                phi_max_int = int(np.rint(phi_max_int))
                n_max = int(np.rint(2 * np.pi * num_basis / phi_max / 2) // 2)
            elif phi_max == 0:
                phi_max_int = num_basis // (n_max * 2)
                phi_max = phi_max_int * np.pi

            # Check the number of basis
            num_basis_new = int(np.rint(2 * n_max * phi_max / np.pi))
            if num_basis != 0:
                if num_basis != num_basis_new:
                    print("num_basis changed from %i to %i"
                          " to be a multiple of max range" %
                          (num_basis, num_basis_new))
                num_basis = num_basis_new
            # Check consistency
            # All information exists, check consistency
            if abs(2 * n_max * phi_max - np.pi * num_basis) > 1e-4:
                print(2 * n_max * phi_max / np.pi, num_basis)
                raise ValueError("phi_max, n_max, num_basis are inconsistent")

            # Ensure this is set
            phi_max_int = int(np.rint(phi_max / np.pi))

            self.phi_max = phi_max
            self.phi_max_int = phi_max_int
            self.n_max = n_max

        self.num_basis = num_basis
        self.truncated_dim = truncated_dim
        # print(self.phi_max, self.phi_max_int, self.n_max, self.num_basis)
        # Define phase basis
        self.num_phi = None  # number of phi basis (should be same as num_basis)
        self.ar_phi = None  # array of phi values
        self.delta_phi = None  # Difference between two phi basis
        # Define charge basis
        self.ar_n = None
        self.charge_cos = None
        self.charge_sin = None
        # Define the phase-charge conversion
        self.u1_n_phi = None
        self.u2_n_phi = None

        if self.basis in ("phase", "charge", "phase_only"):
            self.set_phi_basis(phi_max=phi_max, num_phi=num_basis)
        if self.basis in ("phase", "charge"):
            self.set_charge_basis(n_max=n_max, num_n=num_basis)
        if self.basis in ("phase", "charge"):
            self.set_n_phi_transform()

        # fix phase
        self.drive_for_state_phase = drive_for_state_phase
        # cache
        self.eigvec_in_raw_basis = None

    def set_phi_basis(self,
                      phi_max: float,
                      num_phi: Optional[int] = None,
                      phi_step: Optional[float] = None):
        """Initializes parameters for phase basis.

        One may choose to initialize with number of phi values or phi step size.

        Args:
            phi_max : [-phi_max, phi_max-1] as basis
            num_phi : Number of points between phi_max.
            phi_step : Step size of phi.
        """
        self.phi_max = phi_max
        if num_phi is not None:
            self.num_phi = num_phi  # Number of discretized num_phi basis
        elif phi_step is not None:
            self.num_phi = int(phi_max * 2 / phi_step)
            if not self.b_transmon:
                self.num_phi += 1

        # is_basis_even control if end point is included or not
        self.ar_phi = np.linspace(-self.phi_max,
                                  self.phi_max,
                                  self.num_basis,
                                  endpoint=self.is_basis_sym)

        self.delta_phi = self.ar_phi[1] - self.ar_phi[0]

    def set_charge_basis(self, n_max, num_n=0):
        """Initializes parameters for charge basis.

        cos/sin works as exp(ipt) which increase/decrease n by 1

        Note num_n will not be exact as x+any integer belongs to one group for cos/sin
        so a multiple of 2*n_max is used

        Args:
            n_max : Basis as [-n_max, n_max-1]
                (or n_max depends on *is_basis_sym*)
            num_n : Number of basis, not used for transmon
                (where n basis is periodic and must be integer,
                so fully determined by *n_max*)
        """
        n_set = self.phi_max_int  # Number of points per integer is the same as phi_max/pi
        n_group = n_max * 2 + (1 if self.is_basis_sym else 0)
        n = n_set * n_group
        self.ar_n = np.linspace(-n_max, n_max, n, endpoint=self.is_basis_sym)

        mcos = np.zeros((n, n), dtype=np.complex128)
        msin = np.zeros((n, n), dtype=np.complex128)
        # Add block-diaogonal matrix element
        # (only values differ by 1 are connected)
        for i1 in range(n_group - 1):
            i2 = i1 + 1
            for j in range(n_set):
                mcos[i1 * n_set + j, i2 * n_set + j] = 1
                mcos[i2 * n_set + j, i1 * n_set + j] = 1
                msin[i1 * n_set + j, i2 * n_set + j] = -1 / 1j
                msin[i2 * n_set + j, i1 * n_set + j] = 1 / 1j
        self.charge_cos = mcos / 2
        self.charge_sin = msin / 2

    def create_d2phi(self) -> np.ndarray:
        r"""Computes :math:`\frac{d^2}{d \phi^2}`

        This is from finite difference, which is an approximation.
        Used only in **phase_only** basis.

        Returns:
            matrix of :math:`\frac{d^2}{d \phi^2}`
        """
        m = np.eye(self.num_phi, dtype=np.complex128) * -2
        for i in range(0, self.num_phi):
            if i - 1 >= 0:
                m[i, i - 1] = 1
            if i + 1 < self.num_phi:
                m[i, i + 1] = 1

        # Periodic boundary condition
        if self.b_transmon:
            m[0, -1] = 1
            m[-1, 0] = 1

        return m / self.delta_phi**2

    def create_dphi(self) -> np.ndarray:
        r"""Computes :math:`\frac{d}{d \phi}`

        This is from finite difference, which is an approximation.
        Used only in **phase_only** basis.

        Returns:
            matrix of :math:`\frac{d}{d \phi}`
        """
        m = np.zeros((self.num_phi, self.num_phi), dtype=np.complex128)
        for i in range(0, self.num_phi):
            if i - 1 >= 0:
                m[i, i - 1] = -1
            if i + 1 < self.num_phi:
                m[i, i + 1] = 1

        # Periodic boundary condition
        if self.b_transmon:
            m[0, -1] = -1
            m[-1, 0] = 1

        return m / (2 * self.delta_phi)

    def create_phi(self) -> np.ndarray:
        """Computes the phi matrix.

        Returns:
            matrix of phase
        """
        if self.basis == "phase" or self.basis == "phase_only":
            return np.diag(self.ar_phi)
        elif self.basis == "charge":
            return self.transform_phi_to_n(np.diag(self.ar_phi))

    def create_n(self):
        r"""Computes charge matrix.

        Follows :math:`n = -i\frac{\partial}{\partial \phi}`

        For **phase_only** basis, use center difference,
        :math:`\frac{d}{d\phi}(N) = \phi(N+1)-\phi(N-1)` for phase basis.
        This is not accurate and this basis set should not be used.

        Use charge -> phase basis transformation for better accuracy,
        which is in **phase** and **charge** basis.

        Returns:
            matrix of charge
        """
        if self.basis == "phase_only":
            m = self.create_dphi()
            m = -1j * m
            return m
        elif self.basis == "charge":
            m = np.diag(self.ar_n.copy())
            return m
        elif self.basis == "phase":
            m = np.diag(self.ar_n.copy())
            return self.transform_n_to_phi(m)

    def set_n_phi_transform(self):
        """Computes the unitary transformation between n and phi.

        Store as u_n_phi
        phi within -pi, pi and n is bounded to n_max

        Note this transformation is ill defined if number of basis are different

        Require different normalization factor from phi->n and n->phi

        Require both set_charge_basis and set_phase_basis are done
        """
        x, y = np.meshgrid(self.ar_n, self.ar_phi)
        self.u1_n_phi = np.exp(1j * x * y)
        self.u2_n_phi = np.conj(self.u1_n_phi.T)

    def transform_n_to_phi(self, mat: np.ndarray) -> np.ndarray:
        """Run change of basis of matrix from charge basis to phase basis

        Args:
            mat: the matrix in charge basis

        Returns:
            the matrix in phase basis
        """
        return (self.u1_n_phi @ mat @ self.u2_n_phi) / self.ar_phi.size

    def transform_phi_to_n(self, mat: np.ndarray) -> np.ndarray:
        """Run change of basis of matrix from phi basis to n basis

        Args:
            mat: the matrix in phase basis

        Returns:
            the matrix in charge basis
        """
        return (self.u2_n_phi @ mat @ self.u1_n_phi) / self.ar_n.size

    @property
    def qdevice_type(self):
        return 'qubit'

    @property
    def dim(self) -> int:
        """Returns truncated Hilbert space dimension"""
        return self.truncated_dim

    @abstractmethod
    def idling_hamiltonian(self) -> jnp.ndarray:
        """Create the Hamiltonian matrix of the qubit.

        Returns:
            ndarray of the Hamiltonian matrix
        """

    def eigenenergies(self, unify_state_phase=True):
        """Returns array of eigenvalues.

        Args:
            evals_count:
                number of desired eigenvalues (default value = 6)
            unify_state_phase(bool): if True, adjusts the phase of eigenstates
                to meet the condition <i|m|i+1> matrix elements are real positive.
        """
        self.eigval, self.eigvec_in_raw_basis = self._calc_eigsys()
        if unify_state_phase:
            # Convert the basis to let n be real positive in 0-1 1-2 ...
            if self.drive_for_state_phase == "charge":
                opt = self.create_n()
            elif self.drive_for_state_phase == "phase":
                opt = self.create_phi()
            opt_eigenbasis = jnp.conj(
                self.eigvec_in_raw_basis).T @ opt @ self.eigvec_in_raw_basis
            self.unify_state_phase(opt_eigenbasis)

        return self.eigval[:self.dim]

    def unify_state_phase(self, operator: jnp.ndarray):
        """Adjusts the phase of eigenstates to meet the condition <i|m|i+1>
        matrix elements are real positive.

        This is to follow the simple idea we
        are always looking at phase defined by pulse.
        So for charge drive, let the <0|n|1>  be real positive is natural.
        For flux drive similar condition exists.

        Only bands within truncated_dim are considered.

        Require eigvec_in_raw_basis computed before calling this function.

        Args:
            operator: This is to specify which drive is used, so
                eigenstates are rotated to let <i|drive|i+1> be real positive
                Possible values are ``n_operator`` (in eigenbasis) and
                ``phi_operator`` (in eigenbasis).

        Returns:
            Modified eigvec_in_raw_basis.
        """
        if self.eigvec_in_raw_basis is None:
            raise ValueError("Must compute eigvec_in_raw_basis before rotating")

        ar_angle = jnp.cumsum(jnp.angle(jnp.diag(operator, k=1)))
        ar_angle = jnp.concatenate([jnp.zeros(1), ar_angle])
        # Apply the angle to basis
        ar_phase_eigenbasis = jnp.exp(ar_angle * -1j)

        # numpy broadcast
        self.eigvec_in_raw_basis = self.eigvec_in_raw_basis * ar_phase_eigenbasis

        return self.eigvec_in_raw_basis
