import numpy as np
import jax.numpy as jnp

from .base import QuantumSystem
from .circuit import CircuitLCJ
from ..utils.utility import const_init
from ..utils.operators import destroy, create, qeye


class Fluxonium(CircuitLCJ):
    """Class for the fluxonium qubit.

    When considering time-varying external flux,
    the phiext should be on EL. See PRB 99, 174512 (2019)
    Note that Transmon EL=0.

    Args:
        ec (float, optional): charging energy in unit GHz
        ej (float, optional): Josephson energy in unit GHz
        el (float, optional): inductive energy in unit GHz
        phiext: phi external parameters for fluxonium, in unit of flux quanta
        put_phiext_on_inductor: Whether to put the external flux term
            in the EL term (True) or EJ term (False).
        num_basis: Number of basis functions
        truncated_dim: Number of eigenenergies to construct the Hilbert space.
        basis: Basis set, can be {'charge', 'phase', 'phase_only'}.
        n_max: charge basis range [-n_max, n_max).
                Override phi_max if both specified
        phi_max: phi basis range [-phi_max, phi_max), must be multiple of pi
        is_basis_sym: Is basis forced to be [-n, ... +n]
            instead of [-n, ... +n-1].
        name: module name
        drive_for_state_phase: This is to specify which drive is used, so
            eigenstates are rotated to let <i|drive|i+1> be real positive
            Possible values are "charge" and "phase". Default is "charge".
    """

    def __init__(self,
                 ec: float,
                 ej: float,
                 el: float,
                 phiext: float,
                 put_phiext_on_inductor: bool = True,
                 num_basis: int = 400,
                 truncated_dim: int = 10,
                 basis: str = 'phase',
                 n_max: int = 0,
                 phi_max: float = 5 * np.pi,
                 is_basis_sym: bool = False,
                 name: str = 'fluxonium',
                 drive_for_state_phase: str = 'charge') -> None:
        super().__init__(False, basis, num_basis, truncated_dim, n_max, phi_max,
                         is_basis_sym, drive_for_state_phase, name)

        self.ec = ec
        self.ej = ej
        self.el = el
        self.phiext = phiext

        if basis not in ['phase', 'charge', 'phase_only']:
            raise NotImplementedError(f'Do not support basis {basis}.')

        self.put_phiext_on_inductor = put_phiext_on_inductor

    def create_t(self) -> jnp.ndarray:
        """Computes kinetic matrix in phase/charge basis.

        Returns:
            the kinetic matrix
        """
        # Ec
        if self.basis == "charge":
            return jnp.diag(self.ec * 4 * self.ar_n**2)
        # Use central difference
        elif self.basis == "phase_only":
            m = -self.create_d2phi()
            return m * (self.ec * 4)
        elif self.basis == "phase":
            n_square = self.transform_n_to_phi(jnp.diag(self.ar_n**2))
            return n_square * (self.ec * 4)

    def create_v(self) -> jnp.ndarray:
        """Computes potential matrix.

        Returns:
            potential matrix.
        """
        # Create the matrix in phase basis
        if self.put_phiext_on_inductor:
            # EJ
            m = -1.0 * self.ej * jnp.cos(self.ar_phi)
            # EL
            m += self.el * (self.ar_phi + self.phiext)**2 / 2
        else:
            # EJ
            m = -1.0 * self.ej * jnp.cos(self.ar_phi - self.phiext)
            # EL
            m += self.el * (self.ar_phi)**2 / 2

        if self.basis == "charge":
            m = self.transform_phi_to_n(jnp.diag(m))
            return m
        elif self.basis in ("phase", "phase_only"):
            return jnp.diag(m)

    def idling_hamiltonian(self) -> jnp.ndarray:
        """Create the Hamiltonian matrix of the qubit.

        Returns:
            ndarray of the Hamiltonian matrix
        """

        if self.basis in ("phase", "charge", "phase_only"):
            mt = self.create_t()
            mv = self.create_v()
            return mt + mv
        else:
            raise ValueError(f"Unknown basis {self.basis}")

    def n_operator(self, **kwargs) -> jnp.ndarray:
        """
        Return:
            Returns the :math:`n = - i d/d\\phi` operator in the phase basis.
        """

        if self.eigvec_in_raw_basis is None:
            self.eigenenergies()
        n_in_raw_basis = self.create_n()
        n_eigenbasis = jnp.conj(self.eigvec_in_raw_basis
                                ).T @ n_in_raw_basis @ self.eigvec_in_raw_basis

        return n_eigenbasis[:self.dim, :self.dim]

    def phi_operator(self, **kwargs) -> jnp.ndarray:
        """
        Return:
            Returns the phi operator in the phase basis.
        """

        if self.eigvec_in_raw_basis is None:
            self.eigenenergies()
        phi_in_raw_basis = self.create_phi()
        phi_eigenbasis = jnp.conj(
            self.eigvec_in_raw_basis
        ).T @ phi_in_raw_basis @ self.eigvec_in_raw_basis
        phi_truncated_eigenbasis = phi_eigenbasis[:self.dim, :self.dim]
        if self.put_phiext_on_inductor:
            phi_truncated_eigenbasis += jnp.eye(
                phi_truncated_eigenbasis.shape[0]) * self.phiext

        return phi_truncated_eigenbasis


class Transmon(CircuitLCJ):
    """Class for the transmon qubit.

    Args:
        ec (float, optional): charging energy
        ej (float, optional): Josephson energy
        ng (float, optional): ng parameter for transmon
        d (float, optional): junction asymmetry parameter, as :math:`(E_{J2}-E_{J1})/(E_{J2}+E_{J1})`.
            Set to 1 if it's not a Tunable Transmon.
        phiej: phi for tunable EJ for transmon, in unit of flux quanta
        num_basis: Number of basis functions
        truncated_dim: Number of eigenenergies to construct the Hilbert space.
        basis: Basis set, can be {'charge', 'phase', 'phase_only'}.
        n_max: charge basis range [-n_max, n_max).
                Override phi_max if both specified
        is_basis_sym: Is basis forced to be [-n, ... +n]
            instead of [-n, ... +n-1].
        name: module name
        drive_for_state_phase: This is to specify which drive is used, so
            eigenstates are rotated to let <i|drive|i+1> be real positive
            Possible values are "charge" and "phase". Default is "charge".
    """

    def __init__(self,
                 ec: float,
                 ej: float,
                 ng: float,
                 d: float = 1,
                 phiej: float = 0,
                 num_basis: int = 400,
                 truncated_dim: int = 10,
                 basis: str = 'phase',
                 n_max: int = 0,
                 is_basis_sym: bool = False,
                 name: str = 'transmon',
                 drive_for_state_phase: str = 'charge') -> None:
        phi_max = np.pi  # Periodicity of 2pi
        super().__init__(True, basis, num_basis, truncated_dim, n_max,
                         phi_max, is_basis_sym, drive_for_state_phase, name)

        self.ec = ec
        self.ej = ej
        self.ng = ng
        self.d = d
        self.phiej = phiej

        if basis not in ['phase', 'charge', 'phase_only']:
            raise NotImplementedError(f'Do not support basis {basis}.')

        self.put_phiext_on_inductor = False  # Transmon do not have inductor


    def create_t(self) -> jnp.ndarray:
        """Computes kinetic matrix in phase/charge basis.

        Returns:
            the kinetic matrix
        """
        # Ec
        if self.basis == "charge":
            return jnp.diag(self.ec * 4 * (self.ar_n - self.ng)**2)
        # Use central difference
        elif self.basis == "phase_only":
            m = -self.create_d2phi()
            m += 2j * self.ng * self.create_dphi() + np.eye(
                self.ar_phi.size) * self.ng**2
            return m * (self.ec * 4)
        elif self.basis == "phase":
            n1 = self.transform_n_to_phi(jnp.diag(self.ar_n))
            n_square = self.transform_n_to_phi(jnp.diag(self.ar_n**2))
            m = n_square - 2 * self.ng * n1 + jnp.eye(self.ar_phi.size) * self.ng**2
            return m * (self.ec * 4)

    @staticmethod
    def get_ej_eff(ej: float, d: float, phiej: float) -> jnp.ndarray:
        '''Computes the effective EJ from tunable transmon parameters.

        Args:
            ej: the maximum EJ of the transmon
            d (float, optional): junction asymmetry parameter, as :math:`(E_{J2}-E_{J1})/(E_{J2}+E_{J1})`.
            phiej: the :math:`Phi_{EJ}` in the transmon SQUID loop.

        Returns:
            the effective EJ
        '''
        return ej * jnp.sqrt(jnp.cos(phiej / 2.0) ** 2 + d ** 2 * jnp.sin(phiej / 2.0) ** 2)

    def create_v(self) -> jnp.ndarray:
        """Computes potential matrix.

        Returns:
            potential matrix.
        """
        ej_eff = self.get_ej_eff(self.ej, self.d, self.phiej)

        if self.basis == "charge":
            m = -1.0 * ej_eff * self.charge_cos
            return m
        elif self.basis in ("phase", "phase_only"):
            m = -1.0 * ej_eff * jnp.cos(self.ar_phi)  # EJ
            return jnp.diag(m)

    def idling_hamiltonian(self) -> jnp.ndarray:
        """Create the Hamiltonian matrix of the qubit.

        Returns:
            ndarray of the Hamiltonian matrix
        """

        if self.basis in ("phase", "charge", "phase_only"):
            mt = self.create_t()
            mv = self.create_v()
            return mt + mv
        else:
            raise ValueError(f"Unknown basis {self.basis}")

    def n_operator(self, **kwargs) -> jnp.ndarray:
        """
        Return:
            Returns the :math:`n = - i d/d\\phi` operator in the phase basis.
        """

        if self.eigvec_in_raw_basis is None:
            self.eigenenergies()
        n_in_raw_basis = self.create_n()
        n_eigenbasis = jnp.conj(self.eigvec_in_raw_basis
                                ).T @ n_in_raw_basis @ self.eigvec_in_raw_basis

        return n_eigenbasis[:self.dim, :self.dim]

    def phi_operator(self, **kwargs) -> jnp.ndarray:
        """
        Return:
            Returns the phi operator in the phase basis.
        """

        if self.eigvec_in_raw_basis is None:
            self.eigenenergies()
        phi_in_raw_basis = self.create_phi()
        phi_eigenbasis = jnp.conj(
            self.eigvec_in_raw_basis
        ).T @ phi_in_raw_basis @ self.eigvec_in_raw_basis
        phi_truncated_eigenbasis = phi_eigenbasis[:self.dim, :self.dim]

        return phi_truncated_eigenbasis

    def cosphi_operator(self, **kwargs) -> jnp.array:
        """

        Args:
            **kwargs:

        Returns:
            The cos(phi) operator in the truncated basis
        """
        if self.eigvec_in_raw_basis is None:
            self.eigenenergies()
        cosphi_in_raw_basis = self.create_cosphi()
        cosphi_eigenbasis = jnp.conj(
            self.eigvec_in_raw_basis
        ).T @ cosphi_in_raw_basis @ self.eigvec_in_raw_basis
        cosphi_truncated_eigenbasis = cosphi_eigenbasis[:self.dim, :self.dim]
        return cosphi_truncated_eigenbasis

    def phiej_coef_cosphi_operator(self, phiej) -> jnp.array:
        """Convert the phiej Value to be a coefficient of cosphi operator.

        Args:
            phiej: the Phiej value of the tunable transmon

        Returns:
            the coefficient of the cosphi operator
        """
        return -self.get_ej_eff(self.ej, self.d, self.phiej + phiej) + self.get_ej_eff(self.ej, self.d, self.phiej)



class Resonator(QuantumSystem):
    """Class representing a harmonic resonator.

    Args:
        f_res: resonator frequency in unit GHz
        remove_zpe: Remove the Zero-point energy(ZPE) of the resonator.
        truncated_dim: Number of eigenenergies to construct the Hilbert space.
        name: module name
    """

    def __init__(self,
                 f_res: float = None,
                 remove_zpe: bool = False,
                 truncated_dim: int = 10,
                 name: str = 'resonator') -> None:
        super().__init__(name=name)

        self.f_res = f_res
        self.remove_zpe = remove_zpe
        # Truncated parameter
        self.truncated_dim = truncated_dim
        # Cached variable
        self.eigval = None
        self.eigvec_in_raw_basis = None

    @property
    def qdevice_type(self):
        return 'resonator'

    def idling_hamiltonian(self):
        """Create the hamiltonian matrix of the resonator.

        Returns:
            ndarray of the Hamiltonian matrix
        """
        a = destroy(self.truncated_dim)
        adag = create(self.truncated_dim)
        if self.remove_zpe:
            ham = adag @ a
        else:
            ham = adag @ a + 0.5 * qeye(self.truncated_dim)
        return ham * self.f_res

    def eigenenergies(self):
        """Returns array of eigenvalues.

        Args:
            evals_count:
                number of desired eigenvalues (default value = 6)
        """
        evals = self._calc_evals()
        return evals[:self.dim]

    @property
    def dim(self) -> int:
        """Returns truncated Hilbert space dimension"""
        return self.truncated_dim

    def creation_operator(self, **kwargs) -> jnp.ndarray:
        """Returns the creation operator"""
        return create(self.dim)

    def annihilation_operator(self, **kwargs) -> jnp.ndarray:
        """Returns the creation operator"""
        return destroy(self.dim)
