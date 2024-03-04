"""
The code is from `qutip_qip.operations`.
Operations on quantum circuits.
"""
import numbers
from collections.abc import Iterable
import numpy as np
import jax.numpy as jnp
import jax.scipy.linalg as jaxLA

from supergrad.utils.operators import identity, sigmax, sigmay, sigmaz
from supergrad.utils.utility import tensor, fock_dm, permute

__all__ = [
    "rx",
    "ry",
    "rz",
    "sqrtnot",
    "snot",
    "phasegate",
    "qrot",
    "x_gate",
    "y_gate",
    "z_gate",
    "cy_gate",
    "cz_gate",
    "s_gate",
    "t_gate",
    "qasmu_gate",
    "cs_gate",
    "ct_gate",
    "cphase",
    "cnot",
    "csign",
    "berkeley",
    "swapalpha",
    "swap",
    "iswap",
    "sqrtswap",
    "sqrtiswap",
    "fredkin",
    "molmer_sorensen",
    "toffoli",
    "rotation",
    "controlled_gate",
    "globalphase",
    "hadamard_transform",
    "gate_expand_1ton",
    "gate_expand_2ton",
    "gate_expand_3toN",
    "expand_operator",
    "_single_qubit_gates",
    "_para_gates",
    "_ctrl_gates",
    "_swap_like",
    "_toffoli_like",
    "_fredkin_like",
]

_single_qubit_gates = [
    "RX",
    "RY",
    "RZ",
    "SNOT",
    "SQRTNOT",
    "PHASEGATE",
    "X",
    "Y",
    "Z",
    "S",
    "T",
    "QASMU",
]
_para_gates = [
    "RX",
    "RY",
    "RZ",
    "CPHASE",
    "SWAPalpha",
    "PHASEGATE",
    "GLOBALPHASE",
    "CRX",
    "CRY",
    "CRZ",
    "QASMU",
]
_ctrl_gates = [
    "CNOT",
    "CSIGN",
    "CRX",
    "CRY",
    "CRZ",
    "CY",
    "CZ",
    "CS",
    "CT",
    "CPHASE",
]
_swap_like = [
    "SWAP",
    "ISWAP",
    "SQRTISWAP",
    "SQRTSWAP",
    "BERKELEY",
    "SWAPalpha",
]
_toffoli_like = ["TOFFOLI"]
_fredkin_like = ["FREDKIN"]

#
# Single Qubit Gates
#


def x_gate(n=None, target=0):
    """Pauli-X gate or sigmax operator.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing
        a single-qubit rotation through pi radians around the x-axis.

    """
    if n is not None:
        return gate_expand_1ton(x_gate(), n, target)
    return sigmax()


def y_gate(n=None, target=0):
    """Pauli-Y gate or sigmay operator.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing
        a single-qubit rotation through pi radians around the y-axis.

    """
    if n is not None:
        return gate_expand_1ton(y_gate(), n, target)
    return sigmay()


def cy_gate(n=None, control=0, target=1):
    """Controlled Y gate.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing the rotation.

    """
    if (control == 1 and target == 0) and n is None:
        n = 2

    if n is not None:
        return gate_expand_2ton(cy_gate(), n, control, target)
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]])


def z_gate(n=None, target=0):
    """Pauli-Z gate or sigmaz operator.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing
        a single-qubit rotation through pi radians around the z-axis.

    """
    if n is not None:
        return gate_expand_1ton(z_gate(), n, target)
    return sigmaz()


def cz_gate(n=None, control=0, target=1):
    """Controlled Z gate.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing the rotation.

    """
    if (control == 1 and target == 0) and n is None:
        n = 2

    if n is not None:
        return gate_expand_2ton(cz_gate(), n, control, target)
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])


def s_gate(n=None, target=0):
    """Single-qubit rotation also called Phase gate or the Z90 gate.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing
        a 90 degree rotation around the z-axis.

    """
    if n is not None:
        return gate_expand_1ton(s_gate(), n, target)
    return np.array([[1, 0], [0, 1j]])


def cs_gate(n=None, control=0, target=1):
    """Controlled S gate.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing the rotation.

    """
    if (control == 1 and target == 0) and n is None:
        n = 2

    if n is not None:
        return gate_expand_2ton(cs_gate(), n, control, target)
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1j]])


def t_gate(n=None, target=0):
    """Single-qubit rotation related to the S gate by the relationship S=T*T.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing a phase shift of pi/4.

    """
    if n is not None:
        return gate_expand_1ton(t_gate(), n, target)
    return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])


def ct_gate(n=None, control=0, target=1):
    """Controlled T gate.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing the rotation.

    """
    if (control == 1 and target == 0) and n is None:
        n = 2

    if n is not None:
        return gate_expand_2ton(ct_gate(), n, control, target)
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, np.exp(1j * np.pi / 4)],
    ])


def rx(phi, n=None, target=0):
    """Single-qubit rotation for operator sigmax with angle phi.

    Returns
    -------
    result : qobj
        Quantum object for operator describing the rotation.

    """
    if n is not None:
        return gate_expand_1ton(rx(phi), n, target)
    return jnp.array([
        [jnp.cos(phi / 2), -1j * jnp.sin(phi / 2)],
        [-1j * jnp.sin(phi / 2), jnp.cos(phi / 2)],
    ])


def ry(phi, n=None, target=0):
    """Single-qubit rotation for operator sigmay with angle phi.

    Returns
    -------
    result : qobj
        Quantum object for operator describing the rotation.

    """
    if n is not None:
        return gate_expand_1ton(ry(phi), n, target)
    return jnp.array([
        [jnp.cos(phi / 2), -jnp.sin(phi / 2)],
        [jnp.sin(phi / 2), jnp.cos(phi / 2)],
    ])


def rz(phi, n=None, target=0):
    """Single-qubit rotation for operator sigmaz with angle phi.

    Returns
    -------
    result : qobj
        Quantum object for operator describing the rotation.

    """
    if n is not None:
        return gate_expand_1ton(rz(phi), n, target)
    return jnp.array([[jnp.exp(-1j * phi / 2), 0], [0, jnp.exp(1j * phi / 2)]])


def sqrtnot(n=None, target=0):
    """Single-qubit square root NOT gate.

    Returns
    -------
    result : qobj
        Quantum object for operator describing the square root NOT gate.

    """
    if n is not None:
        return gate_expand_1ton(sqrtnot(), n, target)
    return np.array([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]])


def snot(n=None, target=0):
    """Quantum object representing the SNOT (Hadamard) gate.

    Returns
    -------
    snot_gate : qobj
        Quantum object representation of SNOT gate.

    Examples
    --------
    >>> snot() # doctest: +SKIP
    Quantum object: dims = [[2], [2]], \
shape = [2, 2], type = oper, isHerm = True
    Qobj data =
    [[ 0.70710678+0.j  0.70710678+0.j]
     [ 0.70710678+0.j -0.70710678+0.j]]

    """
    if n is not None:
        return gate_expand_1ton(snot(), n, target)
    return 1 / np.sqrt(2.0) * np.array([[1, 1], [1, -1]])


def phasegate(theta, n=None, target=0):
    """
    Returns quantum object representing the phase shift gate.

    Parameters
    ----------
    theta : float
        Phase rotation angle.

    Returns
    -------
    phase_gate : qobj
        Quantum object representation of phase shift gate.

    Examples
    --------
    >>> phasegate(pi/4) # doctest: +SKIP
    Quantum object: dims = [[2], [2]], \
shape = [2, 2], type = oper, isHerm = False
    Qobj data =
    [[ 1.00000000+0.j          0.00000000+0.j        ]
     [ 0.00000000+0.j          0.70710678+0.70710678j]]

    """
    if n is not None:
        return gate_expand_1ton(phasegate(theta), n, target)
    return jnp.array([[1, 0], [0, jnp.exp(1.0j * theta)]])


def qrot(theta, phi, n=None, target=0):
    """
    Single qubit rotation driving by Rabi oscillation with 0 detune.

    Parameters
    ----------
    phi : float
        The inital phase of the rabi pulse.
    theta : float
        The duration of the rabi pulse.
    n : int
        Number of qubits in the system.
    target : int
        The index of the target qubit.

    Returns
    -------
    qrot_gate : :class:`qutip.Qobj`
        Quantum object representation of physical qubit rotation under
        a rabi pulse.
    """
    if n is not None:
        return expand_operator(qrot(theta, phi), n=n, targets=target)
    return jnp.array([
        [
            jnp.cos(theta / 2.0),
            -1.0j * jnp.exp(-1.0j * phi) * jnp.sin(theta / 2.0),
        ],
        [
            -1.0j * jnp.exp(1.0j * phi) * jnp.sin(theta / 2.0),
            jnp.cos(theta / 2.0),
        ],
    ])


def qasmu_gate(args, n=None, target=0):
    """
    QASM U-gate as defined in the OpenQASM standard.

    Parameters
    ----------

    theta : float
        The argument supplied to the last RZ rotation.
    phi : float
        The argument supplied to the middle RY rotation.
    gamma : float
        The argument supplied to the first RZ rotation.
    n : int
        Number of qubits in the system.
    target : int
        The index of the target qubit.

    Returns
    -------
    qasmu_gate : :class:`qutip.Qobj`
        Quantum object representation of the QASM U-gate as defined in the
        OpenQASM standard.
    """

    theta, phi, gamma = args
    if n is not None:
        return expand_operator(qasmu_gate([theta, phi, gamma]),
                               n=n,
                               targets=target)
    return jnp.array(rz(phi) @ ry(theta) @ rz(gamma))


#
# 2 Qubit Gates
#


def cphase(theta, n=2, control=0, target=1):
    """
    Returns quantum object representing the controlled phase shift gate.

    Parameters
    ----------
    theta : float
        Phase rotation angle.

    n : integer
        The number of qubits in the target space.

    control : integer
        The index of the control qubit.

    target : integer
        The index of the target qubit.

    Returns
    -------
    U : qobj
        Quantum object representation of controlled phase gate.
    """

    if n < 1 or target < 0 or control < 0:
        raise ValueError("Minimum value: n=1, control=0 and target=0")

    if control >= n or target >= n:
        raise ValueError("control and target need to be smaller than N")

    U_list1 = [identity(2)] * n
    U_list2 = [identity(2)] * n

    U_list1[control] = fock_dm(2, 1)
    U_list1[target] = phasegate(theta)

    U_list2[control] = fock_dm(2, 0)

    U = tensor(*U_list1) + tensor(*U_list2)
    return U


def cnot(n=None, control=0, target=1):
    """
    Quantum object representing the CNOT gate.

    Returns
    -------
    cnot_gate : qobj
        Quantum object representation of CNOT gate

    Examples
    --------
    >>> cnot() # doctest: +SKIP
    Quantum object: dims = [[2, 2], [2, 2]], \
shape = [4, 4], type = oper, isHerm = True
    Qobj data =
        [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]
         [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j]]

    """
    if (control == 1 and target == 0) and n is None:
        n = 2

    if n is not None:
        return gate_expand_2ton(cnot(), n, control, target)
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


def csign(n=None, control=0, target=1):
    """
    Quantum object representing the CSIGn gate.

    Returns
    -------
    csign_gate : qobj
        Quantum object representation of CSIGn gate

    Examples
    --------
    >>> csign() # doctest: +SKIP
    Quantum object: dims = [[2, 2], [2, 2]], \
shape = [4, 4], type = oper, isHerm = True
    Qobj data =
        [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  -1.+0.j]]

    """
    if (control == 1 and target == 0) and n is None:
        n = 2

    if n is not None:
        return gate_expand_2ton(csign(), n, control, target)
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])


def berkeley(n=None, targets=[0, 1]):
    """
    Quantum object representing the Berkeley gate.

    Returns
    -------
    berkeley_gate : qobj
        Quantum object representation of Berkeley gate

    Examples
    --------
    >>> berkeley() # doctest: +SKIP
    Quantum object: dims = [[2, 2], [2, 2]], \
shape = [4, 4], type = oper, isHerm = True
    Qobj data =
        [[ cos(pi/8).+0.j  0.+0.j           0.+0.j           0.+sin(pi/8).j]
         [ 0.+0.j          cos(3pi/8).+0.j  0.+sin(3pi/8).j  0.+0.j]
         [ 0.+0.j          0.+sin(3pi/8).j  cos(3pi/8).+0.j  0.+0.j]
         [ 0.+sin(pi/8).j  0.+0.j           0.+0.j           cos(pi/8).+0.j]]

    """
    if (targets[0] == 1 and targets[1] == 0) and n is None:
        n = 2

    if n is not None:
        return gate_expand_2ton(berkeley(), n, targets=targets)
    return np.array([
        [np.cos(np.pi / 8), 0, 0, 1.0j * np.sin(np.pi / 8)],
        [0, np.cos(3 * np.pi / 8), 1.0j * np.sin(3 * np.pi / 8), 0],
        [0, 1.0j * np.sin(3 * np.pi / 8),
         np.cos(3 * np.pi / 8), 0],
        [1.0j * np.sin(np.pi / 8), 0, 0,
         np.cos(np.pi / 8)],
    ])


def swapalpha(alpha, n=None, targets=[0, 1]):
    """
    Quantum object representing the SWAPalpha gate.

    Returns
    -------
    swapalpha_gate : qobj
        Quantum object representation of SWAPalpha gate

    Examples
    --------
    >>> swapalpha(alpha) # doctest: +SKIP
    Quantum object: dims = [[2, 2], [2, 2]], \
shape = [4, 4], type = oper, isHerm = True
    Qobj data =
    [[ 1.+0.j  0.+0.j                    0.+0.j                    0.+0.j]
     [ 0.+0.j  0.5*(1 + exp(j*pi*alpha)  0.5*(1 - exp(j*pi*alpha)  0.+0.j]
     [ 0.+0.j  0.5*(1 - exp(j*pi*alpha)  0.5*(1 + exp(j*pi*alpha)  0.+0.j]
     [ 0.+0.j  0.+0.j                    0.+0.j                    1.+0.j]]

    """
    if (targets[0] == 1 and targets[1] == 0) and n is None:
        n = 2

    if n is not None:
        return gate_expand_2ton(swapalpha(alpha), n, targets=targets)
    return jnp.array([
        [1, 0, 0, 0],
        [
            0,
            0.5 * (1 + jnp.exp(1.0j * jnp.pi * alpha)),
            0.5 * (1 - jnp.exp(1.0j * jnp.pi * alpha)),
            0,
        ],
        [
            0,
            0.5 * (1 - jnp.exp(1.0j * jnp.pi * alpha)),
            0.5 * (1 + jnp.exp(1.0j * jnp.pi * alpha)),
            0,
        ],
        [0, 0, 0, 1],
    ])


def swap(n=None, targets=[0, 1]):
    """Quantum object representing the SWAP gate.

    Returns
    -------
    swap_gate : qobj
        Quantum object representation of SWAP gate

    Examples
    --------
    >>> swap() # doctest: +SKIP
    Quantum object: dims = [[2, 2], [2, 2]], \
shape = [4, 4], type = oper, isHerm = True
    Qobj data =
    [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j]
     [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]]

    """
    if targets != [0, 1] and n is None:
        n = 2

    if n is not None:
        return gate_expand_2ton(swap(), n, targets=targets)
    return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


def iswap(n=None, targets=[0, 1]):
    """Quantum object representing the iSWAP gate.

    Returns
    -------
    iswap_gate : qobj
        Quantum object representation of iSWAP gate

    Examples
    --------
    >>> iswap() # doctest: +SKIP
    Quantum object: dims = [[2, 2], [2, 2]], \
shape = [4, 4], type = oper, isHerm = False
    Qobj data =
    [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  0.+1.j  0.+0.j]
     [ 0.+0.j  0.+1.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]]
    """
    if targets != [0, 1] and n is None:
        n = 2

    if n is not None:
        return gate_expand_2ton(iswap(), n, targets=targets)
    return np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])


def sqrtswap(n=None, targets=[0, 1]):
    """Quantum object representing the square root SWAP gate.

    Returns
    -------
    sqrtswap_gate : qobj
        Quantum object representation of square root SWAP gate

    """
    if targets != [0, 1] and n is None:
        n = 2

    if n is not None:
        return gate_expand_2ton(sqrtswap(), n, targets=targets)
    return np.array([
        [1, 0, 0, 0],
        [0, 0.5 + 0.5j, 0.5 - 0.5j, 0],
        [0, 0.5 - 0.5j, 0.5 + 0.5j, 0],
        [0, 0, 0, 1],
    ])


def sqrtiswap(n=None, targets=[0, 1]):
    """Quantum object representing the square root iSWAP gate.

    Returns
    -------
    sqrtiswap_gate : qobj
        Quantum object representation of square root iSWAP gate

    Examples
    --------
    >>> sqrtiswap() # doctest: +SKIP
    Quantum object: dims = [[2, 2], [2, 2]], \
shape = [4, 4], type = oper, isHerm = False
    Qobj data =
    [[ 1.00000000+0.j   0.00000000+0.j   \
       0.00000000+0.j          0.00000000+0.j]
     [ 0.00000000+0.j   0.70710678+0.j   \
       0.00000000-0.70710678j  0.00000000+0.j]
     [ 0.00000000+0.j   0.00000000-0.70710678j\
       0.70710678+0.j          0.00000000+0.j]
     [ 0.00000000+0.j   0.00000000+0.j   \
       0.00000000+0.j          1.00000000+0.j]]

    """
    if targets != [0, 1] and n is None:
        n = 2

    if n is not None:
        return gate_expand_2ton(sqrtiswap(), n, targets=targets)
    return np.array([
        [1, 0, 0, 0],
        [0, 1 / np.sqrt(2), 1j / np.sqrt(2), 0],
        [0, 1j / np.sqrt(2), 1 / np.sqrt(2), 0],
        [0, 0, 0, 1],
    ])


def molmer_sorensen(theta, n=None, targets=[0, 1]):
    """
    Quantum object of a Mølmer–Sørensen gate.

    Parameters
    ----------
    theta: float
        The duration of the interaction pulse.
    N: int
        Number of qubits in the system.
    target: int
        The indices of the target qubits.

    Returns
    -------
    molmer_sorensen_gate : :class:`qutip.Qobj`
        Quantum object representation of the Mølmer–Sørensen gate.
    """
    if targets != [0, 1] and n is None:
        n = 2

    if n is not None:
        return expand_operator(molmer_sorensen(theta), n, targets=targets)
    return jnp.array([
        [jnp.cos(theta / 2.0), 0, 0, -1.0j * jnp.sin(theta / 2.0)],
        [0, jnp.cos(theta / 2.0), -1.0j * jnp.sin(theta / 2.0), 0],
        [0, -1.0j * jnp.sin(theta / 2.0),
         jnp.cos(theta / 2.0), 0],
        [-1.0j * jnp.sin(theta / 2.0), 0, 0,
         jnp.cos(theta / 2.0)],
    ])


#
# 3 Qubit Gates
#


def fredkin(n=None, control=0, targets=[1, 2]):
    """Quantum object representing the Fredkin gate.

    Returns
    -------
    fredkin_gate : qobj
        Quantum object representation of Fredkin gate.

    Examples
    --------
    >>> fredkin() # doctest: +SKIP
    Quantum object: dims = [[2, 2, 2], [2, 2, 2]], \
shape = [8, 8], type = oper, isHerm = True
    Qobj data =
        [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j]]

    """
    if [control, targets[0], targets[1]] != [0, 1, 2] and n is None:
        n = 3

    if n is not None:
        return gate_expand_3toN(fredkin(), n, [control, targets[0]], targets[1])
    return np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ])


def toffoli(n=None, controls=[0, 1], target=2):
    """Quantum object representing the Toffoli gate.

    Returns
    -------
    toff_gate : qobj
        Quantum object representation of Toffoli gate.

    Examples
    --------
    >>> toffoli() # doctest: +SKIP
    Quantum object: dims = [[2, 2, 2], [2, 2, 2]], \
                    shape = [8, 8], type = oper, isHerm = True
    Qobj data =
        [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j]]


    """
    if [controls[0], controls[1], target] != [0, 1, 2] and n is None:
        n = 3

    if n is not None:
        return gate_expand_3toN(toffoli(), n, controls, target)
    return np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ])


#
# Miscellaneous Gates
#


def rotation(op, phi, n=None, target=0):
    """Single-qubit rotation for operator op with angle phi.

    Returns
    -------
    result : qobj
        Quantum object for operator describing the rotation.

    """
    if n is not None:
        return gate_expand_1ton(rotation(op, phi), n, target)
    return jaxLA.expm(-1j * op * phi / 2)


def controlled_gate(U, n=2, control=0, target=1, control_value=1):
    """
    Create an N-qubit controlled gate from a single-qubit gate U with the given
    control and target qubits.

    Parameters
    ----------
    U : Qobj
        Arbitrary single-qubit gate.

    n : integer
        The number of qubits in the target space.

    control : integer
        The index of the first control qubit.

    target : integer
        The index of the target qubit.

    control_value : integer (1)
        The state of the control qubit that activates the gate U.

    Returns
    -------
    result : qobj
        Quantum object representing the controlled-U gate.

    """

    if [n, control, target] == [2, 0, 1]:
        return tensor(fock_dm(2, control_value), U) + tensor(
            fock_dm(2, 1 - control_value), identity(2))
    U2 = controlled_gate(U, control_value=control_value)
    return gate_expand_2ton(U2, n=n, control=control, target=target)


def globalphase(theta, n=1):
    """
    Returns quantum object representing the global phase shift gate.

    Parameters
    ----------
    theta : float
        Phase rotation angle.

    Returns
    -------
    phase_gate : qobj
        Quantum object representation of global phase shift gate.

    Examples
    --------
    >>> phasegate(pi/4) # doctest: +SKIP
    Quantum object: dims = [[2], [2]], \
shape = [2, 2], type = oper, isHerm = False
    Qobj data =
    [[ 0.70710678+0.70710678j          0.00000000+0.j]
     [ 0.00000000+0.j          0.70710678+0.70710678j]]

    """
    data = jnp.exp(1.0j * theta) * np.eye(2**n, 2**n, dtype=complex)
    return data


#
# Operation on Gates
#


def _hamming_distance(x, bits=32):
    """
    Calculate the bit-wise Hamming distance of x from 0: That is, the number
    1s in the integer x.
    """
    tot = 0
    while x:
        tot += 1
        x &= x - 1
    return tot


def hadamard_transform(n=1):
    """Quantum object representing the N-qubit Hadamard gate.

    Returns
    -------
    q : qobj
        Quantum object representation of the N-qubit Hadamard gate.

    """
    data = [[1, 1], [1, -1]]
    H = np.array(data) / np.sqrt(2)
    return tensor(*([H] * n))


#
# Gate Expand
#


def gate_expand_1ton(U, n, target):
    """
    Create a Qobj representing a one-qubit gate that act on a system with N
    qubits.

    Parameters
    ----------
    U : Qobj
        The one-qubit gate

    n : integer
        The number of qubits in the target space.

    target : integer
        The index of the target qubit.

    Returns
    -------
    gate : qobj
        Quantum object representation of N-qubit gate.

    """

    if n < 1:
        raise ValueError("integer n must be larger or equal to 1")

    if target >= n:
        raise ValueError("target must be integer < integer N")
    return tensor(*([identity(2)] * (target) + [U] + [identity(2)]
                  * (n - target - 1)))


def gate_expand_2ton(U, n, control=None, target=None, targets=None):
    """
    Create a Qobj representing a two-qubit gate that act on a system with N
    qubits.

    Parameters
    ----------
    U : Qobj
        The two-qubit gate

    n : integer
        The number of qubits in the target space.

    control : integer
        The index of the control qubit.

    target : integer
        The index of the target qubit.

    targets : list
        List of target qubits.

    Returns
    -------
    gate : qobj
        Quantum object representation of N-qubit gate.

    """

    if targets is not None:
        control, target = targets

    if control is None or target is None:
        raise ValueError("Specify value of control and target")

    if n < 2:
        raise ValueError("integer n must be larger or equal to 2")

    if control >= n or target >= n:
        raise ValueError("control and not target must be integer < integer n")

    if control == target:
        raise ValueError("target and not control cannot be equal")

    p = list(range(n))

    if target == 0 and control == 1:
        p[control], p[target] = p[target], p[control]

    elif target == 0:
        p[1], p[target] = p[target], p[1]
        p[1], p[control] = p[control], p[1]

    else:
        p[1], p[target] = p[target], p[1]
        p[0], p[control] = p[control], p[0]

    return permute(tensor(*([U] + [identity(2)] * (n - 2))), [2] * n, p)


def gate_expand_3toN(U, n, controls=[0, 1], target=2):
    """
    Create a Qobj representing a three-qubit gate that act on a system with N
    qubits.

    Parameters
    ----------
    U : Qobj
        The three-qubit gate

    n : integer
        The number of qubits in the target space.

    controls : list
        The list of the control qubits.

    target : integer
        The index of the target qubit.

    Returns
    -------
    gate : qobj
        Quantum object representation of N-qubit gate.

    """

    if n < 3:
        raise ValueError("integer n must be larger or equal to 3")

    if controls[0] >= n or controls[1] >= n or target >= n:
        raise ValueError("control and not target is None."
                         " Must be integer < integer N")

    if (controls[0] == target or controls[1] == target
            or controls[0] == controls[1]):

        raise ValueError("controls[0], controls[1], and target"
                         " cannot be equal")

    p = list(range(n))
    p1 = list(range(n))
    p2 = list(range(n))

    if controls[0] <= 2 and controls[1] <= 2 and target <= 2:
        p[controls[0]] = 0
        p[controls[1]] = 1
        p[target] = 2

    #
    # n > 3 cases
    #

    elif controls[0] == 0 and controls[1] == 1:
        p[2], p[target] = p[target], p[2]

    elif controls[0] == 0 and target == 2:
        p[1], p[controls[1]] = p[controls[1]], p[1]

    elif controls[1] == 1 and target == 2:
        p[0], p[controls[0]] = p[controls[0]], p[0]

    elif controls[0] == 1 and controls[1] == 0:
        p[controls[1]], p[controls[0]] = p[controls[0]], p[controls[1]]
        p2[2], p2[target] = p2[target], p2[2]
        p = [p2[p[k]] for k in range(n)]

    elif controls[0] == 2 and target == 0:
        p[target], p[controls[0]] = p[controls[0]], p[target]
        p1[1], p1[controls[1]] = p1[controls[1]], p1[1]
        p = [p1[p[k]] for k in range(n)]

    elif controls[1] == 2 and target == 1:
        p[target], p[controls[1]] = p[controls[1]], p[target]
        p1[0], p1[controls[0]] = p1[controls[0]], p1[0]
        p = [p1[p[k]] for k in range(n)]

    elif controls[0] == 1 and controls[1] == 2:
        #  controls[0] -> controls[1] -> target -> outside
        p[0], p[1] = p[1], p[0]
        p[0], p[2] = p[2], p[0]
        p[0], p[target] = p[target], p[0]

    elif controls[0] == 2 and target == 1:
        #  controls[0] -> target -> controls[1] -> outside
        p[0], p[2] = p[2], p[0]
        p[0], p[1] = p[1], p[0]
        p[0], p[controls[1]] = p[controls[1]], p[0]

    elif controls[1] == 0 and controls[0] == 2:
        #  controls[1] -> controls[0] -> target -> outside
        p[1], p[0] = p[0], p[1]
        p[1], p[2] = p[2], p[1]
        p[1], p[target] = p[target], p[1]

    elif controls[1] == 2 and target == 0:
        #  controls[1] -> target -> controls[0] -> outside
        p[1], p[2] = p[2], p[1]
        p[1], p[0] = p[0], p[1]
        p[1], p[controls[0]] = p[controls[0]], p[1]

    elif target == 1 and controls[1] == 0:
        #  target -> controls[1] -> controls[0] -> outside
        p[2], p[1] = p[1], p[2]
        p[2], p[0] = p[0], p[2]
        p[2], p[controls[0]] = p[controls[0]], p[2]

    elif target == 0 and controls[0] == 1:
        #  target -> controls[0] -> controls[1] -> outside
        p[2], p[0] = p[0], p[2]
        p[2], p[1] = p[1], p[2]
        p[2], p[controls[1]] = p[controls[1]], p[2]

    elif controls[0] == 0 and controls[1] == 2:
        #  controls[0] -> self, controls[1] -> target -> outside
        p[1], p[2] = p[2], p[1]
        p[1], p[target] = p[target], p[1]

    elif controls[1] == 1 and controls[0] == 2:
        #  controls[1] -> self, controls[0] -> target -> outside
        p[0], p[2] = p[2], p[0]
        p[0], p[target] = p[target], p[0]

    elif target == 2 and controls[0] == 1:
        #  target -> self, controls[0] -> controls[1] -> outside
        p[0], p[1] = p[1], p[0]
        p[0], p[controls[1]] = p[controls[1]], p[0]

    #
    # n > 4 cases
    #

    elif controls[0] == 1 and controls[1] > 2 and target > 2:
        #  controls[0] -> controls[1] -> outside, target -> outside
        p[0], p[1] = p[1], p[0]
        p[0], p[controls[1]] = p[controls[1]], p[0]
        p[2], p[target] = p[target], p[2]

    elif controls[0] == 2 and controls[1] > 2 and target > 2:
        #  controls[0] -> target -> outside, controls[1] -> outside
        p[0], p[2] = p[2], p[0]
        p[0], p[target] = p[target], p[0]
        p[1], p[controls[1]] = p[controls[1]], p[1]

    elif controls[1] == 2 and controls[0] > 2 and target > 2:
        #  controls[1] -> target -> outside, controls[0] -> outside
        p[1], p[2] = p[2], p[1]
        p[1], p[target] = p[target], p[1]
        p[0], p[controls[0]] = p[controls[0]], p[0]

    else:
        p[0], p[controls[0]] = p[controls[0]], p[0]
        p1[1], p1[controls[1]] = p1[controls[1]], p1[1]
        p2[2], p2[target] = p2[target], p2[2]
        p = [p[p1[p2[k]]] for k in range(n)]

    return permute(tensor(*([U] + [identity(2)] * (n - 3))), [2] * n, p)


def _check_qubits_oper(oper, dims=None, targets=None):
    """
    Check if the given operator is valid.

    Parameters
    ----------
    oper : :class:`qutip.Qobj`
        The quantum object to be checked.
    dims : list, optional
        A list of integer for the dimension of each composite system.
        e.g ``[2, 2, 2, 2, 2]`` for 5 qubits system. If None, qubits system
        will be the default.
    targets : int or list of int, optional
        The indices of qubits that are acted on.
    """
    # if operator matches N
    if oper.dims[0] != oper.dims[1]:
        raise ValueError("The operator is not an "
                         "Qobj with the same input and output dimensions.")
    # if operator dims matches the target dims
    if dims is not None and targets is not None:
        targ_dims = [dims[t] for t in targets]
        if oper.dims[0] != targ_dims:
            raise ValueError("The operator dims {} do not match "
                             "the target dims {}.".format(
                                 oper.dims[0], targ_dims))


def _targets_to_list(targets, oper=None, n=None):
    """
    transform targets to a list and check validity.

    Parameters
    ----------
    targets : int or list of int
        The indices of qubits that are acted on.
    oper : :class:`qutip.Qobj`, optional
        An operator acts on qubits, the type of the :class:`qutip.Qobj`
        has to be an operator
        and the dimension matches the tensored qubit Hilbert space
        e.g. dims = ``[[2, 2, 2], [2, 2, 2]]``
    n : int, optional
        The number of qubits in the system.
    """
    # if targets is a list of integer
    if targets is None:
        targets = list(range(len(oper.dims[0])))
    if not isinstance(targets, Iterable):
        targets = [targets]
    if not all([isinstance(t, numbers.Integral) for t in targets]):
        raise TypeError("targets should be "
                        "an integer or a list of integer")
    # if targets has correct length
    if oper is not None:
        req_num = len(oper.dims[0])
        if len(targets) != req_num:
            raise ValueError("The given operator needs {} "
                             "target qutbis, "
                             "but {} given.".format(req_num, len(targets)))
    # if targets is smaller than N
    if n is not None:
        if not all([t < n for t in targets]):
            raise ValueError("Targets must be smaller than n={}.".format(n))
    return targets


def expand_operator(oper, n, targets, dims=None, cyclic_permutation=False):
    """
    Expand a qubits operator to one that acts on a N-qubit system.

    Parameters
    ----------
    oper : :class:`qutip.Qobj`
        An operator acts on qubits, the type of the :class:`qutip.Qobj`
        has to be an operator
        and the dimension matches the tensored qubit Hilbert space
        e.g. dims = ``[[2, 2, 2], [2, 2, 2]]``
    n : int
        The number of qubits in the system.
    targets : int or list of int
        The indices of qubits that are acted on.
    dims : list, optional
        A list of integer for the dimension of each composite system.
        E.g ``[2, 2, 2, 2, 2]`` for 5 qubits system. If None, qubits system
        will be the default option.
    cyclic_permutation : boolean, optional
        Expand for all cyclic permutation of the targets.
        E.g. if ``n=3`` and `oper` is a 2-qubit operator,
        the result will be a list of three operators,
        each acting on qubits 0 and 1, 1 and 2, 2 and 0.

    Returns
    -------
    expanded_oper : :class:`qutip.Qobj`
        The expanded qubits operator acting on a system with n qubits.

    Notes
    -----
    This is equivalent to gate_expand_1ton, gate_expand_2ton,
    gate_expand_3ton in ``qutip_qip.gate.py``, but works for any dimension.
    """
    if dims is None:
        dims = [2] * n
    targets = _targets_to_list(targets, oper=oper, n=n)
    # _check_qubits_oper(oper, dims=dims, targets=targets)

    # Call expand_operator for all cyclic permutation of the targets.
    if cyclic_permutation:
        oper_list = []
        for i in range(n):
            new_targets = np.mod(np.array(targets) + i, n)
            oper_list.append(
                expand_operator(oper, n=n, targets=new_targets, dims=dims))
        return oper_list

    # Generate the correct order for qubits permutation,
    # eg. if n = 5, targets = [3,0], the order is [1,2,3,0,4].
    # If the operator is cnot,
    # this order means that the 3rd qubit controls the 0th qubit.
    new_order = [0] * n
    for i, t in enumerate(targets):
        new_order[t] = i
    # allocate the rest qutbits (not targets) to the empty
    # position in new_order
    rest_pos = [q for q in list(range(n)) if q not in targets]
    rest_qubits = list(range(len(targets), n))
    for i, ind in enumerate(rest_pos):
        new_order[ind] = rest_qubits[i]
    id_list = [identity(dims[i]) for i in rest_pos]
    return permute(tensor(*([oper] + id_list)), [2] * n, new_order)
