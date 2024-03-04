"""
The code is from `qutip.operators`.
This module contains functions for generating a variety of commonly occuring
quantum operators.
"""

__all__ = [
    'jmat', 'spin_Jx', 'spin_Jy', 'spin_Jz', 'spin_Jm', 'spin_Jp', 'spin_J_set',
    'sigmap', 'sigmam', 'sigmax', 'sigmay', 'sigmaz', 'destroy', 'create',
    'qeye', 'identity', 'position', 'momentum', 'num', 'squeeze', 'squeezing',
    'displace', 'commutator', 'qutrit_ops', 'qdiags', 'phase', 'qzero',
    'charge', 'tunneling'
]

import numbers
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jaxLA

from supergrad.utils.utility import qutrit_basis


#
# Spin operators
#
def jmat(j, *args):
    """Higher-order spin operators:

    Parameters
    ----------
    j : float
        Spin of operator

    args : str
        Which operator to return 'x','y','z','+','-'.
        If no args given, then output is ['x','y','z']

    Returns
    -------
    jmat : ndarray
        ndarray for requested spin operator(s).


    Examples
    --------
    >>> jmat(1) # doctest: +SKIP
    [ Quantum object: dims = [[3], [3]], \
shape = [3, 3], type = oper, isHerm = True
    Qobj data =
    [[ 0.          0.70710678  0.        ]
     [ 0.70710678  0.          0.70710678]
     [ 0.          0.70710678  0.        ]]
     Quantum object: dims = [[3], [3]], \
shape = [3, 3], type = oper, isHerm = True
    Qobj data =
    [[ 0.+0.j          0.-0.70710678j  0.+0.j        ]
     [ 0.+0.70710678j  0.+0.j          0.-0.70710678j]
     [ 0.+0.j          0.+0.70710678j  0.+0.j        ]]
     Quantum object: dims = [[3], [3]], \
shape = [3, 3], type = oper, isHerm = True
    Qobj data =
    [[ 1.  0.  0.]
     [ 0.  0.  0.]
     [ 0.  0. -1.]]]


    Notes
    -----
    If no 'args' input, then returns array of ['x','y','z'] operators.

    """
    if (np.fix(2 * j) != 2 * j) or (j < 0):
        raise TypeError('j must be a non-negative integer or half-integer')

    if not args:
        return jmat(j, 'x'), jmat(j, 'y'), jmat(j, 'z')

    if args[0] == '+':
        A = _jplus(j)
    elif args[0] == '-':
        A = np.conj(_jplus(j)).T
    elif args[0] == 'x':
        A = 0.5 * (_jplus(j) + np.conj(_jplus(j)).T)
    elif args[0] == 'y':
        A = -0.5 * 1j * (_jplus(j) - np.conj(_jplus(j)).T)
    elif args[0] == 'z':
        A = _jz(j)
    else:
        raise TypeError('Invalid type')

    return A


def _jplus(j):
    """
    Internal functions for generating the data representing the J-plus
    operator.
    """
    m = np.arange(j, -j - 1, -1, dtype=complex)
    data = (np.sqrt(j * (j + 1.0) - (m + 1.0) * m))[1:]
    return np.diag(data, k=1)


def _jz(j):
    """
    Internal functions for generating the data representing the J-z operator.
    """
    N = int(2 * j + 1)
    data = np.array([j - k for k in range(N) if (j - k) != 0], dtype=complex)
    # Even shaped matrix
    if (N % 2 != 0):
        data = np.insert(data, int(j), 0)
    return np.diag(data)


#
# Spin j operators:
#
def spin_Jx(j):
    """Spin-j x operator

    Parameters
    ----------
    j : float
        Spin of operator

    Returns
    -------
    op : Qobj
        ``qobj`` representation of the operator.

    """
    return jmat(j, 'x')


def spin_Jy(j):
    """Spin-j y operator

    Parameters
    ----------
    j : float
        Spin of operator

    Returns
    -------
    op : Qobj
        ``qobj`` representation of the operator.

    """
    return jmat(j, 'y')


def spin_Jz(j):
    """Spin-j z operator

    Parameters
    ----------
    j : float
        Spin of operator

    Returns
    -------
    op : Qobj
        ``qobj`` representation of the operator.

    """
    return jmat(j, 'z')


def spin_Jm(j):
    """Spin-j annihilation operator

    Parameters
    ----------
    j : float
        Spin of operator

    Returns
    -------
    op : Qobj
        ``qobj`` representation of the operator.

    """
    return jmat(j, '-')


def spin_Jp(j):
    """Spin-j creation operator

    Parameters
    ----------
    j : float
        Spin of operator

    Returns
    -------
    op : Qobj
        ``qobj`` representation of the operator.

    """
    return jmat(j, '+')


def spin_J_set(j):
    """Set of spin-j operators (x, y, z)

    Parameters
    ----------
    j : float
        Spin of operators

    Returns
    -------
    list : list of Qobj
        list of ``qobj`` representating of the spin operator.

    """
    return jmat(j)


#
# Pauli spin 1/2 operators:
#
def sigmap():
    """Creation operator for Pauli spins.

    Examples
    --------
    >>> sigmap() # doctest: +SKIP
    Quantum object: dims = [[2], [2]], \
shape = [2, 2], type = oper, isHerm = False
    Qobj data =
    [[ 0.  1.]
     [ 0.  0.]]

    """
    return jmat(1 / 2., '+')


def sigmam():
    """Annihilation operator for Pauli spins.

    Examples
    --------
    >>> sigmam() # doctest: +SKIP
    Quantum object: dims = [[2], [2]], \
shape = [2, 2], type = oper, isHerm = False
    Qobj data =
    [[ 0.  0.]
     [ 1.  0.]]

    """
    return jmat(1 / 2., '-')


def sigmax():
    """Pauli spin 1/2 sigma-x operator

    Examples
    --------
    >>> sigmax() # doctest: +SKIP
    Quantum object: dims = [[2], [2]], \
shape = [2, 2], type = oper, isHerm = False
    Qobj data =
    [[ 0.  1.]
     [ 1.  0.]]

    """
    return 2 * jmat(1 / 2, 'x')


def sigmay():
    """Pauli spin 1/2 sigma-y operator.

    Examples
    --------
    >>> sigmay() # doctest: +SKIP
    Quantum object: dims = [[2], [2]], \
shape = [2, 2], type = oper, isHerm = True
    Qobj data =
    [[ 0.+0.j  0.-1.j]
     [ 0.+1.j  0.+0.j]]

    """
    return 2 * jmat(1 / 2, 'y')


def sigmaz():
    """Pauli spin 1/2 sigma-z operator.

    Examples
    --------
    >>> sigmaz() # doctest: +SKIP
    Quantum object: dims = [[2], [2]], \
shape = [2, 2], type = oper, isHerm = True
    Qobj data =
    [[ 1.  0.]
     [ 0. -1.]]

    """
    return 2 * jmat(1 / 2, 'z')


#
# DESTROY returns annihilation operator for N dimensional Hilbert space
# out = destroy(N), N is integer value &  N>0
#
def destroy(N, offset=0):
    '''Destruction (lowering) operator.

    Parameters
    ----------
    N : int
        Dimension of Hilbert space.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the operator.

    Returns
    -------
    oper : qobj
        Qobj for lowering operator.

    Examples
    --------
    >>> destroy(4) # doctest: +SKIP
    Quantum object: dims = [[4], [4]], \
shape = [4, 4], type = oper, isHerm = False
    Qobj data =
    [[ 0.00000000+0.j  1.00000000+0.j  0.00000000+0.j  0.00000000+0.j]
     [ 0.00000000+0.j  0.00000000+0.j  1.41421356+0.j  0.00000000+0.j]
     [ 0.00000000+0.j  0.00000000+0.j  0.00000000+0.j  1.73205081+0.j]
     [ 0.00000000+0.j  0.00000000+0.j  0.00000000+0.j  0.00000000+0.j]]

    '''
    if not isinstance(N, (int, np.integer)):  # raise error if N not integer
        raise ValueError("Hilbert space dimension must be integer value")
    data = np.sqrt(np.arange(offset + 1, N + offset, dtype=complex))
    return np.diag(data, k=1)


#
# create returns creation operator for N dimensional Hilbert space
# out = create(N), N is integer value &  N>0
#
def create(N, offset=0):
    '''Creation (raising) operator.

    Parameters
    ----------
    N : int
        Dimension of Hilbert space.

    Returns
    -------
    oper : qobj
        Qobj for raising operator.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the operator.

    Examples
    --------
    >>> create(4) # doctest: +SKIP
    Quantum object: dims = [[4], [4]], \
shape = [4, 4], type = oper, isHerm = False
    Qobj data =
    [[ 0.00000000+0.j  0.00000000+0.j  0.00000000+0.j  0.00000000+0.j]
     [ 1.00000000+0.j  0.00000000+0.j  0.00000000+0.j  0.00000000+0.j]
     [ 0.00000000+0.j  1.41421356+0.j  0.00000000+0.j  0.00000000+0.j]
     [ 0.00000000+0.j  0.00000000+0.j  1.73205081+0.j  0.00000000+0.j]]

    '''
    if not isinstance(N, (int, np.integer)):  # raise error if N not integer
        raise ValueError("Hilbert space dimension must be integer value")
    qo = destroy(N, offset=offset)  # create operator using destroy function
    return np.conj(qo).T


def _implicit_tensor_dimensions(dimensions):
    """
    Total flattened size and operator dimensions for operator creation routines
    that automatically perform tensor products.

    Parameters
    ----------
    dimensions : (int) or (list of int) or (list of list of int)
        First dimension of an operator which can create an implicit tensor
        product.  If the type is `int`, it is promoted first to `[dimensions]`.
        From there, it should be one of the two-elements `dims` parameter of a
        `qutip.Qobj` representing an `oper` or `super`, with possible tensor
        products.

    Returns
    -------
    size : int
        Dimension of backing matrix required to represent operator.
    """
    flat = jax.tree_util.tree_leaves(dimensions)
    if not all(isinstance(x, numbers.Integral) and x >= 0 for x in flat):
        raise ValueError("All dimensions must be integers >= 0")
    return np.prod(flat)


def qzero(dimensions):
    """
    Zero operator.

    Parameters
    ----------
    dimensions : (int) or (list of int) or (list of list of int)
        Dimension of Hilbert space. If provided as a list of ints, then the
        dimension is the product over this list, but the ``dims`` property of
        the new Qobj are set to this list.  This can produce either `oper` or
        `super` depending on the passed `dimensions`.

    Returns
    -------
    qzero : qobj
        Zero operator Qobj.

    """
    size = _implicit_tensor_dimensions(dimensions)
    # A sparse matrix with no data is equal to a zero matrix.
    return np.zeros((size, size))


#
# QEYE returns identity operator for a Hilbert space with dimensions dims.
# a = qeye(N), N is integer or list of integers & all elements >= 0
#
def qeye(dimensions):
    """
    Identity operator.

    Parameters
    ----------
    dimensions : (int) or (list of int) or (list of list of int)
        Dimension of Hilbert space. If provided as a list of ints, then the
        dimension is the product over this list, but the ``dims`` property of
        the new Qobj are set to this list.  This can produce either `oper` or
        `super` depending on the passed `dimensions`.

    Returns
    -------
    oper : qobj
        Identity operator Qobj.

    Examples
    --------
    >>> qeye(3) # doctest: +SKIP
    Quantum object: dims = [[3], [3]], shape = (3, 3), type = oper, \
isherm = True
    Qobj data =
    [[ 1.  0.  0.]
     [ 0.  1.  0.]
     [ 0.  0.  1.]]
    >>> qeye([2,2]) # doctest: +SKIP
    Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, \
isherm = True
    Qobj data =
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]

    """
    size = _implicit_tensor_dimensions(dimensions)
    return np.eye(size)


def identity(dims):
    """Identity operator. Alternative name to :func:`qeye`.

    Parameters
    ----------
    dimensions : (int) or (list of int) or (list of list of int)
        Dimension of Hilbert space. If provided as a list of ints, then the
        dimension is the product over this list, but the ``dims`` property of
        the new Qobj are set to this list.  This can produce either `oper` or
        `super` depending on the passed `dimensions`.

    Returns
    -------
    oper : qobj
        Identity operator Qobj.
    """
    return qeye(dims)


def position(N, offset=0):
    """
    Position operator x=1/sqrt(2)*(a+np.conj(a).T)

    Parameters
    ----------
    N : int
        Number of Fock states in Hilbert space.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the operator.

    Returns
    -------
    oper : qobj
        Position operator as Qobj.
    """
    a = destroy(N, offset=offset)
    return 1.0 / np.sqrt(2.0) * (a + np.conj(a).T)


def momentum(N, offset=0):
    """
    Momentum operator p=-1j/sqrt(2)*(a-np.conj(a).T)

    Parameters
    ----------
    N : int
        Number of Fock states in Hilbert space.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the operator.

    Returns
    -------
    oper : qobj
        Momentum operator as Qobj.
    """
    a = destroy(N, offset=offset)
    return -1j / np.sqrt(2.0) * (a - np.conj(a).T)


def num(N, offset=0):
    """Quantum object for number operator.

    Parameters
    ----------
    N : int
        The dimension of the Hilbert space.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the operator.

    Returns
    -------
    oper: qobj
        Qobj for number operator.

    Examples
    --------
    >>> num(4) # doctest: +SKIP
    Quantum object: dims = [[4], [4]], \
shape = [4, 4], type = oper, isHerm = True
    Qobj data =
    [[0 0 0 0]
     [0 1 0 0]
     [0 0 2 0]
     [0 0 0 3]]

    """
    data = np.arange(N, dtype=complex) + offset
    return np.diag(data)


def squeeze(N, z, offset=0):
    """Single-mode Squeezing operator.


    Parameters
    ----------
    N : int
        Dimension of hilbert space.

    z : float/complex
        Squeezing parameter.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the operator.

    Returns
    -------
    oper : :class:`qutip.qobj.Qobj`
        Squeezing operator.


    Examples
    --------
    >>> squeeze(4, 0.25) # doctest: +SKIP
    Quantum object: dims = [[4], [4]], \
shape = [4, 4], type = oper, isHerm = False
    Qobj data =
    [[ 0.98441565+0.j  0.00000000+0.j  0.17585742+0.j  0.00000000+0.j]
     [ 0.00000000+0.j  0.95349007+0.j  0.00000000+0.j  0.30142443+0.j]
     [-0.17585742+0.j  0.00000000+0.j  0.98441565+0.j  0.00000000+0.j]
     [ 0.00000000+0.j -0.30142443+0.j  0.00000000+0.j  0.95349007+0.j]]

    """
    a = destroy(N, offset=offset)
    op = (1 / 2.0) * np.conj(z) * (a @ a) - (1 / 2.0) * z * (np.conj(a).T) @ (
        np.conj(a).T)
    return jaxLA.expm(op)


def squeezing(a1, a2, z):
    """Generalized squeezing operator.

    .. math::

        S(z) = \\exp\\left(\\frac{1}{2}\\left(z^*a_1a_2
        - za_1^\\dagger a_2^\\dagger\\right)\\right)

    Parameters
    ----------
    a1 : :class:`qutip.qobj.Qobj`
        Operator 1.

    a2 : :class:`qutip.qobj.Qobj`
        Operator 2.

    z : float/complex
        Squeezing parameter.

    Returns
    -------
    oper : :class:`qutip.qobj.Qobj`
        Squeezing operator.

    """
    b = 0.5 * (np.conj(z) * (a1 @ a2) - z * (jnp.conj(a1).T @ jnp.conj(a2).T))
    return jaxLA.expm(b)


def displace(N, alpha, offset=0):
    """Single-mode displacement operator.

    Parameters
    ----------
    N : int
        Dimension of Hilbert space.

    alpha : float/complex
        Displacement amplitude.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the operator.

    Returns
    -------
    oper : qobj
        Displacement operator.

    Examples
    ---------
    >>> displace(4,0.25) # doctest: +SKIP
    Quantum object: dims = [[4], [4]], \
shape = [4, 4], type = oper, isHerm = False
    Qobj data =
    [[ 0.96923323+0.j -0.24230859+0.j  0.04282883+0.j -0.00626025+0.j]
     [ 0.24230859+0.j  0.90866411+0.j -0.33183303+0.j  0.07418172+0.j]
     [ 0.04282883+0.j  0.33183303+0.j  0.84809499+0.j -0.41083747+0.j]
     [ 0.00626025+0.j  0.07418172+0.j  0.41083747+0.j  0.90866411+0.j]]

    """
    a = destroy(N, offset=offset)
    D = (alpha * np.conj(a).T - np.conj(alpha) * a)
    return jaxLA.expm(D)


def commutator(A, B, kind="normal"):
    """
    Return the commutator of kind `kind` (normal, anti) of the
    two operators A and B.
    """
    if kind == 'normal':
        return A @ B - B @ A

    elif kind == 'anti':
        return A @ B + B @ A

    else:
        raise TypeError("Unknown commutator kind '%s'" % kind)


def qutrit_ops():
    """
    Operators for a three level system (qutrit).

    Returns
    -------
    opers: array
        `array` of qutrit operators.

    """
    out = np.empty((6,), dtype=object)
    one, two, three = qutrit_basis()
    out[0] = one * np.conj(one).T
    out[1] = two * np.conf(two).T
    out[2] = three * np.conj(three).T
    out[3] = one * np.conj(two).T
    out[4] = two * np.conj(three).T
    out[5] = three * np.conj(one).T
    return out


def qdiags(diagonals, offsets):
    """
    Constructs an operator from an array of diagonals.

    Parameters
    ----------
    diagonals : sequence of array_like
        Array of elements to place along the selected diagonals.

    offsets : sequence of ints
        Sequence for diagonals to be set:
            - k=0 main diagonal
            - k>0 kth upper diagonal
            - k<0 kth lower diagonal
    dims : list, optional
        Dimensions for operator

    shape : list, tuple, optional
        Shape of operator.  If omitted, a square operator large enough
        to contain the diagonals is generated.

    See Also
    --------
    numpy.diag : for usage information.

    Notes
    -----
    This function requires SciPy 0.11+.

    Examples
    --------
    >>> qdiags(sqrt(range(1, 4)), 1) # doctest: +SKIP
    Quantum object: dims = [[4], [4]], \
shape = [4, 4], type = oper, isherm = False
    Qobj data =
    [[ 0.          1.          0.          0.        ]
     [ 0.          0.          1.41421356  0.        ]
     [ 0.          0.          0.          1.73205081]
     [ 0.          0.          0.          0.        ]]

    """
    data = np.diag(diagonals, offsets)
    return data


def phase(N, phi0=0):
    """
    Single-mode Pegg-Barnett phase operator.

    Parameters
    ----------
    N : int
        Number of basis states in Hilbert space.
    phi0 : float
        Reference phase.

    Returns
    -------
    oper : qobj
        Phase operator with respect to reference phase.

    Notes
    -----
    The Pegg-Barnett phase operator is Hermitian on a truncated Hilbert space.

    """
    phim = phi0 + (2.0 * np.pi * np.arange(N)) / N  # discrete phase angles
    n = np.arange(N).reshape((N, 1))
    states = np.array(
        [np.sqrt(kk) / np.sqrt(N) * np.exp(1.0j * n * kk) for kk in phim])
    ops = np.array([np.outer(st, st.conj()) for st in states])
    return np.sum(ops, axis=0)


def charge(Nmax, Nmin=None, frac=1):
    """
    Generate the diagonal charge operator over charge states
    from Nmin to Nmax.

    Parameters
    ----------
    Nmax : int
        Maximum charge state to consider.

    Nmin : int (default = -Nmax)
        Lowest charge state to consider.

    frac : float (default = 1)
        Specify fractional charge if needed.

    Returns
    -------
    C : Qobj
        Charge operator over [Nmin,Nmax].

    Notes
    -----
    .. versionadded:: 3.2

    """
    if Nmin is None:
        Nmin = -Nmax
    diag = np.arange(Nmin, Nmax + 1, dtype=float)
    if frac != 1:
        diag *= frac
    return np.diag(diag)


def tunneling(N, m=1):
    """
    Tunneling operator with elements of the form
    :math:`\\sum |N><N+m| + |N+m><N|`.

    Parameters
    ----------
    N : int
        Number of basis states in Hilbert space.
    m : int (default = 1)
        Number of excitations in tunneling event.

    Returns
    -------
    T : Qobj
        Tunneling operator.

    Notes
    -----
    .. versionadded:: 3.2

    """
    diags = np.ones(N - m, dtype=int)
    return np.diag(diags, m) + np.diag(diags, -m)
