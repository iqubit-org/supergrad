"""
The code is from `qutip.operators`.
This module contains functions for generating a variety of commonly occurring
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
from jax._src.numpy.util import implements

from supergrad.utils.utility import qutrit_basis

try:
    import qutip.core.operators as qop
except ImportError:
    qop = None


#
# Spin operators
#
@implements(getattr(qop, 'jmat', None))
def jmat(j, *args):
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


@implements(getattr(qop, '_jplus', None))
def _jplus(j):
    m = np.arange(j, -j - 1, -1, dtype=complex)
    data = (np.sqrt(j * (j + 1.0) - (m + 1.0) * m))[1:]
    return np.diag(data, k=1)


@implements(getattr(qop, '_jz', None))
def _jz(j):
    n = int(2 * j + 1)
    data = np.array([j - k for k in range(n) if (j - k) != 0], dtype=complex)
    # Even shaped matrix
    if (n % 2 != 0):
        data = np.insert(data, int(j), 0)
    return np.diag(data)


#
# Spin j operators:
#
@implements(getattr(qop, 'spin_Jx', None))
def spin_Jx(j):
    return jmat(j, 'x')


@implements(getattr(qop, 'spin_Jy', None))
def spin_Jy(j):
    return jmat(j, 'y')


@implements(getattr(qop, 'spin_Jz', None))
def spin_Jz(j):
    return jmat(j, 'z')


@implements(getattr(qop, 'spin_Jm', None))
def spin_Jm(j):
    return jmat(j, '-')


@implements(getattr(qop, 'spin_Jp', None))
def spin_Jp(j):
    return jmat(j, '+')


@implements(getattr(qop, 'spin_J_set', None))
def spin_J_set(j):
    return jmat(j)


#
# Pauli spin 1/2 operators:
#
@implements(getattr(qop, 'sigmap', None))
def sigmap():
    return jmat(1 / 2., '+')


@implements(getattr(qop, 'sigmam', None))
def sigmam():
    return jmat(1 / 2., '-')


@implements(getattr(qop, 'sigmax', None))
def sigmax():
    return 2 * jmat(1 / 2, 'x')


@implements(getattr(qop, 'sigmay', None))
def sigmay():
    return 2 * jmat(1 / 2, 'y')


@implements(getattr(qop, 'sigmaz', None))
def sigmaz():
    return 2 * jmat(1 / 2, 'z')


#
# DESTROY returns annihilation operator for n dimensional Hilbert space
# out = destroy(n), n is integer value &  n>0
#
@implements(getattr(qop, 'destroy', None))
def destroy(n, offset=0):
    if not isinstance(n, (int, np.integer)):  # raise error if n not integer
        raise ValueError("Hilbert space dimension must be integer value")
    data = np.sqrt(np.arange(offset + 1, n + offset, dtype=complex))
    return np.diag(data, k=1)


#
# create returns creation operator for n dimensional Hilbert space
# out = create(n), n is integer value &  n>0
#
@implements(getattr(qop, 'create', None))
def create(n, offset=0):
    if not isinstance(n, (int, np.integer)):  # raise error if n not integer
        raise ValueError("Hilbert space dimension must be integer value")
    qo = destroy(n, offset=offset)  # create operator using destroy function
    return np.conj(qo).T


@implements(getattr(qop, '_implicit_tensor_dimensions', None))
def _implicit_tensor_dimensions(dimensions):
    flat = jax.tree_util.tree_leaves(dimensions)
    if not all(isinstance(x, numbers.Integral) and x >= 0 for x in flat):
        raise ValueError("All dimensions must be integers >= 0")
    return np.prod(flat)


@implements(getattr(qop, 'qzero', None))
def qzero(dimensions):
    size = _implicit_tensor_dimensions(dimensions)
    # A sparse matrix with no data is equal to a zero matrix.
    return np.zeros((size, size))


#
# QEYE returns identity operator for a Hilbert space with dimensions dims.
# a = qeye(n), n is integer or list of integers & all elements >= 0
#
@implements(getattr(qop, 'qeye', None))
def qeye(dimensions):
    size = _implicit_tensor_dimensions(dimensions)
    return np.eye(size)


@implements(getattr(qop, 'identity', None))
def identity(dims):
    return qeye(dims)


@implements(getattr(qop, 'position', None))
def position(n, offset=0):
    a = destroy(n, offset=offset)
    return 1.0 / np.sqrt(2.0) * (a + np.conj(a).T)


@implements(getattr(qop, 'momentum', None))
def momentum(n, offset=0):
    a = destroy(n, offset=offset)
    return -1j / np.sqrt(2.0) * (a - np.conj(a).T)


@implements(getattr(qop, 'num', None))
def num(n, offset=0):
    data = np.arange(n, dtype=complex) + offset
    return np.diag(data)


@implements(getattr(qop, 'squeeze', None))
def squeeze(n, z, offset=0):
    a = destroy(n, offset=offset)
    op = (1 / 2.0) * np.conj(z) * (a @ a) - (1 / 2.0) * z * (np.conj(a).T) @ (
        np.conj(a).T)
    return jaxLA.expm(op)


@implements(getattr(qop, 'squeezing', None))
def squeezing(a1, a2, z):
    b = 0.5 * (np.conj(z) * (a1 @ a2) - z * (jnp.conj(a1).T @ jnp.conj(a2).T))
    return jaxLA.expm(b)


@implements(getattr(qop, 'displace', None))
def displace(n, alpha, offset=0):
    a = destroy(n, offset=offset)
    d = (alpha * np.conj(a).T - np.conj(alpha) * a)
    return jaxLA.expm(d)


@implements(getattr(qop, 'commutator', None))
def commutator(A, B, kind="normal"):
    if kind == 'normal':
        return A @ B - B @ A

    elif kind == 'anti':
        return A @ B + B @ A

    else:
        raise TypeError("Unknown commutator kind '%s'" % kind)


@implements(getattr(qop, 'qutrit_ops', None))
def qutrit_ops():
    out = np.empty((6,), dtype=object)
    one, two, three = qutrit_basis()
    out[0] = one * np.conj(one).T
    out[1] = two * np.conf(two).T
    out[2] = three * np.conj(three).T
    out[3] = one * np.conj(two).T
    out[4] = two * np.conj(three).T
    out[5] = three * np.conj(one).T
    return out


@implements(getattr(qop, 'qdiags', None))
def qdiags(diagonals, offsets):
    data = np.diag(diagonals, offsets)
    return data


@implements(getattr(qop, 'phase', None))
def phase(n, phi0=0):
    phim = phi0 + (2.0 * np.pi * np.arange(n)) / n  # discrete phase angles
    n = np.arange(n).reshape((n, 1))
    states = np.array(
        [np.sqrt(kk) / np.sqrt(n) * np.exp(1.0j * n * kk) for kk in phim])
    ops = np.array([np.outer(st, st.conj()) for st in states])
    return np.sum(ops, axis=0)


@implements(getattr(qop, 'charge', None))
def charge(nmax, nmin=None, frac=1):
    if nmin is None:
        nmin = -nmax
    diag = np.arange(nmin, nmax + 1, dtype=float)
    if frac != 1:
        diag *= frac
    return np.diag(diag)


@implements(getattr(qop, 'tunneling', None))
def tunneling(n, m=1):
    diags = np.ones(n - m, dtype=int)
    return np.diag(diags, m) + np.diag(diags, -m)
