from typing import Tuple
import jax
from jax.numpy import pi
import jax.numpy as jnp
import jax.scipy.linalg as jlin
from functools import partial

from supergrad.utils.gates import sigmax, sigmay, sigmaz, tensor, identity
from supergrad.utils.fidelity import conv_sq_angles_to_u

# Convert angles to xx,yy,zz
# Eq. 8 magic basis
m = jnp.array([[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]]) / jnp.sqrt(2)

list_pauli = [identity(2), sigmax(), sigmay(), sigmaz()]
ar_pauli_xyz = jnp.stack([sigmax(), sigmay(), sigmaz()], axis=0)
ar_half_xyz = jnp.stack([jlin.expm(1j * pi / 4 * x) for x in ar_pauli_xyz], axis=0)


def so4_to_su2su2(u: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Converts a SO4 matrix to SU2.

    Following R.R. Tucci, quant-ph/0412072. Theorem 2.

    Args:
        u: a square 4x4 matrix follows the condition in the paper

    Returns:
        two 2x2 matrices following :math:`M^{\\dagger}(A\\otimes B) M = U`
    """
    u = m @ u @ m.conj().T

    tol_same = 1e-7

    a = jnp.zeros((2, 2), dtype=jnp.complex128)
    b = jnp.zeros((2, 2), dtype=jnp.complex128)
    a11 = (u[0:2, 0:2] @ u[2:4, 2:4].conj().T)[0, 0]
    a12 = -(u[0:2, 2:4] @ u[2:4, 0:2].conj().T)[0, 0]
    a11a12 = (u[0:2, 0:2] @ u[0:2, 2:4].conj().T)[0, 0]
    a = a.at[0, 0].set(jnp.sqrt(a11))
    a = a.at[0, 1].set(jnp.sqrt(a12))

    def keep_a01(a):
        return a

    def reverse_a01(a):
        a = a.at[0, 1].set(-a[0, 1])
        return a

    a = jax.lax.cond(jnp.abs(a[0, 0] * a[0, 1].conj() - a11a12) > tol_same, reverse_a01, keep_a01, a)

    a = a.at[1, 0].set(-a[0, 1].conj())
    a = a.at[1, 1].set(a[0, 0].conj())

    def large_a00(u, a):
        return u[0:2, 0:2] / a[0, 0]

    def small_a00(u, a):
        return u[0:2, 2:4] / a[0, 1]

    b = jax.lax.cond(jnp.abs(a[0, 0]) > jnp.abs(a[0, 1]), large_a00, small_a00, u, a)

    return a, b


@partial(jax.jit, static_argnums=[1, 2])
def kak1_decomposition(u: jax.Array, all_positive: bool = True, canonical: bool = False):
    r"""Decomposes a 4x4 unitary as KAK1 of Cartan's KAK decomposition.

    Following R.R. Tucci, quant-ph/0412072.

    The result follows
    :math:`tensor(A1, A0)exp(i(I k_0+\sigma_{xx}k_1+\sigma_{yy}k_2+\sigma_{zz}k_3))tensor(B1,B0)=U`

    With `all_positive=True`, the canonical class vector are all positive,
    while `all_positive=False` is to mimic the behavior in Cirq KAK decomposition that the largest value < :math:`\pi/4`.

    Args:
        u: 4x4 unitary
        all_positive: False: largest value in [0, pi/4],  the smallest value can be negative.
            True: largest value in [0, pi/2], all values, all values positive
        canonical: whether to generate the canonical class vector

    Returns:
        KAK vector, matrix A1, A0, B1, B0 if `output_1q` is True.
    """
    if len(u.shape) != 2 or u.shape[0] != 4 or u.shape[1] != 4:
        raise ValueError("U is not a 4x4 matrix")

    # if not is_unitary(u):
    #    raise ValueError("U is not an unitary matrix")

    # Normalize U
    unorm = u / jlin.det(u) ** 0.25

    # Transform to magic basis
    u = m.conj().T @ unorm @ m

    # Create orthogonal matrices
    xr = (u + u.conj()) / 2
    xi = (u - u.conj()) / 2j
    # q = jnp.concatenate([jnp.concatenate([xr, xi], axis=1), jnp.concatenate([-xi, xr], axis=1)], axis=0)
    a = xr
    b = xi
    ua, sa, vah = jlin.svd(a, full_matrices=False)

    # Note we must ensure Uc and Vc in SO(4)
    # which requires det(U)=1
    # Here we only get the orthogonality, O(4), det(U)=+-1
    # if not, we need to transform to make them in SO(4).

    # 2 Cases of  Det(U) , Det(V)
    # same signs: one can change P to adjust both sign to +1
    # different signs: S must be changed, instead of fully positive, real values in  SVD convention
    # We change the first one to be negative
    def same_sign(sa, vah):
        return sa, vah

    def opposite_sign(sa, vah):
        sa = sa.at[0].multiply(-1)
        vah = vah.at[0, :].multiply(-1)
        return sa, vah

    sa, vah = jax.lax.cond(jlin.det(ua) * jlin.det(vah) < 0, opposite_sign, same_sign, sa, vah)

    # Construct common U V from Eckart-Young
    d = ua.conj().T @ a @ vah.conj().T
    g = ua.conj().T @ b @ vah.conj().T

    eigs, eigv = jlin.eigh(g)
    p = eigv
    uc = ua @ p

    # Reverse the sign of P if det(Uc) is -1, which means det(Vc) is -1, too)
    def change_o4(p, uc):
        p = p.at[:, 1].multiply(-1)
        uc = ua @ p
        return p, uc

    def change_so4(p, uc):
        return p, uc

    p, uc = jax.lax.cond(jlin.det(uc) < 0, change_o4, change_so4, p, uc)

    # Now SO(4) should be guaranteed
    vc = vah.conj().T @ p

    # Rewrite U as Uc, Vc and a diagonal with angles (i, zi, iz and zz)
    eitheta = uc.conj().T @ u @ vc
    theta = jnp.angle(eitheta.diagonal())

    # Eq 33
    gamma = jnp.array([[1, 1, -1, 1], [1, 1, 1, -1], [1, -1, -1, -1], [1, -1, 1, 1]])
    gammainv = gamma.T / 4
    k = gammainv @ theta
    k = k[1:4]

    a1, a0 = so4_to_su2su2(uc)
    b1, b0 = so4_to_su2su2(vc.conj().T)

    if canonical:
        k, a1, a0, b1, b0 = canonicalize_kak1_with_1q(k, a1, a0, b1, b0, all_positive=all_positive)
    return k, a1, a0, b1, b0


@partial(jax.jit, static_argnums=[0, 1], donate_argnames=["k", "a1", "a0", "b1", "b0"])
def class_vector_int_shift(ix: int, v: jax.Array, k: jax.Array, a1: jax.Array, a0: jax.Array, b1: jax.Array,
                           b0: jax.Array):
    """Shift the class vector (int form) by pi/2.

    Args:
        k: the class vector
        ix: the index to shift
        v: shift value
        a1: U(2) after on Q1
        a0: U(2) after on Q0
        b1: U(2) before on Q1
        b0: U(2) before on Q0

    Returns:
        the shifted vector k and a1,a0,b1,b0.
    """
    k = k.at[ix].add(v)
    s, r = jnp.divmod(v, 2)

    # Shift pi/2 (pi means rotated back and
    tol_same = 1e-7

    def shift1(a1, a0):
        a1 = a1 @ ar_pauli_xyz[ix]
        a0 = a0 @ ar_pauli_xyz[ix]
        return a1, a0

    def shift0(a1, a0):
        return a1, a0

    a1, a0 = jax.lax.cond(jnp.abs(r - 1) < tol_same, shift1, shift0, a1, a0)

    return k, a1, a0, b1, b0


@partial(jax.jit, static_argnums=[0, 1], donate_argnames=["k", "a1", "a0", "b1", "b0"])
def class_vector_int_reverse(ix1: int, ix2: int, k: jax.Array, a1: jax.Array, a0: jax.Array, b1: jax.Array,
                             b0: jax.Array):
    """Reverse two values in the class vector.

    Note one must reverse two simultaneously to keep only 1Q terms.

    Args:
        k: the class vector
        ix1: the index to reverse
        ix2: the index to reverse
        a1: U(2) after on Q1
        a0: U(2) after on Q0
        b1: U(2) before on Q1
        b0: U(2) before on Q0

    Returns:
        the shifted vector k and a1,a0,b1,b0.
    """
    k = k.at[ix1].multiply(-1)
    k = k.at[ix2].multiply(-1)

    # Apply a reverse on third axis and reverse back
    ix3 = 3 - ix1 - ix2
    p = ar_pauli_xyz[ix3]
    a1 = a1 @ p
    b1 = p @ b1
    return k, a1, a0, b1, b0


@partial(jax.jit, static_argnums=[0, 1], donate_argnames=["k", "a1", "a0", "b1", "b0"])
def class_vector_int_swap(ix1: int, ix2: int, k: jax.Array, a1: jax.Array, a0: jax.Array, b1: jax.Array, b0: jax.Array):
    """Swap two values in the class vector.

    Args:
        k: the class vector
        ix1: the index to swap
        ix2: the index to swap
        a1: U(2) after on Q1
        a0: U(2) after on Q0
        b1: U(2) before on Q1
        b0: U(2) before on Q0

    Returns:
        the shifted vector k and a1,a0,b1,b0.
    """
    t = k[ix2]
    k = k.at[ix2].set(k[ix1])
    k = k.at[ix1].set(t)
    ix3 = 3 - ix1 - ix2

    # Apply pi/4 shift on the third axis of two qubits
    h = ar_half_xyz[ix3]
    a1 = a1 @ h.conj().T
    a0 = a0 @ h.conj().T
    b1 = h @ b1
    b0 = h @ b0
    return k, a1, a0, b1, b0


@partial(jax.jit, static_argnums=[5])
def canonicalize_kak1_with_1q(k0: jax.Array, a1: jax.Array, a0: jax.Array, b1: jax.Array, b0: jax.Array,
                              all_positive: bool = True):
    r"""Shifts the KAK1 vector to a canonical class vector.

    Following R.R. Tucci, quant-ph/0412072. Sec 4
    With `all_positive=True`, the canonical class vector are all positive,
    while with `all_positive=False` the largest value :math:`\le\pi/4`.

    Args:
        k0: the input vector
        a1: U(2) after on Q1
        a0: U(2) after on Q0
        b1: U(2) before on Q1
        b0: U(2) before on Q0
        all_positive: False: largest value in [0, pi/4],  the smallest value can be negative.
            True: largest value in [0, pi/2], all values, all values positive

    Returns:
        a canonical class vector k,  and adjusted a1,a0,b1,b0.
    """
    # Turn to integers and shift kx, ky, kz to 0-2
    k = k0 / (pi / 2)
    shift, k1 = jnp.divmod(k, 1)
    shift = shift.astype(jnp.int64)
    for ix in range(3):
        k, a1, a0, b1, b0 = class_vector_int_shift(ix, shift[ix] * -1, k, a1, a0, b1, b0)

    def keep_swap(ix1, ix2, k, a1, a0, b1, b0):
        return k, a1, a0, b1, b0

    # Swap to let kx>ky>kz
    for ix1, ix2 in [(0, 1), (1, 2), (0, 1)]:
        k, a1, a0, b1, b0 = jax.lax.cond(k[ix1] < k[ix2], class_vector_int_swap, keep_swap, ix1, ix2, k, a1, a0, b1, b0)

    def swap_and_reverse_and_shift(k, a1, a0, b1, b0):
        k, a1, a0, b1, b0 = class_vector_int_swap(0, 1, k, a1, a0, b1, b0)
        k, a1, a0, b1, b0 = class_vector_int_reverse(0, 1, k, a1, a0, b1, b0)
        k, a1, a0, b1, b0 = class_vector_int_shift(0, 1, k, a1, a0, b1, b0)
        k, a1, a0, b1, b0 = class_vector_int_shift(1, 1, k, a1, a0, b1, b0)
        return k, a1, a0, b1, b0

    def keep(k, a1, a0, b1, b0):
        return k, a1, a0, b1, b0

    # Reverse if too large, sort again
    k, a1, a0, b1, b0 = jax.lax.cond(k[0] + k[1] > 1, swap_and_reverse_and_shift, keep, k, a1, a0, b1, b0)
    k, a1, a0, b1, b0 = jax.lax.cond(k[1] < k[2], class_vector_int_swap, keep_swap, 1, 2, k, a1, a0, b1, b0)
    k, a1, a0, b1, b0 = jax.lax.cond(k[0] < k[1], class_vector_int_swap, keep_swap, 0, 1, k, a1, a0, b1, b0)

    def reverse_and_shift(k, a1, a0, b1, b0):
        k, a1, a0, b1, b0 = class_vector_int_reverse(0, 2, k, a1, a0, b1, b0)
        k, a1, a0, b1, b0 = class_vector_int_shift(0, 1, k, a1, a0, b1, b0)
        return k, a1, a0, b1, b0

    # Special case
    k, a1, a0, b1, b0 = jax.lax.cond(jnp.logical_and(k[0] > 0.5, jnp.logical_or(k[2] == 0, not all_positive)),
                                     reverse_and_shift, keep, k, a1, a0, b1, b0)
    k, a1, a0, b1, b0 = jax.lax.cond(k[0] < k[1], class_vector_int_swap, keep_swap, 0, 1, k, a1, a0, b1, b0)

    return k * (pi / 2), a1, a0, b1, b0

def conv_k_to_u_interaction(k: jax.Array) -> jax.Array:
    r"""Construct the interaction part :math:`exp(i \vec{k} \cdot \vec{\Sigma}) in KAK1 decomposition.

    Args:
        k: KAK1 class vector

    Returns:
        the 4x4 unitary as the interaction part.
    """
    terms = [k[ix] * tensor(ar_pauli_xyz[ix], ar_pauli_xyz[ix]) for ix in range(len(ar_half_xyz))]
    mid = jlin.expm(1j * (terms[0] + terms[1] + terms[2]))
    return mid

def conv_k_1q_to_u(k: jax.Array, a1: jax.Array, a0: jax.Array, b1: jax.Array, b0: jax.Array) -> jax.Array:
    """Constructs the unitary matrix from KAK1 coefficients and 1Q operations.

    Args:
        k: KAK1 class vector
        a1: U(2) after on Q1
        a0: U(2) after on Q0
        b1: U(2) before on Q1
        b0: U(2) before on Q0

    Returns:
        the 4x4 unitary
    """
    mid = conv_k_to_u_interaction(k)
    u = tensor(a1, a0) @ mid @ tensor(b1, b0)
    return u


def conv_k_1q_angles_to_u(k: jax.Array, a1: jax.Array, a0: jax.Array, b1: jax.Array, b0: jax.Array) -> jax.Array:
    """Constructs the unitary matrix from KAK1 coefficients and 1Q angles.

    The angle definition is identical to :func`supergrad.utils.fidelity.construct_sq_from_angles`.

    Args:
        k: KAK1 class vector
        a1: angles of U(2) after on Q1
        a0: angles of U(2) after on Q0
        b1: angles of U(2) before on Q1
        b0: angles of U(2) before on Q0

    Returns:
        the 4x4 unitary
    """
    return conv_k_1q_to_u(k, *[conv_sq_angles_to_u(x) for x in (a1, a0, b1, b0)])
