"""
The code is from `qutip_qip.operations`.
Operations on quantum circuits.
"""
import numbers
from collections.abc import Iterable
import numpy as np
import jax.numpy as jnp
import jax.scipy.linalg as jaxLA
from jax._src.numpy.util import implements

from supergrad.utils.operators import identity, sigmax, sigmay, sigmaz
from supergrad.utils.utility import tensor, fock_dm, permute

try:
    import qutip_qip.operations as qop
except ImportError:
    qop = None

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
    "expand_operator",
]

#
# Single Qubit Gates
#


@implements(getattr(qop, 'x_gate', None))
def x_gate(n=None, target=0):
    if n is not None:
        return gate_expand_1ton(x_gate(), n, target)
    return sigmax()


@implements(getattr(qop, 'y_gate', None))
def y_gate(n=None, target=0):
    if n is not None:
        return gate_expand_1ton(y_gate(), n, target)
    return sigmay()


@implements(getattr(qop, 'cy_gate', None))
def cy_gate(n=None, control=0, target=1):
    if (control == 1 and target == 0) and n is None:
        n = 2

    if n is not None:
        return gate_expand_2ton(cy_gate(), n, control, target)
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]])


@implements(getattr(qop, 'z_gate', None))
def z_gate(n=None, target=0):
    if n is not None:
        return gate_expand_1ton(z_gate(), n, target)
    return sigmaz()


@implements(getattr(qop, 'cz_gate', None))
def cz_gate(n=None, control=0, target=1):
    if (control == 1 and target == 0) and n is None:
        n = 2

    if n is not None:
        return gate_expand_2ton(cz_gate(), n, control, target)
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])


@implements(getattr(qop, 's_gate', None))
def s_gate(n=None, target=0):
    if n is not None:
        return gate_expand_1ton(s_gate(), n, target)
    return np.array([[1, 0], [0, 1j]])


@implements(getattr(qop, 'cs_gate', None))
def cs_gate(n=None, control=0, target=1):
    if (control == 1 and target == 0) and n is None:
        n = 2

    if n is not None:
        return gate_expand_2ton(cs_gate(), n, control, target)
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1j]])


@implements(getattr(qop, 't_gate', None))
def t_gate(n=None, target=0):
    if n is not None:
        return gate_expand_1ton(t_gate(), n, target)
    return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])


@implements(getattr(qop, 'ct_gate', None))
def ct_gate(n=None, control=0, target=1):
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


@implements(getattr(qop, 'rx', None))
def rx(phi, n=None, target=0):
    if n is not None:
        return gate_expand_1ton(rx(phi), n, target)
    return jnp.array([
        [jnp.cos(phi / 2), -1j * jnp.sin(phi / 2)],
        [-1j * jnp.sin(phi / 2), jnp.cos(phi / 2)],
    ])


@implements(getattr(qop, 'ry', None))
def ry(phi, n=None, target=0):
    if n is not None:
        return gate_expand_1ton(ry(phi), n, target)
    return jnp.array([
        [jnp.cos(phi / 2), -jnp.sin(phi / 2)],
        [jnp.sin(phi / 2), jnp.cos(phi / 2)],
    ])


@implements(getattr(qop, 'rz', None))
def rz(phi, n=None, target=0):
    if n is not None:
        return gate_expand_1ton(rz(phi), n, target)
    return jnp.array([[jnp.exp(-1j * phi / 2), 0], [0, jnp.exp(1j * phi / 2)]])


@implements(getattr(qop, 'sqrtnot', None))
def sqrtnot(n=None, target=0):
    if n is not None:
        return gate_expand_1ton(sqrtnot(), n, target)
    return np.array([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]])


@implements(getattr(qop, 'snot', None))
def snot(n=None, target=0):
    if n is not None:
        return gate_expand_1ton(snot(), n, target)
    return 1 / np.sqrt(2.0) * np.array([[1, 1], [1, -1]])


@implements(getattr(qop, 'phasegate', None))
def phasegate(theta, n=None, target=0):
    if n is not None:
        return gate_expand_1ton(phasegate(theta), n, target)
    return jnp.array([[1, 0], [0, jnp.exp(1.0j * theta)]])


@implements(getattr(qop, 'qrot', None))
def qrot(theta, phi, n=None, target=0):
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


@implements(getattr(qop, 'qasmu_gate', None))
def qasmu_gate(args, n=None, target=0):
    theta, phi, gamma = args
    if n is not None:
        return expand_operator(qasmu_gate([theta, phi, gamma]),
                               n=n,
                               targets=target)
    return jnp.array(rz(phi) @ ry(theta) @ rz(gamma))


#
# 2 Qubit Gates
#


@implements(getattr(qop, 'cphase', None))
def cphase(theta, n=2, control=0, target=1):
    if n < 1 or target < 0 or control < 0:
        raise ValueError("Minimum value: n=1, control=0 and target=0")

    if control >= n or target >= n:
        raise ValueError("control and target need to be smaller than N")

    u_list1 = [identity(2)] * n
    u_list2 = [identity(2)] * n

    u_list1[control] = fock_dm(2, 1)
    u_list1[target] = phasegate(theta)

    u_list2[control] = fock_dm(2, 0)

    u = tensor(*u_list1) + tensor(*u_list2)
    return u


@implements(getattr(qop, 'cnot', None))
def cnot(n=None, control=0, target=1):
    if (control == 1 and target == 0) and n is None:
        n = 2

    if n is not None:
        return gate_expand_2ton(cnot(), n, control, target)
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


@implements(getattr(qop, 'csign', None))
def csign(n=None, control=0, target=1):
    if (control == 1 and target == 0) and n is None:
        n = 2

    if n is not None:
        return gate_expand_2ton(csign(), n, control, target)
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])


@implements(getattr(qop, 'berkeley', None))
def berkeley(n=None, targets=[0, 1]):
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


@implements(getattr(qop, 'swapalpha', None))
def swapalpha(alpha, n=None, targets=[0, 1]):
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


@implements(getattr(qop, 'swap', None))
def swap(n=None, targets=[0, 1]):
    if targets != [0, 1] and n is None:
        n = 2

    if n is not None:
        return gate_expand_2ton(swap(), n, targets=targets)
    return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


@implements(getattr(qop, 'iswap', None))
def iswap(n=None, targets=[0, 1]):
    if targets != [0, 1] and n is None:
        n = 2

    if n is not None:
        return gate_expand_2ton(iswap(), n, targets=targets)
    return np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])


@implements(getattr(qop, 'sqrtswap', None))
def sqrtswap(n=None, targets=[0, 1]):
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


@implements(getattr(qop, 'sqrtiswap', None))
def sqrtiswap(n=None, targets=[0, 1]):
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


@implements(getattr(qop, 'molmer_sorensen', None))
def molmer_sorensen(theta, n=None, targets=[0, 1]):
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


@implements(getattr(qop, 'fredkin', None))
def fredkin(n=None, control=0, targets=[1, 2]):
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


@implements(getattr(qop, 'toffoli', None))
def toffoli(n=None, controls=[0, 1], target=2):
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


@implements(getattr(qop, 'rotation', None))
def rotation(op, phi, n=None, target=0):
    if n is not None:
        return gate_expand_1ton(rotation(op, phi), n, target)
    return jaxLA.expm(-1j * op * phi / 2)


@implements(getattr(qop, 'controlled_gate', None))
def controlled_gate(u, n=2, control=0, target=1, control_value=1):
    if [n, control, target] == [2, 0, 1]:
        return tensor(fock_dm(2, control_value), u) + tensor(
            fock_dm(2, 1 - control_value), identity(2))
    u2 = controlled_gate(u, control_value=control_value)
    return gate_expand_2ton(u2, n=n, control=control, target=target)


@implements(getattr(qop, 'globalphase', None))
def globalphase(theta, n=1):
    data = jnp.exp(1.0j * theta) * np.eye(2**n, 2**n, dtype=complex)
    return data


#
# Operation on Gates
#


@implements(getattr(qop, '_hamming_distance', None))
def _hamming_distance(x, bits=32):
    tot = 0
    while x:
        tot += 1
        x &= x - 1
    return tot


@implements(getattr(qop, 'hadamard_transform', None))
def hadamard_transform(n=1):
    data = [[1, 1], [1, -1]]
    H = np.array(data) / np.sqrt(2)
    return tensor(*([H] * n))


#
# Gate Expand
#


@implements(getattr(qop, 'gate_expand_1ton', None))
def gate_expand_1ton(u, n, target):
    if n < 1:
        raise ValueError("integer n must be larger or equal to 1")

    if target >= n:
        raise ValueError("target must be integer < integer N")
    return tensor(*([identity(2)] * (target) + [u] + [identity(2)] *
                    (n - target - 1)))


@implements(getattr(qop, 'gate_expand_2ton', None))
def gate_expand_2ton(u, n, control=None, target=None, targets=None):
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

    return permute(tensor(*([u] + [identity(2)] * (n - 2))), [2] * n, p)


@implements(getattr(qop, 'gate_expand_3toN', None))
def gate_expand_3toN(u, n, controls=[0, 1], target=2):
    if n < 3:
        raise ValueError("integer n must be larger or equal to 3")

    if controls[0] >= n or controls[1] >= n or target >= n:
        raise ValueError("control and not target is None."
                         " Must be integer < integer N")

    if (controls[0] == target or controls[1] == target or
            controls[0] == controls[1]):

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

    return permute(tensor(*([u] + [identity(2)] * (n - 3))), [2] * n, p)


@implements(getattr(qop, '_check_qubits_oper', None))
def _check_qubits_oper(oper, dims=None, targets=None):
    # if operator matches N
    if oper.dims[0] != oper.dims[1]:
        raise ValueError("The operator is not an "
                         "array with the same input and output dimensions.")
    # if operator dims matches the target dims
    if dims is not None and targets is not None:
        targ_dims = [dims[t] for t in targets]
        if oper.dims[0] != targ_dims:
            raise ValueError("The operator dims {} do not match "
                             "the target dims {}.".format(
                                 oper.dims[0], targ_dims))


@implements(getattr(qop, '_targets_to_list', None))
def _targets_to_list(targets, oper=None, n=None):
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


@implements(getattr(qop, 'expand_operator', None))
def expand_operator(oper, n, targets, dims=None, cyclic_permutation=False):
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
