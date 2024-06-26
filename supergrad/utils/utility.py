from typing import Callable, Optional, List, Union
from functools import reduce
import json
import numbers
import numpy as np
import jax
import jax.numpy as jnp
import scipy.constants as const
import pprint

import supergrad


def _promote_to_zero_list(arg, length):
    """
    Ensure `arg` is a list of length `length`.  If `arg` is None it is promoted
    to `[0]*length`.  All other inputs are checked that they match the correct
    form.

    Returns:
        list_ : list
            A list of integers of length `length`.
    """
    if arg is None:
        arg = [0] * length
    elif not isinstance(arg, list):
        arg = [arg]
    if not len(arg) == length:
        raise ValueError("All list inputs must be the same length.")
    return arg


def basis(dimensions, n=None, offset=None):
    """Generates the vector representation of a Fock state.

    Args:
        dimensions: int or list of ints
            Number of Fock states in Hilbert space.  If a list, then the resultant
            object will be a tensor product over spaces with those dimensions.

        n: int or list of ints, optional (default 0 for all dimensions)
            Integer corresponding to desired number state, defaults to 0 for all
            dimensions if omitted.  The shape must match ``dimensions``, e.g. if
            ``dimensions`` is a list, then ``n`` must either be omitted or a list
            of equal length.

        offset: int or list of ints, optional (default 0 for all dimensions)
            The lowest number state that is included in the finite number state
            representation of the state in the relevant dimension.

    Returns:
        state: `ndarray` representing the requested number state ``|n>``.
    """
    # Promote all parameters to lists to simplify later logic.
    if not isinstance(dimensions, list):
        dimensions = [dimensions]
    n_dimensions = len(dimensions)
    # promote None to zero list
    if n is None:
        n = [0] * n_dimensions
    if offset is None:
        offset = [0] * n_dimensions
    ns = [
        m - off for m, off in zip(_promote_to_zero_list(n, n_dimensions),
                                  _promote_to_zero_list(offset, n_dimensions))
    ]
    if any((not isinstance(x, numbers.Integral)) or x < 0 for x in dimensions):
        raise ValueError("All dimensions must be >= 0.")
    if not all(0 <= n < dimension for n, dimension in zip(ns, dimensions)):
        raise ValueError("All basis indices must be "
                         "`offset <= n < dimension+offset`.")
    # construct Fock basis in KronObj
    location, size = 0, 1
    for m, dimension in zip(reversed(ns), reversed(dimensions)):
        location += m * size
        size *= dimension
    psi = np.zeros(size, dtype=complex)
    psi[location] = 1
    return psi.reshape((-1, 1))


def qutrit_basis():
    """Basis states for a three level system (qutrit)

    Returns:
        qstates : array
            Array of qutrit basis vectors

    """
    out = np.empty((3,), dtype=object)
    out[:] = [basis(3, 0), basis(3, 1), basis(3, 2)]
    return out


def ghz_state(n=3):
    """
    Returns the N-qubit GHZ-state.

    Args:
        N : int (default=3)
            Number of qubits in state

    Returns:
        array: N-qubit GHZ-state
    """
    state = (tensor(*[basis(2) for k in range(n)]) +
             tensor(*[basis(2, 1) for k in range(n)]))
    return state / jnp.sqrt(2)


def fock_dm(dimensions, n=None, offset=None):
    """Density matrix representation of a Fock state

    Constructed via outer product of :func:`supergrad.utils.utility.basis`.

    Args:
        dimensions : int or list of ints
            Number of Fock states in Hilbert space.  If a list, then the resultant
            object will be a tensor product over spaces with those dimensions.

        n : int or list of ints, optional (default 0 for all dimensions)
            Integer corresponding to desired number state, defaults to 0 for all
            dimensions if omitted.  The shape must match ``dimensions``, e.g. if
            ``dimensions`` is a list, then ``n`` must either be omitted or a list
            of equal length.

        offset : int or list of ints, optional (default 0 for all dimensions)
            The lowest number state that is included in the finite number state
            representation of the state in the relevant dimension.

    Returns:
        dm : ndarray
            Density matrix representation of Fock state.
    """
    psi = basis(dimensions, n, offset=offset)

    return psi * np.conj(psi).T


def create_state(ar_truncated_dim: np.ndarray, ar_ix: np.ndarray):
    """Creates N-Q basis.

    Args:
        ar_truncated_dim: the number of bands in each tensor
        ar_ix: the index of band in each tensor

    Returns:
        Array as the state `|` i1 i2 i3...>
    """
    return tensor_np(*[basis(x, ix) for x, ix in zip(ar_truncated_dim, ar_ix)])


def create_state_init(ar_truncated_dim: np.ndarray,
                      ar_ix_max: Union[np.ndarray, List[List[int]]]):
    """Creates initial states in a N-Q system.

    Args:
        ar_truncated_dim: the number of bands in each tensor
        ar_ix_max: [N,2] array as (qubit index, band maximum band index)
            to iterate all bands from 0 to maximum.
            [[0, 2], [1,2]] will create `|` 00> `|` 01> `|` 10> `|` 11> states.

    Returns:
        a list of initial states
    """
    ar_truncated_dim = np.asarray(ar_truncated_dim)
    list_max = np.ones(ar_truncated_dim.size, dtype=int)
    for ix, x in ar_ix_max:
        list_max[ix] = x

    ar_ix = np.stack(np.meshgrid(*[np.arange(x) for x in list_max],
                                 indexing="ij"),
                     axis=-1).reshape((-1, ar_truncated_dim.size))

    list_state_init = [
        basis(list(ar_truncated_dim), list(band_ix)) for band_ix in ar_ix
    ]
    return np.array(list_state_init), ar_ix


def compute_average_photon(power: float,
                           freq: float,
                           qc: Optional[float] = None,
                           kappa_c: Optional[float] = None) -> float:
    """Computes the average photon number in a cavity.

    Note, must specify one and only one of qc and kappa_c.

    Args:
        power (float): the input power, in unit dBm
        freq (float): the cavity frequency (normal frequency), in unit GHz
        qc (float): the coupling quality factor
        kappa_c (float): the coupling decay rate (normal frequency)

    Returns:
        float: The average photon number.
    """

    if (qc is not None and kappa_c is not None) or (qc is None and
                                                    kappa_c is None):
        raise ValueError("Must specify one and only one of qc and kappa_c")

    # Convert to joule/s
    power = 10**(power / 10) * 1e-3
    # Convert to kappa
    if kappa_c is None:
        kappa_c = freq * 1e9 / qc

    return power / (freq * const.h * 1e9) / kappa_c / 2 / np.pi


def tensor(*operators):
    """Calculates the tensor product of input operators by `jax.numpy.kron`.
    """
    return reduce(jnp.kron, operators)


def tensor_np(*operators):
    """Calculates the tensor product of input operators by `numpy.kron`.
    """
    return reduce(np.kron, operators)


def _parse_operator(operator, **kwargs) -> jnp.ndarray:

    if isinstance(operator, Callable):
        return operator(**kwargs)
    if isinstance(operator, jnp.ndarray):
        return operator
    raise TypeError("Unsupported operator type: ", type(operator))


def identity_wrap(
    operator: Union[Callable, jnp.ndarray],
    subsystem_list,
    subsystem=None,
    coeff=1,
    **kwargs,
) -> jnp.ndarray:
    """Wrap given operator in subspace `subsystem` in identity operators to form
    full Hilbert-space operator.

    Args:
        operator:
            operator acting in Hilbert space of `subsystem`
        subsystem_list (list):
            list of all subsystems relevant to the Hilbert space.
        subsystem (string):
            subsystem where diagonal operator is defined. set `None` if operator
            is `Callable`
        coeff (float): coefficient of operator
        kwargs:
            kwargs for calling operator.
    """
    if subsystem is None:
        if isinstance(operator, Callable):
            subsystem = operator.__self__
        else:
            raise ValueError('Unsupported `subsystem` argument')
    operator = _parse_operator(operator, **kwargs)  # Call operator
    dims = [the_subsys.dim for the_subsys in subsystem_list]
    subsystem_index = subsystem_list.index(subsystem)

    return supergrad.KronObj([coeff * operator], dims, [subsystem_index])


def permute(unitary, dims, order):
    """ Permutes the tensor structure of a composite object in the given order.

    Args:
        unitary (array): 2d-array of a composite object
        dims (list): list of int
            dimensions of a composite object
        order (list): list of int
            the given order of the tensor structure. Note that `len(dims)` == `len(order)`,
            and the order in range [0, `len(dims)`-1]
    """
    # construct subscripts and operands
    unitary_dim = list(dims)
    shape = unitary.shape
    unitary_tensor = unitary.reshape(unitary_dim * 2)
    script = list(np.arange(len(order)) + 1) + list(-1 * np.arange(len(order)) -
                                                    1)
    locs = [loc + 1 for loc in order]  # index start from 1
    aux_locs = [-loc for loc in locs]
    target_script = locs + aux_locs
    return jnp.einsum(unitary_tensor, script, target_script).reshape(shape)


def reduced_unitary(unitary, dims, partial_trace):
    """Calculate the partial trace to extract the unitary of target subsystem,
    when dealing with composite systems.

    Args:
        unitary (array): 2d-array of a composite object
        dims (list): list of int
            dimensions of a composite object
        partial_trace (list): list of int
            the order of subsystem will be traced. Note the order in range
            [0, `len(dims)`-1]
    """
    unitary_dim = list(dims)
    assert np.prod(dims) == unitary.shape[0]
    unitary_tensor = unitary.reshape(unitary_dim * 2)
    script = list(np.arange(len(dims)) + 1)
    script_2 = list(-1 * np.arange(len(dims)) - 1)
    locs = list(np.arange(len(dims)))
    for idx in sorted(partial_trace, reverse=True):
        script_2[idx] *= -1  # set script for calculating trace
        locs.pop(idx)
        dims.pop(idx)
    target_script = [loc + 1 for loc in locs] + [-loc - 1 for loc in locs]
    return jnp.einsum(unitary_tensor, script + script_2, target_script).reshape(
        np.prod(dims), np.prod(dims)) / 2**len(partial_trace)


def const_init(val):
    """
    return an intializer which generates constant value or arrays

    Args:
        val: the constant value

    Returns:
        A callable of shape, dtype to generate a constant value or array.
    """

    if val is None:
        val = 1

    def initializer(shape, dtype):
        if shape:
            return jnp.full(shape, val, dtype)
        else:
            return jnp.array(val, dtype=dtype)

    return initializer


def convert_device_array(x):
    """Convert a device array to a list or a number.
    This is for convert parameters to json compatible format.
    """

    if x.size == 1:
        return float(x)
    else:
        return x.tolist()


def convert_to_json_compatible(x):
    """
    Convert input to json compatible format.
    In particular, it will replace DeviceArray or np.array in the input.
    """

    return jax.tree_util.tree_map(convert_device_array, x)


def convert_to_haiku_dict(x):
    """
    Convert input to haiku compatible dictionary.
    In particular, it will replace input float to the DeviceArray.
    """

    def _parse_params(x):
        if isinstance(x, float):
            return jnp.array(x)
        else:
            return x

    return jax.tree_util.tree_map(_parse_params, x)


def dump_params(params, fp):
    """Dump parameters to json."""
    json.dump(convert_to_json_compatible(params), fp)


def load_params(fp):
    """Load parameters from json."""
    return convert_to_haiku_dict(json.load(fp))


def tree_print(t):
    """Print jax pytree in a human readable way."""

    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(convert_to_json_compatible(t))
