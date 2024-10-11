import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint

from supergrad.quantum_system import KronObj
from supergrad.time_evolution.ode import _parse_hamiltonian, ode_expm
from supergrad.utils.sharding import get_sharding


def sesolve(hamiltonian,
            psi0: jnp.ndarray,
            tlist: jnp.ndarray,
            args=None,
            options={},
            solver='ode_expm'):
    """
    Schrödinger equation evolution of a state vector or a set of state vectors
    for a given Hamiltonian.

    Evolve the state vector (``psi0``) using a given Hamiltonian (``H``).
    Alternatively evolve a set of states use vmap transformation.

    **Time-dependence formats**

    There are two major formats for specifying a time-dependent scalar:

    - Python function
    - array

    For function format, the function signature must be
    ``f(t: float, args: dict) -> complex``, for example

    .. code-block:: python

        def f1_t(t, args):
            return np.exp(-1j * t * args["w1"])

        def f2_t(t, args):
            return np.cos(t * args["w2"])

        H = _parse_sesolve([H0, [H1, f1_t], [H2, f2_t]], args={"w1":1., "w2":2.})

    For numpy array format, the array must be an 1d of dtype ``jnp.float64`` or
    ``jnp.complex128``.  A list of times (``jnp.float64``) at which the
    coeffients must be given as ``tlist``.  The coeffients array must have the
    same length as the tlist.  The times of the tlist do not need to be
    equidistant, but must be sorted.  By default, a linear interpolation
    will be used for the coefficient at time t. Examples of array-format usage are:

    .. code-block:: python

        tlist = np.logspace(-5, 0, 100)
        H = _parse_sesolve([H0, [H1, np.exp(-1j * tlist)], [H2, np.cos(2. * tlist)]],
                    tlist=tlist)

    Mixing time formats is allowed.  It is not possible to parse a single
    hamiltonian that contains different ``tlist`` values.

    Args:
        hamiltonian (list, array):
            System Hamiltonian as a :obj:`ndarray` , list of :obj:`ndarray` and
            coefficient. List format and options can be found in description.

        psi0 (array):
            Initial state vector (ket) or initial unitary operator ``psi0 = U``.
            Alternatively evolve a set of states use vmap transformation.

        tlist (list, array):
            List of times for :math:`t`.

        args (dict, optional):
            Dictionary of scope parameters for time-dependent Hamiltonians.

        options (dict, optional):
            Options for the ODE solver.
            `multi_device` (bool):
            `True` for enable multi-device parallelism evolution support,
            if many states are evolved simultaneously, the compiler can
            automatically batch the states to different devices.
            `diag_ops` (bool):
            It is only for `ode_expm`, if `True` the Hermitian Hamiltonian
            will be convert to diagonal form before exponentiation.

        solver (str, optional): select differentiable ODE solver.
            `odeint` for using Dormand-Prince ODE integration,
            `ode_expm` for using the matrix exponentiation at each time step.

    Returns: Values of the solution state vector at each time point in `tlist`,
        represented as an array with the same shape/structure as `psi0` except
        with a new leading axis of length `len(t)`. if a set of states be evolved
        simultaneously, the returns with another new leading axis of length `len(psi0)`.
    """
    diag_ops = options.pop('diag_ops', False)
    if diag_ops:
        assert solver == 'ode_expm'
    # convert options to hashable tuple
    options = tuple(options.items())

    if isinstance(hamiltonian, (KronObj, jnp.ndarray, list)):
        h_td = _parse_hamiltonian(hamiltonian, tlist, args, diag_ops)
    else:
        raise ValueError('Invalid Hamiltonian type.')

    if psi0.ndim == 2:
        return _sesolve(psi0, tlist, h_td, args, solver, options)
    elif psi0.ndim == 3:
        _simd_sesolve = jax.vmap(_sesolve,
                                 in_axes=(0, None, None, None, None, None))
        psi0 = jax.lax.with_sharding_constraint(psi0,
                                                get_sharding('p', None, None))
        return _simd_sesolve(psi0, tlist, h_td, args, solver, options)
    else:
        raise ValueError(
            '`psi0` is array or array of arrays representing the initial state')


def _sesolve(psi0, tlist, h_td, args, solver, options):
    """The private function of `sesolve`"""

    tlist = jnp.asarray(tlist)
    # Start evolution
    if solver == 'ode_expm':
        psi_evo = ode_expm(lambda psi, t: -1.0j * h_td(t, args=args), psi0,
                           tlist, **dict(options))
    elif solver == 'odeint':
        if isinstance(psi0, KronObj):
            psi0 = psi0.full()
        psi_evo = odeint(lambda psi, t: -1.0j * h_td(t, args=args) @ psi, psi0,
                         tlist, **dict(options))
    else:
        raise ValueError('Unknown ode solver.')

    return psi_evo


def sesolve_final_states_w_basis_trans(hamiltonian,
                                       psi_list,
                                       tlist,
                                       transform_matrix=None,
                                       **kwargs):
    """Compute the final states and applying basis transformation according to
    the transform_matrix.
    The time evolution is computed using sesolve in the product basis.

    Args:
        hamiltonian: `jax.numpy.ndarray`, list
            System Hamiltonian as a :obj:`~ndarray`, list of :obj:`ndarray` and
            coefficient. List format is the same as `qutip.sesolve`.

        psi_list: `jax.numpy.ndarray`
            Initial state vector (ket) or array of state vectors.

        tlist: array_like of float
            List of times for :math:`t`.

        transform_matrix: `jax.numpy.ndarray`
            The transform unitary from product basis to the desired basis.

        kwargs:
            Keyword arguments will be pass to `supergrad.time_evolution.sesolve`
    """
    # Find the corresponding basis in psi_list by the population
    # Note it should be convert to real type to compare
    if psi_list.ndim == 3:
        pop = np.abs(psi_list).real**2
        tuple_ar_ix = tuple(np.argmax(pop, axis=1).flatten())
    # evolution in the product basis with cycle transform
    if transform_matrix is not None:
        # Evolve in multi-qubit eigenbasis

        @jax.jit
        def _evolving_states(transform_matrix, psi_list):
            # Fix bug in gradient computation with multi-device
            # we use jit to tell the compiler transform_matrix must be provided
            # before the evolution
            psi_list = transform_matrix @ psi_list
            res = sesolve(hamiltonian, psi_list, tlist, **kwargs)
            if psi_list.ndim == 3:
                states = res[:, -1, :]
            else:
                states = res[-1, :]
            return jnp.conj(transform_matrix).T @ states

        states = _evolving_states(transform_matrix, psi_list)
    else:
        # Evolve in multi-qubit product basis
        res = sesolve(hamiltonian, psi_list, tlist, **kwargs)
        if psi_list.ndim == 3:
            states = res[:, -1, :]
        else:
            states = res[-1, :]
    if psi_list.ndim == 3:
        return states[:, tuple_ar_ix, 0].T
    else:
        return states
