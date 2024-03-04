import copy
import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint

from supergrad.quantum_system import Kronobj, LindbladObj
from supergrad.time_evolution.ode import _parse_hamiltonian, ode_expm


def mesolve(hamiltonian,
            rho0,
            tlist,
            c_ops=None,
            args=None,
            options={},
            solver='ode_expm',
            diag_ops=True):
    """
    JAX-backend implementation of `qutip.mesolve`, the following docstring is
    modified from `qutip`.

    Master equation evolution of a density matrix for a given Hamiltonian and
    set of collapse operators, or a Liouvillian.

    Evolve the density matrix (`rho0`) using a given Hamiltonian or
    Liouvillian (`LindbladObj`) and an optional set of collapse operators
    (`c_ops`), by integrating the set of ordinary differential equations
    that define the system. In the absence of collapse operators the system is
    evolved according to the unitary evolution of the Hamiltonian.

    The output is either the density matrix at arbitrary points in time
    (`tlist`).

    Args:
        hamiltonian (list, array):
            System Hamiltonian as a :obj:`Kronobj` , list of :obj:`Kronobj` and
            coefficient. List format and options can be found in description.
            or alternatively a system Liouvillian :obj:`LindbladObj`.

        rho0 (array):
            Initial density matrix (rho) or initial set of density matrices.
            Alternatively evolve a set of density matrix use vmap transformation.

        tlist (list, array):
            List of times for :math:`t`.

        c_ops : None / list of :class:`Kronobj` / list of :class:`LindbladObj`
            single collapse operator, or list of collapse operators, or a list
            of Liouvillian superoperators.

        args (dict, optional):
            Dictionary of scope parameters for time-dependent Hamiltonian.

        options (dict, optional):
            Options for the ODE solver.

        solver (str, optional): select differentiable ODE solver.
            `odeint` for using Dormand-Prince ODE integration,
            `ode_expm` for using the matrix exponentiation at each time step.

        diag_ops (bool):
            Use diagonal Liouvillian superoperators to accelerate matrix exponential
            this flag only use for `ode_expm` solver.

    Returns: Values of the solution density matrix at each time point in `tlist`,
        represented as an array with the same shape/structure as `rho0` except
        with a new leading axis of length `len(t)`. if a set of states be evolved
        simultaneously, the returns with another new leading axis of length `len(rho0)`.
    """
    if diag_ops:
        assert solver == 'ode_expm'

    if isinstance(hamiltonian, (Kronobj, jnp.ndarray, list)):
        h_td = _parse_hamiltonian(hamiltonian, tlist, args, diag_ops)
    else:
        raise ValueError('Invalid Hamiltonian type.')
    # Construct Liouvillian
    lind = LindbladObj()
    if c_ops is not None:
        for c_op in c_ops:
            if isinstance(c_op, LindbladObj):
                lind += c_op
            if isinstance(c_op, list):
                lind.add_lindblad_operator(*c_op)
            else:
                lind.add_lindblad_operator(c_op)
        if diag_ops:
            lind._diagonalize_super_operator()
    if rho0.ndim == 2:
        return _mesolve(rho0, tlist, h_td, lind, args, solver, options)
    elif rho0.ndim == 3:
        return jax.vmap(_mesolve, in_axes=[0, None, None, None, None, None,
                                           None])(rho0, tlist, h_td, lind, args,
                                                  solver, options)
    else:
        raise ValueError(
            '`rho0` is array or array of arrays representing the initial density'
        )


def _mesolve(rho0, tlist, h_td, lind: LindbladObj, args, solver, options):
    """The private function of `mesolve`"""

    tlist = jnp.asarray(tlist)
    # Start evolution
    if solver == 'ode_expm':

        def _func(rho, t):
            ham = h_td(t, args=args)
            lind_td = copy.deepcopy(lind)
            lind_td.add_liouvillian(ham)
            return lind_td

        rho_evo = ode_expm(_func, rho0, tlist, **options)
    elif solver == 'odeint':
        if isinstance(rho0, Kronobj):
            rho0 = rho0.full()

        def _func(rho, t):
            ham = h_td(t, args=args)
            lind_td = copy.deepcopy(lind)
            lind_td.add_liouvillian(ham)
            return lind_td @ rho

        rho_evo = odeint(_func, rho0, tlist, **options)
    else:
        raise ValueError('Unknown ode solver.')

    return rho_evo


def mesolve_final_states_w_basis_trans(hamiltonian,
                                       rho_list,
                                       tlist,
                                       transform_matrix=None,
                                       **kwargs):
    """Compute the final density matrices and applying basis transformation
    according to the transform_matrix.

    Args:
        hamiltonian: `jax.numpy.ndarray`, list
            System Hamiltonian as a :obj:`~ndarray` , list of :obj:`ndarray` and
            coefficient. List format is the same as `qutip.mesolve`.

        rho_list: `jax.numpy.ndarray`
            Initial density matrix (rho) or array of density matrices.

        tlist: array_like of float
            List of times for :math:`t`.

        transform_matrix: `jax.numpy.ndarray`
            The transform unitary from product basis to the desired basis.

        kwargs:
            Keyword arguments will be pass to `supergrad.time_evolution.mesolve`
    """
    # evolution in the product basis with cycle transform
    if transform_matrix is not None:
        # Evolve in multi-qubit eigenbasis
        rho_list = transform_matrix @ rho_list @ transform_matrix.conjugate(
        ).transpose()
        res = mesolve(hamiltonian, rho_list, tlist, **kwargs)
        density_array = jnp.conj(transform_matrix).T @ res[:, -1, :] @ transform_matrix
    else:
        # Evolve in multi-qubit product basis
        res = mesolve(hamiltonian, rho_list, tlist, **kwargs)
        density_array = res[:, -1, :]
    return density_array
