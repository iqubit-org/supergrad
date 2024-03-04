from functools import partial
import jax
from jax import config
import jax.numpy as jnp
from jax.scipy.linalg import expm, eigh
from jax.numpy import interp

from supergrad.quantum_system import Kronobj, LindbladObj
from supergrad.utils.progress_bar import scan_tqdm

# We need fp64 precision
config.update("jax_enable_x64", True)


def _expm_mul_y(mat, y=None, *args, **kwargs):
    """Combine matrix exponentiation with matrix-vector multiplication.

    Args:
        mat: the matrix
        y: the vector be multiplied by the matrix on its left
        args: arguments be passed to `Kronobj.expm`
        kwargs: the keyword arguments be passed to `Kronobj.expm`
    """
    if isinstance(mat, (Kronobj, LindbladObj)):
        return mat.expm(y, *args, **kwargs)
    else:
        return expm(mat) @ y


def ode_expm(func,
             y0,
             ts,
             *args,
             astep=100,
             trotter_order=None,
             progress_bar=False,
             fwd_ad=False):
    """ODE solver using the matrix exponentiation for the propagators at each time
    step.

    Args:
        func: function to evaluate the time derivative of the solution `y` at
            time `t` as `func(y, t, *args)`, producing the same shape/structure
            as `y0`.
        y0: array or pytree of arrays representing the initial value for the state.
        ts: array of float times for evaluation, like `jnp.linspace(0., 10., 101)`,
            in which the values must be strictly increasing.
        *args: tuple of additional arguments for `func`, which must be arrays
            scalars, or (nested) standard Python containers (tuples, lists,
            dicts, namedtuples, i.e. pytrees) of those types.
        astep: int, absolute number of steps to take for each timepoint (optional).
        trotter_order (complex int): the order of suzuki-trotter decomposition.
            The following arguments are supported,
            a) `None`, calculating matrix exponentiation without trotter decomposition
            b) `1`, first order trotter decomposition
            c) `2`, second order trotter decomposition
            d) `4`, 4th order real decomposition
            e) `4j`, 4th order complex decomposition
        progress_bar (bool) : whether to display a progress bar (optional).
        fwd_ad (bool) : whether to use forward-mode automatic differentiation
            (optional, only used for progress bar).


    Returns:
        Values of the solution `y` (i.e. integrated system values) at each time
        point in `t`, represented as an array (or pytree of arrays) with the
        same shape/s

    """
    return _ode_expm(func, astep, trotter_order, progress_bar, fwd_ad, y0, ts,
                     *args)


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
def _ode_expm(func, astep, trotter_order, progress_bar, fwd_ad, y0, ts, *args):
    """The private function of the matrix exponentiation ODE solver.
    Operations that only depend on static arguments will be constant-folded in
    Python (during tracing), and so the corresponding argument values can be
    any Python object.
    """

    def func_(y, t):
        return func(y, t, *args)

    if isinstance(y0, Kronobj):
        # convert y0 to the dense form
        y0 = y0.full()

    def scan_func(carry, target_t):
        """Evolve the propagator K to evaluation state at time point in ts."""

        def body_fun(state, next_nest_t):
            """Evolution of the discrete time sequence."""
            nest_y, nest_t = state
            # it's reasonable to choose the middle point of every time slice
            evo_term = func_(jnp.eye(dim), (nest_t + next_nest_t) / 2)
            # evolution a timestep for each timepoint
            next_nest_y = _expm_mul_y(evo_term * (next_nest_t - nest_t), nest_y,
                                      trotter_order, tn_expr)
            # normalize exp(-iHt)
            # if nest_y.shape[1] == 1:
            #     norm = jnp.linalg.norm(next_nest_y)
            #     next_nest_y = next_nest_y / norm**2
            # elif nest_y.shape[0] == nest_y.shape[1]:
            #     norm = jnp.trace(next_nest_y)
            #     next_nest_y = next_nest_y / norm
            # else:
            #     raise ValueError('Shape is not compatible.')
            carry = [next_nest_y, next_nest_t]
            return carry, next_nest_y

        y, t = carry
        # decompose the total propagator K to the propagator U for the short time
        nest_ts = jnp.linspace(t, target_t, astep)
        state = [y, t]
        if progress_bar:
            wrapped_body_fun, wrapped_nest_ts = scan_tqdm(body_fun,
                                                          nest_ts[1:],
                                                          fwd_ad=fwd_ad)
            states, _ = jax.lax.scan(wrapped_body_fun, state, wrapped_nest_ts)
        else:
            states, _ = jax.lax.scan(body_fun, state, nest_ts[1:])
        next_y = states[0]

        return states, next_y

    # interval
    # initialize evolution
    init_carry = [y0, ts[0]]
    dim = y0.shape[0]
    # compute optimal contraction path
    evo_term = func_(jnp.eye(dim), ts[0])
    if trotter_order is not None and isinstance(evo_term,
                                                (Kronobj, LindbladObj)):
        tn_expr = evo_term.compute_contraction_path(trotter_order=trotter_order)
    else:
        tn_expr = None
    _, ys = jax.lax.scan(scan_func, init_carry, ts[1:])

    return jnp.concatenate((y0[None], ys))


class EvoElement:
    """
    JAX-backend implementation of `qutip.sesolve`

    Internal type used to represent the time-dependent parts of a
    :class:`~QobjEvo`.

    Availables "types" are

    1. ``jnp.ndarray``
    2. function

    Args:
        qobj: the time-independent hamiltonian
        get_coeff: a callable that take (t, args) and return the coeff at that t
        coeff: The coeff as a ndarray as provided by the user.
        _type: type of `EvoElement`
    """

    def __init__(self, qobj, get_coeff, coeff: jnp.ndarray, _type: str):
        self.qobj = qobj
        self.get_coeff = get_coeff
        self.coeff = coeff
        self._type = _type

    @classmethod
    def make(cls, list_):
        return cls(*list_)

    def __getitem__(self, i):
        if i == 0:
            return self.qobj
        if i == 1:
            return self.get_coeff
        if i == 2:
            return self.coeff
        if i == 3:
            return self._type


class _InterpolateWrapper:
    """Using JAX interpolate since ODE solver only accept
    linearly distributed tlist.
    """

    def __init__(self, tlist, coeff, args=None):
        self.coeff = coeff
        self.tlist = tlist

    def __call__(self, t, args={}):
        return interp(t, self.tlist, self.coeff, left=0., right=0.)


def _parse_hamiltonian(q_object, tlist: jnp.ndarray, args: dict,
                       diag_ops: bool):
    """Prepare the time dependent Hamiltonian for the solver.

    Args:
        q_object (list, array):
            The time-dependent description of the quantum object.  This is of
            the same format as the first parameter to the general ODE solvers;
            in general, it is a list of ``[ndarray, time_dependence]`` pairs
            that are summed to make the whole object.  The ``time_dependence``
            can be any of the formats discussed in the previous section.

        args (dict, optional):
            Mapping of ``{str: object}``, discussed in greater detail above.
            The strings can be any valid Python identifier, and the objects are
            of the consumable types.  See the previous section for details on
            the "magic" names used to access solver internals.

        tlist (list, array):
            List of the times any numpy-array coefficients describe.  This is
            used only in at least one of the time dependence in ``H`` is given
            in Numpy-array format.  The times must be sorted, but need not be
            equidistant.  Values inbetween will be interpolated.

    Returns:
        func(Callable): Return a single `ndarray` at the given time `t`.
    """

    def _td_op_type(element):
        if isinstance(element, (Kronobj, jnp.ndarray)):
            return 0
        try:
            op, td = element
        except (TypeError, ValueError) as exc:
            raise TypeError('Incorrect q_object specification.') from exc
        if callable(td):
            out = 1
        elif isinstance(td, (Kronobj, jnp.ndarray)):
            if td.shape != tlist.shape:
                raise ValueError('Time sequences are not compatible.')
            out = 3
        else:
            raise TypeError('Incorrect q_object specification.')
        return out

    def _td_format_check(q_object):
        if isinstance(q_object, (Kronobj, jnp.ndarray)):
            return 0
        if isinstance(q_object, list):
            return [_td_op_type(element) for element in q_object]
        raise TypeError('Incorrect q_object specification.')

    # Initialize parameters
    const = False
    cte = None
    tlist = jnp.asarray(tlist)

    # Attempt to determine if a 2-element list is a single, time-dependent
    # operator, or a list with 2 possibly time-dependent elements.
    if isinstance(q_object, list) and len(q_object) == 2:
        try:
            # Test if parsing succeeds on this as a single element.
            _td_op_type(q_object)
            q_object = [q_object]
        except (TypeError, ValueError):
            pass

    op_type = _td_format_check(q_object)
    ops = []

    if isinstance(op_type, int):
        if op_type == 0:
            cte = q_object
            const = True
    else:
        for type_, op in zip(op_type, q_object):
            if type_ == 0:
                if cte is None:
                    cte = op
                else:
                    cte += op
            elif type_ == 1:
                ops.append(EvoElement(op[0], op[1], op[1], 'func'))
            elif type_ == 3:
                ops.append(
                    EvoElement(op[0],
                               _InterpolateWrapper(tlist, op[1], args=args),
                               jnp.copy(op[1]), tlist))

        if not ops:
            const = True

    if diag_ops and isinstance(cte, Kronobj):
        cte._diagonalize_operator()
    eig_info = []
    if diag_ops:
        for part in ops:
            part_eig_info = []
            if isinstance(part.qobj, Kronobj):
                for nest_mat in part.qobj.data:
                    part_eig_info.append(eigh(nest_mat[0]))
            eig_info.append(part_eig_info)

    # Define a callable Hamiltonian function.
    def hamiltonian(t, imul=1, args={}):
        """Return a single `ndarray` at the given time ``t``.

        Args:
            t: the given time.
            imul: the term should multiply to hamiltonian.
        """
        op_t = imul * cte
        if not const:
            if diag_ops:
                for partid, part in enumerate(ops):
                    # diagonalize time dependent hamiltonian
                    part_eig_info = eig_info[partid]
                    if isinstance(part.qobj, Kronobj):
                        # iteration through ham_td
                        for idx, sub_kronobj in enumerate(part.qobj):
                            eig_val, eig_vec = part_eig_info[idx]
                            # construct Kronobj in diag unitary method
                            op_t += Kronobj(
                                [jnp.diag(eig_val)],
                                sub_kronobj.dims,
                                sub_kronobj.locs[0],
                                diag_unitary=[
                                    eig_vec,
                                    part.get_coeff(t, args) * imul
                                ],
                            )
                    else:
                        mat = part.qobj
                        eig_val, eig_vec = eigh(mat)
                        # construct Kronobj in diag unitary method
                        op_t += Kronobj(
                            [eig_val],
                            diag_unitary=[
                                eig_vec,
                                part.get_coeff(t, args) * imul
                            ],
                        )
            else:
                for part in ops:
                    op_t += part.qobj * part.get_coeff(t, args) * imul

        return op_t

    return hamiltonian
