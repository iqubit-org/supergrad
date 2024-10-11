from functools import partial
import operator

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm, eigh
from jax.numpy import interp
from jax.tree_util import tree_map
from jax.experimental.ode import odeint

import supergrad
from supergrad.quantum_system import KronObj, LindbladObj
from supergrad.utils.progress_bar import scan_tqdm


def _check_compatibility_mode(compatibility_mode, dim):
    if compatibility_mode:
        return jnp.eye(dim)
    else:
        return 0.


def _expm_mul_y(mat, y=None, *args, **kwargs):
    """Combine matrix exponentiation with matrix-vector multiplication.

    Args:
        mat: the matrix
        y: the vector be multiplied by the matrix on its left
        args: arguments be passed to `KronObj.expm`
        kwargs: the keyword arguments be passed to `KronObj.expm`
    """
    if isinstance(mat, (KronObj, LindbladObj)):
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
             custom_vjp=True,
             pb_fwd_ad=False,
             compatibility_mode=False):
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
        custom_vjp (string) : choose custom automatic differentiation `VJP` rule.
            Default is `None`, which means using the JAX framework to derive `VJP`.
            The following arguments are supported:
            a) True or `LCAM` (Recommended) : using the local continuous adjoint
            method to compute the inverse time evolution for both state and
            adjoint state.
            b) `CAM` : using the continuous adjoint method.
        pb_fwd_ad (bool) : whether to config progress bar as forward mode
            automatic differentiation.
        compatibility_mode (bool) : whether to use the compatible mode for `func`.
            We disable compatible mode to reduce computational cost when the evolution
            operator is not depend on `y`, so we could let `y` equals to 0 in the `func`.


    Returns:
        Values of the solution `y` (i.e. integrated system values) at each time
        point in `t`, represented as an array (or pytree of arrays) with the
        same shape/s

    """
    # convert the func to closure to make it custom_vjp compatible
    converted, consts = jax.custom_derivatives.closure_convert(
        func, y0, ts[0], *args)
    y_eye = _check_compatibility_mode(compatibility_mode, y0.shape[0])
    # compute optimal contraction path
    evo_term = func(y_eye, ts[0], *args)
    if trotter_order is not None and isinstance(evo_term,
                                                (KronObj, LindbladObj)):
        tn_expr = evo_term.compute_contraction_path(trotter_order=trotter_order)
    else:
        tn_expr = None
    if custom_vjp is not None:
        assert not pb_fwd_ad, 'Forward mode automatic differentiation is not supported.'
        _custom_ode_expm = jax.custom_vjp(_ode_expm,
                                          nondiff_argnums=(0, 1, 2, 3, 4, 5, 6))
        if custom_vjp is True or custom_vjp == 'LCAM':
            _custom_ode_expm.defvjp(_ode_expm_fwd, _ode_expm_rev)
        elif custom_vjp == 'CAM':
            _custom_ode_expm.defvjp(_ode_expm_fwd, _ode_expm_rev_continuous)
        else:
            raise ValueError('Invalid custom_vjp type.')
        return _custom_ode_expm(converted, astep, trotter_order, tn_expr,
                                progress_bar, False, compatibility_mode, y0, ts,
                                *args, *consts)
    else:
        return _ode_expm(converted, astep, trotter_order, tn_expr, progress_bar,
                         pb_fwd_ad, compatibility_mode, y0, ts, *args, *consts)


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 5, 6))
def _ode_expm(func, astep, trotter_order, tn_expr, progress_bar, pb_fwd_ad,
              compatibility_mode, y0, ts, *args):
    """The private function of the matrix exponentiation ODE solver.
    Operations that only depend on static arguments will be constant-folded in
    Python (during tracing), and so the corresponding argument values can be
    any Python object.
    """
    if isinstance(y0, KronObj):
        # convert y0 to the dense form
        y0 = y0.full()

    def scan_func(carry, target_t):
        """Evolve the propagator K to evaluation state at time point in ts."""

        def body_fun(state, next_nest_t):
            """Evolution of the discrete time sequence."""
            nest_y, nest_t = state
            # it's reasonable to choose the middle point of every time slice
            evo_term = func(y_eye, (nest_t + next_nest_t) / 2, *args)
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
            return carry, next_nest_t

        y, t = carry
        # decompose the total propagator K to the propagator U for the short time
        nest_ts = jnp.linspace(t, target_t, astep)
        state = [y, t]
        if progress_bar:
            wrapped_body_fun, wrapped_nest_ts = scan_tqdm(body_fun,
                                                          nest_ts[1:],
                                                          fwd_ad=pb_fwd_ad)
            final_state, _ = jax.lax.scan(wrapped_body_fun, state,
                                          wrapped_nest_ts)
        else:
            final_state, _ = jax.lax.scan(body_fun, state, nest_ts[1:])
        next_y = final_state[0]

        return final_state, next_y

    # initialize evolution
    init_carry = [y0, ts[0]]
    # check compatible mode
    y_eye = _check_compatibility_mode(compatibility_mode, y0.shape[0])
    _, ys = jax.lax.scan(scan_func, init_carry, ts[1:])

    return jnp.concatenate((y0[None], ys))


def _ode_expm_fwd(func, astep, trotter_order, tn_expr, progress_bar, pb_fwd_ad,
                  compatibility_mode, y0, ts, *args):
    """Forward pass of the matrix exponentiation ODE solver."""
    yf = _ode_expm(func, astep, trotter_order, tn_expr, progress_bar, pb_fwd_ad,
                   compatibility_mode, y0, ts, *args)
    return yf, (yf, ts, args)


def _ode_expm_rev(func, astep, trotter_order, tn_expr, progress_bar, pb_fwd_ad,
                  compatibility_mode, res, ys_bar):
    """Reverse pass of the matrix exponentiation ODE solver.
    Inverse the origin time evolution augmented with vjp_y, This approach only
    works for the Hermitian Hamiltonian .
    """

    def evolution_step(y, t, next_t, *args):
        evo_term = func(y_eye, (next_t + t) / 2, *args).dag()
        next_y = _expm_mul_y(evo_term * (t - next_t), y, trotter_order, tn_expr)
        return next_y

    def aug_evolution_step(y_bar, t, next_t, *args):
        evo_term = func(y_eye, (next_t + t) / 2, *args).conjugate()
        next_y = _expm_mul_y(evo_term * (next_t - t), y_bar, trotter_order,
                             tn_expr)
        return next_y

    def aug_dynamics(augmented_state, t, *args):
        """Backward ODE about vjp_t and vjp_args."""
        y, y_bar, *_ = augmented_state
        y_dot, vjpfun = jax.vjp(
            lambda _y, _t, _args: func(y_eye, _t, *_args) @ _y, y, t, args)
        return (-y_dot, *vjpfun(y_bar))

    def scan_func(carry, idx):
        """Evolve the propagator K to evaluation state at time point in ts."""

        def body_fun(state, next_nest_t):
            """Evolution of the discrete time sequence."""
            nest_y, nest_y_bar, nest_t_bar, nest_args_bar, nest_t = state
            # compute the vjp of hamiltonian function, back-propagation vjp_y
            next_nest_y, vjpfun = jax.vjp(evolution_step, nest_y, nest_t,
                                          next_nest_t, *args)
            # inverse the time evolution
            next_nest_y_bar = aug_evolution_step(nest_y_bar, nest_t,
                                                 next_nest_t, *args)

            # a) chain rule (accuracy but high cost)
            _, rev_t_bar, next_nest_t_bar, *next_nest_args_bar = vjpfun(
                next_nest_y_bar)
            next_nest_t_bar = nest_t_bar - rev_t_bar - next_nest_t_bar
            next_nest_args_bar = jax.tree_util.tree_map(lambda x, y: x - y,
                                                        nest_args_bar,
                                                        next_nest_args_bar)

            # b) compute augmented dynamics (trapezoid rule, low accuracy)
            # aug_state_l = [nest_y, nest_y_bar, nest_t_bar, nest_args_bar]
            # aug_state_h = [
            #     next_nest_y, next_nest_y_bar, nest_t_bar, nest_args_bar
            # ]
            # _, _, t_dy_l, args_dy_l = aug_dynamics(aug_state_l,
            #                                        (nest_t + next_nest_t) / 2,
            #                                        *args)
            # _, _, t_dy_h, args_dy_h = aug_dynamics(aug_state_h,
            #                                        (nest_t + next_nest_t) / 2,
            #                                        *args)
            # # cumulative vjp_t and vjp_args
            # next_nest_t_bar = nest_t_bar - (t_dy_l + t_dy_h) / 2 * (nest_t -
            #                                                         next_nest_t)
            # next_nest_args_bar = jax.tree_util.tree_map(
            #     lambda x, y, z: x + (y + z) / 2 * (nest_t - next_nest_t),
            #     nest_args_bar, list(args_dy_l), list(args_dy_h))
            # rev_t_bar = next_nest_t_bar

            carry = [
                next_nest_y, next_nest_y_bar, next_nest_t_bar,
                next_nest_args_bar, next_nest_t
            ]
            return carry, -rev_t_bar

        _, y_bar, t0_bar, args_bar = carry
        nest_ts = jnp.linspace(ts[idx], ts[idx - 1], astep)
        # Compute effect of moving measurement time
        # `t_bar` should not be complex as it represents time
        mea_t_bar = jnp.dot((func(y_eye, ts[idx], *args) @ ys[idx]).T,
                            ys_bar[idx]).real.reshape(-1)[0]
        t0_bar = t0_bar - mea_t_bar
        # state is [y, y_bar, t_bar, args_bar, t]
        state = [ys[idx], y_bar, t0_bar, args_bar, ts[idx]]
        if progress_bar:
            wrapped_body_fun, wrapped_nest_ts = scan_tqdm(body_fun,
                                                          nest_ts[1:],
                                                          fwd_ad=True)
            final_state, _ = jax.lax.scan(wrapped_body_fun, state,
                                          wrapped_nest_ts)
        else:
            final_state, _ = jax.lax.scan(body_fun, state, nest_ts[1:])
        # Add gradient from current output
        final_state[1] += ys_bar[idx - 1]
        # check symmetry by compare cached y with inverse computed one
        # jax.debug.print(
        #     f'Check symmetry: {jnp.allclose(final_state[0], ys[idx - 1])}')
        # update carry
        return final_state[:-1], mea_t_bar

    ys, ts, args = res
    # check compatible mode
    y_eye = _check_compatibility_mode(compatibility_mode, ys[-1].shape[0])
    # carry is [y, y_bar, t_bar, args_bar]
    # compute initial states
    init_carry = [
        ys[-1], ys_bar[-1], 0.,
        jax.tree_util.tree_map(jnp.zeros_like, list(args))
    ]
    # inverse the time evolution for adjoint state
    (_, y_bar, t0_bar,
     args_bar), rev_ts_bar = jax.lax.scan(scan_func, init_carry,
                                          jnp.arange(len(ts) - 1, 0, -1))
    ts_bar = jnp.concatenate([jnp.array([t0_bar]), rev_ts_bar[::-1]])
    return y_bar, ts_bar, *args_bar


def _ode_expm_rev_continuous(func, astep, trotter_order, tn_expr, progress_bar,
                             pb_fwd_ad, compatibility_mode, res, ys_bar):
    """Reverse pass of the matrix exponentiation ODE solver using continuous
    adjoint method.
    """

    def aug_dynamics(augmented_state, t, *args):
        """Original system augmented with vjp_y, vjp_t and vjp_args."""
        y, y_bar, *_ = augmented_state
        # `t` here is negatice time, so we need to negate again to get back to
        # normal time. See the `odeint` invocation in `scan_fun` below.
        y_dot, vjpfun = jax.vjp(
            lambda _y, _t, _args: func(y_eye, _t, *_args) @ _y, y, -t, args)
        return (-y_dot, *vjpfun(y_bar))

    ys, ts, args = res
    # check compatible mode
    y_eye = _check_compatibility_mode(compatibility_mode, ys[-1].shape[0])
    y_bar = ys_bar[-1]
    ts_bar = []
    t0_bar = 0.

    def scan_fun(carry, i):
        y_bar, t0_bar, args_bar = carry
        # Compute effect of moving measurement time
        # `t_bar` should not be complex as it represents time
        t_bar = jnp.dot((func(y_eye, ts[i], *args) @ ys[i]).T,
                        ys_bar[i]).real.reshape(-1)[0]
        t0_bar = t0_bar - t_bar
        # Run augmented system backwards to previous observation

        _, y_bar, t0_bar, args_bar = odeint(aug_dynamics,
                                            (ys[i], y_bar, t0_bar, args_bar),
                                            jnp.array([-ts[i], -ts[i - 1]]),
                                            *args,
                                            rtol=1e-10,
                                            atol=1e-10)
        # the tolerance of the ODE solve is set to 1e-10 to ensure the accuracy
        # it's hard to determine tolerance based on `astep` and `trotter_order`
        y_bar, t0_bar, args_bar = tree_map(operator.itemgetter(1),
                                           (y_bar, t0_bar, args_bar))
        # Add gradient from current output
        y_bar = y_bar + ys_bar[i - 1]
        return (y_bar, t0_bar, args_bar), t_bar

    init_carry = (ys_bar[-1], 0., tree_map(jnp.zeros_like, args))
    (y_bar, t0_bar,
     args_bar), rev_ts_bar = jax.lax.scan(scan_fun, init_carry,
                                          jnp.arange(len(ts) - 1, 0, -1))
    ts_bar = jnp.concatenate([jnp.array([t0_bar]), rev_ts_bar[::-1]])
    return (y_bar, ts_bar, *args_bar)


class EvoElement:
    """
    Internal type used to represent the time-dependent parts of a class.

    Availables "types" are

    1. ``jnp.ndarray``
    2. function

    Args:
        oper: the time-independent hamiltonian
        get_coeff: a callable that take (t, args) and return the coeff at that t
        coeff: The coeff as a ndarray as provided by the user.
        _type: type of `EvoElement`
    """

    def __init__(self, oper, get_coeff, coeff: jnp.ndarray, _type: str):
        self.oper = oper
        self.get_coeff = get_coeff
        self.coeff = coeff
        self._type = _type

    @classmethod
    def make(cls, list_):
        return cls(*list_)

    def __getitem__(self, i):
        if i == 0:
            return self.oper
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
        if isinstance(element, (KronObj, jnp.ndarray)):
            return 0
        try:
            op, td = element
        except (TypeError, ValueError) as exc:
            raise TypeError('Incorrect q_object specification.') from exc
        if callable(td):
            out = 1
        elif isinstance(td, (KronObj, jnp.ndarray)):
            if td.shape != tlist.shape:
                raise ValueError('Time sequences are not compatible.')
            out = 3
        else:
            raise TypeError('Incorrect q_object specification.')
        return out

    def _td_format_check(q_object):
        if isinstance(q_object, (KronObj, jnp.ndarray)):
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

    if diag_ops and isinstance(cte, KronObj):
        cte = cte.diagonalize_operator()
    eig_info = []
    if diag_ops:
        # diagonalize time dependent hamiltonian
        for part in ops:
            part_eig_info = []
            if isinstance(part.oper, KronObj):
                for nest_mat in part.oper.data:
                    mat = supergrad.tensor(*nest_mat)
                    part_eig_info.append(eigh(mat))
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
                    if isinstance(part.oper, KronObj):
                        # iteration through ham_td
                        for idx, sub_kronobj in enumerate(part.oper):
                            eig_val, eig_vec = part_eig_info[idx]
                            # construct KronObj in diag unitary method
                            op_t += KronObj(
                                [eig_val],
                                sub_kronobj.dims,
                                sub_kronobj.locs[0],
                                diag_unitary=eig_vec,
                            ) * part.get_coeff(t, args) * imul
                    else:
                        mat = part.oper
                        eig_val, eig_vec = eigh(mat)
                        # construct KronObj in diag unitary method
                        op_t += KronObj(
                            [eig_val],
                            diag_unitary=eig_vec,
                        ) * part.get_coeff(t, args) * imul
            else:
                for part in ops:
                    op_t += part.oper * part.get_coeff(t, args) * imul

        return op_t

    return hamiltonian
