from typing import Callable
from functools import partial
import numpy as np
from scipy.optimize import minimize
import jax
import jax.numpy as jnp
from jax._src import linear_util as lu
from jax._src.api import _jacfwd_unravel, _std_basis, _jvp
from jax._src.api_util import argnums_partial
from jax.flatten_util import ravel_pytree
from jax.example_libraries import optimizers
from jax._src.numpy.util import implements
import pprint

from supergrad.utils.utility import convert_to_json_compatible


def value_and_gradfwd(fun: Callable, argnums=0):
    """Constructed by analogy to value_and_grad
    """

    def jacfun(*args, **kwargs):
        f = lu.wrap_init(fun, kwargs)
        f_partial, dyn_args = argnums_partial(
            f, argnums, args, require_static_args_hashable=False)
        pushfwd: Callable = partial(_jvp, f_partial, dyn_args)
        y, jac = jax.vmap(pushfwd, out_axes=(None, -1))(_std_basis(dyn_args))
        jac_tree = jax.tree_util.tree_map(partial(_jacfwd_unravel, dyn_args), y,
                                          jac)
        return y, jac_tree[0]

    return jacfun


def scipy_optimize_wrapper(fun: Callable,
                           unflatten: Callable,
                           jac: bool = True,
                           jit: bool = True,
                           fwd_ad: bool = False,
                           logging: bool = False):
    """A wrapper transform function (written in JAX) to scipy compatible version.

    Args:
        fun (callable): input function
        unflatten (callable): a callable for unflattening a 1D vector of the
            same length back to a pytree.
        jac (bool): True for using Gradient-based optimize algorithm,
            False for Gradient-free optimize algorithm.
        jit (bool): True for just-in-time compile.
        fwd_ad(bool): True for using forward-mode auto diff.
        logging (bool): True for logging current loss and parameters.
    """
    if jac:
        if fwd_ad:
            val_grad = value_and_gradfwd(fun)
        else:
            val_grad = jax.value_and_grad(fun)
        if jit:
            val_grad = jax.jit(val_grad)

        def scipy_fun(*args, **kwargs):
            """Decorated scipy function

            Return:
                a tuple (f, g) containing the objective function and the gradient.
            """
            # ravel x0 parameters pytree
            args_list = list(args)
            args_list[0] = unflatten(args[0])
            val, grad = val_grad(*args_list, **kwargs)
            # transform DeviceArray to numpy array
            scipy_val = np.asarray(jax.tree_util.tree_leaves(val),
                                   dtype=np.float64)
            # fix ValueError
            grad = jax.tree_util.tree_map(lambda leaf: leaf.tolist(), grad)
            scipy_grad = np.asarray(jax.tree_util.tree_leaves(grad),
                                    dtype=np.float64)
            if logging:
                global step
                pp = pprint.PrettyPrinter(indent=2)
                print(f'step: {step}')
                print('parameters:')
                pp.pprint(convert_to_json_compatible(args_list[0]))
                print('gradient:')
                pp.pprint(grad)
                print(f'loss: {float(scipy_val):f}')
                step += 1
            return scipy_val, scipy_grad

        return scipy_fun
    else:
        if jit:
            fun = jax.jit(fun)

        def scipy_fun(*args, **kwargs):
            """Decorated scipy function

            Return:
                the objective function
            """
            # ravel x0 parameters pytree
            args_list = list(args)
            args_list[0] = unflatten(args[0])
            val = fun(*args_list, **kwargs)
            # transform DeviceArray to numpy array
            scipy_val = np.asarray(jax.tree_util.tree_leaves(val),
                                   dtype=np.float64)
            if logging:
                pp = pprint.PrettyPrinter(indent=2)
                print('parameters:')
                pp.pprint(convert_to_json_compatible(args_list[0]))
                print(f'loss: {float(scipy_val):f}')
            return scipy_val

        return scipy_fun


def scipy_hess_wrapper(fun: Callable, unflatten: Callable, jit: bool = True):
    """A wrapper transform function (written in JAX) to scipy compatible version.

    Args:
        fun (callable): input function
        unflatten (callable): a callable for unflattening a 1D vector of the
            same length back to a pytree.
        jit (bool): True for just-in-time compile.
    """
    # using a composition of `jacfwd` and `jacrev` to compute dense Hessian
    hess = jax.hessian(fun)
    if jit:
        hess = jax.jit(hess)

    def scipy_hess(*args, **kwargs):
        """Decorated scipy function

        Return:
            (n, n) dimension dense Hessian matrix
        """
        # ravel x0 parameters pytree
        args_list = list(args)
        args_list[0] = unflatten(args[0])
        # calculate Hessian matrix
        hess_matrix = hess(*args_list, **kwargs)
        # fix ValueError
        hess_matrix = jax.tree_util.tree_map(lambda leaf: leaf.tolist(),
                                             hess_matrix)
        # flatten Pytree and reshape
        dim = len(
            jax.tree_util.tree_leaves(
                jax.tree_util.tree_map(lambda leaf: leaf.tolist(),
                                       args_list[0])))
        hess_matrix = jnp.reshape(
            jnp.array(jax.tree_util.tree_leaves(hess_matrix)), (dim, dim))
        return hess_matrix

    return scipy_hess


def scipy_callback_wrapper(fun: Callable, unflatten: Callable):
    """A wrapper transform unflattening callback function's current parameter
    vector `xk` back to Pytree.

    Args:
        fun (callable): input function
        unflatten (callable): a callable for unflattening a 1D vector of the
            same length back to a pytree.
    """

    def unflatten_callback(xk, *args):
        return fun(unflatten(xk), *args)

    return unflatten_callback


@implements(
    minimize, """
New methods are implemented in arguments ``jac`` and ``hess``.
If ``jac`` is 'jax', the gradient vector will be calculate by JAX.
If ``hess`` is 'jax', the Hessian matrix will be calculate by JAX.
jit : bool, optional
    True for enable just-in-time compile.
fwd_ad : bool, optional
    True for using forward-mode auto diff.
logging : bool, optional
    Whether to output logging messages. Defaults to :py:obj:`False`.
""")
def scipy_minimize(fun,
                   x0,
                   args=(),
                   method=None,
                   jac='jax',
                   hess=None,
                   hessp=None,
                   bounds=None,
                   constraints=(),
                   tol=None,
                   callback=None,
                   options=None,
                   *,
                   jit: bool = True,
                   fwd_ad: bool = False,
                   logging: bool = False):
    global step
    step = 0
    scipy_x0, unflatten = ravel_pytree(x0)
    if jac == 'jax':
        jac = True
        scipy_fun = scipy_optimize_wrapper(fun,
                                           unflatten,
                                           jac=True,
                                           jit=jit,
                                           fwd_ad=fwd_ad,
                                           logging=logging)
    else:
        scipy_fun = scipy_optimize_wrapper(fun,
                                           unflatten,
                                           jac=False,
                                           jit=jit,
                                           logging=logging)
    if callback is not None:
        scipy_callback = scipy_callback_wrapper(callback, unflatten)
    else:
        scipy_callback = None
    if hess == 'jax':
        hess = scipy_hess_wrapper(fun, unflatten, jit=jit)
    # Minimize with scipy
    res = minimize(scipy_fun,
                   scipy_x0,
                   args=args,
                   method=method,
                   jac=jac,
                   hess=hess,
                   hessp=hessp,
                   callback=scipy_callback,
                   bounds=bounds,
                   constraints=constraints,
                   tol=tol,
                   options=options)
    # pack the output back into a PyTree
    res['x'] = unflatten(res['x'])
    if 'jac' in res.keys():
        res['jac'] = unflatten(res['jac'])

    return res


def adam_opt(fun, x0, args=(), options={}, jit=True, fwd_ad=False):
    """The JAX-backend implementation of the Adam optimization algorithm.

    Args:
        fun: the cost function f(params, all_params, `**kwargs`)
        x0: initial guess will be optimized iteratively.
        args (tuple, optional): Extra arguments passed to the objective
            function and its derivative. (fun, jac and hess functions)
        options (dict, optional): optimizer's hyper parameters dictionary.
            The default dict is
            training_params = {
            'adam_lr': 0.001,
            'adam_lr_decay_rate': 1000,
            'steps': 2000,
            'adam_b1': 0.9,
            'adam_b2': 0.999}
        jit (bool): True for just-in-time compile.
        fwd_ad(bool): True for using forward-mode auto diff.
    """
    # initialize hyper parameters
    training_params = {
        'adam_lr': 0.001,
        'adam_lr_decay_rate': 1000,
        'steps': 2000,
        'adam_b1': 0.9,
        'adam_b2': 0.999
    }
    training_params.update(options)
    print(training_params)

    lr = training_params['adam_lr']

    def schedual_lr(step):
        return training_params['adam_lr'] * 2**(
            -step / training_params['adam_lr_decay_rate'])

    opt_init, opt_update, get_params = optimizers.adam(
        schedual_lr,
        b1=training_params['adam_b1'],
        b2=training_params['adam_b2'])
    opt_state = opt_init(x0)
    if fwd_ad:
        grad_obj = value_and_gradfwd(fun)
    else:
        grad_obj = jax.value_and_grad(fun)
    if jit:
        grad_obj = jax.jit(grad_obj)

    def step(step, opt_state):

        def grad_clip(p, g):
            return np.clip(g, -0.001 * np.abs(p) / lr, 0.001 * np.abs(p) / lr)

        value, grads = grad_obj(get_params(opt_state), *args)
        grads = jax.tree_util.tree_map(grad_clip, get_params(opt_state), grads)
        pp = pprint.PrettyPrinter(indent=2)
        print('gradient:')
        pp.pprint(convert_to_json_compatible(grads))
        opt_state = opt_update(step, grads, opt_state)
        return value, opt_state

    for i in range(training_params['steps']):
        pp = pprint.PrettyPrinter(indent=2)
        print(f'step: {i}')
        print('parameters:')
        pp.pprint(convert_to_json_compatible(get_params(opt_state)))
        value, opt_state = step(i, opt_state)
        print(f'loss: {value}')

    return get_params(opt_state)
