from functools import partial
import jax
import jax.numpy as jnp
from tqdm import tqdm


def pass_operands(operands):
    return operands


def build_tqdm(n: int, print_rate: int = None, fwd_ad: bool = False):
    """Build the tqdm progress bar with backward mode auto-diff support.
    Args:
        n (int): total number of steps
        print_rate (int): the integer rate at which the progress bar will be updated
            by default the print rate will be 1/20 of the total number of steps
        fwd_ad (bool): whether to use forward mode auto-diff, default to False
    Returns:
        Callable: a wrapping function for progress bar
    """
    tqdm_bars = {}

    if print_rate is None:
        print_rate = n // 20 if n > 20 else 1
    else:
        if print_rate < 1:
            raise ValueError("print_rate should be a positive integer")
        print_rate = max(n, print_rate)

    remainder = n % print_rate

    def _id_tap_tqdm_progress(iter_num, message):

        def _define_tqdm(operand):
            if jax.process_index() == 0:
                tqdm_bars[0] = tqdm(range(n))
                tqdm_bars[0].set_description(message, refresh=False)

        def _update_tqdm(arg, operand):
            if jax.process_index() == 0:
                tqdm_bars[0].update(int(arg))

        _ = jax.lax.cond(
            iter_num == 0,
            partial(jax.debug.callback, _define_tqdm),
            pass_operands,
            operand=None,
        )

        _ = jax.lax.cond(
            # update tqdm by print rate
            (iter_num % print_rate == 0) & (iter_num != n - remainder),
            partial(jax.debug.callback, _update_tqdm, print_rate),
            pass_operands,
            operand=None,
        )

        _ = jax.lax.cond(
            # update tqdm by remainder
            iter_num == n - remainder,
            partial(jax.debug.callback, _update_tqdm, remainder),
            pass_operands,
            operand=None,
        )

    def _close_tqdm(operand):
        if tqdm_bars and jax.process_index() == 0:
            tqdm_bars[0].close()

    if fwd_ad:

        def _update_progress_bar(iter_num, result):
            """Updates tqdm in a JAX scan or loop."""
            message = f'Time evolution for {n:,} steps'
            _id_tap_tqdm_progress(iter_num, message)
            _ = jax.lax.cond(
                iter_num == n - 1,
                partial(jax.debug.callback, _close_tqdm),
                pass_operands,
                operand=None,
            )
            return iter_num, result
    else:

        @jax.custom_vjp
        def _update_progress_bar(iter_num, result):
            """Updates tqdm in a JAX scan or loop."""
            message = f'Forward time evolution for {n:,} steps'
            _id_tap_tqdm_progress(iter_num, message)
            _ = jax.lax.cond(
                iter_num == n - 1,
                partial(jax.debug.callback, _close_tqdm),
                pass_operands,
                operand=None,
            )
            return iter_num, result

        def _update_progress_bar_fwd(iter_num, result):
            return _update_progress_bar(iter_num, result), iter_num

        def _update_progress_bar_bwd(residual, cot_result):
            message = f'Backward time evolution for {n:,} steps'
            iter_num = n - residual - 1
            _id_tap_tqdm_progress(iter_num, message)
            _ = jax.lax.cond(
                iter_num == n - 1,
                partial(jax.debug.callback, _close_tqdm),
                pass_operands,
                operand=None,
            )
            return cot_result

        _update_progress_bar.defvjp(_update_progress_bar_fwd,
                                    _update_progress_bar_bwd)

    return _update_progress_bar


def scan_tqdm(func, xs, print_rate: int = None, fwd_ad: bool = False):
    """tqdm progress bar for a JAX scan.
    Args:
        func: function to be wrapped
        xs: scan steps
        print_rate (int): the integer rate at which the progress bar will be updated
            by default the print rate will be 1/20 of the total number of steps
        fwd_ad (bool): whether to use forward mode auto-diff, default to False
    Returns:
        Callable: a wrapping function for progress bar
        enumerate_xs: tuple(indices, xs)
    """
    n = len(xs)
    _update_progress_bar = build_tqdm(n, print_rate, fwd_ad)

    def wrapper_progress_bar(carry, enumerate_xs):
        iter_num, x = enumerate_xs
        result = func(carry, x)
        _, result = _update_progress_bar(iter_num, result)
        return result

    return wrapper_progress_bar, (jnp.arange(n), xs)
