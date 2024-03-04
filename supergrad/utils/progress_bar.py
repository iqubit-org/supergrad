import jax
import jax.numpy as jnp
from jax.experimental import host_callback
from tqdm import tqdm


def build_tqdm(n: int, print_rate: int = None, fwd_ad: bool = False):
    """Build th etqdm progress bar with backward mode auto-diff support.
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

        def _define_tqdm(arg, transform):
            tqdm_bars[0] = tqdm(range(n))
            tqdm_bars[0].set_description(message, refresh=False)

        def _update_tqdm(arg, transform):
            tqdm_bars[0].update(arg)

        _ = jax.lax.cond(
            iter_num == 0,
            lambda _: host_callback.id_tap(_define_tqdm, None, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

        _ = jax.lax.cond(
            # update tqdm by print rate
            (iter_num % print_rate == 0) & (iter_num != n - remainder),
            lambda _: host_callback.id_tap(
                _update_tqdm, print_rate, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

        _ = jax.lax.cond(
            # update tqdm by remainder
            iter_num == n - remainder,
            lambda _: host_callback.id_tap(
                _update_tqdm, remainder, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

    def _close_tqdm(*args):
        if tqdm_bars:
            tqdm_bars[0].close()

    if fwd_ad:

        def _update_progress_bar(iter_num, results):
            """Updates tqdm in a JAX scan or loop."""
            message = f'Forward time evolution for {n:,} steps'
            _id_tap_tqdm_progress(iter_num, message)
            results = jax.lax.cond(
                iter_num == n - 1,
                lambda _: host_callback.id_tap(
                    _close_tqdm, None, result=results),
                lambda _: results,
                operand=None,
            )
            return iter_num, results
    else:

        @jax.custom_vjp
        def _update_progress_bar(iter_num, results):
            """Updates tqdm in a JAX scan or loop."""
            message = f'Forward time evolution for {n:,} steps'
            _id_tap_tqdm_progress(iter_num, message)
            results = jax.lax.cond(
                iter_num == n - 1,
                lambda _: host_callback.id_tap(
                    _close_tqdm, None, result=results),
                lambda _: results,
                operand=None,
            )
            return iter_num, results

        def _update_progress_bar_fwd(iter_num, results):
            return _update_progress_bar(iter_num, results), iter_num

        def _update_progress_bar_bwd(residual, cot_results):
            message = f'Backward time evolution for {n:,} steps'
            iter_num = n - residual - 1
            _id_tap_tqdm_progress(iter_num, message)
            cot_results = jax.lax.cond(
                iter_num == n - 1,
                lambda _: host_callback.id_tap(
                    _close_tqdm, None, result=cot_results),
                lambda _: cot_results,
                operand=None,
            )
            return cot_results

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
