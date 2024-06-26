from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax.scipy.special import erf
import haiku as hk


class PulseBase(hk.Module, ABC):
    """Generic pulse for various gate operations.

    Args:
        length (float): the length of pulse, set `None` for automatically collect
            parameter by haiku.
        amp (float): the max amplitude of pulse, set `None` for automatically
            collect parameter by haiku.
        delay (float): the delay time of pulse
        modulate_wave (bool): modulate the wave packet or not
        name (string): the name of module
    """

    def __init__(self,
                 length=None,
                 amp=None,
                 delay=0.,
                 modulate_wave=False,
                 name: str = 'pulse'):
        super().__init__(name=name)

        if length is None:
            self.length = hk.get_parameter('length', [], init=jnp.ones)
        else:
            self.length = length
        if amp is None:
            self.amp = hk.get_parameter('amp', [], init=jnp.ones)
        else:
            self.amp = amp
        self.delay = delay
        self.modulate_wave = modulate_wave
        if self.modulate_wave:
            self.omega_d = hk.get_parameter('omega_d', [], init=jnp.ones)
            self.phase = hk.get_parameter('phase', [], init=jnp.zeros)

    @abstractmethod
    def create_envelope_pulse(self, t, args={}):
        """Create envelope pulse shape for various gate operations."""

    def create_pulse(self, t, args={}):
        """Create desired pulse shape for various gate operations."""
        t_pulse = t - self.delay
        shape = self.create_envelope_pulse(t, args)
        if self.modulate_wave:
            shape *= jnp.cos(self.omega_d * t_pulse + self.phase)
        return shape

    @property
    def pulse_endtime(self):
        """The time of last pulse frame"""
        return self.length + self.delay


class PulseTrapezoid(PulseBase):
    """The trapezoid pulse.

    Args:
        length: the length of pulse
        amp: the amplitude of pulse
        delay: time delay for waiting other gate operations
        modulate_wave(bool): True for generate modulated sine wave
        name: module name

    Attributes:
        amp: max amplitude of the trapezoid
        t_ramp: time of the rising and falling edges
        t_plateau: the holding time of the plateau of the waveform
        length: the length of pulse shape
    """

    def __init__(
            self,
            length=False,  # unused argument
            amp=None,
            delay=0,
            modulate_wave=False,
            name: str = 'pulse_trapezoid'):
        super().__init__(length, amp, delay, modulate_wave, name)

        # construct activation function
        self.t_ramp = hk.get_parameter(
            't_ramp', [], init=jnp.ones)  # time for rising and falling edge
        self.t_plateau = hk.get_parameter('t_plateau', [], init=jnp.ones)
        self.length = self.t_plateau + 2 * self.t_ramp

    def create_envelope_pulse(self, t, args={}):
        t_pulse = t - self.delay
        time1 = jax.nn.relu(t_pulse / self.t_ramp)
        time2 = jax.nn.relu(
            (2 * self.t_ramp + self.t_plateau - t_pulse) / self.t_ramp)
        shape = self.amp * jnp.min(
            jnp.array([jnp.ones_like(time1), time1, time2]), axis=0)

        return shape * ((t_pulse >= 0) & (t_pulse <= 0 + self.length))


class PulseCosineRamping(PulseBase):
    """A flat-top pulse with cosine ramping at both ends.

    Args:
        length: the length of pulse
        amp: the amplitude of pulse
        delay: time delay for waiting other gate operations
        modulate_wave(bool): True for generate modulated sine wave
        name: module name

    Attributes:
        amp: max amplitude of the flat-top
        t_ramp: time of the rising and falling edges
        t_plateau: the holding time of the plateau of the waveform
        length: the length of pulse shape
    """

    def __init__(
            self,
            length=False,  # unused argument
            amp=None,
            delay=0,
            modulate_wave=False,
            name: str = 'pulse_rampcos'):
        super().__init__(length, amp, delay, modulate_wave, name)

        # construct activation function
        self.t_ramp = hk.get_parameter(
            't_ramp', [], init=jnp.ones)  # time for rising and falling edge
        self.t_plateau = hk.get_parameter('t_plateau', [], init=jnp.ones)
        self.length = self.t_plateau + 2 * self.t_ramp

    def create_envelope_pulse(self, t, args={}):
        t_pulse = t - self.delay
        pulse_1 = (1 - jnp.cos(jnp.pi * t_pulse / self.t_ramp)) / 2 * (
            t_pulse < self.t_ramp)
        pulse_2 = (t_pulse >= self.t_ramp) & (t_pulse
                                              <= self.t_ramp + self.t_plateau)
        pulse_3 = (1 - jnp.cos(jnp.pi *
                               (self.length - t_pulse) / self.t_ramp)) / 2 * (
                                   t_pulse > self.t_ramp + self.t_plateau)
        shape = self.amp * (pulse_1 + pulse_2 + pulse_3)

        return shape * ((t_pulse >= 0) & (t_pulse <= 0 + self.length))


class PulseGaussian(PulseBase):
    """Creates normalized Gaussian pulse, center at center of time range.
    The duration of the pulse is determined by ``2*cutoff*sigma``.
    If ``length < 2*cutoff*sigma``, the waveform will be truncated.
    If ``length > 2*cutoff*sigma``, zeros will be appended to the waveform.

    Note head/tail are directly truncated

    Args:
        length: the length of pulse
        amp: the amplitude of pulse
        modulate_wave(bool): True for generate modulated sine wave
        name: module name

    Attributes:
        sigma: the sigma of Gaussian function
        cutoff: the cutoff of Gaussian function

    Returns:
        an array of amplitudes
    """

    def __init__(
            self,
            length=False,  # unused argument
            amp=None,
            delay=0,
            modulate_wave=False,
            name: str = 'pulse_gaussian'):
        super().__init__(length, amp, delay, modulate_wave, name)

        self.sigma = hk.get_parameter('sigma', [], init=jnp.ones)
        self.cutoff = hk.get_parameter('cutoff', [], init=jnp.ones)
        self.length = 2 * self.cutoff * self.sigma

    def create_envelope_pulse(self, t, args={}):
        t_pulse = t - self.delay
        t1 = self.cutoff * self.sigma
        offset = jnp.exp(-self.cutoff**2 / 2)

        shape = (jnp.exp(-(t_pulse - t1)**2 / self.sigma**2 / 2) -
                 offset) / (1 - offset)

        shape = shape * ((t_pulse >= 0) & (t_pulse <= self.length)) * self.amp

        return shape


class PulseGaussianSquare(PulseBase):
    """Creates Gaussian square pulse shape.

    Args:
        length: the length of pulse
        amp: the amplitude of pulse
        delay: time delay for waiting other gate operations, unit in ns
        modulate_wave(bool): True for generate modulated sine wave
        name: module name

    Attributes:
        sigma: the sigma of Gaussian square
        cutoff: the cutoff of Gaussian square

    Returns:
        an array of amplitudes
    """

    def __init__(self,
                 length=None,
                 amp=None,
                 delay=0,
                 modulate_wave=False,
                 name: str = 'pulse_gaussian_envelope'):
        super().__init__(length, amp, delay, modulate_wave, name)

        self.sigma = hk.get_parameter('sigma', [], init=jnp.ones)
        self.cutoff = hk.get_parameter('cutoff', [], init=jnp.ones)

    def create_envelope_pulse(self, t, args={}):
        t_pulse = t - self.delay
        t1 = self.cutoff * self.sigma
        t2 = self.length - self.cutoff * self.sigma
        offset = jnp.exp(-self.cutoff**2 / 2)

        shape = (jnp.exp(-(t_pulse - t1)**2 / self.sigma**2 / 2) -
                 offset) / (1 - offset) * (t_pulse <= t1)
        shape += 1.0 * ((t_pulse > t1) & (t_pulse <= t2))
        shape += (jnp.exp(-(t_pulse - t2)**2 / self.sigma**2 / 2) -
                  offset) / (1 - offset) * (t_pulse > t2)

        shape = shape * ((t_pulse >= 0) & (t_pulse <= self.length)) * self.amp

        return shape


class PulseCosine(PulseBase):
    """Creates cosine style pulse from phase 0 to 2pi,

    This is centered at the center of time range normalized.

    Args:
        length: the length of pulse
        amp: the amplitude of pulse
        delay: time delay for waiting other gate operations, unit in ns
        modulate_wave(bool): True for generate modulated sine wave
        name: module name

    Returns:
        an array of amplitudes
    """

    def __init__(self,
                 length=None,
                 amp=None,
                 delay=0,
                 modulate_wave=False,
                 name: str = 'pulse_1mcos'):
        super().__init__(length, amp, delay, modulate_wave, name)

    def create_envelope_pulse(self, t, args={}):
        t_pulse = t - self.delay
        i_quad = 0.5 * (1 + jnp.cos(2 * jnp.pi / self.length *
                                    (t_pulse - self.length / 2)))

        shape = (i_quad) * (t_pulse >= 0) * (t_pulse <= self.length) * self.amp

        return shape


class PulseTanh(PulseBase):
    """Creates tanh up/down pulse.

    Note:
    - The pulse shape will reach a flat top of value 1 only when
    ``length`` is large enough, e.g., ``length > (10 + 2*cutoff) * sigma``.

    Args:
        length: the length of pulse
        amp: the amplitude of pulse
        delay: time delay for waiting other gate operations, unit in ns
        modulate_wave(bool): True for generate modulated sine wave
        name: module name

    Attributes:
        sigma: the sigma of tanh function
        cutoff: the cutoff of tanh function

    Returns:
        an array of amplitudes
    """

    def __init__(self,
                 length=None,
                 amp=None,
                 delay=0,
                 modulate_wave=False,
                 name: str = 'pulse_tanh'):
        super().__init__(length, amp, delay, modulate_wave, name)

        self.sigma = hk.get_parameter('sigma', [], init=jnp.ones)
        self.cutoff = hk.get_parameter('cutoff', [], init=jnp.ones)

    def create_envelope_pulse(self, t, args={}):
        t_pulse = t - self.delay
        t1 = self.cutoff * self.sigma
        t2 = self.length - self.cutoff * self.sigma
        offset = 0.5 * (jnp.tanh(-1.0 * self.cutoff) +
                        jnp.tanh(self.length / self.sigma - self.cutoff))
        peak = jnp.tanh(self.length / 2 / self.sigma - self.cutoff)

        shape = (0.5 * jnp.tanh((t_pulse - t1) / self.sigma) + 0.5 * jnp.tanh(
            (t2 - t_pulse) / self.sigma) - offset) / (peak - offset)

        shape = shape * ((t_pulse >= 0) & (t_pulse <= self.length)) * self.amp

        return shape


class PulseErf(PulseBase):
    """Creates error function up/down pulse, integral 1.

    Both up and down parts are erf style, controlled by sigma.
    the erf function is truncated at sigma=-cutoff(as t=0)
    and sigma=+cutoff(as t=tmax)
    The pulse is nearly flat in the middle if sigma is small.

    Args:
        length: the length of pulse
        amp: the amplitude of pulse
        delay: time delay for waiting other gate operations, unit in ns
        modulate_wave(bool): True for generate modulated sine wave
        name: module name

    Attributes:
        sigma: the sigma of erf function
        cutoff: the cutoff of erf function

    Returns:
        an array of amplitudes
    """

    def __init__(self,
                 length=None,
                 amp=None,
                 delay=0,
                 modulate_wave=False,
                 name: str = 'pulse_erf'):
        super().__init__(length, amp, delay, modulate_wave, name)

        self.sigma = hk.get_parameter('sigma', [], init=jnp.ones)
        self.cutoff = hk.get_parameter('cutoff', [], init=jnp.ones)

    def create_envelope_pulse(self, t, args={}):
        t_pulse = t - self.delay
        # t/2/sigma means the time corresponds to dx=1 in erf(x)
        sigma = self.length / 2 / self.sigma
        t1 = t_pulse - sigma * self.cutoff
        t2 = (self.length - t_pulse) - sigma * self.cutoff
        c1 = -self.cutoff
        c2 = -self.cutoff + self.length / sigma

        y = erf(t1 / sigma) + erf(t2 / sigma)
        y = y - erf(c1) - erf(c2)

        shape = y * self.amp / 2

        return shape * ((t_pulse >= 0) & (t_pulse <= self.length))


class PulseWithDRAG(PulseBase):
    """Creates pulse for single qubit operation.

    Args:
        envelope: pulse envelope
        delay: time delay for waiting other gate operations
        drive_dir: 0 for X and pi/2 for Y
        omega_d: driving frequency
        drag: using the derivative removal by adiabatic gate
        name: module name

    Attributes:
        length: the length of pulse shape
        amp: the max amplitude
        omega_d: the drive frequency
    """

    def __init__(self,
                 envelope: PulseBase = PulseCosine,
                 length=None,
                 amp=None,
                 delay=0,
                 modulate_wave=True,
                 name: str = 'pulse_drag'):
        assert modulate_wave is True
        super().__init__(length, amp, delay, modulate_wave, name)

        # DRAG codes
        self.lam = hk.get_parameter('lambda', [], init=jnp.zeros)

        self.envelope = envelope(length=self.length,
                                 amp=1.0,
                                 delay=delay,
                                 modulate_wave=False)
        # the length of pulse determinate by envelope
        self.length = self.envelope.length

    def create_envelope_pulse(self, t, args={}):
        envelope_pulse = self.amp * self.envelope.create_pulse(t)

        return envelope_pulse

    def create_pulse(self, t, args={}):
        """Overload `create_pulse` function."""
        t_pulse = t - self.delay
        shape = self.create_envelope_pulse(t, args)
        pulse = jnp.cos(self.omega_d * t_pulse + self.phase)
        output = shape * pulse

        if isinstance(t_pulse, (np.ndarray, jnp.ndarray)) and t_pulse.ndim > 0:
            d_envelope = jax.vmap(jax.grad(self.envelope.create_pulse))
        else:
            d_envelope = jax.grad(self.envelope.create_pulse)
        output -= self.lam * d_envelope(t) * jnp.sin(self.omega_d * t_pulse +
                                                     self.phase)
        return output * ((t_pulse >= 0) & (t_pulse <= 0 + self.length))


def draw(
    pulse: PulseBase,
    ts,
    function: Optional[str] = "pulse",
    title: Optional[str] = None,
):
    """Plot the pulse over an array `ts`.
    The ``function`` arg specifies which function to plot:

        - ``function == 'pulse'`` plots the full driving pulse.
        - ``function == 'envelope'`` plots the envelope.

    Args:
        ts: The time values array to plot.
        function: Which function to plot.
        title: Title of plot.
    """

    @hk.without_apply_rng
    @hk.transform
    def _draw(ts, function, title):
        if function == "pulse":
            y_vals = pulse.create_pulse(ts)
            title = title or "Value of " + pulse.name
        elif function == "envelope":
            y_vals = pulse.create_envelope_pulse(ts)
            title = title or "Envelope of " + pulse.name

        plt.figure(figsize=(8, 4))
        plt.plot(ts, y_vals)
        plt.title(title)
        plt.tight_layout()

    _draw.apply({}, ts, function, title)


if __name__ == '__main__':

    @hk.without_apply_rng
    @hk.transform
    def plot_pulse():
        pulse = PulseWithDRAG(delay=0, modulate_wave=True, name='pulse')
        ts = np.linspace(0, pulse.pulse_endtime + 4, 500)
        draw(pulse, ts, 'pulse')
        draw(pulse, ts, 'envelope')
        plt.show()

    plot_pulse.apply({
        'pulse': {
            'amp': jnp.array(0.5),
            'omega_d': jnp.array(2.),
            'phase': jnp.array(0.),
            't_ramp': jnp.array(30.),
            't_plateau': jnp.array(30.),
            'sigma': jnp.array(1.),
            'cutoff': jnp.array(2.),
            'length': jnp.array(60.),
            'lambda': jnp.array(0.)
        }
    })
