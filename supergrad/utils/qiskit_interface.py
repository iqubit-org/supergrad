import numpy as np
import jax
from qiskit_dynamics import Signal
import haiku as hk


def to_qiskit_static_hamiltonian(superObj):
    """Convert the static Hamiltonian to qiskit format.

    Args:
        superObj: The static Hamiltonian in `KronObj` or `LindbladObj` format.
    Returns:
        The static Hamiltonian array in qiskit format.
    """
    return superObj.full()


def to_qiskit_drive_hamiltonian(hamiltonian_component_and_pulseshape):
    """Convert the drive Hamiltonian to qiskit format.

    Args:
        hamiltonian_component_and_pulseshape: The list of pair containing the
            time-dependent components of the Hamiltonian and corresponding pulse
            shape.
    Returns:
        (list, list)
        The list of Hamiltonian operators and the list of signal in qiskit format.
    """
    drive_ham_list = []
    signals_list = []
    for drive, create_pulse in hamiltonian_component_and_pulseshape:
        # convert pulse as qiskit signal
        pulse = create_pulse.__self__
        drive_ham_list.append(drive.full())
        carrier_freq = pulse.omega_d / 2 / np.pi

        def signal_closure(pulse):
            # Instantiation same envelope function

            @hk.without_apply_rng
            @hk.transform
            def envelope(t):
                new_pulse = type(pulse)(modulate_wave=False, name=pulse.name)
                return new_pulse.create_envelope_pulse(t)

            # get parameters template
            rng = jax.random.PRNGKey(0)
            params = envelope.init(rng, 0.)
            # filling params by existed pulse
            existed_params = vars(pulse)
            existed_params = hk.data_structures.filter(
                lambda module_name, name, value: name in params[pulse.name].
                keys(), {pulse.name: existed_params})
            params = hk.data_structures.merge(params, existed_params)
            return lambda t: envelope.apply(params, t)

        signals_list.append(
            Signal(signal_closure(pulse), carrier_freq, pulse.phase,
                   pulse.name))
    return drive_ham_list, signals_list
