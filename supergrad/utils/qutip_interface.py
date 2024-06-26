import numpy as np
import jax
import haiku as hk
import qutip as qt
from qutip import Qobj
from supergrad import KronObj


def to_qutip_operator(superObj: KronObj):
    """Convert the Hamiltonian in `superObj` to qutip format.

    Args:
        superObj: The Hamiltonian in `KronObj` format.
    Returns:
        The Hamiltonian in qutip format.
    """
    # firstly downcast diag unitary
    if any(superObj.diag_status):
        superObj._downcast_diagonal_unitary()
    data = []
    for mats, local_list in zip(superObj.data, superObj.locs):
        if local_list is not None:  # downcasting
            # construct identity matrix list
            mat_list = [Qobj(np.eye(dim)) for dim in superObj.dims]
            for mat, local in zip(mats, local_list):
                mat_list[local] = Qobj(np.array(mat))
            # calculate tensor product
            data.append(qt.tensor(*mat_list))
        else:
            data.append(Qobj(np.array(mats), [
                list(superObj.dims),
            ] * 2))
    return sum(data)


def to_qutip_operator_function_pair(hamiltonian_component_and_pulseshape):
    """Convert the drive Hamiltonian to qutip operator-function pairs.

    Args:
        hamiltonian_component_and_pulseshape: The list of pair containing the
            time-dependent components of the Hamiltonian and corresponding pulse
            shape.
    Returns:
        The list of operator-function pairs.
    """
    operator_function_pair_list = []
    for drive, create_pulse in hamiltonian_component_and_pulseshape:
        oper = to_qutip_operator(drive)
        pulse = create_pulse.__self__

        def pulse_closure(pulse):

            @hk.without_apply_rng
            @hk.transform
            def new_create_pulse(t, args):
                new_pulse = type(pulse)(modulate_wave=True, name=pulse.name)
                return new_pulse.create_pulse(t, args)

            # get parameters template
            rng = jax.random.PRNGKey(0)
            params = new_create_pulse.init(rng, 0., {})
            # filling params by existed pulse
            existed_params = vars(pulse)
            existed_params = hk.data_structures.filter(
                lambda module_name, name, value: name in params[pulse.name].
                keys(), {pulse.name: existed_params})
            params = hk.data_structures.merge(params, existed_params)

            return lambda t, args: new_create_pulse.apply(params, t, args)

        operator_function_pair_list.append([oper, pulse_closure(pulse)])

    return operator_function_pair_list
