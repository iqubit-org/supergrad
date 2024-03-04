import numpy as np
import jax.numpy as jnp
import qutip as qt

import supergrad
from supergrad.utils import create_state_init
from supergrad.quantum_system import Fluxonium, parse_interaction, InteractingSystems

__all__ = ['create_x_gate', 'create_cnot_gate']


def create_x_gate():
    """Time evolution for a X gate."""

    class SingleGate(supergrad.Helper):

        def _init_quantum_system(self):
            self.fq = Fluxonium(phiext=0.5 * 2 * np.pi, phi_max=5 * np.pi)

        def hamiltonian_operator(self):
            return np.diag(self.fq.eigenenergies()), np.asarray(
                self.fq.phi_operator())

    sg = SingleGate()

    params = {
        'fluxonium': {
            'ec': jnp.array(1. * 2 * jnp.pi),
            'ej': jnp.array(4. * 2 * jnp.pi),
            'el': jnp.array(1. * 2 * jnp.pi)
        }
    }
    # calculate eigenenergies and operator matrix
    ham_diag, opt = sg.hamiltonian_operator(params)
    ham_diag = qt.Qobj(ham_diag, shape=[10, 10])
    opt = qt.Qobj(opt, shape=[10, 10])

    # pulse shape
    def create_pulse(t,
                     args,
                     length=40.0,
                     amp=0.51753615,
                     omega_d=3.65911655,
                     phase=0.64509906):
        t_pulse = t
        i_quad = 0.5 * (1 + jnp.cos(2 * jnp.pi / length *
                                    (t_pulse - length / 2)))

        shape = (i_quad) * (t_pulse >= 0) * (t_pulse <= length) * amp
        shape *= jnp.cos(omega_d * t_pulse + phase)
        return shape

    # construct initialize states
    psi_list = [qt.basis(10, 0), qt.basis(10, 1)]
    tlist = np.linspace(0, 40., 200)
    list_res = [
        qt.sesolve([ham_diag, [opt, create_pulse]], psi, tlist)
        for psi in psi_list
    ]

    res = np.stack(
        [np.stack([y.full() for y in x.states], axis=0) for x in list_res],
        axis=0)
    states = res[:, -1, :2, 0]
    np.save('data_x_gate.npy', states)
    print('Create X gate is finished.')


def create_cnot_gate():
    """Time evolution for CNOT gate. Noted the arbitrary single qubit gate
    compensation is used to replace physical x gate.
    """

    class MultiGate(supergrad.Helper):

        def _init_quantum_system(self):
            self.control = Fluxonium(phiext=0.5 * 2 * np.pi,
                                     phi_max=5 * np.pi,
                                     name='control')
            self.target = Fluxonium(phiext=0.5 * 2 * np.pi,
                                    phi_max=5 * np.pi,
                                    name='target')
            inter_list = []

            inter_list.append(
                parse_interaction(op1=self.control.n_operator,
                                  op2=self.target.n_operator,
                                  name='jc'))
            inter_list.append(
                parse_interaction(op1=self.control.phi_operator,
                                  op2=self.target.phi_operator,
                                  name='jl'))

            self.hilbertspace = InteractingSystems([self.control, self.target],
                                                   inter_list)
            self.dims = self.hilbertspace.dim

        def hamiltonian_operator(self):
            energy_nd = self.hilbertspace.compute_energy_map()
            # multi-qubit truncated eigenbasis
            energy_1d = energy_nd.flatten()
            energy_1d = energy_1d - energy_1d[0]  # subtract ground state
            ham_static = np.diag(energy_1d)
            opt = self.hilbertspace.transform_operator(
                self.control.phi_operator)
            return ham_static, np.asarray(opt)

    mg = MultiGate()
    params = {
        'control': {
            'ec': jnp.array(1. * 2 * jnp.pi),
            'ej': jnp.array(4. * 2 * jnp.pi),
            'el': jnp.array(1.0 * 2 * jnp.pi),
        },
        'target': {
            'ec': jnp.array(1. * 2 * jnp.pi),
            'ej': jnp.array(4. * 2 * jnp.pi),
            'el': jnp.array(0.8 * 2 * jnp.pi),
        },
        'jc': {
            'strength': jnp.array(11.5e-3 * 2 * jnp.pi)
        },
        'jl': {
            'strength': jnp.array(-1.0 * 2e-3 * 2 * jnp.pi)
        }
    }
    ham, control_phi = mg.hamiltonian_operator(params)
    # convert to Qobj
    ham_diag = qt.Qobj(ham, dims=[[10, 10], [10, 10]])
    control_phi = qt.Qobj(control_phi, dims=[[10, 10], [10, 10]])

    # pulse shape
    def create_pulse(t,
                     args={},
                     amp=0.19362652,
                     omega_d=2.63342338,
                     phase=-0.56882493,
                     t_plateau=69.9963988,
                     t_ramp=30.20130596):
        t_pulse = t
        length = t_plateau + 2 * t_ramp
        pulse_1 = (1 -
                   jnp.cos(jnp.pi * t_pulse / t_ramp)) / 2 * (t_pulse < t_ramp)
        pulse_2 = (t_pulse >= t_ramp) & (t_pulse <= t_ramp + t_plateau)
        pulse_3 = (1 - jnp.cos(jnp.pi * (length - t_pulse) / t_ramp)) / 2 * (
            t_pulse > t_ramp + t_plateau)
        shape = amp * (pulse_1 + pulse_2 + pulse_3)
        shape *= jnp.cos(omega_d * t_pulse + phase)
        return shape * ((t_pulse >= 0) & (t_pulse <= 0 + length))

    # construct initialize states
    length = 130.39901071999998
    psi_list, ar_ix = create_state_init([10, 10], [[0, 2], [1, 2]])
    tlist = np.linspace(0, length, 200)
    list_res = [
        qt.sesolve([ham_diag, [control_phi, create_pulse]],
                   qt.Qobj(psi, dims=[[10, 10], [1, 1]]), tlist)
        for psi in psi_list
    ]

    res = np.stack(
        [np.stack([y.full() for y in x.states], axis=0) for x in list_res],
        axis=0)
    # Pick the computational space
    tuple_band_ix = tuple(np.ravel_multi_index(x, [10, 10]) for x in ar_ix)
    sim_u = res[:, -1, tuple_band_ix, 0]
    np.save('data_cnot_gate.npy', sim_u)
    print('Create CNOT gate is finished.')
