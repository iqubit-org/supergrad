# %%
import os
import numpy as np
import jax.numpy as jnp
import haiku as hk
import pytest

import supergrad
from supergrad import basis
from supergrad.quantum_system import Fluxonium, parse_interaction, InteractingSystem
from supergrad.time_evolution.pulseshape import PulseCosine, PulseCosineRamping
from supergrad.time_evolution import sesolve, sesolve_final_states_w_basis_trans
from supergrad.utils.fidelity import compute_fidelity_with_1q_rotation_axis
from supergrad.utils import create_state_init, gates, identity_wrap

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)


def test_x_gate(solver='ode_expm', options={'astep': 1280}):

    class SingleGate(supergrad.Helper):

        def _init_quantum_system(self):
            self.fq = Fluxonium(phiext=0.5 * 2 * np.pi, phi_max=5 * np.pi)
            self.pulseshape = PulseCosine(modulate_wave=True)

        def evo(self):
            eig_val = self.fq.eigenenergies()
            ham_diag = jnp.diag(eig_val)

            opt = self.fq.phi_operator()
            ham = [ham_diag, [opt, self.pulseshape.create_pulse]]

            psi_list = jnp.array([basis(10, 0), basis(10, 1)])
            res = sesolve(ham,
                          psi_list, [0, self.pulseshape.length],
                          solver=solver,
                          options=options)
            return res

    def infidelity(params, all_params={}):
        params = hk.data_structures.merge(all_params, params)
        res = sg.evo(params)
        states = res[:, -1, :2, 0]
        fidelity, _ = compute_fidelity_with_1q_rotation_axis(gates.sigmax(),
                                                             states,
                                                             opt_method=None)
        return 1 - fidelity, states

    sg = SingleGate()

    params = {
        'fluxonium': {
            'ec': jnp.array(1. * 2 * jnp.pi),
            'ej': jnp.array(4. * 2 * jnp.pi),
            'el': jnp.array(1. * 2 * jnp.pi)
        },
        'pulse_1mcos': {
            'length': jnp.array(40.),
            'amp': jnp.array(0.51753615),
            'omega_d': jnp.array(3.65911655),
            'phase': jnp.array(0.64509906)
        }
    }

    # load cached data
    u0 = np.load('data/data_x_gate.npy')
    infid0 = 1 - compute_fidelity_with_1q_rotation_axis(
        gates.sigmax(), u0, opt_method=None,
        compensation_option='arbit_single')[0]
    infid1, u1 = infidelity(params)
    relative_fid = compute_fidelity_with_1q_rotation_axis(u0,
                                                          u1,
                                                          opt_method=None)[0]
    assert np.allclose(infid0, infid1, atol=1e-4)
    assert 1 - relative_fid < 0.01


def test_x_gate_odeint():
    test_x_gate(solver='odeint', options={})


def test_cr_gate(solver='ode_expm',
                 options={
                     'astep': 2500,
                     'trotter_order': 1,
                     'diag_ops': False
                 }):

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

            self.hilbertspace = InteractingSystem([self.control, self.target],
                                                  inter_list)
            self.pulseshape = PulseCosineRamping(modulate_wave=True)

        def evo(self):
            # Single qubit product basis
            # construct static hamiltonian
            ham_static = self.hilbertspace.idling_hamiltonian_in_prod_basis()
            # add CR pulse
            opt = self.control.phi_operator
            wrap_opt = identity_wrap(opt, self.hilbertspace.subsystem_list)

            ham = [ham_static, [wrap_opt, self.pulseshape.create_pulse]]
            self.tlist = [0, self.pulseshape.pulse_endtime]
            u_to_eigen = self.hilbertspace.compute_transform_matrix()
            psi_list, _ = create_state_init(self._get_dims(), [[0, 2], [1, 2]])
            sim_u = sesolve_final_states_w_basis_trans(ham,
                                                       psi_list,
                                                       self.tlist,
                                                       u_to_eigen,
                                                       solver=solver,
                                                       options=options)
            return sim_u

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
        },
        'pulse_rampcos': {
            'amp': jnp.array(0.19362652),
            'omega_d': jnp.array(2.63342338),
            'phase': jnp.array(-0.56882493),
            't_plateau': jnp.array(69.9963988),
            't_ramp': jnp.array(30.20130596),
        }
    }

    def infidelity(params, all_params={}):
        params = hk.data_structures.merge(all_params, params)
        sim_u = mg.evo(params)
        fidelity, u_optimal = compute_fidelity_with_1q_rotation_axis(
            gates.cnot(),
            sim_u,
            opt_method='gd',
            compensation_option='arbit_single')
        return 1 - fidelity, sim_u

    # load cached data
    u0 = np.load('data/data_cnot_gate.npy')
    infid0 = 1 - compute_fidelity_with_1q_rotation_axis(
        gates.cnot(), u0, opt_method='gd',
        compensation_option='arbit_single')[0]
    infid1, u1 = infidelity(params)
    relative_fid = compute_fidelity_with_1q_rotation_axis(u0,
                                                          u1,
                                                          opt_method=None)[0]
    assert np.allclose(infid0, infid1, atol=1e-4)
    assert 1 - relative_fid < 0.01


@pytest.mark.slow
def test_cr_gate_odeint():
    test_cr_gate(solver='odeint', options={})


def test_cr_gate_diag():
    test_cr_gate(solver='ode_expm',
                 options={
                     'astep': 2500,
                     'trotter_order': 1,
                     'diag_ops': True
                 })


def test_cr_gate_trotter_2():
    test_cr_gate(solver='ode_expm', options={'astep': 2500, 'trotter_order': 2})


def test_cr_gate_trotter_4():
    test_cr_gate(solver='ode_expm', options={'astep': 2500, 'trotter_order': 4})


def test_cr_gate_trotter_4j():
    test_cr_gate(solver='ode_expm',
                 options={
                     'astep': 2500,
                     'trotter_order': 4j
                 })


@pytest.mark.slow
def test_cr_gate_no_trotter():
    test_cr_gate(solver='ode_expm',
                 options={
                     'astep': 2500,
                     'trotter_order': None
                 })
