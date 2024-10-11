from supergrad.helper import Evolve
from supergrad.scgraph.graph_mpc_fluxonium_1d import MPCFluxonium1D


def create_hadamard_transform(n_qubit,
                              astep,
                              trotter_order,
                              diag_ops,
                              minimal_approach=False,
                              custom_vjp=None):
    chain = MPCFluxonium1D(n_qubit, periodic=False, seed=42)
    chain.create_single_qubit_pulse(range(n_qubit), [50.0] * n_qubit,
                                    True,
                                    factor=0.25,
                                    minimal_approach=minimal_approach)

    return Evolve(chain,
                  truncated_dim=2,
                  compensation_option='no_comp',
                  options={
                      'astep': astep,
                      'trotter_order': trotter_order,
                      'diag_ops': diag_ops,
                      'custom_vjp': custom_vjp
                  })


def create_simultaneous_x(n_qubit,
                          astep,
                          trotter_order,
                          diag_ops,
                          minimal_approach=False,
                          custom_vjp=None):
    chain = MPCFluxonium1D(n_qubit, periodic=False, seed=42)
    chain.create_single_qubit_pulse(range(n_qubit), [50.0] * n_qubit,
                                    True,
                                    minimal_approach=minimal_approach)

    return Evolve(chain,
                  truncated_dim=2,
                  compensation_option='no_comp',
                  options={
                      'astep': astep,
                      'trotter_order': trotter_order,
                      'diag_ops': diag_ops,
                      'custom_vjp': custom_vjp
                  })


def create_simultaneous_cnot(n_qubit,
                             astep,
                             trotter_order,
                             diag_ops,
                             minimal_approach=False,
                             custom_vjp=None):
    n_cnot = n_qubit // 2
    chain = MPCFluxonium1D(n_qubit, periodic=False, seed=42)
    chain.create_cr_pulse([2 * i for i in range(n_cnot)],
                          [2 * i + 1 for i in range(n_cnot)], [100.0] * n_cnot,
                          True,
                          minimal_approach=minimal_approach)

    return Evolve(chain,
                  truncated_dim=2,
                  compensation_option='no_comp',
                  options={
                      'astep': astep,
                      'trotter_order': trotter_order,
                      'diag_ops': diag_ops,
                      'custom_vjp': custom_vjp
                  })
