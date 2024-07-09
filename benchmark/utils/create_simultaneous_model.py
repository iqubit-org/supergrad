from supergrad.helper import Evolve
from supergrad.scgraph.graph_mpc_fluxonium_1d import MPCFluxonium1D


def create_simultaneous_x(n_qubit,
                          astep,
                          trotter_order,
                          diag_ops,
                          minimal_approach=False,
                          custom_vjp=None):
    chain = MPCFluxonium1D(n_qubit, periodic=False, seed=42)
    chain.create_single_qubit_pulse(range(n_qubit), [50.0] * n_qubit,
                                    minimal_approach=minimal_approach)
    chain.set_all_node_attr(truncated_dim=2)
    chain.set_compensation("no_comp")

    return Evolve(chain,
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
                          minimal_approach=minimal_approach)
    chain.set_all_node_attr(truncated_dim=2)
    chain.set_compensation("no_comp")

    return Evolve(chain,
                  options={
                      'astep': astep,
                      'trotter_order': trotter_order,
                      'diag_ops': diag_ops,
                      'custom_vjp': custom_vjp
                  })
