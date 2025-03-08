import numpy as np

import scqubits as scq


def create_fluxonium(label, params, truncated_dim):
    return scq.Fluxonium(
        EC=params[label]['ec'],
        EJ=params[label]['ej'],
        EL=params[label]['el'],
        flux=params[label]['phiext'] / 2 / np.pi,
        cutoff=400,  # same as the supergrad using
        truncated_dim=truncated_dim,
        evals_method='evals_jax_dense',  # JAX for the performance evaluation
        id_str=label)


def create_qubit_chain(params, n_qubit, truncated_dim):
    """Hands-on create chain model by scqubits"""

    def add_multi_coupling_term(fm_list, label1, label2,
                                hilbertspace: scq.HilbertSpace, params):
        hilbertspace.add_interaction(
            g=float(
                params[f'capacitive_coupling_{label1}_{label2}']['strength']),
            op1=fm_list[int(label1.lstrip('fm'))].n_operator,
            op2=fm_list[int(label2.lstrip('fm'))].n_operator,
            id_str=f'capacitive_coupling_{label1}_{label2}')
        hilbertspace.add_interaction(
            g=float(
                params[f'inductive_coupling_{label1}_{label2}']['strength']),
            op1=fm_list[int(label1.lstrip('fm'))].phi_operator,
            op2=fm_list[int(label2.lstrip('fm'))].phi_operator,
            id_str=f'inductive_coupling_{label1}_{label2}')

    fm_list = [
        create_fluxonium(label=f'fm{i}',
                         params=params,
                         truncated_dim=truncated_dim) for i in range(n_qubit)
    ]
    # add coupling terms and create the hilbertspace
    hilbertspace = scq.HilbertSpace(fm_list)

    for i in range(n_qubit - 1):
        add_multi_coupling_term(fm_list,
                                label1=f'fm{i}',
                                label2=f'fm{i + 1}',
                                hilbertspace=hilbertspace,
                                params=params)
    return hilbertspace, fm_list
