import numpy as np
from pylatex import Document, Package, MultiColumn, Tabular, TextColor
from pylatex.utils import NoEscape

from supergrad.scgraph.graph import SCGraph, _parse_pulse_name, _parse_edges_name
from supergrad.utils.fidelity import u_to_pauli


def generate_params_tabu(graph: SCGraph,
                         add_random=True,
                         gradient=None,
                         unify_coupling=True):
    """Generate latex source codes about parameters table."""

    geometry_options = {
        "head": "40pt",
        "margin": "0.5in",
        "bottom": "0.6in",
        "includeheadfoot": True
    }
    doc = Document(geometry_options=geometry_options)
    doc.packages.append(Package('amsmath'))

    qubit_type = {0: '3', 1: '4', 2: '5', 3: '1', 4: '2', 5: '3'}
    qubit_name = {
        0: '$\mathrm{Q}_{02}$',
        1: '$\mathrm{Q}_{03}$',
        2: '$\mathrm{Q}_{12}$',
        3: '$\mathrm{Q}_{13}$',
        4: '$\mathrm{Q}_{22}$',
        5: '$\mathrm{Q}_{23}$'
    }

    # Add statement table
    with doc.create(Tabular("| l | r | r || l | l | r |",
                            row_height=1.5)) as data_table:
        data_table.add_hline()
        data_table.add_row(
            MultiColumn(3, align='|c||', data='Device parameters'),
            MultiColumn(3, align='c|', data='Control parameters'))
        data_table.add_hline()
        data_table.add_row([
            "Parameter", "Value (GHz)", "Gradient", "Parameter", "Value",
            "Gradient"
        ])
        data_table.add_hline()
        # collect device paramters
        dev_params = []
        nodes_order = graph.sorted_nodes
        for i, node in enumerate(nodes_order):
            for par_name in ['ec', 'ej', 'el']:
                data = graph.nodes[node][par_name]
                if add_random:
                    var = graph.nodes[node].get('variance', None)
                    data *= var.get('ec', 1.0)
                if gradient is not None:
                    # unpack gradient data
                    data_grad = gradient[0][node][par_name]
                    str_grad = f'{data_grad / 2 / np.pi:.3e}'
                else:
                    str_grad = ''
                row = [
                    NoEscape('$E_{' + f'{par_name[-1].upper()}, ' +
                             qubit_type[i] + '}$'), f'{data / 2 / np.pi:.3e}',
                    str_grad
                ]
                dev_params.append(row)
        # collect coupling params
        flag_list = []
        for edge in graph.sorted_edges:
            attr = graph.edges[edge]
            q1, q2 = sorted(edge)
            for k, v in dict(attr).items():
                if k in flag_list and unify_coupling:
                    continue
                flag_list.append(k)
                if unify_coupling:
                    key = _parse_edges_name(k, 'all', 'unify')
                else:
                    key = _parse_edges_name(k, q1, q2)
                data = v['strength']
                if gradient is not None:
                    # unpack gradient data
                    data_grad = gradient[0][key]['strength']
                    str_grad = f'{data_grad / 2 / np.pi:.3e}'
                else:
                    str_grad = ''
                if k == 'capacitive_coupling':
                    type_coup = '$J_{C,'
                elif k == 'inductive_coupling':
                    type_coup = '$J_{L,'
                if unify_coupling:
                    row = [
                        NoEscape(type_coup.split(',')[0] + '}$'),
                        f'{data / 2 / np.pi:.3e}', str_grad
                    ]
                else:
                    row = [
                        NoEscape(
                            type_coup +
                            f'({nodes_order.index(q1)}, {nodes_order.index(q2)})'
                            + '}$'), f'{data / 2 / np.pi:.3e}', str_grad
                    ]
                dev_params.append(row)
        # collect control params
        control_params = []
        for idx, node in enumerate(graph.sorted_nodes):
            data = graph.nodes[node]
            pulse_dict = dict(data).pop('pulse', None)
            if pulse_dict is not None:
                pulse_params = ['amp', 't_ramp', 't_plateau', 'omega_d']
                pulse_name = [
                    '$\epsilon_{{d}', '$t_{\\text{ramp}', '$t_{\\text{plateau}',
                    '$\omega_{{d}'
                ]
                unit_list = ['GHz', 'ns', 'ns', 'GHz']

                # pulse_params = ['amp', 'length', 'omega_d', ]
                # pulse_name = ['$\epsilon_{{d}', '$t_{\\text{single}', '$\omega_{{d}']
                # unit_list = ['GHz', 'ns', 'GHz']

                for i, par_name in enumerate(pulse_params):
                    # unpack gradient data
                    if gradient is not None:
                        pulse = _parse_pulse_name(node, 'pulse',
                                                  pulse_dict['pulse_type'])
                        # unpack gradient data
                        try:
                            data_grad = gradient[1][pulse][par_name]
                        except KeyError:
                            data_grad = gradient[2][pulse][par_name]
                        if par_name in ['omega_d', 'amp']:
                            data_grad = data_grad / np.pi / 2
                        str_grad = f'{data_grad:.3e} '
                    else:
                        str_grad = ''

                    data_par = pulse_dict[par_name]
                    if par_name == 'omega_d':
                        data_par = data_par / np.pi / 2
                        eof = '}/2 \pi$'
                    elif par_name == 'amp':
                        data_par = data_par / np.pi / 2
                        eof = '}$'
                    else:
                        eof = '}$'
                    row = [
                        NoEscape(pulse_name[i] + eof + f'({qubit_name[idx]})'),
                        f'{data_par:.3e} ' + unit_list[i], str_grad
                    ]
                    control_params.append(row)
        # filling
        if len(dev_params) >= len(control_params):
            target_col = dev_params
            fill_col = control_params
        else:
            target_col = control_params
            fill_col = dev_params
        for i, row in enumerate(target_col):
            if i < len(fill_col):
                row.extend(fill_col[i])
                data_table.add_row(row)
            else:
                row.extend(['', '', ''])
                data_table.add_row(row)
        data_table.add_hline()

    doc.generate_pdf("complex_report", clean_tex=False)


def generate_pauli_error_tabu(target_u,
                              sim_u,
                              num_print=10,
                              part_lst=[[0, 1], [2, 3], [4, 5]],
                              default_color='black',
                              average_error=None):
    """Generate latex source codes about pauli error analysis.
    Pauli error diagnose. Color the print result by how many partitions have errors

    Args:
        sim_u: unitary
        target_u: target unitary
        num_print: The top num_print pauli error operators will be printed
    """

    def pauli_error_string(error_list, prob_str):
        count = 0
        if part_lst is not None:
            for p in part_lst:
                if np.sum(error_list[p]) >= 1:
                    count += 1
        if count < 4:
            color = color_dict[count]
        else:
            color = 'blue'

        err_string = []
        # construct error list pair
        for p in part_lst:
            str_part = [pauli_dict[key] for key in error_list[p]]
            err_string.append(TextColor(color, ''.join(str_part)))

        err_string.append(TextColor(color, prob_str))

        return err_string

    pauli_dict = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z'}
    color_dict = {0: default_color, 1: 'teal', 2: 'orange', 3: 'purple'}
    qubit_name = {
        0: '$\mathrm{Q}_{02}$',
        1: '$\mathrm{Q}_{03}$',
        2: '$\mathrm{Q}_{12}$',
        3: '$\mathrm{Q}_{13}$',
        4: '$\mathrm{Q}_{22}$',
        5: '$\mathrm{Q}_{23}$'
    }

    err = sim_u.conj().T @ target_u
    pe = u_to_pauli(err)
    ind = np.unravel_index(np.argsort(pe, axis=None), pe.shape)
    ind_array = np.array(ind)

    geometry_options = {
        "head": "40pt",
        "margin": "0.5in",
        "bottom": "0.6in",
        "includeheadfoot": True
    }
    doc = Document(geometry_options=geometry_options)
    doc.packages.append(Package('amsmath'))

    # Add statement table
    num_part = len(part_lst)
    column_name = [
        NoEscape(' '.join([qubit_name[idx] for idx in p])) for p in part_lst
    ]
    with doc.create(Tabular('| c ' * num_part + '| c |',
                            row_height=1.5)) as data_table:
        data_table.add_hline()
        data_table.add_row(column_name + ['Error Rate'])
        data_table.add_hline()
        for i in range(1, num_print + 1):
            data_table.add_row(
                pauli_error_string(ind_array[:, -i], f'{pe[ind][-i]:10.4e}'))
        data_table.add_hline()
        if average_error is not None:
            data_table.add_hline()
            for i, error in enumerate(average_error, 1):
                color = color_dict[i]
                data_table.add_row([
                    MultiColumn(size=num_part,
                                align='| c |',
                                data=TextColor(
                                    color, f'Average weight-{i} Pauli error')),
                    TextColor(color, f'{error:10.4e}')
                ])
            data_table.add_hline()

    doc.generate_pdf("complex_report", clean_tex=False)
