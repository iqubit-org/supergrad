import copy
from typing import List
import warnings
import jax
import jax.numpy as jnp
import networkx as nx
from networkx.classes.reportviews import EdgeView, NodeView

from supergrad.time_evolution.pulseshape import (PulseCosine, PulseTrapezoid,
                                                 PulseCosineRamping, PulseTanh,
                                                 PulseErf, PulseGaussian)
from supergrad.utils.utility import identity_wrap, convert_to_haiku_dict

from supergrad.quantum_system.artificial_atom import (Fluxonium, Transmon,
                                                      Resonator)
from supergrad.quantum_system.interaction import (InteractingSystem,
                                                  InteractionTerm,
                                                  parse_interaction)

# please register module in the following dictionary for graph converting
artificial_atom_dict = {
    'fluxonium': Fluxonium,
    'transmon': Transmon,
    'resonator': Resonator
}
pulse_shape_dict = {
    'trapezoid': PulseTrapezoid,
    'cos': PulseCosine,
    'rampcos': PulseCosineRamping,
    'tanh': PulseTanh,
    'erf': PulseErf,
    'gaussian': PulseGaussian
}


def parse_pre_comp_name(q):
    """Parse the previous compensation name. The compensation name consist of
    three parts separated by underscores '_', the third part is nodes' name.

    Args:
        q (str): the node name
    """
    key = '_'.join(['pre', 'comp', q])
    return key


def parse_post_comp_name(q):
    """Parse the post compensation name. The compensation name consist of
    three parts separated by underscores '_', the third part is nodes' name.

    Args:
        q (str): the node name
    """
    key = '_'.join(['post', 'comp', q])
    return key


def _parse_edges_name(k, q1, q2):
    """Parse interaction name. The compensation name consist of four parts
    separated by underscores '_'. The first and second part could be either
    'capacitive_coupling' or 'inductive_coupling', the third and fourth
    part are sorted qubits' name.

    Args:
        k (str): coupling type in ['capacitive_coupling', 'inductive_coupling']
        q1 (str): a node name
        q2 (str): a node name
    """
    key = '_'.join(sorted([q1, q2]))  # use string intrinsic order
    key = '_'.join([k, key])
    return key


def _parse_pulse_name(q: str, name: str, pulse: str):
    """Parse pulseshape name. The compensation name consist of three parts
    separated by underscores '_', they are node name, 'pulse' and the registered
    pulse type, respectively.

    Args:
        q (str): a node name
        name (str): the pulse name
        pulse (str): the registered pulse type
    """
    if name.find("_") != -1:
        raise ValueError("Pulse name should not contain '_'.")
    key = '_'.join([q, name, pulse])
    return key


class SCGraph(nx.Graph):
    """Graph for storing parameters of qubits, their coupling strength
    and the control pulses.
    It is inherited from the `networkx.Graph`, which provides an implementation
    of graph structure.
    """

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)

        # register the global seed and PRNGKey to support random number.
        self._seed = None
        self.key = None

    @property
    def sorted_nodes(self):
        """A list contains all the nodes in deterministic order."""
        return sorted(list(NodeView(self)))

    @property
    def sorted_edges(self):
        """A list contains all the edges in deterministic order."""
        return sorted(list(EdgeView(self)))

    @property
    def seed(self):
        """The seed for random device parameters variance."""
        return self._seed

    @seed.setter
    def seed(self, new_seed):
        if isinstance(new_seed, int):
            self._seed = new_seed
            # update the PRNGKey
            self.key = jax.random.PRNGKey(self._seed)
        else:
            raise ValueError('Please use integer as random seed.')

    def add_lcj_params_variance_to_graph(self, multi_err=0.01, seed=1):
        """Assign (random) values of variance to superconducting qubits' parameters
        ['ec', 'ej', 'el'] in the graph. Note that here we only generate and store
        the multiplicative factors in the graph. They are not applied yet.

        Args:
            multi_err (float): the strength of the multiplicative error.
                More concretely, a different factor 1+ N * multi_error will be
                multiplied to each parameter, where N is sampled from normal
                distribution.
            seed (int): the random seed to generate parameters variance.
        """
        # generate PRNGKey by setter
        self.seed = seed
        for node_name in self.sorted_nodes:
            self.key, subkey = jax.random.split(self.key)
            node = NodeView(self)[node_name]
            params_variance = 1 + jax.random.normal(subkey,
                                                    shape=(3,)) * multi_err
            variance = {}
            for i, item in enumerate(['ec', 'ej', 'el']):
                variance[item] = params_variance[i]
            node['variance'] = variance

    def _convert_graph_to_pulse_parameters(self, node, node_name, params_haiku):
        """Convert a networkX graph to Haiku's parameters dictionary.
        Only for internal use.
        """
        # update pulse parameters
        for k, v in dict(node).items():
            if str(k).startswith('pulse'):
                pulse_type = v.pop('pulse_type')
                key = _parse_pulse_name(node_name, k, pulse_type)
                # parameters tidy up
                for unused_params in [
                        'delay', 'operator_type', 'crosstalk', 'arguments'
                ]:
                    v.pop(unused_params, None)
                params_haiku.update({key: v})
        return params_haiku

    def _convert_graph_to_compensation(self, node, node_name,
                                       compensation_params):
        """Convert a networkX graph to Haiku's parameters dictionary.
        Only for internal use.
        """
        for k, v in dict(node).items():
            if str(k).startswith('compensation'):
                pre_comp_val = v['pre_comp']
                post_comp_val = v['post_comp']
                compensation_params.update(
                    {parse_pre_comp_name(node_name): pre_comp_val})
                compensation_params.update(
                    {parse_post_comp_name(node_name): post_comp_val})
        return compensation_params

    def convert_graph_to_pulse_parameters_haiku(self, load_compensation=True):
        """Convert a networkX graph to Haiku's parameters dictionary.

        Args:
            load_compensation: whether to load the compensation parameters.
        """
        graph = copy.deepcopy(self)
        params_haiku = {}
        # update artificial atom parameters
        compensation_params = {}
        for node_name in self.sorted_nodes:
            node: dict = graph.nodes[node_name]
            self._convert_graph_to_pulse_parameters(node, node_name,
                                                    params_haiku)
            self._convert_graph_to_compensation(node, node_name,
                                                compensation_params)
        # update compensation parameters
        if compensation_params and load_compensation:
            params_haiku.update({'single_q_compensation': compensation_params})
        return convert_to_haiku_dict(params_haiku)

    def convert_graph_to_parameters_haiku(self,
                                          share_params=False,
                                          unify_coupling=False,
                                          only_device_params=False):
        """Convert a networkX graph to Haiku's parameters dictionary.

        Args:
            share_params (bool): If True, two sets of parameters will be shared
                according to the shared_param_mark attribute in the graph. This means
                that the shared parameters will remain the same during optimization.
                First, qubits with the same shared_param_mark will have their parameters
                shared. Second, the coupling strengths between qubits of
                type i and j are shared for all such pairs.
            unify_coupling (bool): all coupling sharing the same parameters.
                Valid only if `share_params` is True.
            only_device_params (bool): whether to only load device parameters.
        """
        graph = copy.deepcopy(self)
        params_haiku = {}
        # update artificial atom parameters
        type_list = []
        compensation_params = {}
        for node_name in self.sorted_nodes:
            node: dict = graph.nodes[node_name]
            node_type = node.get('shared_param_mark', node_name)
            # update pulse parameters
            if not only_device_params:
                self._convert_graph_to_pulse_parameters(node, node_name,
                                                        params_haiku)
                self._convert_graph_to_compensation(node, node_name,
                                                    compensation_params)
            if node_type not in type_list or not share_params:
                qubit_params = {}
                attr_dict = dict(node)
                for kw in ['ec', 'ej', 'el', 'phiext']:
                    val = attr_dict.pop(kw, None)
                    if val is not None:
                        qubit_params.update({kw: val})
                params_haiku.update({node_name: qubit_params})
            type_list.append(node_type)
        # update compensation parameters
        if compensation_params and not only_device_params:
            params_haiku.update({'single_q_compensation': compensation_params})
        # update interaction parameters
        edge_type_list = []
        for edge in self.sorted_edges:
            q1, q2 = edge
            attr = graph.edges[edge]
            edge_type = tuple(
                sorted([
                    graph.nodes[node].get('shared_param_mark', node)
                    for node in [q1, q2]
                ]))
            for k, v in dict(attr).items():
                if unify_coupling:
                    key = _parse_edges_name(k, 'all', 'unify')
                elif edge_type_list and unify_coupling:
                    continue
                else:
                    if edge_type in edge_type_list and share_params:
                        continue
                    else:
                        key = _parse_edges_name(k, q1, q2)
                params_haiku.update({key: v})
            edge_type_list.append(edge_type)
        return convert_to_haiku_dict(params_haiku)

    def convert_graph_to_comp_initial_guess(self,
                                            compensation_option='only_vz',
                                            name='single_q_compensation'):
        """Convert a networkX graph to virtual compensation initial guess.

        Args:
            compensation_option: single qubit compensation strategy, should be
                in ['only_vz', 'arbit_single]
            name: name of the single qubit compensation module, default name is
                'single_q_compensation'.
        """
        assert compensation_option in ['only_vz', 'arbit_single']
        shape = [] if compensation_option == 'only_vz' else [3]
        comp_dict = {}
        for node in self.sorted_nodes:
            comp_dict.update({parse_pre_comp_name(node): jnp.zeros(shape)})
            comp_dict.update({parse_post_comp_name(node): jnp.zeros(shape)})
        return {name: comp_dict}

    def convert_graph_to_quantum_system(self,
                                        add_random: bool = True,
                                        share_params: bool = True,
                                        unify_coupling: bool = False,
                                        **kwargs):
        """Convert a networkX graph to SuperGrad quantum system.

        Args:
            add_random: whether adding random device parameters deviations or not
            share_params : If True, two sets of parameters will be shared
                according to the shared_param_mark attribute in the graph. This means
                that the shared parameters will remain the same during optimization.
                First, qubits with the same shared_param_mark will have their parameters
                shared. Second, the coupling strengths between qubits of
                type i and j are shared for all such pairs.
            unify_coupling: all coupling sharing the same parameters. Valid
                only if `share_params` is True.
        """
        # check arguments
        if unify_coupling:
            assert share_params is True
        # convert nodes to artificial atoms
        subsystem_list = []
        type_list = []
        for node_name in self.sorted_nodes:
            node = NodeView(self)[node_name]
            node_type = node.get('shared_param_mark', node_name)
            node_kwargs = copy.deepcopy(kwargs)
            # update global kwargs with graph kwargs
            node_kwargs.update(node.get('arguments', {}))
            if node_type in type_list and share_params:
                type_idx = type_list.index(node_type)
                mirrored_subsystem = copy.deepcopy(subsystem_list[type_idx])
                # reconstruct class
                mirrored_subsystem.name = node_name
                if add_random:
                    var = node.get('variance', None)
                    mirrored_subsystem.add_lcj_params_variance(var)
                subsystem_list.append(mirrored_subsystem)
            else:
                var = None
                if add_random:
                    var = node.get('variance', None)
                subsystem_list.append(artificial_atom_dict[node['system_type']](
                    name=node_name, var=var, **node_kwargs))
            type_list.append(node_type)
        subsystem_name_list = self.sorted_nodes
        interaction_list = []
        edge_type_list = []
        for edge in self.sorted_edges:
            q1, q2 = edge
            attr = EdgeView(self)[edge]
            edge_type = tuple(
                sorted([
                    NodeView(self)[node].get('shared_param_mark', node)
                    for node in [q1, q2]
                ]))
            idx_q1 = subsystem_name_list.index(q1)
            idx_q2 = subsystem_name_list.index(q2)
            if unify_coupling:
                predicate = interaction_list and share_params  # TODO what is predicate
            else:
                predicate = (edge_type in edge_type_list and share_params)

            if predicate:
                if unify_coupling:
                    mirrored_interaction_pair: List[
                        InteractionTerm] = copy.deepcopy(interaction_list[0])
                else:
                    edge_type_idx = edge_type_list.index(edge_type)
                    mirrored_interaction_pair: List[
                        InteractionTerm] = copy.deepcopy(
                            interaction_list[edge_type_idx])
                # reconstruct class
                for mirror in mirrored_interaction_pair:
                    old_name = mirror.name
                    k = '_'.join(old_name.split('_')[0:2])
                    mirror.name = _parse_edges_name(k, q1, q2)
                    if k == 'capacitive_coupling':
                        mirror.operator_list = [
                            subsystem_list[idx_q1].n_operator,
                            subsystem_list[idx_q2].n_operator
                        ]
                    elif k == 'inductive_coupling':
                        mirror.operator_list = [
                            subsystem_list[idx_q1].phi_operator,
                            subsystem_list[idx_q2].phi_operator
                        ]
                interaction_list.append(mirrored_interaction_pair)
            else:
                interaction_pair = []
                if unify_coupling:
                    for k in dict(attr).keys():
                        if k == 'capacitive_coupling':
                            unify_cap_coupling = parse_interaction(
                                op1=subsystem_list[idx_q1].n_operator,
                                op2=subsystem_list[idx_q2].n_operator,
                                name=_parse_edges_name(k, 'all', 'unify'))
                            unify_cap_coupling.name = _parse_edges_name(
                                k, q1, q2)
                            interaction_pair.append(unify_cap_coupling)
                        elif k == 'inductive_coupling':
                            unify_ind_coupling = parse_interaction(
                                op1=subsystem_list[idx_q1].phi_operator,
                                op2=subsystem_list[idx_q2].phi_operator,
                                name=_parse_edges_name(k, 'all', 'unify'))
                            unify_ind_coupling.name = _parse_edges_name(
                                k, q1, q2)
                            interaction_pair.append(unify_ind_coupling)
                else:
                    for k in dict(attr).keys():
                        if k == 'capacitive_coupling':
                            interaction_pair.append(
                                parse_interaction(
                                    op1=subsystem_list[idx_q1].n_operator,
                                    op2=subsystem_list[idx_q2].n_operator,
                                    name=_parse_edges_name(k, q1, q2)))
                        elif k == 'inductive_coupling':
                            interaction_pair.append(
                                parse_interaction(
                                    op1=subsystem_list[idx_q1].phi_operator,
                                    op2=subsystem_list[idx_q2].phi_operator,
                                    name=_parse_edges_name(k, q1, q2)))
                interaction_list.append(interaction_pair)

            edge_type_list.append(edge_type)
        # unflatten interaction list
        inter_list = []
        for inter_pair in interaction_list:
            inter_list.extend(inter_pair)
        return InteractingSystem(subsystem_list, inter_list)

    def convert_graph_to_pulse_lst(self, hilbert_space: InteractingSystem,
                                   **kwargs):
        """Obtain the pulse list and the max length of all pulses.
        The pulse list contain a list of [wrap_opt, pulse.create_pulse].
        wrap_opt is the drive operator on the whole Hilbert space.
        pulse.create_pulse gives function for computing the shape of the pulses.

        Args:
            hilbert_space: the hilbert space class
            kwargs: The keyword arguments be passed to module `PulseShape`

        """
        graph = copy.deepcopy(self)
        pulse_lst = []
        pulse_endtime = [0.]
        for node_name in self.sorted_nodes:
            node = graph.nodes[node_name]
            for k, v in dict(node).items():
                if str(k).startswith('pulse'):
                    opt_type = v.pop('operator_type')
                    opt = getattr(hilbert_space[node_name], opt_type)
                    pulse_type = v.pop('pulse_type')
                    delay = v.pop('delay', 0.)
                    # update global kwargs with graph kwargs
                    pulse_kwargs = copy.deepcopy(kwargs)
                    pulse_kwargs.update(v.pop('arguments', {}))
                    pulse = pulse_shape_dict[pulse_type](delay=delay,
                                                         name=_parse_pulse_name(
                                                             node_name, k,
                                                             pulse_type),
                                                         **pulse_kwargs)
                    wrap_opt = identity_wrap(opt, hilbert_space.subsystem_list)
                    pulse_lst.append([wrap_opt, pulse.create_pulse])
                    pulse_endtime.append(pulse.pulse_endtime)
                    crosstalk = v.pop('crosstalk', None)
                    if crosstalk is not None:
                        for xterm, coeff in crosstalk.items():
                            opt = getattr(hilbert_space[xterm], opt_type)
                            wrap_opt = identity_wrap(
                                opt, hilbert_space.subsystem_list, coeff=coeff)
                            pulse_lst.append([wrap_opt, pulse.create_pulse])
        return pulse_lst, jnp.max(jnp.array(pulse_endtime))

    def update_params(self,
                      params: dict,
                      share_params=True,
                      unify_coupling=False):
        """Update parameters to networkX Graph

        Args:
            params: new parameters dictionary
            share_params : If True, two sets of parameters will be shared
                according to the shared_param_mark attribute in the graph. This means
                that the shared parameters will remain the same during optimization.
                First, qubits with the same shared_param_mark will have their parameters
                shared. Second, the coupling strengths between qubits of
                type i and j are shared for all such pairs.
            unify_coupling: all coupling sharing the same parameters. Valid
                only if `share_params` is True.
        """
        # check arguments
        if unify_coupling:
            assert share_params is True
        if share_params:
            # load parameters dict
            type_dict = {}
            pulse_dict = {}
            edge_type_dict = {}
            for k, v in params.items():
                parse_k = k.split('_')
                # artificial atom
                if len(parse_k) == 1:
                    q = parse_k[0]
                    q_type = NodeView(self)[q].get('shared_param_mark', q)
                    type_dict.update({q_type: v})
                # pulse shape
                elif len(parse_k) == 3:
                    pulse_dict.update({k: v})
                # interaction
                elif len(parse_k) == 4:
                    k1, k2, q1, q2 = parse_k
                    coupling_name = '_'.join([k1, k2])
                    if '_'.join([q1, q2]) == 'all_unify':
                        try:
                            edge_type_dict[('all',
                                            'unify')].update({coupling_name: v})
                        except KeyError:
                            edge_type_dict.update({
                                ('all', 'unify'): {
                                    coupling_name: v
                                }
                            })
                    else:
                        edge_type = tuple(
                            sorted(
                                NodeView(self)[node].get(
                                    'shared_param_mark', node)
                                for node in [q1, q2]))
                        try:
                            edge_type_dict[edge_type].update({coupling_name: v})
                        except KeyError:
                            edge_type_dict.update(
                                {edge_type: {
                                    coupling_name: v
                                }})
            # update graph nodes parameters
            for node in NodeView(self):
                node_type = NodeView(self)[node].get('shared_param_mark', node)
                try:
                    NodeView(self)[node].update(type_dict[node_type])
                except KeyError:
                    warnings.warn(
                        f'The parameters dictionary does not contain nodes '
                        f'type {node_type}, keeping node {node} unchanged.')

            # update graph edges parameters
            for edge in EdgeView(self):
                q1, q2 = edge
                edge_type = tuple(
                    sorted([
                        NodeView(self)[node].get('shared_param_mark', node)
                        for node in [q1, q2]
                    ]))
                # update interaction
                try:
                    if unify_coupling:
                        EdgeView(self)[edge].update(
                            list(edge_type_dict.values())[0])
                    else:
                        EdgeView(self)[edge].update(edge_type_dict[edge_type])
                except KeyError and IndexError:
                    warnings.warn(
                        f'The parameters dictionary does not contain edges '
                        f'type {edge_type}, keeping edge {edge} unchanged.')
        else:
            pulse_dict = {}
            for k, v in params.items():
                parse_k = k.split('_')
                # artificial atom
                if len(parse_k) == 1:
                    q = parse_k[0]
                    NodeView(self)[q].update(v)
                # pulse shape
                elif len(parse_k) == 3:
                    pulse_dict.update({k: v})
                # interaction
                elif len(parse_k) == 4:
                    k1, k2, q1, q2 = parse_k
                    coupling_name = '_'.join([k1, k2])
                    EdgeView(self)[(q1, q2)].update({coupling_name: v})
        # update pulse shape parameters
        for node_name in self.sorted_nodes:
            node = NodeView(self)[node_name]
            for k, v in dict(node).items():
                if str(k).startswith('pulse'):
                    pulse_type = v['pulse_type']
                    pulse_info = pulse_dict.get(
                        _parse_pulse_name(node_name, k, pulse_type), None)
                    if pulse_info is not None:
                        v.update(pulse_info)

    def remove_pulse(self):
        """Remove pulse parameters from the graph"""
        for node_name in self.sorted_nodes:
            node = NodeView(self)[node_name]
            for k, v in dict(node).items():
                if str(k).startswith('pulse'):
                    # remove key and value pair
                    node.pop(k, None)
