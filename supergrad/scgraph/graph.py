import copy
from typing import Dict, Any, Optional, Literal, List
import jax
import jax.numpy as jnp
import networkx as nx
from networkx.classes.reportviews import EdgeView, NodeView

from supergrad.time_evolution.pulseshape import (PulseCosine, PulseTrapezoid,
                                                 PulseCosineRamping, PulseTanh,
                                                 PulseErf, PulseGaussian)
from supergrad.utils.utility import identity_wrap
from supergrad.utils.format_conv import deep_partial_dict, deep_update_dict

from supergrad.quantum_system.artificial_atom import (Fluxonium, Transmon,
                                                      Resonator)
from supergrad.quantum_system.interaction import (InteractingSystem,
                                                  InteractionTerm,
                                                  parse_interaction)

CompensationType = Literal["no_comp", "only_vz", "arbit_single"]

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
    pulse name, respectively.

    Args:
        q (str): a node name
        name (str): the pulse name
        pulse (str): the registered pulse type
    """
    if name.find("_") != -1:
        raise ValueError("Pulse name should not contain '_'.")
    key = '_'.join([q, name, pulse])
    return key


#: Keys to indicate the shared parameters
key_share_mark = "shared_param_mark"

#: Key of the deviation group
key_deviation = "deviation"

#: Key of compensation group
key_compensation = "compensation"

#: Keys to host sub-node, control related (not device related)
list_top_key_control = ["pulse", "compensation"]

#: Top groups that are fully not differentiable
list_not_differentiable_top = ["deviation", "arguments"]
#: Keys that are not differentiable and excluded from node parameter passing
list_not_differentiable_key = ["system_type", "operator_type", "pulse_type", "delay", "crosstalk"] + [key_share_mark]
list_not_differentiable_key_top = list_not_differentiable_key + list_not_differentiable_top


def is_key_device_param(key: str) -> bool:
    """Checks if a key belongs to device category group instead of control category.

    Args:
        key: the key of the item

    Returns:
        if the key belongs to device category
    """
    return key not in list_top_key_control and key != key_deviation


def is_differentiable_param(key: str) -> bool:
    """Checks if a key points differentiable parameters.

    Only used when the value is not a dictionary.

    Args:
        key: the key of the item

    Returns:
        if the value is a differentiable variable
    """
    return key not in list_not_differentiable_key_top


def get_quantum_system_params(dic: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts key-value pairs related to quantum system parameters.

    Args:
        dic: a dictionary with all possible parameters in the graph

    Returns:
        a dictionary with only quantum system related parameters
    """
    return dict([(key, val) for key, val in dic.items() if is_differentiable_param(key) and is_key_device_param(key)])


class SCGraph(nx.Graph):
    """Graph for storing parameters of qubits, their coupling strength
    and the control pulses.
    It is inherited from the `networkx.Graph`, which provides an implementation
    of graph structure.
    """

    def __init__(self, incoming_graph_data=None, seed=None, **attr):
        super().__init__(incoming_graph_data, **attr)

        # Set default values
        for key in ["share_params", "unify_coupling"]:
            if self.graph.get(key) is None:
                self.graph[key] = False

        # register the global seed and PRNGKey to support random number.
        self._seed = None
        self.key = None

        # Data to deal with shared parameters for nodes and edges
        self.map_share_node: Optional[Dict[str, Any]] = None
        self.map_share_edge: Optional[Dict[str, Any]] = None

        self._prepare_share_param()

        # Set seed to initialize
        if seed is not None:
            self.seed = seed

        # Set version
        version = attr.get("version", 0)
        self.version = version

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
            seed (int): the random seed to generate parameters deviation.
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
            node['deviation'] = variance

    def set_compensation(self, compensation_option: CompensationType = "only_vz"):
        """Sets all compensation to a given type in the graph, values to 0.

        Note, only devices already have compensation will be updated.

        Args:
            compensation_option: Set single qubit compensation strategy, should be in
                ['no_comp', 'only_vz', 'arbit_single']. 'no_comp' means we do no
                compensation. 'only_vz' means we will do single-qubit Z-rotation
                before and after the time evolution. 'arbit_single' means we will do
                arbitrary single-qubit rotation before and after the time evolution.

        """
        dic_shape = {"no_comp": 0, "only_vz": [], "arbit_single": [3]}
        shape = dic_shape[compensation_option]
        for node_name in self.sorted_nodes:
            node = NodeView(self)[node_name]
            if key_compensation in node:
                node[key_compensation] = jnp.zeros(shape)
        return

    @property
    def share_params(self):
        """
        If True, two sets of parameters will be shared according to the shared_param_mark attribute in the graph.
        This means that the shared parameters will remain the same during optimization.
        First, qubits with the same shared_param_mark will have their parameters shared.
        Second, the coupling strengths between qubits of type i and j are shared for all such pairs.

        Returns:
            whether the params are shared
        """
        return self.graph["share_params"]

    @share_params.setter
    def share_params(self, val: bool):
        # Note we must clean the cache
        if self.graph["share_params"] != val:
            self.map_share_edge = None
            self.map_share_node = None
            self.graph["share_params"] = val

    @property
    def unify_coupling(self):
        """
        If true, all coupling sharing the same parameters.
        True is valid only if `share_params` is True.
        """
        return self.graph["unify_coupling"]

    @unify_coupling.setter
    def unify_coupling(self, val: bool):
        # Note we must clean the cache
        if self.graph["unify_coupling"] != val:
            self.map_share_edge = None
            self.map_share_node = None
            self.graph["unify_coupling"] = val

    def subgraph(self, nodes: list) -> "SCGraph":
        """Returns a SubGraph view of the subgraph induced on `nodes` and apply functions in SCGraph.

        Args:
            nodes: A container of nodes which will be iterated through once.

        Returns:
            A subgraph view of the graph.
        """
        g: SCGraph = super().subgraph(nodes)
        g._prepare_share_param()
        return g

    def _prepare_share_param(self):
        """Computes necessary data structure to share the parameters between nodes/edges.

        Note this result is cached in the class. Change `share_params` of the object can clean the cache.

        """
        if self.map_share_node is not None and self.map_share_edge is not None:
            return

        if not self.share_params:
            self.map_share_node = {}
            self.map_share_edge = {}
            return

        # Build the map of shared parameters
        nv = NodeView(self)
        list_node_type = [nv[node_name].get(key_share_mark, node_name) for node_name in self.sorted_nodes]
        dic_share_node: Dict[str, str] = {}
        for node_name, node_type in zip(self.sorted_nodes, list_node_type):
            if node_type not in dic_share_node:
                dic_share_node[node_type] = node_name
        dic_share_edge: Dict[(str, str), (str, str)] = {}

        edge_first = None
        list_edge_type = [tuple(sorted((nv[u].get(key_share_mark, u), nv[v].get(key_share_mark, v)))) for u, v in
                          self.sorted_edges]
        for (u, v), edge_type in zip(self.sorted_edges, list_edge_type):
            if edge_first is None:
                edge_first = (u, v)
            if edge_type not in dic_share_edge:
                if self.unify_coupling:
                    dic_share_edge[edge_type] = edge_first
                else:
                    dic_share_edge[edge_type] = (u, v)

        self.map_share_node = dict(
            [(node_name, dic_share_node[node_type]) for node_name, node_type in zip(self.sorted_nodes, list_node_type)
             if node_name != dic_share_node[node_type]])
        self.map_share_edge = dict(
            [(edge_name, dic_share_edge[edge_type]) for edge_name, edge_type in zip(self.sorted_edges, list_edge_type)
             if edge_name != dic_share_edge[edge_type]])

    def _update_share_param(self):
        """Updates the shared parameters based on the shared setting.

        Note only device parameters are used.

        This automatically reconstruct the `map_share_*` data structure to update.

        TODO: This function severely slow down JAX backpropagation
        """
        for view, sorted_names, map_share in [(NodeView(self), self.sorted_nodes, self.map_share_node),
                                              (EdgeView(self), self.sorted_edges, self.map_share_edge)]:
            for name in sorted_names:
                name_root = map_share[name]
                deep_update_dict(view[name],
                                 deep_partial_dict(view[name_root], f_key=is_key_device_param))

        return

    def convert_graph_to_quantum_system(self, **kwargs):
        if self.version == 0:
            return self.convert_graph_to_quantum_system_v0(**kwargs)
        if self.version == 1:
            return self.convert_graph_to_quantum_system_v1(**kwargs)
        elif self.version == 2:
            return self.convert_graph_to_quantum_system_v2(**kwargs)
        else:
            raise ValueError(f"Wrong version {self.version}")

    def convert_graph_to_quantum_system_v2(self,
                                        **kwargs):
        """Convert a networkX graph to SuperGrad quantum system.
        """
        # convert nodes to artificial atoms
        subsystem_list = []
        for node_name in self.sorted_nodes:
            node = NodeView(self)[node_name]
            node_kwargs = copy.deepcopy(kwargs)
            # update global kwargs with graph kwargs
            node_kwargs.update(node.get('arguments', {}))
            # Update quantum system parameters from shared dictionary
            node_root = NodeView(self)[self.map_share_node.get(node_name, node_name)]
            node_kwargs.update(get_quantum_system_params(node_root))
            var = node.get('deviation', None)
            if var is not None:
                for key, val in var.items():
                    if key in node_kwargs:
                        node_kwargs[key] *= val
            subsystem_list.append(artificial_atom_dict[node['system_type']](
                name=node_name, **node_kwargs))

        subsystem_name_list = self.sorted_nodes
        interaction_list = []
        for edge in self.sorted_edges:
            q1, q2 = edge
            #  Get attributes from shared map
            edge_root = self.map_share_edge.get(edge, edge)
            attr = EdgeView(self)[edge_root]
            idx_q1 = subsystem_name_list.index(q1)
            idx_q2 = subsystem_name_list.index(q2)
            interaction_pair = []
            for k in dict(attr).keys():
                if k == 'capacitive_coupling':
                    op1 = subsystem_list[idx_q1].n_operator
                    op2 = subsystem_list[idx_q2].n_operator
                elif k == 'inductive_coupling':
                    op1 = subsystem_list[idx_q1].phi_operator
                    op2 = subsystem_list[idx_q2].phi_operator
                else:
                    raise ValueError(f"Unknown coupling type {k}")
                interaction_pair.append(
                    parse_interaction(
                        **attr[k],
                        op1=op1,
                        op2=op2,
                        name=_parse_edges_name(k, q1, q2)))
                interaction_list.append(interaction_pair)

        # unflatten interaction list
        inter_list = []
        for inter_pair in interaction_list:
            inter_list.extend(inter_pair)
        return InteractingSystem(subsystem_list, inter_list)

    def convert_graph_to_quantum_system_v1(self, **kwargs):
        unify_coupling = self.unify_coupling
        share_params = self.share_params
        add_random = False

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
            node_kwargs.update(get_quantum_system_params(node))
            var = None
            if add_random:
                var = node.get('deviation', None)
            subsystem_list.append(artificial_atom_dict[node['system_type']](
                name=node_name, var=var, **node_kwargs))
            type_list.append(node_type)
        subsystem_name_list = self.sorted_nodes
        interaction_list = []
        edge_type_list = []
        for edge in self.sorted_edges:
            q1, q2 = edge
            attr = EdgeView(self)[edge]
            nodes_pair = sorted([q1, q2])
            edge_type = tuple(
                NodeView(self)[node].get('shared_param_mark', node)
                for node in nodes_pair)
            idx_q1 = subsystem_name_list.index(q1)
            idx_q2 = subsystem_name_list.index(q2)


            interaction_pair = []
            for k in dict(attr).keys():
                if k == 'capacitive_coupling':
                    interaction_pair.append(
                        parse_interaction(
                            **attr[k],
                            op1=subsystem_list[idx_q1].n_operator,
                            op2=subsystem_list[idx_q2].n_operator,
                            name=_parse_edges_name(k, q1, q2)))
                elif k == 'inductive_coupling':
                    interaction_pair.append(
                        parse_interaction(
                            **attr[k],
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

    def convert_graph_to_quantum_system_v0(self, **kwargs):
        unify_coupling = self.unify_coupling
        share_params = self.share_params
        add_random = False

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
            node_kwargs.update(get_quantum_system_params(node))
            if node_type in type_list and share_params:
                type_idx = type_list.index(node_type)
                mirrored_subsystem = copy.deepcopy(subsystem_list[type_idx])
                # reconstruct class
                mirrored_subsystem.name = node_name
                if add_random:
                    var = node.get('deviation', None)
                    mirrored_subsystem.add_lcj_params_variance(var)
                subsystem_list.append(mirrored_subsystem)
            else:
                var = None
                if add_random:
                    var = node.get('deviation', None)
                subsystem_list.append(artificial_atom_dict[node['system_type']](
                    name=node_name, var=var, **node_kwargs))
            type_list.append(node_type)
        subsystem_name_list = self.sorted_nodes
        interaction_list = []
        edge_type_list = []
        for edge in self.sorted_edges:
            q1, q2 = edge
            attr = EdgeView(self)[edge]
            nodes_pair = sorted([q1, q2])
            edge_type = tuple(
                NodeView(self)[node].get('shared_param_mark', node)
                for node in nodes_pair)
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
                                **attr[k],
                                op1=subsystem_list[idx_q1].n_operator,
                                op2=subsystem_list[idx_q2].n_operator,
                                name=_parse_edges_name(k, 'all', 'unify'))
                            unify_cap_coupling.name = _parse_edges_name(
                                k, q1, q2)
                            interaction_pair.append(unify_cap_coupling)
                        elif k == 'inductive_coupling':
                            unify_ind_coupling = parse_interaction(
                                **attr[k],
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
                                    **attr[k],
                                    op1=subsystem_list[idx_q1].n_operator,
                                    op2=subsystem_list[idx_q2].n_operator,
                                    name=_parse_edges_name(k, q1, q2)))
                        elif k == 'inductive_coupling':
                            interaction_pair.append(
                                parse_interaction(
                                    **attr[k],
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
            if "pulse" not in node:
                continue
            for k, v in node["pulse"].items():
                opt_type = v.pop('operator_type')
                opt = getattr(hilbert_space[node_name], opt_type)
                pulse_type = v.pop('pulse_type')
                delay = v.pop('delay', 0.)
                # update global kwargs with graph kwargs
                pulse_kwargs = copy.deepcopy(kwargs)
                pulse_kwargs.update(v.pop('arguments', {}))
                # Update pulse differentiable parameters
                pulse_kwargs.update(get_quantum_system_params(v))
                pulse = pulse_shape_dict[pulse_type](delay=delay,
                                                     name=_parse_pulse_name(
                                                         node_name, k,
                                                         "pulse"),
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

    def update_parameters(self, params: Dict[str, Any]):
        """Updates the parameters in the graph by a dict-represented graph (with nodes and edges).

        All parameters in the input dictionary must be JAX array,
        and they will overwrite the original graph.

        Args:
            params: the parameters to update.

        """
        # We assume the graph is frozen, so graph update/add cannot be used
        # Only attributes are modified
        if "nodes" in params:
            nv = NodeView(self)
            # We modify in-place instead of replace any dict to avoid data loss
            for node_name, node_data in params["nodes"].items():
                deep_update_dict(nv[node_name], node_data)

        if "edges" in params:
            ev = EdgeView(self)
            for (u, v), edge_data in params["edges"].items():
                deep_update_dict(ev[u, v], edge_data)

        return

    def convert_graph_to_parameters(self,
                                    only_device_params=False,
                                    only_compensation=False,
                                    ) -> dict:
        """Convert a networkX graph to dictionary that contains all JAX array parameters.

        Args:
            only_device_params (bool): whether to only load device parameters.
            only_compensation (bool): whether to only load compensation parameters

        Raises:
            ValueError: both only_* options are true

        Returns:
            a dictionary with all JAX array parameters
        """
        if only_compensation and only_compensation:
            raise ValueError("At most one only_* can be specified")
        self._prepare_share_param()

        f_key = is_differentiable_param
        if only_device_params:
            def f_key(x):
                return is_key_device_param(x) and is_differentiable_param(x)
        elif only_compensation:
            def f_key(x):
                return x in [key_compensation, "pre_comp", "post_comp"]

        nodes = self.sorted_nodes
        edges = self.sorted_edges

        if self.share_params:
            nodes_root = self.map_share_node.values()
            edges_root = self.map_share_edge.values()
            nodes = [x for x in nodes if x in nodes_root]
            edges = [x for x in edges if x in edges_root]

        nv = NodeView(self)
        ev = EdgeView(self)

        return deep_partial_dict(
            {"nodes": dict([(node_name, nv[node_name]) for node_name in nodes]),
             "edges": dict([(edge_name, ev[edge_name]) for edge_name in edges])},
            f_val=lambda x: isinstance(x, jax.Array), f_key=f_key)
