from collections import deque
import jax
import jax.numpy as jnp
import networkx as nx

from supergrad.scgraph.graph import SCGraph

# Below we try to specify the structure and parameters of a quantum processor
# This is done step by step.

# A detail explanation of all parameters in the resulting graph can be found in the end of this file.
# The explanation is roughly in the form of json schemas.

# For the examples in the notebook, we use parameters in PeriodicGraphOpt and its child class
# The parameters of PeriodicGraphOpt are obtained through some optimization
# This will be explained in a related paper.


class PeriodicGraph(SCGraph):
    """Device parameters for 5x5 periodic quantum processor graph.


    Args:
        seed : (int) The seed for random device parameters variance.
            default seed is None, means no variance.
    """

    def __init__(self, seed=None):
        super().__init__(None)
        # initialize graph
        temp_graph = nx.grid_2d_graph(5, 5, periodic=True)

        params_1 = {
            'ec': jnp.array(1. * 2 * jnp.pi),
            'ej': jnp.array(4. * 2 * jnp.pi),
            'el': jnp.array(1.1 * 2 * jnp.pi),
            'system_type': 'fluxonium',
            'shared_param_mark': 1,
            'arguments': {
                'phiext': 0.5 * 2 * jnp.pi
            }
        }
        params_2 = {
            'ec': jnp.array(1. * 2 * jnp.pi),
            'ej': jnp.array(4. * 2 * jnp.pi),
            'el': jnp.array(1.2 * 2 * jnp.pi),
            'system_type': 'fluxonium',
            'shared_param_mark': 2,
            'arguments': {
                'phiext': 0.5 * 2 * jnp.pi
            }
        }
        params_3 = {
            'ec': jnp.array(1. * 2 * jnp.pi),
            'ej': jnp.array(4. * 2 * jnp.pi),
            'el': jnp.array(1.0 * 2 * jnp.pi),
            'system_type': 'fluxonium',
            'shared_param_mark': 3,
            'arguments': {
                'phiext': 0.5 * 2 * jnp.pi
            }
        }
        params_4 = {
            'ec': jnp.array(1. * 2 * jnp.pi),
            'ej': jnp.array(4. * 2 * jnp.pi),
            'el': jnp.array(0.8 * 2 * jnp.pi),
            'system_type': 'fluxonium',
            'shared_param_mark': 4,
            'arguments': {
                'phiext': 0.5 * 2 * jnp.pi
            }
        }
        params_5 = {
            'ec': jnp.array(1. * 2 * jnp.pi),
            'ej': jnp.array(4. * 2 * jnp.pi),
            'el': jnp.array(0.9 * 2 * jnp.pi),
            'system_type': 'fluxonium',
            'shared_param_mark': 5,
            'arguments': {
                'phiext': 0.5 * 2 * jnp.pi
            }
        }
        coupling_params = {
            'capacitive_coupling': {
                'strength': jnp.array(11.5e-3 * 2 * jnp.pi)
            },
            'inductive_coupling': {
                'strength': jnp.array(-1.0 * 2e-3 * 2 * jnp.pi)
            },
        }

        # adding attributes to nodes
        params = deque([params_1, params_2, params_3, params_4, params_5])
        for i in range(5):
            for j in range(5):
                temp_graph.nodes[(i, j)].update(params[j])
            # params list right shift
            params.rotate(3)
        # relabel nodes
        label_mapping = dict(
            (label,
             ''.join(['q', str(label[0]), str(label[1])]))
            for label in temp_graph.nodes)
        temp_graph = nx.relabel_nodes(temp_graph, label_mapping)
        # adding attributes to edges
        for edge in temp_graph.edges:
            temp_graph.edges[edge].update(coupling_params)
        # save temp_graph
        self.add_nodes_from(temp_graph.nodes.data())
        self.add_edges_from(temp_graph.edges.data())

        if seed is not None:
            # add variance to el ec ej params
            self.add_lcj_params_variance_to_graph(multi_err=0.01, seed=seed)


class XGatePeriodicGraph(PeriodicGraph):
    """Add pulse parameters for simultaneous X gates.
    In the example, we add 6 pulses on `q02`, `q03`, `q12`,
    `q13`, `q22` and `q23`.

    Args:
        seed : (int) The seed for random device parameters variance.
            default seed is None, means no variance.
    """

    def __init__(self, seed=None):
        super().__init__(seed)

        # note that the keywords of other pulse types are not always the same
        self.add_node(
            'q02', **{
                'pulse': {
                    'amp': jnp.array(0.07439612),
                    'length': jnp.array(40.05643032),
                    'omega_d': jnp.array(3.58624499),
                    'phase': jnp.array(1.29078953),
                    'pulse_type': 'cos',
                    'operator_type': 'phi_operator',
                    'delay': 0.
                },
                'compensation': {
                    'pre_comp': jnp.array(-0.33619935),
                    'post_comp': jnp.array(0.33863245)
                }
            })
        self.add_node(
            'q03', **{
                'pulse': {
                    'amp': jnp.array(0.06819317),
                    'length': jnp.array(40.04157743),
                    'omega_d': jnp.array(2.61055019),
                    'phase': jnp.array(0.5435847),
                    'pulse_type': 'cos',
                    'operator_type': 'phi_operator',
                    'delay': 0.
                },
                'compensation': {
                    'pre_comp': jnp.array(0.51024),
                    'post_comp': jnp.array(-0.50827319)
                },
            })

        self.add_node(
            'q12', **{
                'pulse': {
                    'amp': jnp.array(0.07169389),
                    'length': jnp.array(40.0549766),
                    'omega_d': jnp.array(3.17153229),
                    'phase': jnp.array(0.87593417),
                    'pulse_type': 'cos',
                    'operator_type': 'phi_operator',
                    'delay': 0.
                },
                'compensation': {
                    'pre_comp': jnp.array(0.0192981),
                    'post_comp': jnp.array(-0.00608968)
                }
            })
        self.add_node(
            'q13', **{
                'pulse': {
                    'amp': jnp.array(0.07708328),
                    'length': jnp.array(40.04376009),
                    'omega_d': jnp.array(4.19561078),
                    'phase': jnp.array(1.37478176),
                    'pulse_type': 'cos',
                    'operator_type': 'phi_operator',
                    'delay': 0.
                },
                'compensation': {
                    'pre_comp': jnp.array(-0.47187891),
                    'post_comp': jnp.array(0.47470512)
                }
            })

        self.add_node(
            'q22', **{
                'pulse': {
                    'amp': jnp.array(0.08007134),
                    'length': jnp.array(40.04477099),
                    'omega_d': jnp.array(5.00355723),
                    'phase': jnp.array(1.23550966),
                    'pulse_type': 'cos',
                    'operator_type': 'phi_operator',
                    'delay': 0.
                },
                'compensation': {
                    'pre_comp': jnp.array(-0.3009113),
                    'post_comp': jnp.array(0.30283622)
                }
            })
        self.add_node(
            'q23', **{
                'pulse': {
                    'amp': jnp.array(0.07334288),
                    'length': jnp.array(40.04449978),
                    'omega_d': jnp.array(3.51294492),
                    'phase': jnp.array(0.75852119),
                    'pulse_type': 'cos',
                    'operator_type': 'phi_operator',
                    'delay': 0.
                },
                'compensation': {
                    'pre_comp': jnp.array(0.23767629),
                    'post_comp': jnp.array(-0.22839222)
                }
            })


class CNOTGatePeriodicGraph(PeriodicGraph):
    """Add pulse parameters for simultaneous CNOT gates.
    In the example, we add 3 CR pulses on `q02`, `q12` and `q22`.
    Note that the target qubits are `q03`, `q13`, `q23` respectively.

    Args:
        seed : (int) The seed for random device parameters variance.
            default seed is None, means no variance.
    """

    def __init__(self, seed=None):
        super().__init__(seed)
        # adding optimized pulse shape to edges
        self.add_node(
            'q02', **{
                'pulse': {
                    'amp': jnp.array(0.19362652),
                    'omega_d': jnp.array(2.63342338),
                    'phase': jnp.array(-0.56882493),
                    't_plateau': jnp.array(69.9963988),
                    't_ramp': jnp.array(30.20130596),
                    'pulse_type': 'rampcos',
                    'operator_type': 'phi_operator',
                    'delay': 0.
                },
                'compensation': {
                    'pre_comp':
                        jnp.array([0.26824387, -0.00582382, 0.19798702]),
                    'post_comp':
                        jnp.array([0.01907886, -0.0714353, -0.27239727])
                }
            })

        self.add_node(
            'q03', **{
                'compensation': {
                    'pre_comp':
                        jnp.array([-1.42126234, 0.68387943, 0.09867095]),
                    'post_comp':
                        jnp.array([-0.77910168, 0.16258883, -0.00519836])
                }
            })

        self.add_node(
            'q12', **{
                'pulse': {
                    'amp': jnp.array(0.18108519),
                    'omega_d': jnp.array(4.18157275),
                    'phase': jnp.array(0.27647924),
                    't_plateau': jnp.array(70.00154013),
                    't_ramp': jnp.array(29.98031495),
                    'pulse_type': 'rampcos',
                    'operator_type': 'phi_operator',
                    'delay': 0.
                },
                'compensation': {
                    'pre_comp':
                        jnp.array([0.00466943, 0.78741568, -0.0862072]),
                    'post_comp':
                        jnp.array([-0.00220794, 0.38970001, -1.51744927])
                }
            })

        self.add_node(
            'q13', **{
                'compensation': {
                    'pre_comp':
                        jnp.array([-1.09492337, -1.34049224, -1.11393925]),
                    'post_comp':
                        jnp.array([-0.40020693, 0.7668882, -0.74054753])
                },
            })

        self.add_node(
            'q22', **{
                'pulse': {
                    'amp': jnp.array(0.21301889),
                    'omega_d': jnp.array(3.51378339),
                    'phase': jnp.array(1.08374892),
                    't_plateau': jnp.array(69.94150468),
                    't_ramp': jnp.array(30.06869994),
                    'pulse_type': 'rampcos',
                    'operator_type': 'phi_operator',
                    'delay': 0.
                },
                'compensation': {
                    'pre_comp':
                        jnp.array([-0.00160062, -1.50823985, 0.16763715]),
                    'post_comp':
                        jnp.array(
                            [-1.03353877e-03, -1.75575364e+00, -1.10831887e-02])
                }
            })

        self.add_node(
            'q23', **{
                'compensation': {
                    'pre_comp':
                        jnp.array([-0.01605929, -0.1141277, -0.69661291]),
                    'post_comp':
                        jnp.array([1.99985832, -0.92168355, -0.49487715])
                },
            })


class CNOTGatePeriodicGraph2(PeriodicGraph):
    """Add pulse parameters for simultaneous CNOT gates.
    In the example, we add 3 CR pulses on `q02`, `q12` and `q22`.
    Note that the target qubits are `q01`, `q11`, `q21` respectively.
    The target qubits are different compared to CNOTGatePeriodicGraph.

    Args:
        seed : (int) The seed for random device parameters variance.
            default seed is None, means no variance.
    """

    def __init__(self, seed=None):
        super().__init__(seed)
        # adding optimized pulse shape to edges
        self.add_node(
            'q01', **{
                'compensation': {
                    'pre_comp':
                        jnp.array([-0.18729245, -0.26026559, 2.79556595]),
                    'post_comp':
                        jnp.array([-0.78930838, 1.19644611, -1.06109615]),
                },
            })

        self.add_node(
            'q02', **{
                'pulse': {
                    'amp': jnp.array(0.18788192),
                    'omega_d': jnp.array(4.607684),
                    'phase': jnp.array(-1.79225586),
                    't_plateau': jnp.array(69.95600166),
                    't_ramp': jnp.array(30.20569655),
                    'pulse_type': 'rampcos',
                    'operator_type': 'phi_operator',
                    'delay': 0.
                },
                'compensation': {
                    'pre_comp':
                        jnp.array(
                            [-2.34791156e-04, 1.48276568e+00, 8.99221791e-01]),
                    'post_comp':
                        jnp.array([0.55455219, 0.00073962, 0.20388206]),
                }
            })

        self.add_node(
            'q11', **{
                'compensation': {
                    'pre_comp':
                        jnp.array([-0.09253156, 0.41565964, 0.26632383]),
                    'post_comp':
                        jnp.array([-0.75489047, 1.68428959, -1.71317095]),
                },
            })

        self.add_node(
            'q12', **{
                'pulse': {
                    'amp': jnp.array(0.10219415),
                    'omega_d': jnp.array(2.75130128),
                    'phase': jnp.array(-0.05679767),
                    't_plateau': jnp.array(69.25554591),
                    't_ramp': jnp.array(29.07256922),
                    'pulse_type': 'rampcos',
                    'operator_type': 'phi_operator',
                    'delay': 0.
                },
                'compensation': {
                    'pre_comp':
                        jnp.array([-0.0400265, -0.29959335, -3.0960532]),
                    'post_comp':
                        jnp.array([1.37150942, -0.0108739, -1.19826934]),
                }
            })

        self.add_node(
            'q21', **{
                'compensation': {
                    'pre_comp':
                        jnp.array([0.50715299, -0.8000848, -0.72776202]),
                    'post_comp':
                        jnp.array([2.32881958, -0.63164007, -0.46285065]),
                },
            })

        self.add_node(
            'q22', **{
                'pulse': {
                    'amp': jnp.array(0.13038021),
                    'omega_d': jnp.array(4.19534038),
                    'phase': jnp.array(0.08719563),
                    't_plateau': jnp.array(70.01800142),
                    't_ramp': jnp.array(30.16955306),
                    'pulse_type': 'rampcos',
                    'operator_type': 'phi_operator',
                    'delay': 0.
                },
                'compensation': {
                    'pre_comp':
                        jnp.array(
                            [-6.06081433e-04, -1.64773922e+00,
                             -2.04536211e+00]),
                    'post_comp':
                        jnp.array([3.13436934, 0.53835986, -0.79312629]),
                }
            })


class PeriodicGraphOpt(PeriodicGraph):
    """Device parameters for 5x5 periodic quantum processor graph after one-step
    optimization based on fidelities of simultaneous X gates and CNOT gates.

    Args:
        seed : (int) The seed for random device parameters variance.
            default seed is None, means no variance.
    """

    def __init__(self, seed=None):
        super().__init__(seed)
        # update device params by its gradient
        device_params = {
            'q02': {
                'ec': jnp.array(6.28318531),
                'ej': jnp.array(25.13274123),
                'el': jnp.array(6.28318531)
            },
            'q03': {
                'ec': jnp.array(6.28318531),
                'ej': jnp.array(25.13274123),
                'el': jnp.array(5.02654825)
            },
            'q12': {
                'ec': jnp.array(6.28318531),
                'ej': jnp.array(25.13274123),
                'el': jnp.array(5.65486678)
            },
            'q13': {
                'ec': jnp.array(6.28318531),
                'ej': jnp.array(25.13274123),
                'el': jnp.array(6.91150384)
            },
            'q22': {
                'ec': jnp.array(6.28318531),
                'ej': jnp.array(25.13274123),
                'el': jnp.array(7.53982237)
            }
        }
        device_grad = {
            'q02': {
                'ec': jnp.array(-7.00540723),
                'ej': jnp.array(2.24324625),
                'el': jnp.array(-6.1602346)
            },
            'q03': {
                'ec': jnp.array(2.81242553),
                'ej': jnp.array(-0.83988738),
                'el': jnp.array(2.5584191)
            },
            'q12': {
                'ec': jnp.array(5.98396226),
                'ej': jnp.array(-1.85285452),
                'el': jnp.array(5.35874343)
            },
            'q13': {
                'ec': jnp.array(-0.36019005),
                'ej': jnp.array(0.10861791),
                'el': jnp.array(-0.29491562)
            },
            'q22': {
                'ec': jnp.array(0.03386484),
                'ej': jnp.array(-0.02553209),
                'el': jnp.array(0.05359291)
            }
        }
        device_params = jax.tree_util.tree_map(lambda g, v: v - 0.01 * g,
                                               device_grad, device_params)
        device_params.update({
            'capacitive_coupling_all_unify': {
                'strength': jnp.array(0.07225663)
            },
            'inductive_coupling_all_unify': {
                'strength': jnp.array(-0.01256637)
            },
        })
        self.update_params(device_params, unify_coupling=True)


class XGatePeriodicGraphOpt(PeriodicGraphOpt):
    """Add pulse parameters for simultaneous X gates.
    In the example, we add 6 pulses on `q02`, `q03`, `q12`,
    `q13`, `q22` and `q23`.

    Args:
        seed : (int) The seed for random device parameters variance.
            default seed is None, means no variance.
    """

    def __init__(self, seed=None):
        super().__init__(seed)
        # adding optimized pulse shape to edges
        self.add_node(
            'q02', **{
                'pulse': {
                    'amp': jnp.array(0.07467672),
                    'length': jnp.array(39.82459069),
                    'omega_d': jnp.array(3.70771137),
                    'phase': jnp.array(0.7558433),
                    'pulse_type': 'cos',
                    'operator_type': 'phi_operator',
                    'delay': 0.
                },
                'compensation': {
                    'pre_comp': jnp.array(0.42591073),
                    'post_comp': jnp.array(-0.41941967)
                }
            })
        self.add_node(
            'q03', **{
                'pulse': {
                    'amp': jnp.array(0.06839676),
                    'length': jnp.array(39.81302627),
                    'omega_d': jnp.array(2.56897537),
                    'phase': jnp.array(0.85390569),
                    'pulse_type': 'cos',
                    'operator_type': 'phi_operator',
                    'delay': 0.
                },
                'compensation': {
                    'pre_comp': jnp.array(0.11757419),
                    'post_comp': jnp.array(-0.11531302)
                }
            })

        self.add_node(
            'q12', **{
                'pulse': {
                    'amp': jnp.array(0.07142831),
                    'length': jnp.array(39.85070846),
                    'omega_d': jnp.array(3.07586917),
                    'phase': jnp.array(1.58941357),
                    'pulse_type': 'cos',
                    'operator_type': 'phi_operator',
                    'delay': 0.
                },
                'compensation': {
                    'pre_comp': jnp.array(-0.74646573),
                    'post_comp': jnp.array(0.74593951)
                }
            })
        self.add_node(
            'q13', **{
                'pulse': {
                    'amp': jnp.array(0.07763859),
                    'length': jnp.array(39.85217838),
                    'omega_d': jnp.array(4.20299531),
                    'phase': jnp.array(1.43135504),
                    'pulse_type': 'cos',
                    'operator_type': 'phi_operator',
                    'delay': 0.
                },
                'compensation': {
                    'pre_comp': jnp.array(-0.58423741),
                    'post_comp': jnp.array(0.58729911)
                }
            })

        self.add_node(
            'q22', **{
                'pulse': {
                    'amp': jnp.array(0.08054434),
                    'length': jnp.array(39.80346128),
                    'omega_d': jnp.array(5.00262381),
                    'phase': jnp.array(1.39100452),
                    'pulse_type': 'cos',
                    'operator_type': 'phi_operator',
                    'delay': 0.
                },
                'compensation': {
                    'pre_comp': jnp.array(-0.48806854),
                    'post_comp': jnp.array(0.49022838)
                }
            })
        self.add_node(
            'q23', **{
                'pulse': {
                    'amp': jnp.array(0.07422721),
                    'length': jnp.array(39.85217838),
                    'omega_d': jnp.array(3.63596315),
                    'phase': jnp.array(0.26554983),
                    'pulse_type': 'cos',
                    'operator_type': 'phi_operator',
                    'delay': 0.
                },
                'compensation': {
                    'pre_comp': jnp.array(1.03746918),
                    'post_comp': jnp.array(-1.02909962)
                }
            })


class CNOTGatePeriodicGraphOpt(PeriodicGraphOpt):
    """Add pulse parameters for simultaneous CNOT gates.
    In the example, we add 3 CR pulses on `q02`, `q12`
    and `q22`.

    Args:
        seed : (int) The seed for random device parameters variance.
            default seed is None, means no variance.
    """

    def __init__(self, seed=None):
        super().__init__(seed)
        # adding optimized pulse shape to edges
        self.add_node(
            'q02', **{
                'pulse': {
                    'amp': jnp.array(0.18128846),
                    'omega_d': jnp.array(2.58934559),
                    'phase': jnp.array(-0.24290228),
                    't_plateau': jnp.array(69.93608145),
                    't_ramp': jnp.array(29.92806488),
                    'pulse_type': 'rampcos',
                    'operator_type': 'phi_operator',
                    'delay': 0.
                },
                'compensation': {
                    'pre_comp':
                        jnp.array([0.05671005, 0.02682845, 0.0395099]),
                    'post_comp':
                        jnp.array([-0.01528741, 0.04418319, -0.374539]),
                }
            })

        self.add_node(
            'q03', **{
                'compensation': {
                    'pre_comp':
                        jnp.array([-1.46070525, 0.66029355, 0.06603344]),
                    'post_comp':
                        jnp.array([-1.00629771, 0.16411027, -0.00797514]),
                },
            })

        self.add_node(
            'q12', **{
                'pulse': {
                    'amp': jnp.array(0.17872194),
                    'omega_d': jnp.array(4.19989714),
                    'phase': jnp.array(-0.01543561),
                    't_plateau': jnp.array(69.95327862),
                    't_ramp': jnp.array(29.94562879),
                    'pulse_type': 'rampcos',
                    'operator_type': 'phi_operator',
                    'delay': 0.
                },
                'compensation': {
                    'pre_comp':
                        jnp.array([0.00185588, 1.00099539, -0.22439037]),
                    'post_comp':
                        jnp.array([0.00200104, 0.58679953, -1.74804011]),
                }
            })

        self.add_node(
            'q13', **{
                'compensation': {
                    'pre_comp':
                        jnp.array([-1.02837338, -1.10918103, -0.80985523]),
                    'post_comp':
                        jnp.array([-0.11690826, 1.09973631, -1.0941268]),
                },
            })

        self.add_node(
            'q22', **{
                'pulse': {
                    'amp': jnp.array(0.23370657),
                    'omega_d': jnp.array(3.65018191),
                    'phase': jnp.array(-0.50006247),
                    't_plateau': jnp.array(69.96302405),
                    't_ramp': jnp.array(29.98769304),
                    'pulse_type': 'rampcos',
                    'operator_type': 'phi_operator',
                    'delay': 0.
                },
                'compensation': {
                    'pre_comp':
                        jnp.array(
                            [-1.90040736e-04, -1.34752499e+00, 3.15152817e-01]),
                    'post_comp':
                        jnp.array(
                            [-1.42308406e-03, -1.59853235e+00,
                             -1.88352743e-01]),
                }
            })

        self.add_node(
            'q23', **{
                'compensation': {
                    'pre_comp':
                        jnp.array([0.36516325, -0.38763126, -0.41575102]),
                    'post_comp':
                        jnp.array([1.65404972, -0.67251125, -0.04376101]),
                },
            })


"""
// JSON schema for nodes

{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "ec": {
      "type": "array",
      "items": {
        "description": "charging energy in unit GHz.
                We multiply 2pi to them so that we follow the same convention of
                setting hbar=1 as done in qutip's sesolve.
                Same for ej and el",
        "type": "number"
      }
    },
    "ej": {
      "type": "array",
      "items": {
        "description": "Josephson energy in unit GHz",
        "type": "number"
      }
    },
    "el": {
      "type": "array",
      "items": {
        "description": "inductive energy in unit GHz",
        "type": "number"
      }
    },
    "system_type": {
      "type": "string",
      "enum": [
        "fluxonium",
        "transmon",
        "resonator"
      ]
    },
    "shared_param_mark": {
      "type": "integer"
      "description": "We will set the parameters to be the same for
                    fluxoniums with same shared_param_mark.
                    This is intended to only effect gradient computation when share_params = True in Evolve,
                    (see hamiltonian/common_functions/compute_unitary_evolve.py)
                    but to be safe just make sure to set shared fluxoniums to the same value.""
    },
    "arguments": {
      "type": "object",
      "properties": {
        "phiext": {
          "description": "phiext is the value of phi_ext used in the fluxonium,
                            e.g. phiext = 0.5 * 2 * pi means that the fluxonium is at the sweetspot",
          "type": "number"
        },
        "phi_max": {
          "description": "phi basis range [-phi_max, phi_max), must be multiple of pi",
          "type": "number"
        },
        "truncated_dim": {
          "description": "How many eigenvectors we keep for each subsystem.
                            Currently we always keep the eigenvectors with lowest k energies",
          "type": "number"
        }
      }
    },
    "variance": {
      "type": "object",
      "description": "The variance of ec, ej, el parameters.
                        The values here are multiplied to the ec ej el if add_random = True in Evolve
                        see hamiltonian/common_functions/compute_unitary_evolve.py",
      "properties": {
        "ec": {
          "type": "array",
          "items": {
            "type": "number"
          }
        },
        "ej": {
          "type": "array",
          "items": {
            "type": "number"
          }
        },
        "el": {
          "type": "array",
          "items": {
            "type": "number"
          }
        }
      }
    },
    "pulse": {
      "type": "object",
      "description": "the parameters below depend on the pulse type",
      "properties": {
        "amp": {
          "type": "array",
          "items": {
            "description": "the amplitude of pulse",
            "type": "number"
          }
        },
        "length": {
          "type": "array",
          "items": {
            "description": "the length of pulse",
            "type": "number"
          }
        },
        "omega_d": {
          "type": "array",
          "items": {
            "description": "drive frequency of modulate wave",
            "type": "number"
          }
        },
        "phase": {
          "type": "array",
          "items": {
            "description": "phase of modulate wave",
            "type": "number"
          }
        },
        "pulse_type": {
          "type": "string",
          “description": "see hamiltonian/time_evolution/pulseshape.py"
          "enum": [
            "trapezoid",
            "cos",
            "rampcos",
            "tanh",
            "erf",
            "gaussian"
          ]
        },
        "operator_type": {
          "type": "string",
          "enum": [
            "n_operator",
            "phi_operator"
          ]
        },
        "delay": {
          "description": "time delay for waiting other gate operations",
          "type": "number"
        },
        "crosstalk": {
          "description": "the qubit has crosstalk",
          "type": "object",
          "additionalProperties": {
            "description": "the factor of crosstalk amplitude",
            "type": "number"
          }
        },
        "arguments": {
          "type": "object",
          "properties": {
            "modulate_wave": {
              "type": "boolean"
              "description": "if true, a modulate wave of frequency omega_d is
                                multiplied to the waveform"
            }
          }
        }
      }
    }
  }
}

// JSON schema for edges

{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "capacitive_coupling": {
      "type": "object",
      "properties": {
        "strength": {
          "description": "coefficient for the capacitive interaction strength",
          "type": "array",
          "items": {
            "type": "number"
          }
        }
      }
    },
    "inductive_coupling": {
      "type": "object",
      "properties": {
        "strength": {
          "description": "coefficient for the inductive interaction strength",
          "type": "array",
          "items": {
            "type": "number"
          }
        }
      }
    }
  }
}

"""
