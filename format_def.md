
# Format description

The core of the SuperGrad program is the description of the Hamiltonian. 
We have a large data structure to hold it, and it is treated as a "parameter tree" in JAX,
is processed to create the Hamiltonian for later calculations.

This description of the Hamiltonian can be represented in 3 forms in the SuperGrad.

The difference mostly comes from the method to "address" the parameters. 
To represent a tree data structure, we can have a dictionary with names as keys (*dic-with-key*), 
or a list with items including a key-value pair as "name:X" (*list-with-key*).
For consistency in addressing, the *dic-with-key* is used inside the program. 
However, this structure is nested in multiple levels and hard to read. 
So the input file is organized in the *list-with-key* style.

## Input format (SuperGradModel, SGM)
This is a json format that supports file I/O. This format is input-only. 
It is more human-readable and in *list-with-key* format.
It supports some one-time conversions to SGG format (e.g units).

## Internal format (SuperGradGraph, SGG)
This is the core data structure used in the SuperGrad. 
The quantum system can be described by a graph, with parameters on both nodes and edges.
Any subgraph of the graph data can also be used as a valid data set of a quantum system.
This graph (stored in the `SCGraph`)and its equivalent dictionary format (used in parametere transfer) are defined as SGG.
For easier to use, all items can be uniquely accessed by a series of keys, so the dictionary is in a *dic-with-key* format.

## Internal Json format (SuperGradGraphJson, SGGJ)
This is a json-compatible format to represent the data to modify/update the graph data. 
The notable difference between the internal format and internal-json format,
is that the edge keys are strings instead of source/target tuples that is not supported in JSON.

For conversion between the formats, see `utils/format_conv.py` and `utils/format_sgm.py`.

# Compare with Haiku
Haiku wraps an object or class into a pure function and a dictionary of parameter set.

Advantage:
* The code is guaranteed to be pure.
* In the wrapping, any functions can register a module name to directly fetch parameters from the input dictionary.
* No parameter passing necessary from one function to another, which generally nested lots of levels.

Disadvantage
* Functions implicitly accept an input argument which is not listed in python. This is not intuitive.
* The input dictionary definition has nothing to do with the code structure, all belongs to the scatterd functions.
* One must write an elaborated function to read the input dictionary if there are many parameters.
* It is hard to run a function and reuse its results to do something later, as it must be pure.
* The Haiku pure function parameter and other parameters require explicitly treatment, which is troublesome when some parameters may or may not from Haiku modules.

On the other side, the graph-style parameters:

Advantage:
* The parameter update / use are fully on the graph-style parameters which is easy to understand.
* No need to treat arguments for "pure Haiku function" explicitly.
* All functions can be reused easily.
* Functions do not accept an input dictionary which is not listed in python. 

Disadvantage:
* The "pureness" is not guaranteed, user must pay attention to it
* Functions cannot fetch the "global" parameters from the Haiku input dictionary. They must be passed one by one.


# Dataflow in the SuperGrad

Here is a pseudo-code to show how to use SuperGrad framework:

```python
# Initialize
sgg0 = ...

# Pure function for JAX auto-diff
def pure_function(dict_input):
    sgg1 = sgg0.copy()
    sgg1.update_parameters(dict_input)
    ham = sgg1.convert_graph_to_quantum_system()
    # Do something to Hamiltonian
    r = f(ham) 
    return r

``` 

## Input

The input *SGM* data is converted to *SGG* format and used in the SCGraph.
One can also directly define the data in *SGG* dict format in Python, or implement a class to inherit `SCGraph` and create graph data.

## SCGraph

This class is a wrapper of a SGG data that supports modification (to implement functions)
and generate Hamiltonian accordingly. It supports:

1. Initialized by a SGG data, which includes the graph structure and all default parameters. 
The default parameters will never be reused again by users explicitly.
2. Set shared parameters, unify coupling by `share_params` and `unify_coupling`.
3. Add random deviation to devices by `add_lcj_params_variance_to_graph()`.
4. When all prepared, use `convert_graph_to_parameters()` to check the dictionary of all parameters that are differentiable.
5. Accept partial-SGG data to update the graph (the input of a function for JAX, by `update_parameters(input)` after `copy()`).
    This partial-SSG should be a subset of the dictionary from `convert_graph_to_parameters()`.
5. Create Hamiltonian based on the current data, by `convert_graph_to_quantum_system()` to `InteractingSystem`.
6. Export control parameters, e.g. pulse and compensation parameters with the Hamiltonian by `convert_graph_to_pulse_lst()`.

Note, the original `SCGraph` itself should NOT change to avoid side effect in a pure function.
The most easy way is copy the original one, update it and create the Hamiltonian.
This ensures that a one can implement a pure function from this interface.

`SCGraph` is inherited from `NetworkX.Graph`, which is simply a graph with a graph-wide state (how parameters are shared).
So its `subgraph()` is not a valid `SCGraph`, and should not be used to construct the Hamiltonian / update parameters. 
The subgraph in most cases are used to modify the graph.
To create a sub system to construct the Hamiltonian, use `subscgraph()`.


It also provides some functions to modify parameters more easily.

1. `set_all_node_attr`: This can modify parameters of all nodes together.
2. `set_compensation`: This can set compensation (see section below) of all relevant nodes. 
   One can select to change all nodes already with "compensation" parameter or all "data" device-category nodes.

## Evolve(Helper)

This class is also a wrapper of a SGG data, which requires the Hamiltonian and do more later.
It can run evolutions (final states and trajectories) in different basis (product or eigen).

`Helper` class:
1. Must implement a `init_quantum_system` function which creates the Hamiltonian from the graph and compute necessary properties.
2. In lots of cases `init_quantum_system` is time-consuming, so we put this function a standalone function that the user should call it manually instead of old `_init_quantum_system`.
3. A sanity check is applied on other functions in `Helper` so they must run after the `init_quantum_system`.
4. One can use a decorator to define if `init_quantum_system` is required. 
   1. Without any decorator, the function should call `init_quantum_system` itself. Otherwise, an error is raised.
   2. With `Helper.decorator_bypass_init`,  the function can do anything freely.
   3. With `Helper.decorator_auto_init`, the function calls `init_quantum_system` automatically, and behave similar to a Haiku module function.


# Parameter definition

The ``
To let the program easier to understand, we classify the parameters into several different categories.

Both nodes and edges can hold parameters.

## Category

### Node: Non-physical parameters

* `include_param`: control which part in `common` in SGM format is included in this node
* `shared_param_mark`: control whether two nodes/edges share the parameters. See next section.

### Node/Edge: Device physical parameters

* `system_type`: indicate the quantum system type of this node
* `ec`, `ej`, `el`, ...: Quantum device parameters (differentiable). 
* `inductive_coupling`, `capacitive_coupling`, ...: Coupling parameters (differentiable).

Note all parameters that does not belong to other groups are included in this group.

### Node: Control physical parameters

* `pulse`: a dict of the pulse parameters.
  * `pulse_type`: indicate the pulse type
  * `operator_type`: indicate the operator to add pulse
  * `amp_type`: Some pulses are applied on a variable that is not linear on an operator (e.g. tunable transmon Phi_EJ).
      We can use `amp_type` to select how to convert to linear coefficients.
  * `delay`: the delay of the pulse from initial time. This is NOT differentiable now.
  * `amp`, `length`, `omega_d`, `phase`, ... : Pulse parameters (differentiable).
* `compensation`: a dict of the compensation parameters (differentiable).
  * `compensation` is special because we sometimes want to modify its structure, due to 3 possible selections
    * No (0 parameter)
    * Virtual-Z (1 parameter)
    * Arbitrary single qubit gate (3 parameters)
  * Generally we want to apply one method to several devices. So we provide an interface to modify the `SCGraph` compensation data all together. 
 This is ``
  * Note we can have different compensation methods for different devices. This is selected by the `SCGraph`.
  * There is a `device_category` 

### Node: Deviation of parameters

* `deviation` a dictionary of device parameter deviations. This is not shared and not differentiable.
  * In old version, `deviation` is saved, but only used if `add_random` is set to True when calling `convert_graph_to_quantum_system`
  * In this version, the data fully controlled the process, so if `deviation` presents then the randomness is always used.

### Other tags
  * `arguments`: this dictionary can be included in any dictionary to host static parameters that never change after initialization.

## Share parameter / unify coupling

`SCGraph` supports parameter sharing between qubits, marked by the `share_param_mark` tag. 

* All nodes with same `share_param_mark` will share the same parameters.
* All edges with same source/target (undirected) will share the same parameters.

When this is activated, `SCGraph` will only treat the first one in the sorted nodes / edges with given `share_param_mark` as parameters.
Parameters of other nodes / edges in SGG will be ignored. 
The parameters in first node / edge will be copied to other nodes / edges.
Note only device / coupling parameters will be copied, others are not copied, including `pulse`, `compensation`, `deviation`.
Especially, `deviation` is not copied, so the deviated device parameters can be different even the device parameters are shared.

The keyword `unify_coupling` is a special case that all edges copies the first edge.

To know which node or edge is the one we can set when considering the sharing, 
one can check use `convert_graph_to_parameters()` to see what is left in the partial-SGG format dictionary.

### Pitfall

The parameter sharing must be computed before the `SCGraph` is used in the pure function for JAX. 
So any change to the graph structure, especially the `subgraph`, requires recalculation of the sharing status to be used later in JAX.
So calling `SCGraph.subgraph()` creates an object that cannot be directly used. 
Call `SCGraph.subscgraph()` instead to create a subgraph to be used later, see introduction to `SCGraph` in previous section.
