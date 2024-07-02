


# Format description

The "parameter tree" can be represented in 3 forms in the SuperGrad.

The difference mostly comes from the method to "address" the parameters. 
To represent a tree data structure, we can have a dictionary with names as keys (*dic-with-key*), 
or a list with items including a key-value pair as "name:X" (*list-with-key*).
For consistency in addressing, the *dic-with-key* is used inside the program. 
However, this structure is nested in multiple levels and hard to read. 
So the input file is organized in the *list-with-key* style.

## Input formata (SuperGradModel, SGM)
This is a json format that supports file I/O. This format is input-only. 
It is more human-readable and in *list-with-key* format, and supports some one-time conversion (e.g units).

## Internal format (SuperGradGraph, SGG)
This is the core data structure used in the SuperGrad. 
The quantum system can be described by a graph, with parameters on both nodes and edges.
Any subgraph of the graph data can also be used as a valid data set of a quantum system.
For easier to use, all items can be uniquely accessed by a series of keys. 
This graph can be export-import as a dictionary in a *dic-with-key* format.

## Internal Json format (SuperGradGraphJson, SGGJ)
This is a json-compatible format to represent the data to modify/update the graph data. 
The notable difference between the internal format and internal-json format,
is that the edge keys are strings instead of source/target tuples that is not supported in JSON.


# Compare with Haiku
Haiku wraps an object or class into a pure function and a dictionary of parameter set.

Advantage:
* The code is guaranteed to be pure.
* In the wrapping, any functions can register a module name to directly fetch parameter from the input dictionary.

Disadvantage
* Functions must accept an input dictionary which is not listed in python. This is not intuitive.
* The input dictionary definition has nothing to do with the code structure, all belongs to the scatterd functions.
* One must write an elaborated function to read the input dictionary if there are many parameters.
* It is hard to run a function and reuse its results to do something later, as it must be pure.
* The Haiku pure function parameter and other parameters require explicitly treatment.

On the other side, the graph-style parameters:

Advantage:
* The parameter update / use are fully on the graph-style parameters which is easy to understand.
* No need to treat arguments for "pure Haiku function" explicitly.
* All functions can be reused easily.
* Functions do not accept an input dictionary which is not listed in python. 

Disadvantage:
* The "pureness" is not guaranteed, user must pay attention to it
* Function cannot fetch the "global" parameter from the Haiku input dictionary.


# Dataflow in the SuperGrad

## Input

The input *SGM* data is converted to *SGG* format and used in the SCGraph.

## SCGraph

This class is a wrapper of a SGG data that supports modification (to implement functions)
and generate Hamiltonian accordingly.

1. Initialized by a SGG data, which includes the graph structure and all default parameters. 
The default parameters will never be reused again by users explicitly.
2. Accept partial-SGG data to update the graph (the entry point of JAX function)
3. Create Hamiltonian based on the current data (`InteractingSystem`)
4. Deal with shared parameters, unify coupling and other parameter level.

Note, everything except those the input *partial-SSG* data in `SCGraph` will NOT change.
This ensures that a one can implement a pure function from this interface.

## Evolve(Helper)

This class is another wrapper of a SGG data, which requires the Hamiltonian and do more later.

1. Run evolutions (final states and trajectories) in different basis (product or eigen).

`Helper` class:
1. Must implement a `init_quantum_system` function which creates the Hamiltonian from the graph and compute necessary properties.
2. In lots of cases `init_quantum_system` is time-consuming, so we put this function a standalone function that the user should call it manually instead of old `_init_quantum_system`.
3. A sanity check is applied on other functions in `Helper` so they must run after the `init_quantum_system`.
4. One can use a decorator to remove the check, in that case the function should call `init_quantum_system` itself, and behave as a Haiku module function.

# Parameter definition

To let the program easier to understand, we classify the parameters into several different categories.

Both nodes and edges can hold parameters.

## Category

### Device physical parameters

* `ec`, `ej`, `el`, ...: Quantum device parameters. 
* `inductive_coupling`, `capacitive_coupling`, ...: Coupling parameters.

Note all parameters that does not belong to other groups are included in this group.

### Control physical parameters

* `pulse`: a dict of the pulse parameters.
* `compensation`: a dict of the compensation parameters.
  * `compensation` is special because we sometimes want to modify its structure, due to 3 possible selections
    * No (0 parameter)
    * Virtual-Z (1 parameter)
    * Arbitrary single qubit gate (3 parameter)
  * Generally we want to apply one method to several devices. So we provide an interface to modify the `SCGraph` compensation data all together. 
  * Note we can have different compensation method for different devices. This is selected by the `SCGraph`
  

### Deviation of parameters

* `deviation` a dictionary of device parameter deviations. This is not shared and not differentiable.

### Not a number 
  * `system_type`: indicate the quantum system type
  * `shared_param_mark`: control whether two nodes share the parameters
  * `pulse_type`: indicate the pulse type
  * `operator_type`: indicate the operator to add pulse

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

The keyworkd `unify_coupling` is a special case that all edges copies the first edge.

