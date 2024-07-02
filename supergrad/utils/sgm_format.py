from typing import Union, Tuple
import networkx as nx
import jsonschema
import jax.numpy as jnp
from supergrad.scgraph import SCGraph
import json
from pathlib import Path

def is_sgm_valid(data: dict) -> Tuple[bool, str]:
    """Checks if the json data is a valid SGM format.

    Args:
        data: the json data

    Returns:
        if the data follows SGM format, the error string
    """
    schema = json.loads((Path(__file__).parent / "sgm_schema.json").read_text())
    try:
        jsonschema.validate(data, schema=schema)
    except jsonschema.exceptions.ValidationError as e:
        return False, str(e)
    # Further check values

    return True, ""


def convert_children_to_data_array(dic: dict, energy_scale: float) -> dict:
    """Converts the children that are numbers/lists to JAX float array and remove graph properties.

    Graph properties "id", "source", "target" are all removed.

    Except some specific keys, all int/float values are converted into Jax float array.

    Note this function is not recursive.

    Args:
        dic: a dictionary to convert
        energy_scale: multiply the input value by this to get the energy in the unit in the later calculation.

    Returns:
        converted dictionary
    """
    list_skip_key = ["id", "source", "target", "name"]
    list_fix_key = ["shared_param_mark", "delay"]
    list_key_scale = ["strength", "ec", "ej", "el"]
    dic2 = {}
    for key, val in dic.items():
        if key in list_skip_key:
            continue
        if (isinstance(val, float) or isinstance(val, int)) and (key not in list_fix_key):
            if key in list_key_scale:
                val = jnp.array(val * energy_scale)
            else:
                val = jnp.array(val * 1.0)
        if isinstance(val, list):
            if len(val) == 0:
                raise ValueError(f"SGM not support empty lists: {key}")
            if isinstance(val[0], float) or isinstance(val[0], int):
                val = jnp.asarray([x * 1.0 for x in val])
        dic2[key] = val
    return dic2


def convert_sgm_to_networkx(data: dict) -> nx.Graph:
    """Converts SGM format json to NetworkX.

    Returns:
        NetworkX graph
    """
    g = data["global"]
    energy_unit = g["energy_unit"]
    frequency_unit = g["frequency_unit"]

    if frequency_unit != "GHz":
        raise ValueError("Only support frequency_unit: GHz")

    if energy_unit == "h":
        energy_scale = 2 * jnp.pi
    elif energy_unit == "hbar":
        energy_scale = 1.0
    else:
        raise ValueError(f"Unsupported energy_unit: {energy_unit}")

    b_valid, error = is_sgm_valid(data)
    if not b_valid:
        print(error)
        raise ValueError("Input data is not a valid SGM")
    temp_graph = nx.Graph()

    # Put global attr to graph itself
    for key, val in g.items():
        if key not in ["energy_unit", "frequency_unit"]:
            temp_graph.graph[key] = val

    # Keys that used in graph, not data
    list_top = ["pulse", "compensation"]

    dic_common = dict([(x["name"], x["arguments"]) for x in data["common"]])
    for node1 in data["nodes"]:
        id1 = node1["id"]
        arg = node1.get("arguments")
        if isinstance(arg, str):
            arguments = dic_common.get(arg)
            if arguments is None:
                raise ValueError(f"Cannot find common arguments with name : {arg}")
            node1["arguments"] = arguments
        node1 = convert_children_to_data_array(node1, energy_scale)
        # Check other top level data
        for top in list_top:
            if top not in data:
                continue
            for node2 in data[top]:
                if node2["id"] == id1:
                    # For pulse, add to the dictionary "pulse" with name
                    # For compensation, direct add
                    if top == "compensation":
                        node1[top] = convert_children_to_data_array(node2, energy_scale)
                    elif top == "pulse":
                        if top not in node1:
                            node1[top] = {}
                        node1[top][node2["name"]] = convert_children_to_data_array(node2, energy_scale)
                    else:
                        raise ValueError(f"Unknown top level name {top}")

        temp_graph.add_node(id1, **node1)

    for link1 in data["edges"]:
        source = link1.pop("source")
        target = link1.pop("target")
        link1 = {key: convert_children_to_data_array(val, energy_scale) for key, val in link1.items()}
        temp_graph.add_edge(source, target, **link1)

    return temp_graph


def read_sgm_data(s: Union[str, Path]) -> SCGraph:
    """Reads SGM file to SCGraph.

    Args:
        s: the path to the data file

    Returns:
        SCGraph with data from file
    """
    data = convert_sgm_to_networkx(json.loads(Path(s).read_text()))
    return SCGraph(incoming_graph_data=data)