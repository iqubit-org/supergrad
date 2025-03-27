from typing import Callable, Any
import json
import jax
import pprint
import jax.numpy as jnp


def deep_update_dict(dic0: dict, dic1: dict, append=False):
    """Merges all values from a nested dictionary to another.

    This modification is in-place.

    Args:
        dic0: the original dict, to put new data into it
        dic1: the new data
        append: whether to allow a new key to be added to the dictionary.

    Raises:
        ValueError: any key not exists or type mismatch in the original dictionary

    Returns:
        the original dictionary updated
    """
    for key, val1 in dic1.items():
        if key not in dic0:
            if not append:
                raise ValueError(f"{key} not found in original data")
            else:
                dic0[key] = val1
                continue
        val0 = dic0[key]
        is_dict0 = isinstance(val0, dict)
        is_dict1 = isinstance(val1, dict)
        if is_dict0 != is_dict1:
            raise ValueError(f"{key} type mismatch in original/new data")
        if is_dict0 and is_dict1:
            deep_update_dict(val0, val1, append)
        else:
            dic0[key] = val1
    return dic0


def deep_partial_dict(dic0: dict, f_key: Callable[[str], bool] = None, f_val: Callable[[Any], bool] = None):
    """Get partial of the dict with items follows the predicate.

    Note we can provide predicates of both/either of key and values.

    If a key points to a nested dictionary and the key does not follow te predicate,
    then the nested one is ignored, too.

    Args:
        dic0: the original dictionary
        f_key: determine whether the item should be included based on the key
        f_val: determine whether the item should be included based on the value

    Raises:
        ValueError: None of the predicates is provided.

    Returns:
        a new dictionary with only items that follows the predicate.
    """
    if f_key is None and f_val is None:
        raise ValueError("None of the predicates is provided")
    f_key = (lambda x: True) if f_key is None else f_key
    f_val = (lambda x: True) if f_val is None else f_val
    dic1 = {}
    for key, val0 in dic0.items():
        if not f_key(key):
            continue
        if isinstance(val0, dict):
            val1 = deep_partial_dict(val0, f_key, f_val)
            if len(val1) > 0:
                dic1[key] = val1
        else:
            if f_val(val0):
                dic1[key] = val0
    return dic1


def convert_device_array(x):
    """Convert a device array to a list or a number.
    This is for convert parameters to json compatible format.
    """

    if len(x.shape) == 0:
        return float(x)
    else:
        return x.tolist()


def convert_nested_dic_keys(dic: dict, f_key: Callable[[Any], Any]) -> dict:
    """Converts all key in a nested dictionary as a deep copy.

    Args:
        dic: a nested dictionary
        f_key: the function to convert the key

    Returns:
        a dictionary with key converted, items deep copied.
    """
    keys = list(dic)
    dic2 = {}
    for key in keys:
        # Recursive check
        val2 = dic[key]
        if isinstance(dic[key], dict):
            val2 = convert_nested_dic_keys(val2, f_key)
        # Check the key and convert
        key2 = f_key(key)
        dic2[key2] = val2
    return dic2


def convert_edge_key_tuple_to_str(dic: dict):
    """Converts all edge key as tuples to a single string in a nested dictionary.

    Args:
        dic: the dictionary with possible string or Tuple[string,string] as key

    Returns:
        a dictionary with only string keys, items deep copied.
    """
    return convert_nested_dic_keys(dic, lambda x: (f"{x[0]}:{x[1]}" if isinstance(x, tuple) else x))


def convert_edge_key_str_to_tuple(dic: dict):
    """Converts all edge key as tuples to a single string in a nested dictionary.

    Args:
        dic: the dictionary with possible string or Tuple[string,string] as key

    Returns:
        a dictionary with only string keys, items deep copied.
    """
    return convert_nested_dic_keys(dic, lambda x: (tuple(x.split(":")) if ":" in x else x))


def convert_to_json_compatible(x):
    """
    Convert input to json compatible format.
    In particular, it will replace DeviceArray or np.array in the input.
    Edge names are replaced by a single string connected.
    """

    return jax.tree_util.tree_map(convert_device_array, convert_edge_key_tuple_to_str(x))


def convert_to_device_array_dict(x):
    """
    Convert input to haiku compatible dictionary.
    In particular, it will replace input float to the DeviceArray.
    """

    def _parse_params(x):
        if isinstance(x, float):
            return jnp.array(x)
        else:
            return x

    return jax.tree_util.tree_map(_parse_params, convert_edge_key_str_to_tuple(x))


def dump_params(params, fp):
    """Dump parameters to json."""
    json.dump(convert_to_json_compatible(params), fp)


def load_params(fp):
    """Load parameters from json."""
    return convert_to_device_array_dict(json.load(fp))


def tree_print(t):
    """Print jax pytree in a human readable way."""

    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(convert_to_json_compatible(t))
