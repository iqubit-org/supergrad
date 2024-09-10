import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.experimental import mesh_utils


def sharding_put(data, partition_spec):
    """Put data to multi-devices with sharding.

    Args:
        data: `jax.numpy.ndarray`
            The data to be distributed batched.
        partition_spec: `jax.sharding.PartitionSpec`
            The partition specification. The sharding is done along the "p"
            axis_name.
    """
    devices = mesh_utils.create_device_mesh((jax.local_device_count(),))
    mesh = Mesh(devices, 'p')
    sharding = NamedSharding(mesh, partition_spec)
    return jax.device_put(data, sharding)


def distributed_state_fidelity(target_states, computed_states):
    """Compute the state fidelity or the composite state (unitary) fidelity.
    The fidelity could be computed in a multi-devices cluster with the same sharding.

    Args:
        target_states: The target states.
        computed_states: The computed states.
    """
    # assert target_states.sharding == computed_states.sharding
    ar_state_fidelity = jnp.sum(jnp.conj(target_states) * computed_states,
                                axis=0)
    return jnp.abs(ar_state_fidelity.mean())**2
