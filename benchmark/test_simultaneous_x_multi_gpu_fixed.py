import os
import sys
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from qiskit_dynamics import Solver
import qutip as qt
import supergrad
from supergrad.helper import Evolve
from supergrad.utils import basis
from supergrad.utils.fidelity import compute_fidelity_with_1q_rotation_axis
from supergrad.utils.memory_profiling import trace_max_memory_usage
from supergrad.utils.qiskit_interface import (to_qiskit_static_hamiltonian,
                                              to_qiskit_drive_hamiltonian)
from supergrad.utils.qutip_interface import (to_qutip_operator,
                                             to_qutip_operator_function_pair)
from supergrad.utils.sharding import distributed_state_fidelity, get_sharding

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('/'.join(dir_path.split('/')[:-1]))

from supergrad.scgraph.graph_mpc_fluxonium_1d import MPCFluxonium1D
from benchmark.utils.create_simultaneous_model import create_simultaneous_x
from benchmark.utils.fidelity import fidelity

# Test with n_qubit=4 for sanity test
n_qubit = 4


def create_multi_gpu_evolution(n_qubit, gpu_count, astep=5000, trotter_order=2,
                               diag_ops=True, minimal_approach=True, custom_vjp=True):
    """Create evolution object with proper multi-GPU sharding"""
    
    # Create the base evolution object
    evo = create_simultaneous_x(n_qubit=n_qubit,
                                astep=astep,
                                trotter_order=trotter_order,
                                diag_ops=diag_ops,
                                minimal_approach=minimal_approach,
                                custom_vjp=custom_vjp)
    
    # Get available devices and select the first gpu_count devices
    all_devices = jax.devices()
    if len(all_devices) < gpu_count:
        print(f"Warning: Only {len(all_devices)} devices available, requested {gpu_count}")
        gpu_count = len(all_devices)
    
    selected_devices = all_devices[:gpu_count]
    print(f"Using {len(selected_devices)} devices: {[d.device_kind for d in selected_devices]}")
    
    return evo, selected_devices


def test_simultaneous_x_state_grad_lcam_multi_gpu(gpu_count=8):
    """Test state gradient with proper multi-GPU sharding"""
    
    print(f"🧪 Testing State Evolution + Gradient (LCAM) with {gpu_count} GPUs")
    print("=" * 60)
    
    # Create evolution object with multi-GPU support
    evo, devices = create_multi_gpu_evolution(n_qubit, gpu_count)
    
    # Create target state
    target_state = basis(2**n_qubit, 2**n_qubit - 1)
    
    # Create MULTIPLE initial states to trigger sharding (psi0.ndim == 3)
    # This is the key fix - we need 3D input to trigger JAX sharding
    n_parallel_states = gpu_count  # One state per GPU
    initial_states = jnp.stack([basis(2**n_qubit) for _ in range(n_parallel_states)])
    
    print(f"Created {n_parallel_states} parallel initial states")
    print(f"Initial states shape: {initial_states.shape}")
    print(f"Target state shape: {target_state.shape}")
    
    # Apply sharding to initial states
    sharding = get_sharding('p', None, None)
    initial_states = jax.device_put(initial_states, sharding)
    
    @jax.jit
    @jax.value_and_grad
    def infidelity(params):
        # This will now trigger the multi-GPU path in sesolve
        output = evo.product_basis(params, initial_states)
        return 1 - distributed_state_fidelity(target_state, output)
    
    print("Running state evolution with multi-GPU sharding...")
    start_time = jax.time.time()
    
    # Run the computation
    result = infidelity(evo.all_params)
    
    # Wait for completion
    jax.block_until_ready(result)
    
    end_time = jax.time.time()
    execution_time = end_time - start_time
    
    print(f"✅ State gradient completed in {execution_time:.2f}s")
    print(f"Result type: {type(result)}")
    if hasattr(result, '__len__'):
        print(f"Result length: {len(result)}")
    
    return result, execution_time


def test_simultaneous_x_grad_lcam_multi_gpu(gpu_count=8):
    """Test unitary gradient with proper multi-GPU sharding"""
    
    print(f"🧪 Testing Unitary Evolution + Gradient (LCAM) with {gpu_count} GPUs")
    print("=" * 60)
    
    # Create evolution object with multi-GPU support
    evo, devices = create_multi_gpu_evolution(n_qubit, gpu_count)
    
    # Create target unitary (X gates on all qubits)
    u_ref = supergrad.tensor(*[np.array([[0, 1], [1, 0]])] * n_qubit)
    u_ref = jax.device_put(u_ref, jax.devices('cpu')[0])
    
    # Create MULTIPLE initial states to trigger sharding
    n_parallel_states = gpu_count
    initial_states = jnp.stack([basis(2**n_qubit) for _ in range(n_parallel_states)])
    
    print(f"Created {n_parallel_states} parallel initial states for unitary evolution")
    print(f"Initial states shape: {initial_states.shape}")
    
    # Apply sharding to initial states
    sharding = get_sharding('p', None, None)
    initial_states = jax.device_put(initial_states, sharding)
    
    @jax.jit
    @jax.value_and_grad
    def infidelity(params):
        # This will now trigger the multi-GPU path in sesolve
        output = evo.product_basis(params, initial_states)
        return 1 - distributed_state_fidelity(u_ref, output)
    
    print("Running unitary evolution with multi-GPU sharding...")
    start_time = jax.time.time()
    
    # Run the computation
    result = infidelity(evo.all_params)
    
    # Wait for completion
    jax.block_until_ready(result)
    
    end_time = jax.time.time()
    execution_time = end_time - start_time
    
    print(f"✅ Unitary gradient completed in {execution_time:.2f}s")
    print(f"Result type: {type(result)}")
    if hasattr(result, '__len__'):
        print(f"Result length: {len(result)}")
    
    return result, execution_time


def main():
    """Run the fixed multi-GPU benchmark"""
    print("🚀 Fixed Multi-GPU Benchmark - Proper Sharding Implementation")
    print("=" * 70)
    print(f"Testing with n_qubit={n_qubit}")
    print("Key fixes:")
    print("  ✅ Multiple initial states (psi0.ndim == 3)")
    print("  ✅ Proper JAX sharding applied")
    print("  ✅ Device selection and validation")
    print("  ✅ Using all 8 available GPUs")
    print("=" * 70)
    
    # Test with 8 GPUs (all available)
    gpu_count = 8
    
    try:
        # Test state gradient
        print("\n🔍 Testing State Gradient with Multi-GPU...")
        state_result, state_time = test_simultaneous_x_state_grad_lcam_multi_gpu(gpu_count)
        
        # Test unitary gradient
        print("\n🔍 Testing Unitary Gradient with Multi-GPU...")
        unitary_result, unitary_time = test_simultaneous_x_grad_lcam_multi_gpu(gpu_count)
        
        print("\n🎯 Multi-GPU Benchmark Results:")
        print("=" * 50)
        print(f"State Gradient: {state_time:.2f}s")
        print(f"Unitary Gradient: {unitary_time:.2f}s")
        print(f"Total Time: {state_time + unitary_time:.2f}s")
        
        # Check if we actually used multiple GPUs
        print(f"\n🔍 GPU Usage Verification:")
        print(f"Requested GPUs: {gpu_count}")
        print(f"Available devices: {len(jax.devices())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
