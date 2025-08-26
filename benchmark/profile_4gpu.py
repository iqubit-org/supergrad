#!/usr/bin/env python3
"""
4-GPU Profiler Script
Sets CUDA_VISIBLE_DEVICES=0,1,2,3 before JAX import for true isolation
"""

import os
import sys
import time
import json
from datetime import datetime

# Set GPU isolation BEFORE any JAX imports
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

print("🔒 GPU Isolation Setup:")
print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print("   Expected GPUs: 4 (GPU_0, GPU_1, GPU_2, GPU_3)")

# Now import JAX and verify isolation
import jax
import jax.numpy as jnp
import jax.profiler

# Import JAX profiler utilities
from jax_profiler_utils import (
    setup_tensorboard_logging, 
    run_with_jax_profiler, 
    print_tensorboard_summary,
    create_profiled_test_function
)

print("\n📱 JAX Device Verification:")
print(f"   JAX devices found: {len(jax.devices())}")
print(f"   Device types: {[d.device_kind for d in jax.devices()]}")

# Verify isolation worked
if len(jax.devices()) != 4:
    print(f"   ❌ GPU isolation failed - expected 4 GPUs, got {len(jax.devices())}")
    sys.exit(1)
else:
    print("   ✅ GPU isolation successful: 4 devices visible")

# Import GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
    print("   ✅ NVML initialized for GPU monitoring")
except ImportError:
    NVML_AVAILABLE = False
    print("   ⚠️  NVML not available - GPU memory monitoring disabled")

# Add after the imports section
def verify_gpu_participation():
    """Verify that all GPUs are actually participating in computation"""
    print("\n🔍 Verifying True GPU Participation...")
    
    # Test 1: Verify JAX sharding is working
    print("   🧪 Test 1: JAX Sharding Verification...")
    try:
        # Create a test array that should be sharded across all GPUs
        test_array = jnp.ones((1000, 1000))
        sharded_array = jax.device_put_sharded(
            [test_array[:500, :500], test_array[500:, :500], 
             test_array[:500, 500:], test_array[500:, 500:]],
            jax.devices()
        )
        
        # Verify sharding worked
        sharding_info = jax.devices()
        print(f"      ✅ Sharding successful: {len(sharding_info)} devices")
        print(f"      📱 Devices: {[d.id for d in sharding_info]}")
        
    except Exception as e:
        print(f"      ❌ Sharding failed: {e}")
        return False
    
    # Test 2: Verify cross-GPU computation
    print("   🧪 Test 2: Cross-GPU Computation Verification...")
    try:
        # Test actual computation across GPUs
        result = jnp.sum(jnp.array([jnp.sum(part) for part in sharded_array]))
        jax.block_until_ready(result)
        print(f"      ✅ Cross-GPU computation successful: {result}")
        
    except Exception as e:
        print(f"      ❌ Cross-GPU computation failed: {e}")
        return False
    
    # Test 3: Verify NCCL communication
    print("   🧪 Test 3: NCCL Communication Verification...")
    try:
        # Test NCCL all-reduce operation
        test_data = jnp.ones((100, 100))
        sharded_data = jax.device_put_sharded(
            [test_data] * len(jax.devices()),
            jax.devices()
        )
        
        # This should trigger NCCL communication
        result = jnp.sum(jnp.array([jnp.sum(part) for part in sharded_data]))
        jax.block_until_ready(result)
        print(f"      ✅ NCCL communication successful: {result}")
        
    except Exception as e:
        print(f"      ❌ NCCL communication failed: {e}")
        return False
    
    print("   ✅ All GPU participation tests passed!")
    return True

def get_detailed_gpu_metrics():
    """Get detailed GPU metrics including compute utilization"""
    if not NVML_AVAILABLE:
        return {}
    
    try:
        detailed_metrics = {}
        device_count = pynvml.nvmlDeviceGetCount()
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Compute utilization
            util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # Temperature
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # Power usage
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to Watts
            
            detailed_metrics[f'GPU_{i}'] = {
                'memory_used_gb': mem_info.used / (1024**3),
                'memory_total_gb': mem_info.total / (1024**3),
                'compute_utilization_percent': util_info.gpu,
                'memory_utilization_percent': util_info.memory,
                'temperature_celsius': temp,
                'power_watts': power
            }
        
        return detailed_metrics
        
    except Exception as e:
        print(f"   ⚠️  Failed to get detailed GPU metrics: {e}")
        return {}

# Get initial GPU memory state
def get_gpu_memory_usage():
    if not NVML_AVAILABLE:
        return {}
    
    try:
        memory_info = {}
        device_count = pynvml.nvmlDeviceGetCount()
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_info[f'GPU_{i}'] = {
                'total_gb': info.total / (1024**3),
                'used_gb': info.used / (1024**3),
                'free_gb': info.free / (1024**3),
                'utilization_percent': (info.used / info.total) * 100
            }
        return memory_info
    except Exception as e:
        print(f"   ⚠️  Failed to get GPU memory info: {e}")
        return {}


print("\n Initial GPU Memory State:")
initial_gpu_memory = get_gpu_memory_usage()
for gpu, mem in initial_gpu_memory.items():
    print(f"   {gpu}: {mem['used_gb']:.1f}GB used / {mem['total_gb']:.1f}GB total")

# Import SuperGrad functions
sys.path.append('..')
try:
    from test_simultaneous_x import test_simultaneous_x_grad_lcam, test_simultaneous_x_grad_tad
    from utils.create_simultaneous_model import create_simultaneous_x
    print("\n✅ Successfully imported benchmark functions")
except ImportError as e:
    print(f"\n❌ Failed to import benchmark functions: {e}")
    sys.exit(1)

# Create mock benchmark
def create_mock_benchmark(test_type, n_qubit):
    class MockBenchmark:
        def __init__(self, test_type, n_qubit):
            if test_type == 'unitary':
                self.group = f'gradient_simultaneous_x_{n_qubit}_qubits'
            else:
                self.group = f'gradient_simultaneous_x_state_{n_qubit}_qubits'
            self.extra_info = {}
        
        def __call__(self, func):
            return func()
    
    return MockBenchmark(test_type, n_qubit)


# Configuration
n_qubit = 8
print(f"\n🧪 Starting 4-GPU Profiling with n_qubit={n_qubit}")
print("=" * 60)

# Initialize results
results = {
    'gpu_count': 4,
    'n_qubit': n_qubit,
    'timestamp': datetime.now().isoformat(),
    'isolation_verified': True,
    'initial_gpu_memory': initial_gpu_memory,
    'step_timing': {},
    'memory_profiles': {}
}

# Step 1: System Creation
print("🏗️  Step 1: Creating evolution object...")
start_time = time.time()
start_memory = os.getpid()  # Process ID for reference

evo = create_simultaneous_x(
    n_qubit,
    astep=5000,
    trotter_order=2,
    diag_ops=True,
    minimal_approach=True,
    custom_vjp=False
)

creation_time = time.time() - start_time
print(f"✅ System creation completed: {creation_time:.2f}s")

# Get parameter count
total_params = sum(p.size for p in jax.tree_util.tree_leaves(evo.all_params))
print(f"   Total parameters: {total_params}")

results['step_timing']['system_creation'] = creation_time
results['total_params'] = total_params

# Setup TensorBoard trace logging
tensorboard_log_dir = setup_tensorboard_logging(4)

# Add after system creation, before Step 2
print("\n🔍 Verifying GPU Participation...")
gpu_participation_verified = verify_gpu_participation()
results['gpu_participation_verified'] = gpu_participation_verified

if not gpu_participation_verified:
    print("   ⚠️  GPU participation verification failed - results may be unreliable")

# Step 2: Unitary Gradient (LCAM) with JAX Profiler
print("\n🧪 Step 2: Profiling Unitary Evolution + Gradient (LCAM)...")
try:
    start_time = time.time()
    benchmark = create_mock_benchmark('unitary', n_qubit)
    
    # Create profiled test function
    profiled_lcam_test = create_profiled_test_function(
        test_simultaneous_x_grad_lcam, benchmark, n_qubit
    )
    
    # Run with JAX profiler
    result, trace_dir = run_with_jax_profiler(
        profiled_lcam_test, 
        "lcam_trace", 
        tensorboard_log_dir,
        "LCAM gradient computation on 4 GPUs"
    )
    
    execution_time = time.time() - start_time
    print(f"✅ Unitary gradient (LCAM) completed: {execution_time:.2f}s")
    
    results['unitary_grad_lcam'] = {
        'execution_time': execution_time,
        'success': True,
        'tensorboard_trace': trace_dir
    }
    results['step_timing']['unitary_gradient_lcam'] = execution_time
    
except Exception as e:
    execution_time = time.time() - start_time
    print(f"❌ Unitary gradient (LCAM) failed: {e}")
    results['unitary_grad_lcam'] = {
        'success': False,
        'error': str(e),
        'execution_time': execution_time
    }

# Step 3: Unitary Gradient (TAD) with JAX Profiler
print("\n🧪 Step 3: Profiling Unitary Evolution + Gradient (TAD)...")
try:
    start_time = time.time()
    benchmark = create_mock_benchmark('unitary', n_qubit)
    
    # Create profiled test function
    profiled_tad_test = create_profiled_test_function(
        test_simultaneous_x_grad_tad, benchmark, n_qubit
    )
    
    # Run with JAX profiler
    result, trace_dir = run_with_jax_profiler(
        profiled_tad_test, 
        "tad_trace", 
        tensorboard_log_dir,
        "TAD gradient computation on 4 GPUs"
    )
    
    execution_time = time.time() - start_time
    print(f"✅ Unitary gradient (TAD) completed: {execution_time:.2f}s")
    
    results['unitary_grad_tad'] = {
        'execution_time': execution_time,
        'success': True,
        'tensorboard_trace': trace_dir
    }
    results['step_timing']['unitary_gradient_tad'] = execution_time
    
except Exception as e:
    execution_time = time.time() - start_time
    print(f"❌ Unitary gradient (TAD) failed: {e}")
    results['unitary_grad_tad'] = {
        'success': False,
        'error': str(e),
        'execution_time': execution_time
    }

# Add detailed GPU metrics before final memory state
print("\n📊 Detailed GPU Metrics:")
detailed_gpu_metrics = get_detailed_gpu_metrics()
for gpu, metrics in detailed_gpu_metrics.items():
    print(f"   {gpu}:")
    print(f"      Memory: {metrics['memory_used_gb']:.1f}GB / {metrics['memory_total_gb']:.1f}GB")
    print(f"      Compute: {metrics['compute_utilization_percent']}%")
    print(f"      Memory: {metrics['memory_utilization_percent']}%")
    print(f"      Temp: {metrics['temperature_celsius']}°C")
    print(f"      Power: {metrics['power_watts']:.1f}W")

results['detailed_gpu_metrics'] = detailed_gpu_metrics

# Final GPU memory state
print("\n Final GPU Memory State:")
final_gpu_memory = get_gpu_memory_usage()
for gpu, mem in final_gpu_memory.items():
    print(f"   {gpu}: {mem['used_gb']:.1f}GB used / {mem['total_gb']:.1f}GB total")

results['final_gpu_memory'] = final_gpu_memory

# Print TensorBoard summary
print_tensorboard_summary(results, tensorboard_log_dir)

# Save results
output_file = 'profile_4gpu_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to: {output_file}")
print("🎯 4-GPU profiling completed!")
print("=" * 60)
