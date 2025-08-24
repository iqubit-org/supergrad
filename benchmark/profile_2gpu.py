#!/usr/bin/env python3
"""
2-GPU Profiler Script
Sets CUDA_VISIBLE_DEVICES=0,1 before JAX import for true isolation
"""

import os
import sys
import time
import json
from datetime import datetime

# Set GPU isolation BEFORE any JAX imports
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

print("🔒 GPU Isolation Setup:")
print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print("   Expected GPUs: 2 (GPU_0, GPU_1)")

# Now import JAX and verify isolation
import jax
import jax.numpy as jnp

print("\n📱 JAX Device Verification:")
print(f"   JAX devices found: {len(jax.devices())}")
print(f"   Device types: {[d.device_kind for d in jax.devices()]}")

# Verify isolation worked
if len(jax.devices()) != 2:
    print(f"   ❌ GPU isolation failed - expected 2 GPUs, got {len(jax.devices())}")
    sys.exit(1)
else:
    print("   ✅ GPU isolation successful: 2 devices visible")

# Import GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
    print("   ✅ NVML initialized for GPU monitoring")
except ImportError:
    NVML_AVAILABLE = False
    print("   ⚠️  NVML not available - GPU memory monitoring disabled")

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
print(f"\n🧪 Starting 2-GPU Profiling with n_qubit={n_qubit}")
print("=" * 60)

# Initialize results
results = {
    'gpu_count': 2,
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

# Step 2: Unitary Gradient (LCAM)
print("\n🧪 Step 2: Profiling Unitary Evolution + Gradient (LCAM)...")
try:
    start_time = time.time()
    benchmark = create_mock_benchmark('unitary', n_qubit)
    
    result = test_simultaneous_x_grad_lcam(
        benchmark=benchmark,
        n_qubit=n_qubit
    )
    
    # Wait for completion
    jax.block_until_ready(result)
    
    execution_time = time.time() - start_time
    print(f"✅ Unitary gradient (LCAM) completed: {execution_time:.2f}s")
    
    results['unitary_grad_lcam'] = {
        'execution_time': execution_time,
        'success': True
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

# Step 3: Unitary Gradient (TAD)
print("\n🧪 Step 3: Profiling Unitary Evolution + Gradient (TAD)...")
try:
    start_time = time.time()
    benchmark = create_mock_benchmark('unitary', n_qubit)
    
    result = test_simultaneous_x_grad_tad(
        benchmark=benchmark,
        n_qubit=n_qubit
    )
    
    # Wait for completion
    jax.block_until_ready(result)
    
    execution_time = time.time() - start_time
    print(f"✅ Unitary gradient (TAD) completed: {execution_time:.2f}s")
    
    results['unitary_grad_tad'] = {
        'execution_time': execution_time,
        'success': True
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

# Final GPU memory state
print("\n Final GPU Memory State:")
final_gpu_memory = get_gpu_memory_usage()
for gpu, mem in final_gpu_memory.items():
    print(f"   {gpu}: {mem['used_gb']:.1f}GB used / {mem['total_gb']:.1f}GB total")

results['final_gpu_memory'] = final_gpu_memory

# Save results
output_file = 'profile_2gpu_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to: {output_file}")
print("🎯 2-GPU profiling completed!")
print("=" * 60)
