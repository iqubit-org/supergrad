#!/usr/bin/env python3
"""
Multi-GPU Performance Profiling Script for SuperGrad

This script helps identify bottlenecks in multi-GPU execution by:
1. Profiling individual components
2. Measuring GPU utilization
3. Analyzing memory transfers
4. Identifying serialization points
"""

import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.experimental import mesh_utils
import psutil
import GPUtil
from functools import partial
import matplotlib.pyplot as plt

# Set environment for multi-GPU testing
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

from supergrad.helper import Evolve
from supergrad.utils.gates import hadamard_transform
from supergrad.scgraph.graph_mpc_fluxonium_1d import MPCFluxonium1D
from supergrad.utils.sharding import (
    get_sharding, distributed_fidelity_with_auto_vz_compensation)


class MultiGPUProfiler:
    """Profiler for analyzing multi-GPU performance bottlenecks."""
    
    def __init__(self):
        self.device_count = jax.local_device_count()
        print(f"Available devices: {self.device_count}")
        self.profiling_data = {}
        
    def get_gpu_utilization(self):
        """Get current GPU utilization across all GPUs."""
        try:
            gpus = GPUtil.getGPUs()
            return {f"GPU_{i}": gpu.load * 100 for i, gpu in enumerate(gpus)}
        except:
            return {"GPU_0": 0.0}  # Fallback for CPU-only testing
            
    def get_memory_usage(self):
        """Get current memory usage."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
        
    def time_function(self, func, *args, **kwargs):
        """Time a function execution with memory and GPU monitoring."""
        start_mem = self.get_memory_usage()
        
        start_time = time.time()
        result = func(*args, **kwargs)
        jax.block_until_ready(result)  # Ensure computation is complete
        end_time = time.time()
        
        end_mem = self.get_memory_usage()
        end_gpu = self.get_gpu_utilization()
        
        return {
            'execution_time': end_time - start_time,
            'memory_delta': end_mem - start_mem,
            'gpu_utilization': end_gpu,
            'result': result
        }
    
    def profile_component(self, name, func, *args, **kwargs):
        """Profile a specific component."""
        print(f"\n=== Profiling {name} ===")
        profile_data = self.time_function(func, *args, **kwargs)
        self.profiling_data[name] = profile_data
        
        print(f"Execution time: {profile_data['execution_time']:.4f}s")
        print(f"Memory delta: {profile_data['memory_delta']:.2f} MB")
        print(f"GPU utilization: {profile_data['gpu_utilization']}")
        
        return profile_data
    
    def analyze_scaling(self, func, *args, **kwargs):
        """Analyze scaling behavior across different device counts."""
        print("\n=== Scaling Analysis ===")
        
        scaling_data = {}
        for device_count in [1, 2, 4, 8]:
            if device_count <= self.device_count:
                print(f"\nTesting with {device_count} devices...")
                
                # Temporarily set device count
                os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={device_count}'
                
                # Reinitialize JAX
                jax.config.update('jax_platform_name', 'cpu')
                
                try:
                    profile_data = self.time_function(func, *args, **kwargs)
                    scaling_data[device_count] = profile_data
                    print(f"  Time: {profile_data['execution_time']:.4f}s")
                except Exception as e:
                    print(f"  Error: {e}")
                    scaling_data[device_count] = None
        
        return scaling_data
    
    def identify_bottlenecks(self):
        """Identify potential bottlenecks based on profiling data."""
        print("\n=== Bottleneck Analysis ===")
        
        bottlenecks = []
        
        # Check for poor scaling
        if 'scaling_analysis' in self.profiling_data:
            scaling = self.profiling_data['scaling_analysis']
            times = [data['execution_time'] for data in scaling.values() if data is not None]
            if len(times) >= 2:
                speedup = times[0] / times[-1]
                expected_speedup = len(times)
                efficiency = speedup / expected_speedup
                
                print(f"Speedup: {speedup:.2f}x (expected: {expected_speedup}x)")
                print(f"Efficiency: {efficiency:.2%}")
                
                if efficiency < 0.5:
                    bottlenecks.append("Poor multi-GPU scaling efficiency")
        
        # Check for memory issues
        for name, data in self.profiling_data.items():
            if data['memory_delta'] > 1000:  # > 1GB
                bottlenecks.append(f"High memory usage in {name}")
        
        # Check for low GPU utilization
        for name, data in self.profiling_data.items():
            avg_gpu_util = np.mean(list(data['gpu_utilization'].values()))
            if avg_gpu_util < 50:
                bottlenecks.append(f"Low GPU utilization in {name}: {avg_gpu_util:.1f}%")
        
        return bottlenecks


def create_test_workload():
    """Create a representative test workload."""
    n_qubit = 4
    chain = MPCFluxonium1D(n_qubit, periodic=False, seed=42)
    chain.create_single_qubit_pulse(range(n_qubit), [50.0] * n_qubit,
                                    True, factor=0.25, minimal_approach=True)
    
    target_unitary = hadamard_transform(n_qubit)
    target_unitary = jax.device_put(target_unitary, get_sharding(None, 'p'))
    
    evo = Evolve(chain,
                 truncated_dim=2,
                 compensation_option='no_comp',
                 solver='ode_expm',
                 options={
                     'astep': 5000,
                     'trotter_order': 2,
                     'diag_ops': True,
                     'progress_bar': True,
                     'custom_vjp': True,
                 })
    
    return evo, target_unitary


def main():
    """Main profiling function."""
    profiler = MultiGPUProfiler()
    
    # Create test workload
    evo, target_unitary = create_test_workload()
    
    # Profile individual components
    print("=== Component Profiling ===")
    
    # 1. Profile quantum system initialization
    profiler.profile_component(
        "quantum_system_init",
        lambda: evo._init_quantum_system()
    )
    
    # 2. Profile Hamiltonian construction
    profiler.profile_component(
        "hamiltonian_construction",
        lambda: evo.construct_hamiltonian_and_pulseshape()
    )
    
    # 3. Profile time evolution
    profiler.profile_component(
        "time_evolution",
        lambda: evo.product_basis(evo.all_params)
    )
    
    # 4. Profile fidelity computation
    output = evo.product_basis(evo.all_params)
    profiler.profile_component(
        "fidelity_computation",
        lambda: distributed_fidelity_with_auto_vz_compensation(target_unitary, output)
    )
    
    # 5. Profile gradient computation
    @jax.jit
    @partial(jax.value_and_grad, has_aux=True)
    def compute_fidelity(params):
        realized_unitary = evo.product_basis(params)
        fidelity, res_unitary = distributed_fidelity_with_auto_vz_compensation(
            target_unitary, realized_unitary)
        return 1 - fidelity, res_unitary
    
    profiler.profile_component(
        "gradient_computation",
        lambda: compute_fidelity(evo.all_params)
    )
    
    # Analyze scaling
    profiler.profiling_data['scaling_analysis'] = profiler.analyze_scaling(
        lambda: evo.product_basis(evo.all_params)
    )
    
    # Identify bottlenecks
    bottlenecks = profiler.identify_bottlenecks()
    
    print("\n=== Identified Bottlenecks ===")
    for i, bottleneck in enumerate(bottlenecks, 1):
        print(f"{i}. {bottleneck}")
    
    if not bottlenecks:
        print("No obvious bottlenecks identified. Consider deeper analysis.")
    
    return profiler.profiling_data


if __name__ == "__main__":
    profiling_data = main() 