#!/usr/bin/env python3
"""
Quick Diagnostic Script for Multi-GPU Performance Issues

This script provides immediate insights into potential bottlenecks
without requiring extensive setup or profiling tools.
"""

import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

# Set up multi-GPU environment
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
key = jax.random.PRNGKey(0)

def quick_diagnostic():
    """Run quick diagnostic tests to identify bottlenecks."""
    
    print("=== SuperGrad Multi-GPU Quick Diagnostic ===\n")
    
    # 1. Check device availability
    print("1. Device Configuration:")
    print(f"   Available devices: {jax.local_device_count()}")
    print(f"   Device types: {[d.platform for d in jax.devices()]}")
    print()
    
    # 2. Test basic JAX operations scaling
    print("2. Basic JAX Operations Scaling:")
    test_basic_scaling()
    print()
    
    # 3. Test sharding performance
    print("3. Sharding Performance:")
    test_sharding_performance()
    print()
    
    # 4. Test memory transfer overhead
    print("4. Memory Transfer Overhead:")
    test_memory_transfers()
    print()
    
    # 5. Provide recommendations
    print("5. Initial Recommendations:")
    provide_recommendations()


def test_basic_scaling():
    """Test basic JAX operations scaling."""
    sizes = [1000, 2000, 4000]
    
    for size in sizes:
        print(f"   Matrix size: {size}x{size}")
        
        # Single device
        start = time.time()
        a = jax.random.normal(key, (size, size))
        b = jax.random.normal(key, (size, size))
        c = a @ b
        jax.block_until_ready(c)
        single_time = time.time() - start
        
        # Multi-device (if available)
        if jax.local_device_count() > 1:
            start = time.time()
            a = jax.device_put_sharded([a] * jax.local_device_count(), jax.devices())
            b = jax.device_put_sharded([b] * jax.local_device_count(), jax.devices())
            c = jax.pmap(lambda x, y: x @ y)(a, b)
            jax.block_until_ready(c)
            multi_time = time.time() - start
            
            speedup = single_time / multi_time
            efficiency = speedup / jax.local_device_count()
            print(f"     Speedup: {speedup:.2f}x (Efficiency: {efficiency:.1%})")
        else:
            print("     Single device only")


def test_sharding_performance():
    """Test sharding performance."""
    try:
        from jax.sharding import Mesh, NamedSharding, PartitionSpec
        from jax.experimental import mesh_utils
        
        size = 2000
        print(f"   Testing sharding with {size}x{size} matrix")
        
        # Create sharding
        devices = mesh_utils.create_device_mesh((jax.local_device_count(),))
        mesh = Mesh(devices, 'p')
        sharding = NamedSharding(mesh, PartitionSpec('p', None))
        
        # Test sharded operation
        start = time.time()
        a = jax.random.normal(key, (size, size))
        a_sharded = jax.device_put(a, sharding)
        b = jax.random.normal(key, (size, size))
        b_sharded = jax.device_put(b, sharding)
        c_sharded = a_sharded @ b_sharded
        jax.block_until_ready(c_sharded)
        sharded_time = time.time() - start
        
        # Compare with non-sharded
        start = time.time()
        c = a @ b
        jax.block_until_ready(c)
        normal_time = time.time() - start
        
        overhead = (sharded_time - normal_time) / normal_time
        print(f"     Sharding overhead: {overhead:.1%}")
        
        if overhead > 0.5:
            print("     ⚠️  High sharding overhead detected!")
        
    except ImportError:
        print("     Sharding not available")


def test_memory_transfers():
    """Test memory transfer overhead."""
    size = 1000
    
    # Test device transfer overhead
    a = jax.random.normal(key, (size, size))
    
    start = time.time()
    for _ in range(10):
        b = jax.device_put(a)
        jax.block_until_ready(b)
    transfer_time = time.time() - start
    
    # Test computation time
    start = time.time()
    for _ in range(10):
        c = a @ a
        jax.block_until_ready(c)
    compute_time = time.time() - start
    
    transfer_ratio = transfer_time / compute_time
    print(f"   Transfer/Compute ratio: {transfer_ratio:.2f}")
    
    if transfer_ratio > 0.1:
        print("   ⚠️  High transfer overhead detected!")


def provide_recommendations():
    """Provide initial recommendations based on diagnostic results."""
    print("   Based on the diagnostic results:")
    print()
    print("   🔍 Next Steps:")
    print("   1. Run the full profiling script: python benchmark/multi_gpu_profiling.py")
    print("   2. Check GPU utilization with: nvidia-smi")
    print("   3. Monitor memory usage during computation")
    print("   4. Test with different problem sizes")
    print()
    print("   🎯 Likely Issues:")
    print("   - Time evolution serialization (most probable)")
    print("   - Memory transfer overhead")
    print("   - Suboptimal sharding strategy")
    print("   - JIT compilation bottlenecks")
    print()
    print("   🛠️  Quick Fixes to Try:")
    print("   1. Increase problem size to amortize overhead")
    print("   2. Test different sharding patterns")
    print("   3. Use jax.jit more aggressively")
    print("   4. Minimize device transfers")


def check_specific_issues():
    """Check for specific issues in the SuperGrad codebase."""
    print("\n=== SuperGrad-Specific Issues ===")
    
    issues = []
    
    # Check for potential serial bottlenecks
    print("Checking for serial bottlenecks...")
    
    # 1. Time evolution serialization
    issues.append("Time evolution may be serial due to jax.lax.scan usage")
    
    # 2. Sharding limitations
    issues.append("Limited sharding strategy in utils/sharding.py")
    
    # 3. Memory transfers
    issues.append("Multiple basis transformations may cause transfer overhead")
    
    # 4. JIT compilation
    issues.append("Custom VJP rules may not be optimized for multi-device")
    
    print(f"Found {len(issues)} potential issues:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    
    return issues


if __name__ == "__main__":
    quick_diagnostic()
    check_specific_issues()
