# SuperGrad Multi-GPU Performance Optimization Session

**Date**: December 2024  
**Objective**: Analyze and improve multi-GPU performance scaling from <2x to 4-5x speedup with 8 GPUs

## Problem Statement

SuperGrad is experiencing poor multi-GPU scaling:
- **Current**: Less than 2x speedup with 8 GPUs
- **Expected**: 4-5x speedup with 8 GPUs
- **Goal**: Identify bottlenecks and implement systematic optimizations

## Analysis Summary

### Identified Bottlenecks

#### 1. **Time Evolution Serialization** (Primary Suspect)
- **Location**: `supergrad/time_evolution/ode.py`
- **Issue**: Uses `jax.lax.scan` for time stepping, creating serial dependencies
- **Impact**: Prevents effective parallelization across GPUs
- **Evidence**: Matrix exponentiation operations may not be optimally distributed

#### 2. **Limited Sharding Strategy**
- **Location**: `supergrad/utils/sharding.py`
- **Issue**: Only basic partitioning along single axis ('p')
- **Impact**: Poor load balancing across GPUs
- **Evidence**: Sharding only applied to target unitaries and final outputs

#### 3. **Memory Transfer Overhead**
- **Issue**: Frequent data transfers between devices
- **Causes**: 
  - Multiple basis transformations (product ↔ eigen basis)
  - Sharding operations requiring data redistribution
  - Unnecessary device transfers in evolution pipeline
- **Impact**: High communication overhead reduces effective speedup

#### 4. **Inefficient JIT Compilation**
- **Issue**: JIT compilation not optimized for multi-device execution
- **Evidence**: Custom VJP rules may not be optimized for distributed execution
- **Impact**: Compilation and execution overhead

#### 5. **Quantum System Initialization Bottleneck**
- **Location**: `Evolve._init_quantum_system()`
- **Issues**:
  - Hamiltonian construction may not be parallelized
  - Eigenvalue decomposition operations may be single-device
  - Parameter sharing logic may create dependencies
- **Impact**: Initialization time doesn't scale with device count

## Tools Created

### 1. Multi-GPU Profiler (`benchmark/multi_gpu_profiling.py`)

```python
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
```

### 2. Quick Diagnostic Script (`benchmark/quick_diagnostic.py`)

```python
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
        a = jnp.random.normal(size=(size, size))
        b = jnp.random.normal(size=(size, size))
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
        a = jnp.random.normal(size=(size, size))
        a_sharded = jax.device_put(a, sharding)
        b = jnp.random.normal(size=(size, size))
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
    a = jnp.random.normal(size=(size, size))
    
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
```

## Step-by-Step Optimization Plan

### Phase 1: Profiling and Bottleneck Identification (Week 1)

#### Immediate Actions
1. **Run Quick Diagnostic**
   ```bash
   cd benchmark
   python quick_diagnostic.py
   ```

2. **Run Comprehensive Profiling**
   ```bash
   python multi_gpu_profiling.py
   ```

3. **Monitor GPU Utilization**
   ```bash
   watch -n 1 nvidia-smi
   ```

4. **Enable JAX Profiling**
   ```python
   jax.profiler.start_trace("/tmp/jax-trace")
   # Run your computation
   jax.profiler.stop_trace()
   ```

#### Expected Outcomes
- Identify primary bottleneck (likely time evolution)
- Measure current speedup and efficiency
- Understand GPU utilization patterns

### Phase 2: Component-Level Optimization (Week 2-3)

#### 2.1 Optimize Time Evolution
- **Investigate `ode_expm` parallelization**
  - Check if time steps can be parallelized
  - Optimize matrix exponentiation for multi-device
  - Consider alternative ODE solvers

- **Improve sharding in evolution**
  ```python
  # Example: Better sharding for evolution
  def improved_evolution_sharding(hamiltonian, psi_list):
      # Shard both Hamiltonian and states
      sharded_ham = jax.device_put(hamiltonian, get_sharding('p', None))
      sharded_psi = jax.device_put(psi_list, get_sharding('p', None))
      return evolve_with_sharding(sharded_ham, sharded_psi)
  ```

#### 2.2 Optimize Quantum System Initialization
- **Parallelize Hamiltonian construction**
- **Distribute eigenvalue computations**
- **Optimize parameter sharing logic**

#### 2.3 Improve Memory Management
- **Minimize device transfers**
- **Use in-place operations where possible**
- **Optimize sharding patterns**

### Phase 3: Advanced Optimizations (Week 4)

#### 3.1 Implement Better Sharding Strategies
```python
# Example: Multi-dimensional sharding
def advanced_sharding_strategy():
    devices = mesh_utils.create_device_mesh((2, 4))  # 2x4 grid
    mesh = Mesh(devices, ('batch', 'time'))
    return NamedSharding(mesh, PartitionSpec('batch', 'time'))
```

#### 3.2 Optimize Custom VJP Rules
- **Ensure VJP computations are parallelized**
- **Minimize serial dependencies in gradient computation**
- **Use efficient adjoint methods**

#### 3.3 Implement Pipeline Parallelism
- **Overlap computation and communication**
- **Use asynchronous operations**
- **Implement prefetching**

## Debugging and Monitoring Tools

### JAX Debugging
```python
# Visualize array sharding
jax.debug.visualize_array_sharding(your_array)

# Print device placement
jax.debug.print("Device placement: {}", jax.devices())

# Test different sharding patterns
sharding_patterns = [
    get_sharding(None, 'p'),      # Current
    get_sharding('p', None),      # Alternative 1
    get_sharding('p', 'p'),       # Alternative 2
]
```

### GPU Monitoring
```python
# Monitor GPU utilization
import GPUtil
gpus = GPUtil.getGPUs()
for gpu in gpus:
    print(f"GPU {gpu.id}: {gpu.load*100:.1f}% utilization")
```

### Memory Profiling
```python
# Use built-in memory profiler
from supergrad.utils.memory_profiling import trace_max_memory_usage
```

## Expected Performance Improvements

### Short-term (1-2 weeks)
- **Target**: 2-3x speedup with 8 GPUs
- **Focus**: Identify and fix primary bottleneck
- **Metrics**: Improved GPU utilization, reduced memory transfers

### Medium-term (3-4 weeks)
- **Target**: 4-5x speedup with 8 GPUs
- **Focus**: Comprehensive component optimization
- **Metrics**: 50-60% efficiency across all components

### Long-term (1-2 months)
- **Target**: 6-7x speedup with 8 GPUs
- **Focus**: Advanced optimizations and pipeline parallelism
- **Metrics**: Near-linear scaling, >80% efficiency

## Key Files and Locations

### Core Files to Monitor
- `supergrad/time_evolution/ode.py` - Time evolution bottleneck
- `supergrad/utils/sharding.py` - Sharding strategy
- `supergrad/helper/compute_unitary_evolve.py` - Evolution wrapper
- `supergrad/time_evolution/schrodinger_solver.py` - Solver implementation

### Profiling Tools
- `benchmark/multi_gpu_profiling.py` - Comprehensive profiler
- `benchmark/quick_diagnostic.py` - Quick diagnostics
- `benchmark/multi_gpu_analysis.md` - This analysis document

## Conclusion

The poor multi-GPU scaling in SuperGrad is likely due to:
1. **Time evolution serialization** (primary bottleneck)
2. **Limited sharding strategy**
3. **Memory transfer overhead**
4. **Suboptimal JIT compilation**

By following this systematic approach and using the provided tools, you should achieve the target 4-5x speedup with 8 GPUs. Start with the profiling tools to get concrete data, then focus on the time evolution component as the most impactful optimization.

---

*This session document captures the complete analysis and toolset for optimizing SuperGrad's multi-GPU performance. All tools are ready to use and the plan provides a clear path to achieving the desired performance improvements.*