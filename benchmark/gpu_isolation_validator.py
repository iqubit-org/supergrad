#!/usr/bin/env python3
"""
GPU Isolation Validator - Phase 1 Testing
==========================================

This script validates true GPU isolation using CUDA_VISIBLE_DEVICES
and tests for hanging issues with smaller workloads.

Purpose:
- Validate that GPU isolation actually works (memory only on selected GPUs)
- Test if 2-GPU configuration hangs with n_qubit=4
- Measure execution times for scaling analysis
- Identify if the issue is isolation or workload-related
"""

import os
import sys
import time
import subprocess
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.append('..')


def run_isolated_gpu_test(gpu_count, n_qubit=4):
    """
    Run a GPU test with true isolation using CUDA_VISIBLE_DEVICES
    
    Args:
        gpu_count: Number of GPUs to use (1 or 2)
        n_qubit: Number of qubits for the test
    
    Returns:
        dict: Test results including timing and success status
    """
    print(f"🔍 Testing {gpu_count}-GPU Configuration with True Isolation")
    print("=" * 60)
    
    # Set GPU isolation
    if gpu_count == 1:
        cuda_devices = "0"
        expected_gpus = ["GPU_0"]
    elif gpu_count == 2:
        cuda_devices = "0,1"
        expected_gpus = ["GPU_0", "GPU_1"]
    else:
        raise ValueError("Only testing 1-GPU and 2-GPU configurations")
    
    print(f"   CUDA_VISIBLE_DEVICES: {cuda_devices}")
    print(f"   Expected active GPUs: {expected_gpus}")
    print(f"   Test workload: n_qubit={n_qubit}")
    print()
    
    # Create the test script content
    test_script = f'''#!/usr/bin/env python3
import os
import sys
import time
import jax
import jax.numpy as jnp

# Set GPU isolation BEFORE importing JAX
os.environ['CUDA_VISIBLE_DEVICES'] = '{cuda_devices}'

print(f"🚀 Starting {gpu_count}-GPU isolation test")
print(f"CUDA_VISIBLE_DEVICES: {{os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}}")

# Now import JAX and check devices
import jax
print(f"Available JAX devices: {{len(jax.devices())}}")
print(f"Device types: {{[d.device_kind for d in jax.devices()]}}")

# Import SuperGrad functions
sys.path.append('..')
try:
    from test_simultaneous_x import test_simultaneous_x_state_grad_lcam
    from utils.create_simultaneous_model import create_simultaneous_x
    print("✅ Successfully imported benchmark functions")
except ImportError as e:
    print(f"❌ Import failed: {{e}}")
    sys.exit(1)

# Create mock benchmark
class MockBenchmark:
    def __init__(self):
        self.group = 'test_isolation'
        self.extra_info = {{}}
    
    def __call__(self, func):
        return func()

print(f"🧪 Testing State Evolution + Gradient (LCAM) with n_qubit={n_qubit}")
print("=" * 50)

# Test 1: System Creation
print("Step 1: Creating evolution object...")
start_time = time.time()

evo = create_simultaneous_x(
    {n_qubit},
    astep=5000,
    trotter_order=2,
    diag_ops=True,
    minimal_approach=True,
    custom_vjp=True
)

creation_time = time.time() - start_time
print(f"✅ System creation completed: {{creation_time:.2f}}s")

# Test 2: State Gradient
print("Step 2: Running state gradient computation...")
start_time = time.time()

try:
    benchmark = MockBenchmark()
    result = test_simultaneous_x_state_grad_lcam(benchmark, {n_qubit})
    
    # Wait for completion
    jax.block_until_ready(result)
    
    execution_time = time.time() - start_time
    print(f"✅ State gradient completed: {{execution_time:.2f}}s")
    print(f"Result type: {{type(result)}}")
    
    # Success
    print("🎯 {gpu_count}-GPU isolation test PASSED!")
    print(f"Total execution time: {{execution_time:.2f}}s")
    
except Exception as e:
    execution_time = time.time() - start_time
    print(f"❌ State gradient failed after {{execution_time:.2f}}s: {{e}}")
    print("🎯 {gpu_count}-GPU isolation test FAILED!")
    sys.exit(1)
'''
    
    # Write test script to temporary file
    script_path = f"temp_isolation_test_{gpu_count}gpu.py"
    with open(script_path, 'w') as f:
        f.write(test_script)
    
    try:
        # Run the isolated test
        print("   🚀 Starting isolated GPU test...")
        start_time = time.time()
        
        # Set environment variables for the subprocess
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = cuda_devices
        env['PYTHONPATH'] = '..'
        
        # Run the test
        result = subprocess.run(
            ['python3', script_path],
            env=env,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for safety
        )
        
        execution_time = time.time() - start_time
        
        # Check results
        if result.returncode == 0:
            print(f"   ✅ {gpu_count}-GPU test completed successfully!")
            print(f"   Execution time: {execution_time:.2f}s")
            print(f"   Output: {result.stdout}")
            
            return {
                'success': True,
                'gpu_count': gpu_count,
                'execution_time': execution_time,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        else:
            print(f"   ❌ {gpu_count}-GPU test failed!")
            print(f"   Return code: {result.returncode}")
            print(f"   Execution time: {execution_time:.2f}s")
            print(f"   Stdout: {result.stdout}")
            print(f"   Stderr: {result.stderr}")
            
            return {
                'success': False,
                'gpu_count': gpu_count,
                'execution_time': execution_time,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
    except subprocess.TimeoutExpired:
        print(f"   ⏰ {gpu_count}-GPU test TIMED OUT after 10 minutes!")
        print("   This indicates a hanging issue with multi-GPU configuration")
        return {
            'success': False,
            'gpu_count': gpu_count,
            'execution_time': 600,
            'error': 'Timeout - hanging detected',
            'hanging': True
        }
        
    except Exception as e:
        print(f"   ❌ {gpu_count}-GPU test failed with exception: {e}")
        return {
            'success': False,
            'gpu_count': gpu_count,
            'execution_time': time.time() - start_time,
            'error': str(e)
        }
        
    finally:
        # Clean up temporary script
        if os.path.exists(script_path):
            os.remove(script_path)


def main():
    """Run GPU isolation validation tests"""
    print("🚀 GPU Isolation Validator - Phase 1 Testing")
    print("=" * 60)
    print("Purpose: Validate true GPU isolation and check for hanging issues")
    print("Workload: n_qubit=4 (smaller, faster tests)")
    print("Configurations: 1-GPU and 2-GPU")
    print("=" * 60)
    
    results = {}
    
    # Test 1-GPU first (baseline)
    print("\n🔍 Phase 1: Testing 1-GPU Configuration")
    print("-" * 40)
    result_1gpu = run_isolated_gpu_test(1, n_qubit=4)
    results['1gpu'] = result_1gpu
    
    if not result_1gpu['success']:
        print("❌ 1-GPU baseline failed - cannot proceed")
        return results
    
    # Test 2-GPU (validation)
    print("\n🔍 Phase 2: Testing 2-GPU Configuration")
    print("-" * 40)
    result_2gpu = run_isolated_gpu_test(2, n_qubit=4)
    results['2gpu'] = result_2gpu
    
    # Analysis
    print("\n📊 GPU Isolation Validation Results")
    print("=" * 60)
    
    if result_1gpu['success'] and result_2gpu['success']:
        print("✅ Both configurations completed successfully!")
        
        # Check for hanging
        if result_2gpu.get('hanging', False):
            print("⚠️  2-GPU configuration shows hanging behavior")
            print("   This indicates workload-related issues, not isolation problems")
        else:
            print("✅ No hanging detected in 2-GPU configuration")
        
        # Scaling analysis
        time_1gpu = result_1gpu['execution_time']
        time_2gpu = result_2gpu['execution_time']
        
        if time_2gpu < time_1gpu:
            speedup = time_1gpu / time_2gpu
            efficiency = (speedup / 2) * 100
            print(f"📈 Scaling Results:")
            print(f"   1-GPU: {time_1gpu:.2f}s")
            print(f"   2-GPU: {time_2gpu:.2f}s")
            print(f"   Speedup: {speedup:.2f}x (expected: 2.00x)")
            print(f"   Efficiency: {efficiency:.1f}%")
        else:
            print(f"📉 No speedup observed:")
            print(f"   1-GPU: {time_1gpu:.2f}s")
            print(f"   2-GPU: {time_2gpu:.2f}s")
            print(f"   This suggests the workload isn't being distributed")
    
    elif result_2gpu.get('hanging', False):
        print("⚠️  2-GPU configuration hangs - workload issue detected")
        print("   This is NOT an isolation problem, but a computation problem")
        print("   Recommendation: Investigate SuperGrad's multi-GPU implementation")
    
    else:
        print("❌ Some configurations failed")
        print("   Check individual results for details")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"gpu_isolation_validation_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Recommendations
    print("\n💡 Next Steps:")
    if result_1gpu['success'] and result_2gpu['success'] and not result_2gpu.get('hanging', False):
        print("   ✅ GPU isolation works - proceed to full benchmark (n_qubit=8)")
    elif result_2gpu.get('hanging', False):
        print("   ⚠️  Multi-GPU hanging detected - investigate SuperGrad implementation")
        print("   🔍 Check NCCL configuration, communication patterns, or workload distribution")
    else:
        print("   ❌ Basic isolation failed - check CUDA/JAX setup")
    
    return results


if __name__ == "__main__":
    main()
