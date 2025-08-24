#!/usr/bin/env python3
"""
Test script for the Unified Multi-GPU Profiler
Runs the complete scaling analysis across 1, 2, 4, and 8 GPUs
"""

import os
import sys

# Add parent directory to path for imports
sys.path.append('..')

from unified_multi_gpu_profiler import MultiGPUProfiler


def test_single_configuration(profiler, gpu_count: int):
    """Test a single GPU configuration using the provided profiler"""
    print(f"🧪 Testing {gpu_count}-GPU Configuration...")
    print("=" * 50)
    
    try:
        result = profiler.profile_gpu_configuration(gpu_count)
        
        if 'error' in result:
            print(f"❌ {gpu_count}-GPU test failed: {result['error']}")
        else:
            print(f"✅ {gpu_count}-GPU test completed successfully!")
            print(f"   State gradient: {result.get('state_grad', {}).get('execution_time', 'N/A')}s")
            print(f"   Unitary gradient: {result.get('unitary_grad', {}).get('execution_time', 'N/A')}s")
        
        return result
        
    except Exception as e:
        print(f"❌ {gpu_count}-GPU test failed: {e}")
        return {'error': str(e)}


def main():
    """Run the enhanced unified multi-GPU profiling with JAX device selection"""
    print("🚀 Testing Enhanced Multi-GPU Profiler with JAX Device Selection")
    print("=" * 60)
    print("Enhanced Features:")
    print("   ✅ Step-by-step timing breakdown")
    print("   ✅ GPU memory usage monitoring")
    print("   ✅ Communication overhead profiling")
    print("   ✅ Individual GPU utilization tracking")
    print("   ✅ Comprehensive bottleneck analysis")
    print("   ✅ JAX built-in device selection (no environment variables)")
    print("   ✅ Real-time device count verification")
    print("=" * 60)
    print("Configuration: n_qubit=8 (main benchmark)")
    print("Expected execution time: ~4-6 hours total")
    print("   - 1-GPU: ~60-90 minutes")
    print("   - 2-GPU: ~30-45 minutes") 
    print("   - 4-GPU: ~15-23 minutes")
    print("   - 8-GPU: ~8-12 minutes")
    print("=" * 60)
    print("Note: This approach uses JAX's built-in device selection")
    print("      and will automatically adjust to available GPUs")
    print("=" * 60)

    try:
        # Initialize profiler with n_qubit=8 for main benchmark
        profiler = MultiGPUProfiler(n_qubit=8)

        # Test individual configurations first
        print("\n🔍 Testing Individual Configurations:")
        print("-" * 40)

        # Test 1-GPU first (baseline)
        result_1gpu = test_single_configuration(profiler, 1)
        if 'error' in result_1gpu:
            print("❌ Cannot proceed: 1-GPU baseline failed")
            return None

        # Test 2-GPU
        result_2gpu = test_single_configuration(profiler, 2)

        # Test 4-GPU
        result_4gpu = test_single_configuration(profiler, 4)

        # Test 8-GPU
        result_8gpu = test_single_configuration(profiler, 8)

        # Collect results
        profiler.results = {
            '1gpu': result_1gpu,
            '2gpu': result_2gpu,
            '4gpu': result_4gpu,
            '8gpu': result_8gpu
        }

        # Generate enhanced scaling analysis
        print("\n📊 Generating Enhanced Scaling Analysis:")
        print("-" * 40)
        profiler._generate_scaling_analysis()

        # Save results
        profiler._save_results()

        print("\n🎯 Enhanced Individual Configuration Testing Complete!")
        print("Check the generated JSON file for detailed results and bottleneck analysis.")
        print("\n💡 Multi-GPU benchmark with n_qubit=8 completed!")
        print("   This used the correct functions: test_simultaneous_x_grad_lcam and test_simultaneous_x_grad_tad")
        print("   These functions should trigger SuperGrad's multi-GPU sharding!")

        return profiler.results

    except Exception as e:
        print(f"❌ Profiling failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
