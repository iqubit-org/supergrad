#!/usr/bin/env python3
"""
JAX Profiler Utilities for SuperGrad Benchmarking
Provides TensorBoard trace generation for GPU kernel analysis
"""

import os
import jax
import jax.profiler
from datetime import datetime


def setup_tensorboard_logging(gpu_count, timestamp=None):
    """Setup TensorBoard log directory for a specific GPU configuration"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    tensorboard_log_dir = f"tensorboard_logs/{gpu_count}gpu_{timestamp}"
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    
    print(f"📊 TensorBoard traces will be saved to: {tensorboard_log_dir}")
    return tensorboard_log_dir


def run_with_jax_profiler(func, trace_name, log_dir, description=""):
    """
    Run function with JAX profiler and generate TensorBoard trace
    
    Args:
        func: Function to profile
        trace_name: Name for the trace (e.g., 'lcam_trace', 'tad_trace')
        log_dir: Base TensorBoard log directory
        description: Optional description for the trace
    
    Returns:
        tuple: (result, trace_dir)
    """
    trace_dir = os.path.join(log_dir, trace_name)
    os.makedirs(trace_dir, exist_ok=True)
    
    print(f"   🔍 Starting JAX profiler trace: {trace_name}")
    if description:
        print(f"      Description: {description}")
    
    # Start JAX profiler
    jax.profiler.start_trace(trace_dir)
    
    try:
        # Run the function
        result = func()
        
        # Stop profiler and generate trace
        jax.profiler.stop_trace()
        
        print(f"   ✅ JAX profiler trace completed: {trace_dir}")
        return result, trace_dir
        
    except Exception as e:
        # Stop profiler even if function fails
        jax.profiler.stop_trace()
        print(f"   ❌ JAX profiler trace failed: {e}")
        raise e


def print_tensorboard_summary(results, tensorboard_log_dir):
    """Print summary of TensorBoard traces"""
    print(f"\n📊 JAX Profiler Summary:")
    print(f"   TensorBoard log directory: {tensorboard_log_dir}")
    print(f"   View traces with: tensorboard --logdir {tensorboard_log_dir}")
    print(f"   Available traces:")
    
    # Check for LCAM trace
    if 'unitary_grad_lcam' in results and results['unitary_grad_lcam'].get('success'):
        trace_path = results['unitary_grad_lcam'].get('tensorboard_trace', 'N/A')
        print(f"      - LCAM: {trace_path}")
    
    # Check for TAD trace
    if 'unitary_grad_tad' in results and results['unitary_grad_tad'].get('success'):
        trace_path = results['unitary_grad_tad'].get('tensorboard_trace', 'N/A')
        print(f"      - TAD: {trace_path}")
    
    print(f"\n💡 TensorBoard Analysis Tips:")
    print(f"   - Look for GPU kernel execution patterns")
    print(f"   - Check memory transfer efficiency (H2D/D2H)")
    print(f"   - Analyze sharding and replication patterns")
    print(f"   - Compare LCAM vs TAD kernel utilization")


def create_profiled_test_function(test_func, benchmark, n_qubit):
    """Create a profiled version of a test function"""
    def profiled_test():
        result = test_func(benchmark=benchmark, n_qubit=n_qubit)
        jax.block_until_ready(result)
        return result
    return profiled_test
