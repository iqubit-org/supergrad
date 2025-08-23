#!/usr/bin/env python3
"""
Unified Multi-GPU Profiler for Paper Benchmark
Tests the SAME workload (n_qubit=8) across different GPU counts (1, 2, 4, 8)
for direct performance scaling analysis.
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
from typing import Dict, Any, List, Optional
import psutil

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("⚠️  pynvml not available. Install with: pip install pynvml")
    print("   GPU memory monitoring will be disabled.")


class MultiGPUProfiler:
    """
    Enhanced Unified Multi-GPU Profiler for SuperGrad benchmarks.
    
    Features:
    - Step-by-step timing breakdown
    - GPU memory usage monitoring
    - Communication overhead profiling
    - Individual GPU utilization tracking
    - Comprehensive bottleneck analysis
    
    Tests the same workload (n_qubit=12) across different GPU configurations
    to analyze scaling efficiency and identify bottlenecks.
    
    Expected execution time: ~1 hour total (10 min per GPU config)
    """
    
    def __init__(self, n_qubit=12):
        """
        Initialize the enhanced profiler.
        
        Args:
            n_qubit (int): Number of qubits for the benchmark (default: 12)
                          This workload takes ~10 minutes on 8 GPUs
        """
        global NVML_AVAILABLE
        
        self.n_qubit = n_qubit
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"🚀 Initialized Enhanced Multi-GPU Profiler for {n_qubit} qubits")
        print("   Expected time per GPU config: ~10 minutes")
        print("   Total profiling time: ~1 hour")
        print(f"   GPU memory monitoring: {'✅ Enabled' if NVML_AVAILABLE else '❌ Disabled'}")
        
        # Initialize NVML if available
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                print("   NVML initialized successfully")
            except Exception as e:
                print(f"   ⚠️  NVML initialization failed: {e}")
                NVML_AVAILABLE = False

    def get_gpu_memory_usage(self) -> Dict[str, Dict[str, float]]:
        """Get current GPU memory usage for all visible GPUs"""
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

    def get_gpu_utilization(self) -> Dict[str, float]:
        """Get current GPU utilization for all visible GPUs"""
        if not NVML_AVAILABLE:
            return {}
        
        try:
            utilization_info = {}
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization_info[f'GPU_{i}'] = util.gpu
            return utilization_info
        except Exception as e:
            print(f"   ⚠️  Failed to get GPU utilization: {e}")
            return {}

    def profile_gpu_configuration(self, gpu_count: int) -> Dict[str, Any]:
        """Profile a specific GPU configuration with forced JAX device refresh"""
        print(f"🔍 Testing {gpu_count}-GPU Configuration with Device Refresh...")
        
        # Store original environment
        original_cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        original_xla_flags = os.environ.get('XLA_FLAGS', '')
        
        try:
            # Set environment variables for this GPU configuration
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(gpu_count))
            os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={gpu_count}'
            
            print(f"   Environment set for {gpu_count} GPUs")
            print(f"   CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
            print(f"   XLA_FLAGS: {os.environ['XLA_FLAGS']}")
            
            # Force JAX to refresh device list
            import jax
            import jax.numpy as jnp
            
            # Clear JAX's internal caches to force device refresh
            print("   Clearing JAX caches and refreshing devices...")
            jax.clear_caches()         # Clear all JAX caches
            
            # Try to clear device cache if available (some JAX versions don't have this)
            try:
                if hasattr(jax.devices, 'cache_clear'):
                    jax.devices.cache_clear()
                elif hasattr(jax, '_clear_device_cache'):
                    jax._clear_device_cache()
            except:
                print("   Note: Device cache clearing not available in this JAX version")
            
            # Get fresh device list
            available_devices = jax.devices()
            device_types = [d.device_kind for d in available_devices]
            
            print(f"   Available devices: {len(available_devices)}")
            print(f"   Device types: {device_types}")
            
            if len(available_devices) != gpu_count:
                print(f"⚠️  Warning: Expected {gpu_count} GPUs, but got {len(available_devices)}")
                print("   This might indicate environment variable issues")
            else:
                print(f"   ✅ Successfully configured {gpu_count} GPUs")
            
            # Import SuperGrad functions
            sys.path.append('..')
            
            # Import the real benchmark functions (qiskit_dynamics is needed)
            try:
                from test_simultaneous_x import test_simultaneous_x_state_grad_lcam, test_simultaneous_x_grad_lcam
                from utils.create_simultaneous_model import create_simultaneous_x
                print("   ✅ Successfully imported benchmark functions")
            except ImportError as e:
                print(f"   ❌ Failed to import benchmark functions: {e}")
                print("   This might be due to missing dependencies like qiskit_dynamics")
                return {'error': f'Import error: {e}'}
            
            # Create mock benchmark
            def create_mock_benchmark(test_type, n_qubit):
                class MockBenchmark:
                    def __init__(self, test_type, n_qubit):
                        if test_type == 'state':
                            self.group = f'gradient_simultaneous_x_state_{n_qubit}_qubits'
                        else:
                            self.group = f'gradient_simultaneous_x_{n_qubit}_qubits'
                        self.extra_info = {}
                    
                    def __call__(self, func):
                        return func()
                
                return MockBenchmark(test_type, n_qubit)
            
            # Initialize results
            results = {
                'gpu_count': gpu_count,
                'n_qubit': self.n_qubit,
                'timestamp': datetime.now().isoformat(),
                'step_timing': {},
                'memory_profiles': {},
                'gpu_profiles': {},
                'actual_devices_found': len(available_devices)
            }
            
            # Step 1: System Creation and Initialization
            print(f"   🏗️  Step 1: Creating evolution object for {self.n_qubit} qubits...")
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            evo = create_simultaneous_x(
                self.n_qubit,
                astep=5000,
                trotter_order=2,
                diag_ops=True,
                minimal_approach=True,
                custom_vjp=False  # Use JAX's built-in VJP for better multi-GPU compatibility
            )
            
            creation_time = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            results['step_timing']['system_creation'] = creation_time
            results['memory_profiles']['system_creation'] = {
                'start_mb': start_memory,
                'end_mb': end_memory,
                'delta_mb': end_memory - start_memory
            }
            
            print(f"   ✅ System creation completed: {creation_time:.2f}s (Memory: +{end_memory - start_memory:.1f}MB)")
            
            # Get parameter count
            total_params = sum(p.size for p in jax.tree_util.tree_leaves(evo.all_params))
            print(f"   Total parameters: {total_params}")
            
            # Step 2: State Gradient Computation
            print(f"   🧪 Step 2: Profiling State Evolution + Gradient (LCAM) - n_qubit={self.n_qubit}...")
            try:
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                # Get initial GPU memory if available
                start_gpu_memory = self.get_gpu_memory_usage()
                
                # Run the real benchmark
                benchmark = create_mock_benchmark('state', self.n_qubit)
                result = test_simultaneous_x_state_grad_lcam(
                    benchmark=benchmark,
                    n_qubit=self.n_qubit
                )
                
                # Wait for completion
                jax.block_until_ready(result)
                
                # Get final metrics
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                end_gpu_memory = self.get_gpu_memory_usage()
                
                execution_time = end_time - start_time
                memory_delta = end_memory - start_memory
                
                results['state_grad'] = {
                    'execution_time': execution_time,
                    'memory_delta_cpu': memory_delta,
                    'memory_delta_gpu': 0.0,  # Will be calculated if GPU memory available
                    'gpu_utilization': self.get_gpu_utilization(),
                    'success': True
                }
                
                results['step_timing']['state_gradient'] = execution_time
                results['memory_profiles']['state_gradient'] = {
                    'start_mb': start_memory,
                    'end_mb': end_memory,
                    'delta_mb': memory_delta,
                    'start_gpu_memory': start_gpu_memory,
                    'end_gpu_memory': end_gpu_memory
                }
                
                print(f"   ✅ State gradient completed: {execution_time:.2f}s (Memory: +{memory_delta:.1f}MB)")
                
            except Exception as e:
                print(f"   ❌ State gradient failed: {e}")
                results['state_grad'] = {'success': False, 'error': str(e)}
            
            # Step 3: Unitary Gradient Computation
            print(f"   🧪 Step 3: Profiling Unitary Evolution + Gradient (LCAM) - n_qubit={self.n_qubit}...")
            try:
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                start_gpu_memory = self.get_gpu_memory_usage()
                
                # Run the real benchmark
                benchmark = create_mock_benchmark('unitary', self.n_qubit)
                result = test_simultaneous_x_grad_lcam(
                    benchmark=benchmark,
                    n_qubit=self.n_qubit
                )
                
                # Wait for completion
                jax.block_until_ready(result)
                
                # Get final metrics
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                end_gpu_memory = self.get_gpu_memory_usage()
                
                execution_time = end_time - start_time
                memory_delta = end_memory - start_memory
                
                results['unitary_grad'] = {
                    'execution_time': execution_time,
                    'memory_delta_cpu': memory_delta,
                    'memory_delta_gpu': 0.0,  # Will be calculated if GPU memory available
                    'gpu_utilization': self.get_gpu_utilization(),
                    'success': True
                }
                
                results['step_timing']['unitary_gradient'] = execution_time
                results['memory_profiles']['unitary_gradient'] = {
                    'start_mb': start_memory,
                    'end_mb': end_memory,
                    'delta_mb': memory_delta,
                    'start_gpu_memory': start_gpu_memory,
                    'end_gpu_memory': end_gpu_memory
                }
                
                print(f"   ✅ Unitary gradient completed: {execution_time:.2f}s (Memory: +{memory_delta:.1f}MB)")
                
            except Exception as e:
                print(f"   ❌ Unitary gradient failed: {e}")
                results['unitary_grad'] = {'success': False, 'error': str(e)}
            
            # Final memory profile
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            final_gpu_memory = self.get_gpu_memory_usage()
            
            results['memory_profiles']['final'] = {
                'process_mb': final_memory,
                'gpu_memory': final_gpu_memory
            }
            
            print(f"   🎯 {gpu_count}-GPU configuration completed!")
            print(f"   Final memory usage: {final_memory:.1f}MB (Process)")
            if final_gpu_memory:
                for gpu, mem in final_gpu_memory.items():
                    print(f"   {gpu}: {mem['used_gb']:.1f}GB used / {mem['total_gb']:.1f}GB total")
            print()
            
            return results
            
        except Exception as e:
            print(f"   ❌ {gpu_count}-GPU configuration failed: {e}")
            return {'error': str(e)}
        
        finally:
            # Restore original environment
            if original_cuda_devices:
                os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_devices
            else:
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            
            if original_xla_flags:
                os.environ['XLA_FLAGS'] = original_xla_flags
            else:
                os.environ.pop('XLA_FLAGS', None)
            
            print("   Environment restored")
    
    def run_full_scaling_analysis(self) -> Dict[str, Any]:
        """Run the complete scaling analysis across all GPU configurations"""
        print("🚀 Starting Multi-GPU Scaling Analysis...")
        print(f"   Workload: {self.n_qubit} qubits")
        print("   GPU configurations: 1, 2, 4, 8")
        print("=" * 60)
        
        # Test each GPU configuration
        gpu_configs = [1, 2, 4, 8]
        
        for gpu_count in gpu_configs:
            try:
                self.results[f'{gpu_count}gpu'] = self.profile_gpu_configuration(gpu_count)
            except Exception as e:
                print(f"❌ Failed to test {gpu_count}-GPU configuration: {e}")
                self.results[f'{gpu_count}gpu'] = {'error': str(e)}
        
        # Generate scaling analysis
        self._generate_scaling_analysis()
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _generate_scaling_analysis(self) -> None:
        """Generate comprehensive scaling efficiency analysis with detailed insights"""
        print("📊 Enhanced Scaling Analysis Results:")
        print("=" * 60)
        
        # Get baseline (1-GPU) times
        baseline_1gpu = self.results.get('1gpu', {})
        if not baseline_1gpu or 'error' in baseline_1gpu:
            print("❌ Cannot analyze scaling: 1-GPU baseline failed")
            return
        
        baseline_state = baseline_1gpu.get('state_grad', {}).get('execution_time', 0)
        baseline_unitary = baseline_1gpu.get('unitary_grad', {}).get('execution_time', 0)
        
        if baseline_state == 0 or baseline_unitary == 0:
            print("❌ Cannot analyze scaling: Invalid baseline times")
            return
        
        print("📈 Baseline (1-GPU) Performance:")
        print(f"   State Gradient: {baseline_state:.4f}s")
        print(f"   Unitary Gradient: {baseline_unitary:.4f}s")
        
        # Show baseline step timing if available
        if 'step_timing' in baseline_1gpu:
            print("   Step-by-step breakdown:")
            for step, time_val in baseline_1gpu['step_timing'].items():
                print(f"     {step}: {time_val:.4f}s")
        print()
        
        # Analyze each configuration
        for gpu_count in [2, 4, 8]:
            config_key = f'{gpu_count}gpu'
            config = self.results.get(config_key, {})
            
            if 'error' in config:
                print(f"❌ {gpu_count}-GPU: Failed - {config['error']}")
                continue
            
            print(f"🔍 {gpu_count}-GPU Configuration:")
            
            # State gradient analysis
            state_grad = config.get('state_grad', {})
            if state_grad.get('success'):
                state_time = state_grad['execution_time']
                state_speedup = baseline_state / state_time
                state_efficiency = (state_speedup / gpu_count) * 100
                print(f"   State Gradient: {state_time:.4f}s "
                      f"({state_speedup:.2f}x speedup, {state_efficiency:.1f}% efficiency)")
                
                # Show step timing comparison if available
                if 'step_timing' in config and 'step_timing' in baseline_1gpu:
                    print("     Step-by-step scaling:")
                    for step in ['system_creation', 'state_gradient']:
                        if step in config['step_timing'] and step in baseline_1gpu['step_timing']:
                            step_time = config['step_timing'][step]
                            step_baseline = baseline_1gpu['step_timing'][step]
                            step_speedup = step_baseline / step_time
                            step_efficiency = (step_speedup / gpu_count) * 100
                            print(f"       {step}: {step_time:.4f}s ({step_speedup:.2f}x, {step_efficiency:.1f}%)")
            else:
                print(f"   State Gradient: FAILED - {state_grad.get('error', 'Unknown error')}")
            
            # Unitary gradient analysis
            unitary_grad = config.get('unitary_grad', {})
            if unitary_grad.get('success'):
                unitary_time = unitary_grad['execution_time']
                unitary_speedup = baseline_unitary / unitary_time
                unitary_efficiency = (unitary_speedup / gpu_count) * 100
                print(f"   Unitary Gradient: {unitary_time:.4f}s "
                      f"({unitary_speedup:.2f}x speedup, {unitary_efficiency:.1f}% efficiency)")
                
                # Show step timing comparison if available
                if 'step_timing' in config and 'step_timing' in baseline_1gpu:
                    print("     Step-by-step scaling:")
                    for step in ['system_creation', 'unitary_gradient']:
                        if step in config['step_timing'] and step in baseline_1gpu['step_timing']:
                            step_time = config['step_timing'][step]
                            step_baseline = baseline_1gpu['step_timing'][step]
                            step_speedup = step_baseline / step_time
                            step_efficiency = (step_speedup / gpu_count) * 100
                            print(f"       {step}: {step_time:.4f}s ({step_speedup:.2f}x, {step_efficiency:.1f}%)")
            else:
                print(f"   Unitary Gradient: FAILED - {unitary_grad.get('error', 'Unknown error')}")
            
            # Memory analysis if available
            if 'memory_profiles' in config:
                print("     Memory usage:")
                for step, mem_profile in config['memory_profiles'].items():
                    if step != 'final' and 'delta_mb' in mem_profile:
                        print(f"       {step}: +{mem_profile['delta_mb']:.1f}MB")
            
            print()
        
        # Generate bottleneck insights
        self._generate_bottleneck_insights()
        
        print("=" * 60)
    
    def _generate_bottleneck_insights(self) -> None:
        """Generate insights about potential bottlenecks based on the data"""
        print("🔍 Bottleneck Analysis:")
        print("-" * 40)
        
        # Check if we have enough data
        if len(self.results) < 2:
            print("   ⚠️  Insufficient data for bottleneck analysis")
            return
        
        # Analyze scaling patterns
        gpu_configs = [1, 2, 4, 8]
        state_times = []
        unitary_times = []
        
        for gpu_count in gpu_configs:
            config_key = f'{gpu_count}gpu'
            config = self.results.get(config_key, {})
            
            if 'error' not in config:
                state_grad = config.get('state_grad', {})
                unitary_grad = config.get('unitary_grad', {})
                
                if state_grad.get('success'):
                    state_times.append((gpu_count, state_grad['execution_time']))
                if unitary_grad.get('success'):
                    unitary_times.append((gpu_count, unitary_grad['execution_time']))
        
        # Analyze state gradient scaling
        if len(state_times) >= 2:
            print("   State Gradient Scaling:")
            for i in range(1, len(state_times)):
                gpu_count, time_val = state_times[i]
                prev_gpu_count, prev_time = state_times[i - 1]
                speedup = prev_time / time_val
                expected_speedup = gpu_count / prev_gpu_count
                efficiency = (speedup / expected_speedup) * 100
                
                if efficiency < 80:
                    print(f"     {prev_gpu_count}→{gpu_count} GPUs: {speedup:.2f}x speedup "
                          f"(expected {expected_speedup:.2f}x, {efficiency:.1f}% efficiency)")
                    print("       ⚠️  Poor scaling detected - possible bottleneck")
                else:
                    print(f"     {prev_gpu_count}→{gpu_count} GPUs: {speedup:.2f}x speedup "
                          f"(expected {expected_speedup:.2f}x, {efficiency:.1f}% efficiency)")
        
        # Analyze unitary gradient scaling
        if len(unitary_times) >= 2:
            print("   Unitary Gradient Scaling:")
            for i in range(1, len(unitary_times)):
                gpu_count, time_val = unitary_times[i]
                prev_gpu_count, prev_time = unitary_times[i - 1]
                speedup = prev_time / time_val
                expected_speedup = gpu_count / prev_gpu_count
                efficiency = (speedup / expected_speedup) * 100
                
                if efficiency < 80:
                    print(f"     {prev_gpu_count}→{gpu_count} GPUs: {speedup:.2f}x speedup "
                          f"(expected {expected_speedup:.2f}x, {efficiency:.1f}% efficiency)")
                    print("       ⚠️  Poor scaling detected - possible bottleneck")
                else:
                    print(f"     {prev_gpu_count}→{gpu_count} GPUs: {speedup:.2f}x speedup "
                          f"(expected {expected_speedup:.2f}x, {efficiency:.1f}% efficiency)")
        
        # Memory bottleneck analysis
        print("   Memory Analysis:")
        for gpu_count in [1, 2, 4, 8]:
            config_key = f'{gpu_count}gpu'
            config = self.results.get(config_key, {})
            
            if 'error' not in config and 'memory_profiles' in config:
                mem_profiles = config['memory_profiles']
                if 'final' in mem_profiles:
                    process_mb = mem_profiles['final']['process_mb']
                    print(f"     {gpu_count}-GPU: {process_mb:.1f}MB process memory")
        
        print()
    
    def _save_results(self) -> None:
        """Save results to JSON file"""
        filename = f"scaling_analysis_{self.n_qubit}q_{self.timestamp}.json"
        
        # Prepare results for JSON serialization
        json_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                json_results[key] = {}
                for subkey, subvalue in value.items():
                    if subkey == 'gpu_utilization':
                        # Convert numpy types to native Python types
                        json_results[key][subkey] = {k: float(v) for k, v in subvalue.items()}
                    else:
                        json_results[key][subkey] = subvalue
            else:
                json_results[key] = value
        
        try:
            with open(filename, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            print(f"💾 Results saved to: {filename}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")


if __name__ == "__main__":
    # Run the complete scaling analysis
    profiler = MultiGPUProfiler(n_qubit=8)
    results = profiler.run_full_scaling_analysis()
    
    print("🎯 Multi-GPU Scaling Analysis Complete!")
    print("Check the generated JSON file for detailed results.")
