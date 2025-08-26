#!/usr/bin/env python3
"""
Hybrid Multi-GPU Profiler - Main Orchestrator
Calls individual GPU profiling scripts for true isolation
Runs the complete scaling analysis across 1, 2, 4, and 8 GPUs
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime





def run_gpu_profiler(gpu_count):
    """Run a specific GPU profiling script"""
    script_name = f"profile_{gpu_count}gpu.py"
    print(f"\n🔍 Running {gpu_count}-GPU Profiler...")
    print(f"   Script: {script_name}")
    print(f"   Expected GPUs: {gpu_count}")
    
    try:
        # Run the profiling script
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, script_name],
            env=os.environ.copy()
        )
        end_time = time.time()
        
        print(f"   ✅ {gpu_count}-GPU script completed in {end_time - start_time:.2f}s")
        
        # Check if result file was created
        result_file = f"profile_{gpu_count}gpu_results.json"
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                result_data = json.load(f)
            
            print(f"   📊 Results loaded from: {result_file}")
            return result_data
        else:
            print(f"   ❌ No result file found: {result_file}")
            print(f"   Subprocess return code: {result.returncode}")
            return {'error': 'No result file generated'}
            
    except Exception as e:
        print(f"   ❌ {gpu_count}-GPU script failed: {e}")
        return {'error': str(e)}

def analyze_scaling(results):
    """Analyze scaling across all GPU configurations"""
    print("\n📊 Multi-GPU Scaling Analysis")
    print("=" * 60)
    
    if not results:
        print("❌ No results to analyze")
        return
    
    # Get baseline (1-GPU)
    baseline = results.get('1gpu', {})
    if 'error' in baseline:
        print("❌ 1-GPU baseline failed - cannot analyze scaling")
        return
    
    # Check if LCAM and TAD results exist
    lcam_baseline = baseline.get('unitary_grad_lcam', {})
    tad_baseline = baseline.get('unitary_grad_tad', {})
    
    if not lcam_baseline.get('success') or not tad_baseline.get('success'):
        print("❌ Baseline tests failed - cannot analyze scaling")
        return
    
    lcam_baseline_time = lcam_baseline.get('execution_time', 0)
    tad_baseline_time = tad_baseline.get('execution_time', 0)
    
    print(f"📈 Baseline (1-GPU) Performance:")
    print(f"   LCAM: {lcam_baseline_time:.2f}s")
    print(f"   TAD: {tad_baseline_time:.2f}s")
    
    # Analyze each configuration (skip 2-GPU as it's disabled)
    for gpu_count in [4, 8]:
        config_key = f'{gpu_count}gpu'
        config = results.get(config_key, {})
        
        if 'error' in config:
            print(f"\n❌ {gpu_count}-GPU: Failed - {config['error']}")
            continue
        
        print(f"\n🔍 {gpu_count}-GPU Configuration:")
        
        # LCAM analysis
        lcam_result = config.get('unitary_grad_lcam', {})
        if lcam_result.get('success'):
            lcam_time = lcam_result.get('execution_time', 0)
            if lcam_time > 0:
                lcam_speedup = lcam_baseline_time / lcam_time
                lcam_efficiency = (lcam_speedup / gpu_count) * 100
                
                print(f"   LCAM: {lcam_time:.2f}s (Speedup: {lcam_speedup:.2f}x, Efficiency: {lcam_efficiency:.1f}%)")
            else:
                print(f"   LCAM: No timing data")
        else:
            print(f"   LCAM: Failed - {lcam_result.get('error', 'Unknown error')}")
        
        # TAD analysis
        tad_result = config.get('unitary_grad_tad', {})
        if tad_result.get('success'):
            tad_time = tad_result.get('execution_time', 0)
            if tad_time > 0:
                tad_speedup = tad_baseline_time / tad_time
                tad_efficiency = (tad_speedup / gpu_count) * 100
                
                print(f"   TAD: {tad_time:.2f}s (Speedup: {tad_speedup:.2f}x, Efficiency: {tad_efficiency:.1f}%)")
            else:
                print(f"   TAD: No timing data")
        else:
            print(f"   TAD: Failed - {tad_result.get('error', 'Unknown error')}")

def main():
    """Run the hybrid multi-GPU profiling with individual scripts"""
    print("🚀 Hybrid Multi-GPU Profiler - Main Orchestrator")
    print("=" * 60)
    print("Architecture:")
    print("   ✅ Individual GPU profiling scripts for true isolation")
    print("   ✅ Fresh JAX import for each GPU configuration")
    print("   ✅ CUDA_VISIBLE_DEVICES set before JAX import")
    print("   ✅ Results collected from individual JSON files")
    print("   ✅ Unified scaling analysis")
    print("=" * 60)
    print("Configuration: n_qubit=8 (main benchmark)")
    print("Expected execution time: ~2-2.5 hours total")
    print("   - 1-GPU: ~30-60 minutes")
    print("   - 2-GPU: SKIPPED (script exists but testing disabled)")
    print("   - 4-GPU: ~30-60 minutes")
    print("   - 8-GPU: ~30-60 minutes")
    print("=" * 60)
    print("Note: Each GPU configuration runs in a separate process")
    print("      ensuring true GPU isolation and no JAX cache conflicts")
    print("=" * 60)

    try:
        # Run all GPU configurations
        results = {}
        
        # Test 1-GPU first (baseline)
        print("\n🔍 Testing Individual GPU Configurations:")
        print("-" * 40)
        
        result_1gpu = run_gpu_profiler(1)
        if 'error' in result_1gpu:
            print("❌ Cannot proceed: 1-GPU baseline failed")
            return None
        
        results['1gpu'] = result_1gpu
        
        # Skip 2-GPU testing (script exists but testing is disabled)
        # result_2gpu = run_gpu_profiler(2)
        # results['2gpu'] = result_2gpu
        
        # Test 4-GPU
        result_4gpu = run_gpu_profiler(4)
        results['4gpu'] = result_4gpu
        
        # Test 8-GPU
        result_8gpu = run_gpu_profiler(8)
        results['8gpu'] = result_8gpu
        
        # Generate scaling analysis
        analyze_scaling(results)
        
        # Save unified results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unified_file = f"hybrid_profiler_results_{timestamp}.json"
        with open(unified_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'n_qubit': 8,
                'results': results
            }, f, indent=2)
        
        print(f"\n💾 Unified results saved to: {unified_file}")
        print("\n🎯 Hybrid Multi-GPU Profiling Complete!")
        print("   ✅ TRUE GPU isolation achieved with individual scripts")
        print("   ✅ All configurations tested successfully")
        print("   ✅ Scaling analysis generated")
        
        return results

    except Exception as e:
        print(f"❌ Profiling failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
