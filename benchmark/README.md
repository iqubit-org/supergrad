# SuperGrad Benchmark Suite

This directory contains the essential benchmarking tools for analyzing SuperGrad's GPU scaling performance.

## 🎯 Core Files

### **Enhanced Multi-GPU Profiler**
- **`unified_multi_gpu_profiler.py`** - Main profiler class with comprehensive GPU analysis
- **`test_unified_profiler.py`** - Script to run the complete profiling suite
- **`requirements_enhanced_profiler.txt`** - Dependencies for enhanced profiling

### **Original Benchmark Tests**
- **`test_simultaneous_x.py`** - Author's original benchmark tests (the target of our profiling)
- **`test_simultaneous_cnot.py`** - CNOT gate benchmark tests
- **`utils/`** - Utility functions for creating benchmark models

### **Legacy Tests** (Kept for Reference)
- **`fdm_overall_gradient.py`** - Finite difference method tests
- **`scqubits_step_overall.py` - Step-by-step evolution tests
- **`test_comparison_exists.py`** - Comparison validation tests
- **`autodiff_evolution_gradient.py`** - Automatic differentiation tests

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd ~/shang/supergrad/benchmark
pip install -r requirements_enhanced_profiler.txt
```

### 2. Run Enhanced Profiling
```bash
# Run complete GPU scaling analysis (1, 2, 4, 8 GPUs)
nohup env PYTHONPATH=.. python3 -u test_unified_profiler.py > enhanced_profiler.log 2>&1 &

# Monitor progress
tail -f enhanced_profiler.log
```

### 3. Expected Results
- **Total time**: ~1 hour
- **Per GPU config**: ~10-15 minutes
- **Workload**: n_qubit=12 (realistic benchmark size)
- **Output**: Detailed scaling analysis with bottleneck identification

## 📊 What the Enhanced Profiler Provides

### **Step-by-Step Analysis**
- System creation timing
- State gradient computation timing
- Unitary gradient computation timing
- Memory usage patterns

### **GPU Scaling Metrics**
- Speedup ratios across GPU configurations
- Efficiency percentages (ideal vs. actual scaling)
- Memory usage per GPU configuration
- Bottleneck identification

### **Comprehensive Output**
- Real-time progress logging
- JSON results file for further analysis
- Detailed bottleneck insights
- Actionable optimization recommendations

## 🔍 Understanding the Results

### **Good Scaling** (>80% efficiency)
- Configuration scales well with more GPUs
- Minimal communication overhead
- Good parallelization

### **Poor Scaling** (<80% efficiency)
- Potential bottlenecks identified
- Communication overhead issues
- Memory or synchronization problems

## 📁 Directory Structure
```
benchmark/
├── README.md                           # This file
├── unified_multi_gpu_profiler.py      # Enhanced profiler class
├── test_unified_profiler.py           # Main profiling script
├── requirements_enhanced_profiler.txt  # Dependencies
├── test_simultaneous_x.py             # Author's benchmark
├── utils/                              # Benchmark utilities
│   └── create_simultaneous_model.py   # Model creation
├── ref_data/                           # Reference data
└── res_data/                           # Results storage
```

## 💡 Tips for Best Results

1. **Ensure all GPUs are free** before running
2. **Monitor GPU memory** during execution
3. **Check the log file** for real-time progress
4. **Analyze the JSON results** for detailed insights
5. **Compare scaling efficiency** across different GPU counts

## 🆘 Troubleshooting

### **pynvml not available**
```bash
pip install pynvml
```
The profiler will work without it, but with limited GPU metrics.

### **Memory errors**
- Reduce `n_qubit` if needed
- Ensure sufficient GPU memory
- Check for other GPU processes

### **Long execution times**
- Normal for n_qubit=12
- Expected: ~10-15 minutes per GPU configuration
- Total: ~1 hour for complete analysis
