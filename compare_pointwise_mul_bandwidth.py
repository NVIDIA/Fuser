import json
import os
import matplotlib.pyplot as plt

# Paths to the benchmark result files
cuda129_path = "/benchmarks/cursor/inference_bm/benchmark_results/pointwise_mul_float16_16384_cuda129.json"
cuda130_path = "/benchmarks/cursor/inference_bm/benchmark_results/pointwise_mul_float16_16384_cuda130.json"

# Helper to extract hidden size and SOL bandwidth from a file
def extract_bandwidths(path):
    with open(path, 'r') as f:
        data = json.load(f)
    results = []
    for bench in data["benchmarks"]:
        # Only consider entries with the expected keys
        if (
            "params" in bench and
            "size" in bench["params"] and
            isinstance(bench["params"]["size"], list) and
            len(bench["params"]["size"]) == 2 and
            "extra_info" in bench and
            "% Peak Bandwidth (SOL)" in bench["extra_info"]
        ):
            hidden_size = bench["params"]["size"][1]
            sol_bandwidth = bench["extra_info"]["% Peak Bandwidth (SOL)"]
            results.append((hidden_size, sol_bandwidth))
    # Sort by hidden size
    results.sort(key=lambda x: x[0])
    return results

# Extract data
cuda129_data = extract_bandwidths(cuda129_path)
cuda130_data = extract_bandwidths(cuda130_path)

# Prepare for plotting
sizes_129, sol_129 = zip(*cuda129_data)
sizes_130, sol_130 = zip(*cuda130_data)

plt.figure(figsize=(10,6))
plt.plot(sizes_129, sol_129, marker='o', label='CUDA 12.9')
plt.plot(sizes_130, sol_130, marker='s', label='CUDA 13.0')
plt.xlabel('Hidden Size')
plt.ylabel('SOL Bandwidth (%)')
plt.title('Pointwise Mul: SOL Bandwidth vs Hidden Size')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('pointwise_mul_sol_bandwidth_comparison.png')
plt.show() 