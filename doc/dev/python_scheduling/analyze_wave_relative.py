from enum import Enum
import math
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Layout(Enum):
    NN = 0
    NT = 1
    TN = 2
    TT = 3
    MAX = 4


def get_layout_enum(layout):
    if layout == "NN":
        return Layout.NN
    elif layout == "NT":
        return Layout.NT
    elif layout == "TN":
        return Layout.TN
    elif layout == "TT":
        return Layout.TT
    else:
        return Layout.MAX


# machine_info - dict
#  * node, cpu, gpu-name - string
#  * gpu_sm_count - int
# commit_info - dict
#  * id, project, branch - string
# benchmarks - list
#  * fullname: string
#  * params - dict
#     * config: [M, N, K, Shape]
#  * stats - dict
#     * median: float


def get_field(json_data, field):
    return pd.DataFrame(json_data[field])


def analyze(json_data):
    benchmarks = get_field(json_data, "benchmarks")
    data = {layout: {} for layout in Layout if layout is not Layout.MAX}
    for row in benchmarks.itertuples():
        M, N, K, layout = row.params["config"]
        shape = (M, N, K)
        time = row.stats["median"]
        data[get_layout_enum(layout)][shape] = time
    return data


def sort_keys_by_waves(json_data, keys):
    gpu_sm_count = json_data["machine_info"]["gpu_sm_count"]

    def key_by_waves(shape):
        M, N, K = shape
        BM = max(M // 64, 1)
        BN = max(N // 256, 1)
        waves = (BM * BN) // gpu_sm_count
        return waves

    return [k for k in sorted(keys, key=key_by_waves)]


def key_by_waves(gpu_sm_count, shape):
    M, N, K = shape
    BM = max(M // 64, 1)
    BN = max(N // 256, 1)
    waves = (BM * BN) / gpu_sm_count
    return waves


def key_by_square(shape):
    M, N, _ = shape
    return math.sqrt(M * N) / max(M, N)


eager_file = "gh200_matmul_eager.json"
nvf_file = "gh200_matmul_nvf.json"

eager_json = json.load(open(eager_file))
nvf_json = json.load(open(nvf_file))

eager_data = analyze(eager_json)
nvf_data = analyze(nvf_json)

gpu_sm_count = eager_json["machine_info"]["gpu_sm_count"]
sorted_keys = [
    k
    for k in sorted(
        eager_data[Layout.NN].keys(), key=lambda x: key_by_waves(gpu_sm_count, x)
    )
]

histogram = [[] for _ in range(10)]
splitk_key = []
splitk_value = []
multiwave_key = []
multiwave_value = []
for key in sorted_keys:
    if key not in nvf_data[Layout.NN]:
        continue
    eager_value = eager_data[Layout.NN][key]
    nvf_value = nvf_data[Layout.NN][key]
    score = eager_value / nvf_value
    num_waves = key_by_waves(gpu_sm_count, key)

    bucket = int(score // 0.1)
    histogram[bucket].append(key)

    if num_waves >= 1:
        multiwave_key.append(num_waves)
        multiwave_value.append(score)
    else:
        splitk_key.append(num_waves)
        splitk_value.append(score)

for shape in histogram[3]:
    num_waves = key_by_waves(gpu_sm_count, shape)
    squareness = key_by_square(shape)
    print(f"{num_waves:2f}, {squareness:2f}, {shape}")

plt.scatter(
    np.array(multiwave_key), np.array(multiwave_value), color="blue", marker="o", s=5
)
plt.scatter(np.array(splitk_key), np.array(splitk_value), color="red", marker="o", s=5)

plt.xlabel("Shape (Number of Waves)")
plt.xscale("log")
plt.ylabel("Relative %")
plt.ylim(0, 1)
plt.title("Hopper Matmul Eager vs. NvFuser")
plt.legend(["multiwave", "splitk"], loc="lower right")
plt.savefig("relative_num_waves.png")
plt.close("all")

# mma macro
# cta-tile => cooperative warp group
# number of shared memory stages

# 1) Pick largest mma macro; Minimize wave quantization
# 2) Tile problem shape by mma macro
# If number of tiles is 2x number of SMs, use cooperative warp group and double
# cta size along axis.
# 3) Increase memory pipeline to use available shared memory
# 4) Row vs Column Order --- if contiguous dim is M and M > N, then Row with (2, 1, 1) CGA. Otherwise, Column with (1, 2, 1) CGA.
# 5) Grid swizzle to improve L2 cache performance
