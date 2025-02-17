from enum import Enum
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


eager_file = "gh200_matmul_eager.json"
nvf_file = "gh200_matmul_nvf.json"

eager_json = json.load(open(eager_file))
nvf_json = json.load(open(nvf_file))

eager_data = analyze(eager_json)
nvf_data = analyze(nvf_json)

relative_scores = []
for layout in Layout:
    if layout is Layout.MAX:
        continue

    nvf_layout_data = nvf_data[layout]
    eager_layout_data = eager_data[layout]
    for key in nvf_layout_data.keys():
        eager_value = eager_layout_data[key]
        nvf_value = nvf_layout_data[key]
        score = eager_value / nvf_value
        relative_scores.append(score)

plt.scatter(
    np.arange(len(relative_scores)),
    np.array(list(sorted(relative_scores))),
    color="blue",
    marker="o",
    s=5,
)
plt.ylabel("Relative %")
plt.ylim(0, 1)
plt.title("Hopper Matmul S-Curve")
plt.savefig("relative_s_curve.png")
plt.close("all")
