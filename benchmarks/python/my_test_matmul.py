# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
from nvfuser import FusionDefinition
import torch
import math
import csv
import os
from enum import Enum


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
def analyze_json(filename):
    import pandas as pd
    import json

    def get_field(json_data, field):
        return pd.DataFrame(json_data[field])

    def _organize_by_layout(json_data):
        benchmarks = get_field(json_data, "benchmarks")
        data = {layout: {} for layout in Layout if layout is not Layout.MAX}
        for row in benchmarks.itertuples():
            M, N, K, layout = row.params["config"]
            shape = (M, N, K)
            time = row.stats["median"]
            data[get_layout_enum(layout)][shape] = time
        return data

    json = json.load(open(filename))
    return _organize_by_layout(json)


def plot_s_curve(eager_data, nvf_data, filename="relative_s_curve.png"):
    def _calculate_relative_scores():
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
        return relative_scores

    import matplotlib.pyplot as plt
    import numpy as np

    relative_scores = _calculate_relative_scores()

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
    plt.savefig(filename)
    plt.close("all")


def matmul_fusion(fd: FusionDefinition, inputs: list[torch.Tensor]) -> None:
    a = fd.from_pytorch(inputs[0])
    b = fd.from_pytorch(inputs[1])
    out = fd.ops.matmul(a, b)
    fd.add_output(out)


def load_matmul_problems():
    with open(os.path.join(os.path.dirname(__file__), "matmul_problems.csv"), "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header row
        return list((int(m), int(n), int(k), layout) for m, n, k, layout in reader)


def estimate_matmul_size(config, dtype):
    def _estimate_size(shape, dtype):
        return math.prod(shape) * dtype.itemsize

    m, n, k, layout = config
    total_in_gbs = 0
    for shape in [[m, k], [n, k], [m, n]]:
        total_in_gbs += _estimate_size(shape, dtype)
    return total_in_gbs


def test_matmul_nvf(
    config: tuple,
    disable_validation: bool,
):
    m, n, k, layout = config
    dtype = torch.bfloat16

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)

    if layout == "NT" or layout == "NN":
        a = a.as_strided(size=[m, k], stride=[1, m])
    if layout == "TN" or layout == "NN":
        b = b.as_strided(size=[k, n], stride=[1, k])

    kwargs = dict(
        _enable_options=["fuse_matmul"],
        _disable_options=["matmul_expr_eval"],
        profile=True,
    )

    with FusionDefinition() as fd:
        matmul_fusion(fd, [a, b])

    try:
        nvf_outputs = fd.execute([a, b], **kwargs)
    except Exception as e:
        return -1

    if not disable_validation:
        eager_output = torch.matmul(a, b)
        assert torch.allclose(nvf_outputs[0], eager_output, atol=1e-2, rtol=1e-2)

    prof = fd.profile()
    # convert to microseconds to match pytorch profiler units
    return prof.kernel_profiles[0].time_ms * 1e-3


def profile_configuration():
    device_properties = torch.cuda.get_device_properties(0)
    configurations = load_matmul_problems()
    data = {layout: {} for layout in Layout if layout is not Layout.MAX}
    for config in configurations:
        # short-circuit: problem does not fit on device
        if (
            estimate_matmul_size(config, torch.bfloat16)
            >= device_properties.total_memory
        ):
            continue
        time = test_matmul_nvf(config, disable_validation=True)

        # short-circuit: failed to run problem
        if time == -1:
            continue

        M, N, K, layout = config
        shape = (M, N, K)
        data[get_layout_enum(layout)][shape] = time
    return data


def main():
    plot_s_curve(
        eager_data=analyze_json("gh200_matmul_eager.json"),
        nvf_data=profile_configuration(),
        filename="foo.png",
    )


if __name__ == "__main__":
    main()
