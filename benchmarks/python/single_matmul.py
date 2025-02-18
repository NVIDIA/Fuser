# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
from nvfuser import FusionDefinition, SchedulerType
import torch
import math
import itertools
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


def estimate_matmul_size(config, dtype):
    def _estimate_size(shape, dtype):
        return math.prod(shape) * dtype.itemsize

    m, n, k, layout = config
    total_in_gbs = 0
    for shape in [[m, k], [n, k], [m, n]]:
        total_in_gbs += _estimate_size(shape, dtype)
    return total_in_gbs


def matmul_fusion(fd: FusionDefinition, inputs: list[torch.Tensor]) -> None:
    a = fd.from_pytorch(inputs[0])
    b = fd.from_pytorch(inputs[1])
    out = fd.ops.matmul(a, b)
    fd.add_output(out)


# These are the parameters we'll optimize
parameter_configurations = [
    splitk_factors := list(range(1, 8)),
    load_stages := list(range(1, 4)),
]


# Apply scheduler with custom parameters using decorator
def custom_matmul_scheduler(fd, config):
    def inner_fn():
        # NOTE Scheduler _matmul_ ***rejected*** because : MatmulOp and LinearOp
        # fusion is disabled by default. Enable it using
        # NVFUSER_ENABLE=fuse_matmul
        status, error = fd.sched.can_schedule(SchedulerType.matmul)
        assert status, error

        schedule_params = fd.sched.compute_matmul_heuristics()

        # Modify original parameters
        if config is not None:
            splitk_factor, stages = config
            schedule_params.circular_buffer_options.circular_buffer_smem_write = (
                stages > 1
            )
            schedule_params.circular_buffer_options.smem_circular_buffer_stage = stages
            schedule_params.splitk_factor = splitk_factor

        # Schedule fusion
        fd.sched.schedule()

    fd.schedule = inner_fn
    return fd


def test_matmul_nvf(
    problem_config: tuple,
    schedule_config: tuple,
    disable_validation: bool,
):
    m, n, k, layout = problem_config
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

    with FusionDefinition() as presched_fd:
        matmul_fusion(presched_fd, [a, b])

    scheduled_fd = custom_matmul_scheduler(presched_fd, schedule_config)

    try:
        nvf_outputs = scheduled_fd.execute([a, b], **kwargs)
    except Exception as e:
        print(e)
        return -1

    if not disable_validation:
        eager_output = torch.matmul(a, b)
        assert torch.allclose(nvf_outputs[0], eager_output, atol=1e-2, rtol=1e-2)

    prof = scheduled_fd.profile()
    # convert to microseconds to match pytorch profiler units
    return prof.kernel_profiles[0].time_ms * 1e-3


def main():
    problem_config = (1752, 4720, 584, "NN")

    device_properties = torch.cuda.get_device_properties(0)
    # short-circuit: problem does not fit on device
    if (
        estimate_matmul_size(problem_config, torch.bfloat16)
        >= device_properties.total_memory
    ):
        assert False

    eager_data = analyze_json("gh200_matmul_eager.json")
    eager_result = eager_data[get_layout_enum(problem_config[3])][problem_config[:3]]

    for scheduler_config in itertools.product(splitk_factors, load_stages):
        nvf_result = test_matmul_nvf(
            problem_config, scheduler_config, disable_validation=True
        )
        normalized_result = eager_result / nvf_result
        print(
            f"{eager_result: .3e} out of {nvf_result: 3e} is {normalized_result: 2f}."
        )


# NOTE Scheduler _matmul_ ***rejected*** because : MatmulOp and LinearOp fusion
# is disabled by default. Enable it using NVFUSER_ENABLE=fuse_matmul
# Use CMD:
# NVFUSER_ENABLE=fuse_matmul NVFUSER_DISABLE=matmul_expr_eval python single_matmul.py
if __name__ == "__main__":
    main()
