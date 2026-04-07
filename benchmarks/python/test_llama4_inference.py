# SPDX-FileCopyrightText: Copyright (c) 2026-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from .benchmark_inference import (
    InferenceBenchmarkConfig,
    InferenceBenchmark,
    _register_nvfp4_ops,
)
from nvfuser_direct.pytorch_utils import DEVICE_PROPERTIES


# NOTE: for some reason, changing the order of nvfp4 and cudagraph parameter breaks thunder benchmark. I suspect there's something with thunder's cache.
@pytest.mark.parametrize("input_length", [4096])
@pytest.mark.parametrize("output_length", [4])
@pytest.mark.parametrize("mode", ["thunder", "inductor"])
@pytest.mark.parametrize("enable_nvfp4", [True, False])
@pytest.mark.parametrize("enable_cudagraph", [False, True])
def test_llama4_inference_benchmark(
    benchmark,
    input_length: int,
    output_length: int,
    mode: str,
    enable_nvfp4: bool,
    enable_cudagraph: bool,
):
    if mode == "inductor" and enable_nvfp4:
        pytest.skip("nvfp4 is not supported by inductor yet.")

    if mode == "thunder" and enable_nvfp4 and enable_cudagraph:
        pytest.skip("FIXME: nvfp4 and cudagraph doesn't work together.")

    if DEVICE_PROPERTIES["gpu_compute_capability_major"] < 10:
        if enable_nvfp4:
            pytest.skip("nvfp4 support requires compute_capability >= 10.0")
        if mode == "thunder" and enable_cudagraph:
            pytest.skip("cudagraph doesn't support grouped matmul")

    if enable_nvfp4:
        _register_nvfp4_ops()

    config = InferenceBenchmarkConfig(
        model_name="meta-llama/Llama-4-Maverick-17B-128E",
        batch_size=1,
        input_length=input_length,
        output_length=output_length,
        num_layers=2,
        num_iterations=5,
        warmup_iterations=5,
        mode=mode,
        enable_nvfp4=enable_nvfp4,
        fx_report_folder=None,
        enable_nv_linear=True,
        disable_moe_replacement=False,
        attn_implementation=None,
        thunder_cache=None,
        enable_cudagraph=enable_cudagraph,
        debug_moe=False,
        use_hardcoded_model=True,
    )
    inference_benchmark = InferenceBenchmark(config)

    inference_benchmark.run_benchmark()
    inference_benchmark.print_results()
