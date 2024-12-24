# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from nvfuser import FusionDefinition
from .core import run_benchmark, with_executor, unary_bwd_torch, clear_dynamo_cache
import torch

from .rope_ops import rope_with_cat_fusion, rope_without_cat_fusion, rope_setup


@pytest.mark.parametrize("use_cat", [True, False], ids=["with_cat", "without_cat"])
def test_rope_benchmark(
    benchmark, use_cat: bool, disable_validation: bool, disable_benchmarking: bool
):
    batch_size = 32
    seq_len = 4096
    num_heads = 32
    features_per_head = 128

    # torch.manual_seed(0)
    q = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        features_per_head,
        dtype=torch.bfloat16,
        device="cuda:0",
    )
    freqs = torch.randn(
        seq_len, features_per_head // 2, dtype=torch.bfloat16, device="cuda:0"
    )
    cos = freqs.cos()
    sin = freqs.sin()

    if use_cat:
        with FusionDefinition() as fd:
            rope_with_cat_fusion(fd, batch_size, seq_len, num_heads, features_per_head)
        inputs = [q, torch.cat([cos, cos], dim=-1), torch.cat([sin, sin], dim=-1)]
    else:
        with FusionDefinition() as fd:
            rope_without_cat_fusion(
                fd, batch_size, seq_len, num_heads, features_per_head
            )
        # [S, F/2, 2]
        cos_and_minus_sin = torch.stack([cos, -sin], dim=-1)
        # [S, F/2, 2]
        sin_and_cos = torch.stack([sin, cos], dim=-1)
        # [S, 2, F/2, 2]
        cos_sin_matrix = torch.stack([cos_and_minus_sin, sin_and_cos], dim=1)
        inputs = [q, cos_sin_matrix]

    if not disable_validation:
        q_real, q_image = q.permute([0, 2, 1, 3]).split(features_per_head // 2, dim=-1)
        q_real = q_real.to(torch.float32)
        q_image = q_image.to(torch.float32)
        ref_out = torch.cat(
            [q_real * cos - q_image * sin, q_image * cos + q_real * sin], dim=-1
        ).to(torch.bfloat16)
        nvf_out = fd.execute(inputs)
        torch.testing.assert_close(nvf_out, [ref_out], atol=0, rtol=0)

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, inputs)


@pytest.mark.parametrize(
    "rope_variation",
    [
        "llama_2_7b_hf_rope",
        "llama_3_8B_rope",
        "hf_qwen2_rope",
        "hf_phi3_rope",
        "hf_mistral_nemo_rope",
    ],
)
@pytest.mark.parametrize("executor", ["eager", "torchcompile", "thunder"])
def test_rope_variations_fwd_benchmark(
    benchmark,
    rope_variation: str,
    executor: str,
):
    kwargs = {}
    if executor == "thunder":
        kwargs["nv_enable_matmul"] = True
    elif executor == "torchcompile":
        clear_dynamo_cache()

    model, inputs, _, _ = rope_setup[rope_variation]()

    def fwd_call(inp):
        return model(*inp)

    # Compile the fwd fn for torchcompile
    benchmark_fn = with_executor(executor, fwd_call, **kwargs)
    run_benchmark(benchmark, benchmark_fn, inputs())


@pytest.mark.parametrize(
    "rope_variation",
    [
        "llama_2_7b_hf_rope",
        "llama_3_8B_rope",
        "hf_qwen2_rope",
        "hf_phi3_rope",
        "hf_mistral_nemo_rope",
    ],
)
@pytest.mark.parametrize("executor", ["eager", "torchcompile", "thunder"])
def test_rope_variations_bwd_benchmark(
    benchmark,
    rope_variation: str,
    executor: str,
):
    kwargs = {}
    if executor == "thunder":
        kwargs["nv_enable_matmul"] = True
    elif executor == "torchcompile":
        clear_dynamo_cache()

    model, fwd_inputs, grad, iobytes = rope_setup[rope_variation]()

    def fwd_call(inp):
        return model(*inp)

    # execute the compiled fwd fn
    fwd_fn = with_executor(executor, fwd_call, **kwargs)
    outputs = fwd_fn(fwd_inputs())

    # accumulate all output, so we can feed a single grad and use the unary bwd function
    output = outputs[0]
    for i in range(1, len(outputs)):
        output += outputs[i]

    # NOTE: the iobytes is computed based on how thunder autograd worked. So this is just
    # a reference point for torchcompile and eager executor for comparison.
    run_benchmark(benchmark, unary_bwd_torch, [output, grad()], iobytes=iobytes())
