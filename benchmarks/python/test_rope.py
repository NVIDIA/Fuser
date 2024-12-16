# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
from nvfuser import FusionDefinition, DataType
from .core import run_benchmark, with_executor, unary_bwd_torch
import torch

from functools import partial

# Mimic the Hugging Face implementation:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L216
def rope_with_cat_fusion(
    fd: FusionDefinition,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    features_per_head: int,
) -> None:
    q = fd.define_tensor(
        shape=[batch_size, seq_len, num_heads, features_per_head],
        dtype=DataType.BFloat16,
    )
    cos = fd.define_tensor(
        shape=[seq_len, features_per_head],
        dtype=DataType.BFloat16,
    )
    sin = fd.define_tensor(
        shape=[seq_len, features_per_head],
        dtype=DataType.BFloat16,
    )

    q = fd.ops.permute(q, dims=[0, 2, 1, 3])
    q_real = fd.ops.slice(
        q,
        start_indices=[0, 0, 0, 0],
        end_indices=[batch_size, num_heads, seq_len, features_per_head // 2],
        strides=[1, 1, 1, 1],
    )
    q_image = fd.ops.slice(
        q,
        start_indices=[0, 0, 0, features_per_head // 2],
        end_indices=[batch_size, num_heads, seq_len, features_per_head],
        strides=[1, 1, 1, 1],
    )

    # nvFuser has problems generating negation for bfloat.
    q_image = fd.ops.cast(q_image, dtype=DataType.Float)
    q_image = -q_image
    q_image = fd.ops.cast(q_image, dtype=DataType.BFloat16)

    q_rotated = fd.ops.cat([q_image, q_real], dim=-1)

    cos = fd.ops.broadcast_in_dim(
        cos, shape=[1, 1, seq_len, features_per_head], broadcast_dims=[2, 3]
    )
    sin = fd.ops.broadcast_in_dim(
        sin, shape=[1, 1, seq_len, features_per_head], broadcast_dims=[2, 3]
    )

    out = q * cos + q_rotated * sin
    out = fd.ops.cast(out, DataType.BFloat16)
    fd.add_output(out)


# Idea from @nikitaved: we split and concatenate the embeddings instead of `q`.
# The embeddings are constant that can be precomputed. So we pay the overhead
# of split and concatenation only once. The actual forward pass is merely
# elementwise+reduction surrounded by some meta ops.
def rope_without_cat_fusion(
    fd: FusionDefinition,
    batch_size: int,  # B
    seq_len: int,  # S
    num_heads: int,  # H
    features_per_head: int,  # F
) -> None:
    q = fd.define_tensor(
        shape=[batch_size, seq_len, num_heads, features_per_head],
        dtype=DataType.BFloat16,
    )
    # `cos_sin_matrix` is essentially a batch (of size S*F/2) of 2x2 matrices
    # laid out in a special way to keep computation simple.
    #
    # Using the notations in Figure 1 in https://arxiv.org/pdf/2104.09864.pdf,
    # cos_sin_matrix[0] contains the following:
    #
    #   cos(θ_1),   -sin(θ1)
    #   cos(θ_2),   -sin(θ2)
    #   ...
    #   cos(θ_F/2), -sin(θ_F/2)
    #   ------------------------
    #   sin(θ_1),   cos(θ_1)
    #   sin(θ_2),   cos(θ_2)
    #   ...
    #   sin(θ_F/2), cos(θ_F/2)
    #
    # cos_sin_matrix[i] is similar but each θ is multiplied by `i+1`.
    cos_sin_matrix = fd.define_tensor(
        shape=[seq_len, 2, features_per_head // 2, 2],
        dtype=DataType.BFloat16,
    )

    q = fd.ops.reshape(
        q, new_shape=[batch_size, seq_len, num_heads, 2, features_per_head // 2]
    )
    q = fd.ops.permute(q, dims=[0, 2, 1, 4, 3])
    q = fd.ops.broadcast_in_dim(
        q,
        shape=[batch_size, num_heads, seq_len, 1, features_per_head // 2, 2],
        broadcast_dims=[0, 1, 2, 4, 5],
    )

    cos_sin_matrix = fd.ops.broadcast_in_dim(
        cos_sin_matrix,
        shape=[batch_size, num_heads, seq_len, 2, features_per_head // 2, 2],
        broadcast_dims=[2, 3, 4, 5],
    )

    out = fd.ops.sum(q * cos_sin_matrix, [-1])
    out = fd.ops.cast(out, DataType.BFloat16)
    out = fd.ops.reshape(
        out, new_shape=[batch_size, num_heads, seq_len, features_per_head]
    )
    fd.add_output(out)


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


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    if cos.dim() > 1:
        # batch dimensions must align
        # sin/cos are (B, T, hs) so we unsqeeze -3 for nh
        # we count from back because all of apply_rope does
        cos = cos.unsqueeze(-3)
        sin = sin.unsqueeze(-3)

    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)


def llama_hf_rope(config_str):

    class Config:
        def __init__(self, n_head, head_size, n_query_groups, rope_n_elem, batches, seq_length):
            self.n_head = n_head
            self.head_size = head_size
            self.n_query_groups = n_query_groups
            self.rope_n_elem = rope_n_elem
            self.batches = batches
            self.seq_length = seq_length
    
    
    class LitGPTRope(torch.nn.Module):
        def __init__(self, config) :
            super(LitGPTRope, self).__init__()
            self.config = config
    
        def forward(self, qkv, cos, sin):
            B, T, _ = qkv.size()
            # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
            q_per_kv = self.config.n_head // self.config.n_query_groups
            total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
            qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size)
            qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)
    
            # split batched computation into three
            q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)
    
            # maybe repeat k and v if for the non multi-head attention cases
            # training: flash attention requires it
            # inference: multi-query would require a full kv cache so avoid it to limit its memory usage
            # if self.config.n_query_groups != self.config.n_head and (input_pos is None or self.config.n_query_groups != 1):
            if self.config.n_query_groups != self.config.n_head and (self.config.n_query_groups != 1):
                k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
                v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
    
            q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
            k = k.reshape(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
            v = v.reshape(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)
    
            q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
            k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
            q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
            k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)
            return q, k
    
    configs = {}
    configs["llama_2_7b_hf_rope"] = Config(n_head=32, head_size=128, n_query_groups=32, rope_n_elem=128, batches=2, seq_length=4096)
    configs["llama_3_8B_rope"] = Config(n_head=32, head_size=128, n_query_groups=8, rope_n_elem=128, batches=2, seq_length=8192)

    cfg = configs[config_str]

    def inputs():
        qkv = torch.randn(cfg.batches, cfg.seq_length, cfg.head_size * (cfg.n_head + 2 * cfg.n_query_groups), device='cuda', dtype=torch.bfloat16, requires_grad=True)
        cos = torch.randn(cfg.seq_length, cfg.rope_n_elem, device='cuda', dtype=torch.bfloat16, requires_grad=False)
        sin = torch.randn(cfg.seq_length, cfg.rope_n_elem, device='cuda', dtype=torch.bfloat16, requires_grad=False)
        return qkv, cos, sin

    def grads():
        grad = torch.randn(cfg.batches, cfg.n_head, cfg.seq_length, cfg.head_size, device='cuda', dtype=torch.bfloat16, requires_grad=False)
        return grad

    return LitGPTRope(cfg).cuda().bfloat16(), inputs, grads


# { 'name_benchmark' : (fn, [[sizes0, optional_strides0, dtype0], [sizes1, dtype1], ...]) }
rope_setup = {
    "llama_2_7b_hf_rope":
        partial(llama_hf_rope, config_str="llama_2_7b_hf_rope"),
    "llama_3_8B_rope":
        partial(llama_hf_rope, config_str="llama_3_8B_rope"),
    # "hf_qwen2_rope": (
    #     hf_qwen2_rope,
    #     [
    #         ((1, 32768, 3584), torch.bfloat16),
    #         ((1, 32768, 512), torch.bfloat16),
    #         ((1, 32768, 512), torch.bfloat16),
    #         ((1, 32768, 128), torch.bfloat16),
    #         ((1, 32768, 128), torch.bfloat16),
    #     ],
    # ),
    # "hf_phi3_rope": (
    #     hf_phi3_rope,
    #     [
    #         ((2, 4096, 9216), torch.bfloat16),
    #         ((48,), torch.bfloat16),
    #         ((1, 4096), torch.int64),
    #     ],
    # ),
    # "hf_mistral_nemo_rope": (
    #     hf_mistral_nemo_rope,
    #     [
    #         ((1, 128000, 4096), torch.bfloat16),
    #         ((1, 128000, 1024), torch.bfloat16),
    #         ((1, 128000, 1024), torch.bfloat16),
    #         ((64,), torch.bfloat16),
    #         ((1, 128000), torch.int64),
    #     ],
    # ),
}


@pytest.mark.parametrize(
    "rope_variation",
    [
        "llama_2_7b_hf_rope",
        "llama_3_8B_rope",
        #"hf_qwen2_rope",
        #"hf_phi3_rope",
        #"hf_mistral_nemo_rope",
    ],
)
@pytest.mark.parametrize("executor", ["eager", "torchcompile", "thunder"])
def test_rope_variations_fwd_benchmark(
    benchmark,
    rope_variation: str,
    executor: str,
):
    if executor == "torchcompile":
        clear_dynamo_cache()

    model, inputs, _ = rope_setup[rope_variation]()

    def fwd_call(inp):
        return model(*inp)

    benchmark_fn = with_executor(executor, fwd_call)
    run_benchmark(benchmark, benchmark_fn, inputs())


@pytest.mark.parametrize(
    "rope_variation",
    [
        "llama_2_7b_hf_rope",
        "llama_3_8B_rope",
        #"hf_qwen2_rope",
        #"hf_phi3_rope",
        #"hf_mistral_nemo_rope",
    ],
)
@pytest.mark.parametrize("executor", ["eager", "torchcompile", "thunder"])
def test_rope_variations_bwd_benchmark(
    benchmark,
    rope_variation: str,
    executor: str,
):
    if executor == "torchcompile":
        clear_dynamo_cache()

    # TODO why not just a random like for grad on output instead of returning a grad function
    model, fwd_inputs, grad = rope_setup[rope_variation]()

    def fwd_call(inp):
        return model(*inp)

    # execute the compiled fwd fn
    fwd_fn = with_executor(executor, fwd_call)
    outputs = fwd_fn(fwd_inputs())

    # NOTE does this look about right?
    output = outputs[0]
    for i in range(1, len(outputs)):
        output += outputs[i]

    benchmark_fn = with_executor(executor, fwd_call)
    # FIXME fix the bytes computation!
    run_benchmark(benchmark, unary_bwd_torch, [output, grad()], iobytes=10)
