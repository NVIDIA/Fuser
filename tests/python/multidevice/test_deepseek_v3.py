# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import transformers
import torch
import torch.distributed as dist
from contextlib import contextmanager
from functools import wraps
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import (
    parallelize_module,
    RowwiseParallel,
    ColwiseParallel,
)


@contextmanager
def default_tensor_type(dtype=torch.float32, device="cpu"):
    # Save
    prev_dtype = torch.get_default_dtype()
    prev_device = torch.get_default_device()

    # Set
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)

    yield

    # Restore
    torch.set_default_dtype(prev_dtype)
    torch.set_default_device(prev_device)


def download_once(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        rank = dist.get_rank()
        if rank == 0:
            # Download only once.
            result = fn(*args, **kwargs)

        dist.barrier()

        if rank != 0:
            # Other ranks load from cache.
            result = fn(*args, **kwargs)

        return result

    return wrapper


@download_once
def load_config(model_name: str) -> transformers.PretrainedConfig:
    return transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)


@download_once
def load_model(config: transformers.PretrainedConfig) -> transformers.PreTrainedModel:
    return transformers.AutoModel.from_config(config, trust_remote_code=True)


# This test timed out once when downloading
# "/deepseek-ai/DeepSeek-V3/resolve/main/configuration_deepseek.py" (cf.
# http://nv/eCm). I consider this a one-off, but please let me know if this
# error becomes consistent.
@pytest.mark.mpi
def test_transformer_layer(setup_default_process_group):
    config = load_config("deepseek-ai/deepseek-v3")
    # Create only one layer which is sufficient for the test.
    config.num_hidden_layers = 1
    # Without this, the first and only layer will have a dense MLP instead of MoE.
    config.first_k_dense_replace = 0
    # Disable quantization so the test can run on A100 and is made easier for nvFuser.
    delattr(config, "quantization_config")

    # This ensures the input tokens are identically replicated on all ranks.
    # Otherwise, some ranks may skip an expert because they have no tokens to
    # send, while other ranks don't. This will cause a deadlock because a NCCL
    # collective is expected to be called by all ranks in the process group.
    torch.manual_seed(0)

    d = dist.get_world_size()
    mesh = dist.device_mesh.init_device_mesh("cuda", [d])

    with default_tensor_type(dtype=config.torch_dtype, device="cuda"):
        model = load_model(config)
        # Training is unavailable (cf. https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py#L439)
        model.eval()

        transformer_layer = model.layers[0]

        # By default, RowwiseParallel and ColwiseParallel output a local tensor
        # and therefore num_heads needs to be adjusted to accomodate the local
        # size. Alternatively, I could RowwiseParallel(use_local_output=False)
        # so the linear layer outputs a DTensor, which can be viewed using the
        # original num_heads. This requires all activations, parameters, and
        # buffers to be DTensor; otherwise aten ops would complain "got mixed
        # torch.Tensor and DTensor". Doing so is challenging because
        # DeepseekV3RotaryEmbedding creates cos_cached and sin_cached during
        # the first forward call (cf.
        # https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py#L143-L144).
        transformer_layer.self_attn.num_heads //= d

        # Create the parallel plan
        parallel_plan = {
            "self_attn.q_b_proj": ColwiseParallel(),
            "self_attn.kv_b_proj": ColwiseParallel(),
            "self_attn.o_proj": RowwiseParallel(),
        }

        for expert in range(config.n_routed_experts):
            parallel_plan[f"mlp.experts.{expert}.gate_proj"] = ColwiseParallel()
            parallel_plan[f"mlp.experts.{expert}.up_proj"] = ColwiseParallel()
            parallel_plan[f"mlp.experts.{expert}.down_proj"] = RowwiseParallel()

        parallel_plan["mlp.shared_experts.gate_proj"] = ColwiseParallel()
        parallel_plan["mlp.shared_experts.up_proj"] = ColwiseParallel()
        parallel_plan["mlp.shared_experts.down_proj"] = RowwiseParallel()

        transformer_layer = parallelize_module(
            transformer_layer,
            mesh,
            parallel_plan,
        )

        # Sanity-check parameters are indeed distributed
        distributed_params: list[str] = [
            name
            for name, parameter in transformer_layer.named_parameters()
            if isinstance(parameter.data, DTensor)
        ]
        assert len(distributed_params) == 3 + (config.n_routed_experts + 1) * 3

        batch_size = 1
        seq_len = 2048
        inp = torch.randn(batch_size, seq_len, config.hidden_size)
        mask = transformers.modeling_attn_mask_utils._prepare_4d_causal_attention_mask(
            None, [batch_size, seq_len], inp, past_key_values_length=0
        )
        (out,) = transformer_layer(inp, attention_mask=mask)
        # Finish all computation and communication. Otherwise,
        # destroy_process_group may deadlock.
        torch.cuda.synchronize()

        assert out.size() == (batch_size, seq_len, config.hidden_size)
        assert out.dtype == config.torch_dtype
        assert out.is_cuda
