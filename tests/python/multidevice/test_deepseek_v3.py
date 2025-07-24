# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


# Run command:
# mpirun -np 1 pytest tests/python/multidevice/test_deepseek_v3.py --only-mpi -s

import pytest
import transformers
import torch
import torch.distributed as dist
from contextlib import contextmanager
from enum import Enum, auto
from functools import wraps
from linear import TensorParallelLinear
from nvfuser.testing.benchmark_utils import get_benchmark_fns
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ParallelStyle,
    RowwiseParallel,
    ColwiseParallel,
)
from torch.distributed.tensor.placement_types import Shard
from typing import Optional


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


# This decorator ensures that the model/config is downloaded only once by rank
# 0. Other ranks will load from the cache that's stored on the same machine.
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


class Executor(Enum):
    # https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html
    TORCH_TP = auto()
    NVFUSER = auto()


def parallelize_linear_with_nvfuser(
    linear: torch.nn.Linear,
    mesh: dist.device_mesh.DeviceMesh,
    parallel_style: ParallelStyle,
) -> torch.nn.Linear:
    assert isinstance(linear, torch.nn.Linear), f"Unsupported layer: {linear}"

    assert len(parallel_style.input_layouts) == 1, "Expect 1D mesh"
    input_layout = parallel_style.input_layouts[0]

    assert len(parallel_style.output_layouts) == 1, "Expect 1D mesh"
    output_layout = parallel_style.output_layouts[0]

    if isinstance(parallel_style, RowwiseParallel):
        # We only support TP at this moment. A row-wise parallel linear is
        # expected to have the input sharded on the contracting dimension and
        # the output replicated.
        assert input_layout.is_shard(-1), f"Unsupported layout: {input_layout}"
        assert output_layout.is_replicate(), f"Unsupported layout: {output_layout}"
        return TensorParallelLinear.distribute(
            linear, mesh, in_placements=[input_layout], weight_placements=[Shard(-1)]
        )

    if isinstance(parallel_style, ColwiseParallel):
        # We only support TP at this moment. A column-wise parallel linear is
        # expected to have the input replicated and the output sharded on the
        # feature dimension.
        assert input_layout.is_replicate(), f"Unsupported layout: {input_layout}"
        assert output_layout.is_shard(-1), f"Unsupported layout: {output_layout}"
        return TensorParallelLinear.distribute(
            linear, mesh, in_placements=[input_layout], weight_placements=[Shard(0)]
        )

    assert False, f"Unsupported parallel style: {parallel_style}"


# Recursively finds all linear modules and replaces them with tensor-parallel
# nvFuser definitions if a parallel plan is found.
def parallelize_module_with_nvfuser(
    module: torch.nn.Module,
    mesh: dist.device_mesh.DeviceMesh,
    parallel_plan: dict[str, ParallelStyle],
    fqn: str,  # stands for fully qualified name
    parent_module: Optional[torch.nn.Module] = None,
):
    for child_module_name, child_module in module.named_children():
        if fqn:
            child_fqn = f"{fqn}.{child_module_name}"
        else:
            child_fqn = child_module_name

        parallelize_module_with_nvfuser(
            child_module, mesh, parallel_plan, child_fqn, module
        )

    if (parallel_style := parallel_plan.get(fqn)) is None:
        return

    new_module = parallelize_linear_with_nvfuser(module, mesh, parallel_style)
    assert parent_module is not None
    module_name = fqn.split(".")[-1]
    setattr(parent_module, module_name, new_module)


# This test timed out once when downloading
# "/deepseek-ai/DeepSeek-V3/resolve/main/configuration_deepseek.py" (cf.
# http://nv/eCm). I consider this a one-off, but please let me know if this
# error becomes consistent.
@pytest.mark.mpi
@pytest.mark.parametrize(
    "executor",
    [Executor.TORCH_TP, Executor.NVFUSER],
    ids=lambda e: e.name,
)
def test_transformer_layer(setup_default_process_group, benchmark, executor: Executor):
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
        # Loading the model under `device="cuda"` makes weight initialization
        # much faster but requires full GPU memory allocation.
        #
        # Alternatively, I think the following may work but haven't tried it:
        # 1. Load the model under torch.nn.utils.init_empty_weights. This skips weight initialization and allocates full weights on CPU not GPU.
        # 2. parallelize_module
        # 3. Load pre-trained parameters.
        # 4. Move the model to CUDA.
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

        match executor:
            case Executor.TORCH_TP:
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
            case Executor.NVFUSER:
                parallelize_module_with_nvfuser(
                    transformer_layer, mesh, parallel_plan, fqn=""
                )

        batch_size = 1
        seq_len = 2048
        inp = torch.randn(batch_size, seq_len, config.hidden_size)
        mask = transformers.modeling_attn_mask_utils._prepare_4d_causal_attention_mask(
            None, [batch_size, seq_len], inp, past_key_values_length=0
        )
        warmup_fn, benchmark_fn = get_benchmark_fns(
            lambda: transformer_layer(inp, attention_mask=mask)
        )

        (out,) = warmup_fn()
        assert out.size() == (batch_size, seq_len, config.hidden_size)
        assert out.dtype == config.torch_dtype
        assert out.is_cuda

        benchmark.pedantic(benchmark_fn, rounds=5)
