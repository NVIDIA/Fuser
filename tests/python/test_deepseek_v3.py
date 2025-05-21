# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import multidevice_fixtures
import pytest
import transformers
import torch
import torch.distributed as dist
import torch.utils._pytree as pytree
from contextlib import contextmanager
from enum import auto, Enum
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import (
    parallelize_module,
    RowwiseParallel,
    ColwiseParallel,
)


multidevice_test = multidevice_fixtures.multidevice_test


@pytest.fixture(scope="module")
def setup_process_group(multidevice_test):
    # The default port as used by https://github.com/pytorch/pytorch/blob/45a8b5682eb69d865cbf68c7f2f689b56b4efd53/torch/csrc/distributed/c10d/TCPStore.hpp#L51.
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://localhost:29500",
        world_size=multidevice_test.size,
        rank=multidevice_test.rank,
    )
    yield
    dist.destroy_process_group()


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


@contextmanager
def cuda_profiler():
    torch.cuda.cudart().cudaProfilerStart()
    try:
        yield
    finally:
        torch.cuda.cudart().cudaProfilerStop()


class ComputeType(Enum):
    FORWARD = auto()
    BACKWARD = auto()
    INFERENCE = auto()


@pytest.mark.mpi
@pytest.mark.parametrize(
    "compute_type",
    [ComputeType.FORWARD, ComputeType.BACKWARD, ComputeType.INFERENCE],
    ids=["forward", "backward", "inference"],
)
def test_transformer_layer(setup_process_group, compute_type: ComputeType):
    if compute_type != ComputeType.INFERENCE:
        pytest.xfail(
            """
Training is unavailable as is (cf.
https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py#L439).
However, you can apply the following patch in your local cache (typically)
~/.cache/huggingface/modules/transformers_modules/deepseek-ai/deepseek-v3/ to
work around that limitation.

```diff
--- a/modeling_deepseek.py 2025-03-12 13:23:40.817961783 -0700
+++ b/modeling_deepseek.py  2025-03-12 13:22:38.499791931 -0700
@@ -436,7 +436,6 @@

         ### select top-k experts
         if self.topk_method == "noaux_tc":
-            assert not self.training
             scores_for_choice = scores.view(bsz * seq_len, -1) + self.e_score_correction_bias.unsqueeze(0)
             group_scores = (
                 scores_for_choice.view(bsz * seq_len, self.n_group, -1).topk(2, dim=-1)[0].sum(dim = -1)
@@ -526,13 +525,11 @@
         topk_idx, topk_weight = self.gate(hidden_states)
         hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
         flat_topk_idx = topk_idx.view(-1)
-        if not self.training:
-            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
+        y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
         if self.config.n_shared_experts is not None:
             y = y + self.shared_experts(identity)
         return y

-    @torch.no_grad()
     def moe_infer(self, x, topk_ids, topk_weight):
         cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
         cnts.scatter_(1, topk_ids, 1)
```
"""
        )

    config = transformers.AutoConfig.from_pretrained(
        "deepseek-ai/deepseek-v3", trust_remote_code=True
    )

    # Create only one layer which is sufficient for the test.
    config.num_hidden_layers = 1
    # Without this, the first and only layer will have a dense MLP instead of MoE.
    config.first_k_dense_replace = 0
    # Disable quantization so the test can run on A100 and is made easier for nvFuser.
    delattr(config, "quantization_config")

    d = dist.get_world_size()
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    mesh = dist.device_mesh.init_device_mesh("cuda", [d])

    with default_tensor_type(dtype=config.torch_dtype, device="cuda"):
        model = transformers.AutoModel.from_config(config, trust_remote_code=True)
        if compute_type == ComputeType.INFERENCE:
            model.eval()
        else:
            model.train()

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

        def print_size_hook(name):
            def hook(module, inp, out):
                def get_size_and_strides(x):
                    if isinstance(x, DTensor):
                        return get_size_and_strides(x.to_local())

                    if isinstance(x, torch.Tensor):
                        return x.size(), x.stride()

                    return x

                inp_sizes = pytree.tree_map(get_size_and_strides, inp)
                weight_sizes = pytree.tree_map(get_size_and_strides, module.weight)
                out_sizes = pytree.tree_map(get_size_and_strides, out)
                print(
                    f"{name}: inp = {inp_sizes}, weight = {weight_sizes}, out = {out_sizes}"
                )

            return hook

        for name, module in transformer_layer.named_modules():
            if isinstance(module, torch.nn.Linear):
                module.register_forward_hook(print_size_hook(name))

        batch_size = 1
        seq_len = 2048
        inp = torch.randn(batch_size, seq_len, config.hidden_size)
        if compute_type == ComputeType.BACKWARD:
            inp.requires_grad_()
        mask = transformers.modeling_attn_mask_utils._prepare_4d_causal_attention_mask(
            None, [batch_size, seq_len], inp, past_key_values_length=0
        )
        (out,) = transformer_layer(inp, attention_mask=mask)

        assert out.size() == (batch_size, seq_len, config.hidden_size)
        assert out.dtype == config.torch_dtype
        assert out.is_cuda

        if compute_type == ComputeType.BACKWARD:
            out.sum().backward()

            assert inp.grad.size() == (batch_size, seq_len, config.hidden_size)
            assert inp.grad.dtype == config.torch_dtype
            assert inp.grad.is_cuda
