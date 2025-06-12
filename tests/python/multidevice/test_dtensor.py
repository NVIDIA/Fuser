# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
import torch.distributed as dist
from enum import Enum, auto
from fusion_definition_wrapper import FusionDefinitionWrapper
from linear import LinearFunction, TensorParallelLinear
from nvfuser import DataType, FusionDefinition
from nvfuser.testing.benchmark_utils import get_benchmark_fns
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard, Replicate


@pytest.mark.mpi
def test_plus_one(setup_default_process_group, multidevice_test):
    def define_fusion(fd: FusionDefinition):
        inp = fd.define_tensor((-1, -1), contiguity=False, dtype=DataType.Float)
        one = fd.define_scalar(1.0, dtype=DataType.Float)
        out = fd.ops.add(inp, one)
        fd.add_output(out)

    op = FusionDefinitionWrapper(define_fusion)

    num_devices = dist.get_world_size()

    in_tensor = torch.randn(num_devices, 4)
    mesh = dist.device_mesh.init_device_mesh("cuda", [num_devices])
    in_dtensor = dist.tensor.distribute_tensor(in_tensor, mesh, [Shard(0)])

    (out_dtensor,) = op([in_dtensor])
    torch.testing.assert_close(out_dtensor.to_local(), in_dtensor.to_local() + 1)
    assert out_dtensor.device_mesh == in_dtensor.device_mesh
    assert out_dtensor.placements == in_dtensor.placements


@pytest.mark.mpi
def test_column_parallel_linear(setup_default_process_group, multidevice_test):
    d, b, s, e = dist.get_world_size(), 2, 3, 5

    mesh = dist.device_mesh.init_device_mesh("cuda", [d])

    inp_tensor = torch.randint(
        -4, 4, (b, s, e), dtype=torch.bfloat16, requires_grad=True
    )
    weight_tensor = torch.randint(
        -4, 4, (d * e, e), dtype=torch.bfloat16, requires_grad=True
    )

    inp_dtensor = dist.tensor.distribute_tensor(inp_tensor, mesh, [Replicate()])
    weight_dtensor = dist.tensor.distribute_tensor(weight_tensor, mesh, [Shard(0)])

    def assert_close(expected_tensor: torch.Tensor, dtensor: DTensor):
        torch.testing.assert_close(expected_tensor, dtensor.to_local().cpu())

    out_tensor = torch.nn.functional.linear(inp_tensor, weight_tensor)
    out_dtensor = LinearFunction.apply(inp_dtensor, weight_dtensor)
    rank = dist.get_rank()
    assert_close(out_tensor.split(e, dim=-1)[rank], out_dtensor)

    (expected_grad_x, expected_grad_w) = torch.autograd.grad(
        out_tensor,
        (inp_tensor, weight_tensor),
        torch.ones_like(out_tensor),
    )
    (grad_x, grad_w) = torch.autograd.grad(
        out_dtensor,
        (inp_dtensor, weight_dtensor),
        torch.ones_like(out_dtensor),
    )
    assert_close(expected_grad_x, grad_x)
    assert_close(expected_grad_w.split(e, dim=0)[rank], grad_w)


class Executor(Enum):
    # https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html
    TORCH_TP = auto()
    NVFUSER = auto()


@pytest.mark.mpi
@pytest.mark.parametrize(
    "executor",
    [Executor.TORCH_TP, Executor.NVFUSER],
    ids=lambda e: e.name,
)
def test_row_parallel_linear(
    setup_default_process_group, multidevice_test, executor: Executor
):
    d, s, k, h, h_intermediate = dist.get_world_size(), 2048, 8, 7168, 2048

    mesh = dist.device_mesh.init_device_mesh("cuda", [d])

    inp = torch.randn((s * k, h), dtype=torch.bfloat16)
    inp = DTensor.from_local(inp, mesh, [Replicate()])

    tokens_per_expert = [
        897,
        923,
        858,
        892,
        875,
        872,
        870,
        862,
        847,
        807,
        848,
        2048,
        481,
        148,
        50,
        9,
        1,
        2048,
        2048,
    ]
    assert sum(tokens_per_expert) == s * k

    unsharded_up_expert = torch.nn.Linear(
        h, h_intermediate, dtype=torch.bfloat16, device="cuda"
    )
    unsharded_down_expert = torch.nn.Linear(
        h_intermediate, h, dtype=torch.bfloat16, device="cuda"
    )

    up_experts = []
    down_experts = []
    for _ in tokens_per_expert:
        up_experts.append(
            TensorParallelLinear.distribute(
                unsharded_up_expert, mesh, [Replicate()], [Shard(0)]
            )
        )
        down_experts.append(
            TensorParallelLinear.distribute(
                unsharded_down_expert, mesh, [Shard(-1)], [Shard(-1)]
            )
        )

    def model():
        expert_outs = []
        offset = 0
        for num_tokens, up_expert, down_expert in zip(
            tokens_per_expert, up_experts, down_experts
        ):
            expert_out = inp[offset : offset + num_tokens]
            match executor:
                case Executor.TORCH_TP:
                    expert_out = torch.nn.functional.linear(
                        expert_out, up_expert.weight
                    )
                    expert_out = torch.nn.functional.linear(
                        expert_out, down_expert.weight
                    )
                    dist.all_reduce(expert_out.to_local())
                case Executor.NVFUSER:
                    expert_out = expert_out.to_local()
                    expert_out = up_expert(expert_out)
                    expert_out = down_expert(expert_out)
                    expert_out = DTensor.from_local(expert_out, mesh, [Replicate()])
            expert_outs.append(expert_out)
            offset += num_tokens
        return torch.cat(expert_outs, dim=0)

    warmup_fn, benchmark_fn = get_benchmark_fns(model)
    out_dtensor = warmup_fn()
    assert out_dtensor.size() == (s * k, h)

    for _ in range(5):
        benchmark_fn()
