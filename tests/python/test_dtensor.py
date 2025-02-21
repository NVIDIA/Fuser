# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import multidevice_fixtures
import nvfuser
import pytest
import torch
import torch.distributed as dist
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from nvfuser import DataType, FusionDefinition
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Placement, Shard, Replicate
from typing import Callable, cast


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


class FusionDefinitionWrapper:
    def __init__(self, define_fusion: Callable[[FusionDefinition], None]):
        """Wraps a function that defines a fusion without `multidevice_schedule`."""
        self._define_fusion = define_fusion

    def _create_fusion_definition(
        self, in_dtensors: Iterable[DTensor]
    ) -> FusionDefinition:
        define_fn = self._define_fusion

        class Model(FusionDefinition):
            def definition(self) -> None:
                define_fn(self)

            def _find_tensor_by_index(self, index: int) -> nvfuser.Tensor:
                for t in self.sched.tensors():
                    if t.index == index:
                        return t
                return None

            def multidevice_schedule(self) -> None:
                for in_tensor_index, in_dtensor in zip(self.inputs(), in_dtensors):
                    in_tensor = self._find_tensor_by_index(in_tensor_index)

                    # Set the device mesh.
                    assert (
                        in_dtensor.device_mesh.ndim == 1
                    ), "nvFuser's Python API only supports 1D meshes."
                    mesh = nvfuser.DeviceMesh(in_dtensor.device_mesh.mesh.tolist())

                    self.sched._set_device_mesh(in_tensor, mesh)

                    # Parallelize.
                    assert len(in_dtensor.placements) == 1, "Expect a 1D mesh"
                    placement: Placement = in_dtensor.placements[0]
                    if placement.is_shard():
                        dim = cast(Shard, placement).dim
                        self.sched.parallelize(
                            in_tensor, dim, nvfuser.ParallelType.mesh_x
                        )

        return Model()

    def __call__(self, in_dtensors: Iterable[DTensor]) -> list[DTensor]:
        fusion_def = self._create_fusion_definition(in_dtensors)

        in_tensors = [in_dtensor.to_local() for in_dtensor in in_dtensors]
        out_tensors = fusion_def.execute(in_tensors)

        for i, out_tensor in enumerate(out_tensors):
            if isinstance(out_tensor, nvfuser.DistributedTensor):
                mesh = dist.device_mesh.init_device_mesh(
                    "cuda", (out_tensor.mesh.size,)
                )
                placements: list[Placement] = []
                for parallel_type in [nvfuser.ParallelType.mesh_x]:
                    axis: int = out_tensor.axis_sharded_on(parallel_type)
                    placements.append(Replicate() if axis == -1 else Shard(axis))
                out_tensors[i] = DTensor.from_local(out_tensor.local, mesh, placements)
        return out_tensors


@pytest.mark.mpi
def test_plus_one(setup_process_group):
    def define_fusion(fd: FusionDefinition):
        inp = fd.define_tensor(
            (-1, -1), contiguity=(False, False), dtype=DataType.Float
        )
        one = fd.define_scalar(1.0, dtype=DataType.Float)
        out = fd.ops.add(inp, one)
        fd.add_output(out)

    op = FusionDefinitionWrapper(define_fusion)

    num_devices = dist.get_world_size()
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    in_tensor = torch.randn(num_devices, 4)
    mesh = dist.device_mesh.init_device_mesh("cuda", [num_devices])
    in_dtensor = dist.tensor.distribute_tensor(in_tensor, mesh, [Shard(0)])

    (out_dtensor,) = op([in_dtensor])
    torch.testing.assert_close(out_dtensor.to_local(), in_dtensor.to_local() + 1)
    assert out_dtensor.device_mesh == in_dtensor.device_mesh
    assert out_dtensor.placements == in_dtensor.placements


@pytest.mark.mpi
def test_linear(setup_process_group):
    @dataclass
    class LinearConfig:
        def __init__(self, num_devices: int, batch: int, sequence: int, hidden: int):
            self.d = num_devices
            self.b = batch
            self.s = sequence
            self.e = hidden

    def define_linear_forward(config: LinearConfig, fd: FusionDefinition) -> None:
        d, b, s, e = config.d, config.b, config.s, config.e
        inp = fd.define_tensor([b, s, e])
        weight = fd.define_tensor([d, e, e], contiguity=[True, True, True])
        bias = fd.define_tensor([d, e], contiguity=[True, True])
        out = fd.ops.linear(inp, weight, bias)
        fd.add_output(out)

    def define_linear_backward(config: LinearConfig, fd: FusionDefinition) -> None:
        d, b, s, e = config.d, config.b, config.s, config.e
        x = fd.define_tensor([b, s, e])
        x = fd.ops.reshape(x, [b * s, e])
        w = fd.define_tensor([d, e, e], contiguity=True)
        grad = fd.define_tensor([d, b, s, e], contiguity=True)
        grad = fd.ops.reshape(grad, [d, b * s, e])

        grad_x_partials = fd.ops.matmul(grad, w)
        grad_x = fd.ops.sum(grad_x_partials, [0])  # all reduce
        grad_t = fd.ops.permute(grad, [0, 2, 1])
        grad_w = fd.ops.matmul(grad_t, x)
        grad_b = fd.ops.sum(grad, [1])

        grad_x = fd.ops.reshape(grad_x, [b, s, e])
        fd.add_output(grad_x)
        fd.add_output(grad_w)
        fd.add_output(grad_b)

    class LinearFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            input: DTensor,
            weight: DTensor,
            bias: DTensor,
        ):
            b, s, e = input._local_tensor.shape
            d = weight.device_mesh.size()
            op = FusionDefinitionWrapper(
                partial(define_linear_forward, LinearConfig(d, b, s, e))
            )
            outputs = op([input, weight, bias])
            ctx.save_for_backward(input, weight)
            return outputs[0]

        @staticmethod
        def backward(ctx, grad_output: DTensor):
            d, b, s, e = grad_output.shape
            op = FusionDefinitionWrapper(
                partial(define_linear_backward, LinearConfig(d, b, s, e))
            )
            input, weight = ctx.saved_tensors
            outputs = op([input, weight, grad_output])
            assert len(outputs) == 3
            return (*outputs,)

    d, b, s, e = dist.get_world_size(), 2, 1024, 768
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    mesh = dist.device_mesh.init_device_mesh("cuda", [d])

    inp_tensor = torch.randn(b, s, e, device="cuda", requires_grad=True)
    weight_tensor = torch.randn(d, e, e, device="cuda", requires_grad=True)
    bias_tensor = torch.randn(d, e, device="cuda", requires_grad=True)

    inp_dtensor = dist.tensor.distribute_tensor(inp_tensor, mesh, [Replicate()])
    weight_dtensor = dist.tensor.distribute_tensor(weight_tensor, mesh, [Shard(0)])
    bias_dtensor = dist.tensor.distribute_tensor(bias_tensor, mesh, [Shard(0)])

    # expected forward
    unsharded_out_tensor = torch.nn.functional.linear(
        inp_tensor, weight_tensor.view([d * e, e]), bias_tensor.view([d * e])
    )
    expected_out_tensor = unsharded_out_tensor.view([b, s, d, e]).permute(2, 0, 1, 3)[
        rank : rank + 1
    ]

    # multidevice forward
    out_dtensor = LinearFunction.apply(inp_dtensor, weight_dtensor, bias_dtensor)

    # expected backward
    (expected_grad_x, expected_grad_w, expected_grad_b) = torch.autograd.grad(
        unsharded_out_tensor,
        (inp_tensor, weight_tensor, bias_tensor),
        torch.ones_like(unsharded_out_tensor),
    )

    # multidevice backward
    (grad_x, grad_w, grad_b) = torch.autograd.grad(
        out_dtensor,
        (inp_dtensor, weight_dtensor, bias_dtensor),
        torch.ones_like(out_dtensor),
    )

    def assert_close(expected_tensor, dtensor):
        torch.testing.assert_close(
            expected_tensor, dtensor.to_local(), rtol=1.3e-6, atol=1e-3
        )

    assert_close(expected_out_tensor, out_dtensor)
    assert_close(expected_grad_x, grad_x)
    assert_close(expected_grad_w[rank : rank + 1], grad_w)
    assert_close(expected_grad_b[rank : rank + 1], grad_b)
