# SPDX-FileCopyrightText: Copyright (c) 2026-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: BSD-3-Clause

import pytest

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed.tensor import distribute_tensor, Shard
from torch._C._autograd import DeviceType
from torch._C._distributed_c10d import _SymmetricMemory


@pytest.mark.mpi
def test_allgather_linear_symmetric_memory(setup_default_process_group):
    """Allgather input into symmetric buffer, then linear. Same sizes as row_parallel_linear_forward_reference."""
    if not _SymmetricMemory.has_multicast_support(
        DeviceType.CUDA, torch.cuda.current_device()
    ):
        pytest.skip("multicast not supported on this GPU")

    h, s, t = 2, 3, 6
    d = dist.get_world_size()
    if (h * 4) % d != 0:
        pytest.skip(
            f"Row-parallel linear requires {h * 4} to be divisible by world size {d}."
        )
    assert t % s == 0

    torch.manual_seed(0)
    inp_ref = torch.testing.make_tensor(t, h * 4, dtype=torch.int32, device="cpu").to(
        torch.bfloat16
    )
    weight_ref = torch.testing.make_tensor(
        h, h * 4, dtype=torch.int32, device="cpu"
    ).to(torch.bfloat16)
    out_ref = torch.nn.functional.linear(inp_ref.cuda(), weight_ref.cuda()).cpu()

    mesh = dist.device_mesh.init_device_mesh("cuda", [d])
    inp = distribute_tensor(inp_ref, mesh, placements=[Shard(-1)]).to_local()
    weight = weight_ref.cuda()

    # Symmetric buffer for allgathered input: (t, h*4). multimem_all_gather_out requires
    # multicast support (e.g. NVLink SHARP) on the GPU; skip the test otherwise.
    allgathered_symm = symm_mem.empty(t, h * 4, device="cuda", dtype=inp.dtype)
    symm_mem.rendezvous(allgathered_symm, group=dist.group.WORLD)

    group_name = dist.group.WORLD.group_name
    torch.ops.symm_mem.multimem_all_gather_out(inp, group_name, allgathered_symm)
    out = torch.nn.functional.linear(allgathered_symm, weight)

    torch.testing.assert_close(out.cpu(), out_ref)


@pytest.mark.mpi
def test_linear_reducescatter_symmetric_memory(setup_default_process_group):
    """Partial linear per rank, then reduce_scatter into symmetric buffer. Same sizes as row_parallel_linear_forward_reference."""
    if not _SymmetricMemory.has_multicast_support(
        DeviceType.CUDA, torch.cuda.current_device()
    ):
        pytest.skip("multicast not supported on this GPU")

    h, s, t = 2, 3, 6
    d = dist.get_world_size()
    if (h * 4) % d != 0:
        pytest.skip(
            f"Row-parallel linear requires {h * 4} to be divisible by world size {d}."
        )
    if h % d != 0:
        pytest.skip(
            f"Linear+reducescatter requires h={h} to be divisible by world size {d}."
        )
    assert t % s == 0

    torch.manual_seed(0)
    inp_ref = torch.testing.make_tensor(t, h * 4, dtype=torch.int32, device="cpu").to(
        torch.bfloat16
    )
    weight_ref = torch.testing.make_tensor(
        h, h * 4, dtype=torch.int32, device="cpu"
    ).to(torch.bfloat16)

    mesh = dist.device_mesh.init_device_mesh("cuda", [d])
    inp = distribute_tensor(inp_ref, mesh, placements=[Shard(-1)]).to_local()
    weight = distribute_tensor(weight_ref, mesh, placements=[Shard(-1)]).to_local()

    group_name = dist.group.WORLD.group_name
    out = torch.ops.symm_mem.fused_matmul_reduce_scatter(
        inp, weight.T, "sum", 1, group_name
    )

    out_ref = torch.nn.functional.linear(inp_ref.cuda(), weight_ref.cuda()).cpu()
    torch.testing.assert_close(
        out, distribute_tensor(out_ref, mesh, placements=[Shard(-1)]).to_local()
    )
