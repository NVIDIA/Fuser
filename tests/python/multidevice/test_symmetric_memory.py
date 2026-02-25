# SPDX-FileCopyrightText: Copyright (c) 2026-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: BSD-3-Clause

import pytest

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed.tensor import distribute_tensor, Shard


@pytest.mark.mpi
def test_allgather_linear_symmetric_memory(setup_default_process_group):
    """Allgather input into symmetric buffer, then linear. Same sizes as row_parallel_linear_forward_reference."""
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
    inp_shard = distribute_tensor(inp_ref, mesh, placements=[Shard(-1)]).to_local()
    inp_shard = inp_shard.cuda()
    weight_ref_cuda = weight_ref.cuda()

    # Symmetric buffer for allgathered input: (t, h*4). Prefer symm_mem multimem allgather
    # (NVLink SHARP) when output has multicast support; else fall back to dist.all_gather.
    allgathered_symm = symm_mem.empty(t, h * 4, device="cuda", dtype=inp_shard.dtype)
    symm_mem.rendezvous(allgathered_symm, group=dist.group.WORLD)

    try:
        group_name = dist.group.WORLD.group_name
        torch.ops.symm_mem.multimem_all_gather_out(
            inp_shard, group_name, allgathered_symm
        )
    except RuntimeError as e:
        if "multicast" in str(e).lower():
            tensor_list = [torch.empty_like(inp_shard) for _ in range(d)]
            dist.all_gather(tensor_list, inp_shard, group=dist.group.WORLD)
            allgathered_symm.copy_(torch.cat(tensor_list, dim=1))
        else:
            raise
    out = torch.nn.functional.linear(allgathered_symm, weight_ref_cuda)

    torch.testing.assert_close(out.cpu(), out_ref)


@pytest.mark.mpi
def test_linear_reducescatter_symmetric_memory(setup_default_process_group):
    """Partial linear per rank, then reduce_scatter into symmetric buffer. Same sizes as row_parallel_linear_forward_reference."""
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
    inp_shard = (
        distribute_tensor(inp_ref, mesh, placements=[Shard(-1)]).to_local().cuda()
    )
    weight_shard = (
        distribute_tensor(weight_ref, mesh, placements=[Shard(-1)]).to_local().cuda()
    )

    # Partial linear per rank: (t, (h*4)//d) @ (h, (h*4)//d).T -> (t, h)
    partial = torch.matmul(inp_shard, weight_shard.T)

    # reduce_scatter_tensor splits along dim 0; we want to scatter along last dim (columns).
    # So transpose: (t, h) -> (h, t), then reduce_scatter -> (1, t) per rank, then transpose -> (t, 1).
    partial_t = partial.T.contiguous()  # (h, t)
    out_buf = torch.empty(h // d, t, device="cuda", dtype=partial.dtype)
    dist.reduce_scatter_tensor(
        out_buf, partial_t, op=dist.ReduceOp.SUM, group=dist.group.WORLD
    )
    out_buf = out_buf.T  # (t, h//d)

    out_shard = symm_mem.empty(t, h // d, device="cuda", dtype=partial.dtype)
    symm_mem.rendezvous(out_shard, group=dist.group.WORLD)
    out_shard.copy_(out_buf)

    out_ref = torch.nn.functional.linear(inp_ref.cuda(), weight_ref.cuda()).cpu()
    out_ref_shard = (
        distribute_tensor(out_ref, mesh, placements=[Shard(-1)]).to_local().cuda()
    )
    torch.testing.assert_close(out_shard, out_ref_shard)
