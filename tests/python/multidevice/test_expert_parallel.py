# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
import torch.distributed as dist


# Partitions n into k integers whose sum is n.
def random_partition(n: int, k: int) -> list[int]:
    offsets = torch.randint(0, n + 1, [k - 1]).tolist()
    # The last partition ends at index n.
    offsets.append(n)
    offsets.sort()

    sizes = []
    prev = 0
    for offset in offsets:
        sizes.append(offset - prev)
        prev = offset
    assert sum(sizes) == n
    assert len(sizes) == k
    return sizes


@pytest.mark.mpi
def test_dispatch(setup_default_process_group):
    rank = dist.get_rank()
    torch.manual_seed(rank)

    d = dist.get_world_size()
    num_experts_per_ep = 2
    n_experts = d * num_experts_per_ep

    n_tokens = 12
    n_tokens_per_expert = random_partition(n_tokens, n_experts)
    # Number of tokens to send to each EP group
    n_tokens_per_ep = (
        torch.tensor(n_tokens_per_expert, device="cuda").view(d, -1).sum(-1)
    )

    # Number of tokens to receive from each EP group
    n_tokens_from_ep = torch.empty(d, dtype=torch.int64, device="cuda")
    dist.all_to_all_single(n_tokens_from_ep, n_tokens_per_ep)

    expert_ids = []
    for expert, n in enumerate(n_tokens_per_expert):
        expert_ids.extend([expert] * n)
    # For illustration, input tokens are complex numbers of value `token_id + expert_id * j`.
    inp = (
        torch.arange(
            rank * n_tokens, (rank + 1) * n_tokens, dtype=torch.float, device="cuda"
        )
        + torch.tensor(expert_ids, dtype=torch.float, device="cuda") * 1j
    )

    out = torch.empty(n_tokens_from_ep.sum(), dtype=torch.complex64, device="cuda")
    dist.all_to_all_single(
        out, inp, n_tokens_from_ep.tolist(), n_tokens_per_ep.tolist()
    )

    assert (torch.imag(out) >= num_experts_per_ep * rank).all()
    assert (torch.imag(out) < num_experts_per_ep * (rank + 1)).all()
