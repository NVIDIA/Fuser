# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
import torch.distributed as dist


# Partitions n into k integers whose sum is n.
def random_partition(n: int, k: int) -> torch.Tensor:
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
    return torch.tensor(sizes, device="cuda")


def list_to_matrix(l: list, rows: int, cols: int) -> list[list]:
    return [l[row * cols : (row + 1) * cols] for row in range(rows)]


def matrix_to_list(m: list[list]) -> list:
    return [item for row in m for item in row]


def rank_first_to_expert_first(
    tokens: torch.Tensor, n_tokens_per_expert_from_rank: torch.Tensor
) -> torch.Tensor:
    d, num_experts_per_rank = n_tokens_per_expert_from_rank.size()
    tokens_per_expert_from_rank = tokens.split(
        n_tokens_per_expert_from_rank.flatten().tolist()
    )
    tokens_per_expert_from_rank = list_to_matrix(
        tokens_per_expert_from_rank, d, num_experts_per_rank
    )
    tokens_per_expert_from_rank = list(zip(*tokens_per_expert_from_rank))
    tokens_per_expert_from_rank = matrix_to_list(tokens_per_expert_from_rank)
    tokens = torch.cat(tokens_per_expert_from_rank, dim=0)
    return tokens


@pytest.mark.mpi
def test_dispatch(setup_default_process_group):
    rank = dist.get_rank()
    torch.manual_seed(rank)

    d = dist.get_world_size()
    num_experts_per_rank = 2
    n_experts = d * num_experts_per_rank

    n_tokens = 12
    n_tokens_per_expert = random_partition(n_tokens, n_experts)

    n_tokens_per_expert_from_rank = torch.empty(
        d, num_experts_per_rank, dtype=torch.int64, device="cuda"
    )
    dist.all_to_all_single(n_tokens_per_expert_from_rank, n_tokens_per_expert)

    # Number of tokens to receive from each rank
    n_tokens_from_rank = n_tokens_per_expert_from_rank.sum(-1)

    # For illustration, input tokens are complex numbers of value `token_id + expert_id * j`.
    expert_ids = []
    for expert, n in enumerate(n_tokens_per_expert.tolist()):
        expert_ids.extend([expert] * n)
    inp = (
        torch.arange(
            rank * n_tokens, (rank + 1) * n_tokens, dtype=torch.float, device="cuda"
        )
        + torch.tensor(expert_ids, dtype=torch.float, device="cuda") * 1j
    )

    # Number of tokens to send to each rank
    n_tokens_per_rank = n_tokens_per_expert.view(d, -1).sum(-1)
    out = torch.empty(n_tokens_from_rank.sum(), dtype=torch.complex64, device="cuda")
    dist.all_to_all_single(
        out, inp, n_tokens_from_rank.tolist(), n_tokens_per_rank.tolist()
    )

    out = rank_first_to_expert_first(out, n_tokens_per_expert_from_rank)

    imag = torch.imag(out)
    assert (imag >= num_experts_per_rank * rank).all()
    assert (imag < num_experts_per_rank * (rank + 1)).all()
    assert (imag[:-1] <= imag[1:]).all()
