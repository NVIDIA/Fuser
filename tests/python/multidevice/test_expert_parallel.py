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


# This can be fused into one kernel.
def rank_first_to_expert_first(
    tokens: torch.Tensor, n_tokens_for_expert_from_rank: torch.Tensor
) -> torch.Tensor:
    d, num_experts_per_rank = n_tokens_for_expert_from_rank.size()
    # Rank first, expert second
    tokens_for_expert_from_rank = tokens.split(
        n_tokens_for_expert_from_rank.flatten().tolist()
    )
    tokens_for_expert_from_rank = list_to_matrix(
        tokens_for_expert_from_rank, d, num_experts_per_rank
    )
    # list(zip(*matrix)) transposes the matrix.
    # Expert first, rank second
    tokens_for_expert_from_rank = list(zip(*tokens_for_expert_from_rank))
    tokens_for_expert_from_rank = matrix_to_list(tokens_for_expert_from_rank)
    tokens = torch.cat(tokens_for_expert_from_rank, dim=0)
    return tokens


def expert_first_to_rank_first(
    tokens: torch.Tensor, n_tokens_for_expert_from_rank: torch.Tensor
) -> torch.Tensor:
    d, num_experts_per_rank = n_tokens_for_expert_from_rank.size()
    # Expert first, rank second
    tokens_for_expert_from_rank = tokens.split(
        n_tokens_for_expert_from_rank.t().flatten().tolist()
    )
    tokens_for_expert_from_rank = list_to_matrix(
        tokens_for_expert_from_rank, num_experts_per_rank, d
    )
    # Rank first, expert second
    tokens_for_expert_from_rank = list(zip(*tokens_for_expert_from_rank))
    tokens_for_expert_from_rank = matrix_to_list(tokens_for_expert_from_rank)
    tokens = torch.cat(tokens_for_expert_from_rank, dim=0)
    return tokens


# This test serves as the reference implementation for the expert parallelism
# dispatch and combine logic. An expert-parallel MoE layer dispatches,
# processes, and combines tokens in the following steps:
# 1. Each token is dispatched to the rank hosting the corresponding expert.
# 2. Each rank sorts the received tokens by expert ID.
# 3. Each rank processes the received tokens with their corresponding experts. This is omitted in the test for simplicity.
# 4. Each rank sorts the processed tokens by rank ID.
# 5. Processed tokens are sent back to the original ranks.
@pytest.mark.mpi
def test_dispatch_and_combine(setup_default_process_group):
    rank = dist.get_rank()
    # So RNG generates different numbers for each rank.
    torch.manual_seed(rank)

    d = dist.get_world_size()
    num_experts_per_rank = 3
    n_experts = d * num_experts_per_rank

    # Input tokens are distributed among ranks, so the effective sequence length is `n_tokens * d`.
    n_tokens = 12
    n_tokens_for_expert = random_partition(n_tokens, n_experts)
    # GPU 0: [n_tokens_for_expert_0_from_rank_0, n_tokens_for_expert_1_from_rank_0, ..., n_tokens_for_expert_5_from_rank_0], sums to `n_tokens`
    # GPU 1: [n_tokens_for_expert_0_from_rank_1, n_tokens_for_expert_1_from_rank_1, ..., n_tokens_for_expert_5_from_rank_1], sums to `n_tokens`

    # For illustration, input tokens are complex numbers of value `token_id + expert_id * j`.
    expert_ids = []
    for expert, n in enumerate(n_tokens_for_expert.tolist()):
        expert_ids.extend([expert] * n)
    tokens = (
        torch.arange(
            rank * n_tokens, (rank + 1) * n_tokens, dtype=torch.float, device="cuda"
        )
        + torch.tensor(expert_ids, dtype=torch.float, device="cuda") * 1j
    )

    n_tokens_for_expert_from_rank = torch.empty(
        d, num_experts_per_rank, dtype=torch.int64, device="cuda"
    )
    # The following all_to_all_single communicates the number of tokens each
    # rank sends to each expert. This way, we can compute `output_split_sizes`
    # of the next all_to_all_single. In a more efficient implementation (e.g.
    # https://www.perplexity.ai/hub/blog/efficient-and-portable-mixture-of-experts-communication),
    # the two `all_to_all_single`s can be fused into one kernel.
    dist.all_to_all_single(n_tokens_for_expert_from_rank, n_tokens_for_expert)
    # GPU 0: [[n_tokens_for_expert_0_from_rank_0, n_tokens_for_expert_1_from_rank_0, n_tokens_for_expert_2_from_rank_0],
    #         [n_tokens_for_expert_0_from_rank_1, n_tokens_for_expert_1_from_rank_1, n_tokens_for_expert_2_from_rank_1]]
    # GPU 1: [[n_tokens_for_expert_3_from_rank_0, n_tokens_for_expert_4_from_rank_0, n_tokens_for_expert_5_from_rank_0],
    #         [n_tokens_for_expert_3_from_rank_1, n_tokens_for_expert_4_from_rank_1, n_tokens_for_expert_5_from_rank_1]]

    # Number of tokens to receive from each rank
    n_tokens_from_rank = n_tokens_for_expert_from_rank.sum(-1)
    # GPU 0: [n_tokens_for_expert_012_from_rank_0, n_tokens_for_expert_012_from_rank_1]
    # GPU 1: [n_tokens_for_expert_345_from_rank_0, n_tokens_for_expert_345_from_rank_1]
    # Number of tokens to send to each rank
    n_tokens_to_rank = n_tokens_for_expert.view(d, -1).sum(-1)
    # GPU 0: [n_tokens_for_expert_012_from_rank_0, n_tokens_for_expert_345_from_rank_0]
    # GPU 1: [n_tokens_for_expert_012_from_rank_1, n_tokens_for_expert_345_from_rank_1]

    tokens_by_rank = torch.empty(
        n_tokens_from_rank.sum(), dtype=torch.complex64, device="cuda"
    )
    dist.all_to_all_single(
        tokens_by_rank, tokens, n_tokens_from_rank.tolist(), n_tokens_to_rank.tolist()
    )
    # GPU 0: tokens_for_expert_0_from_rank_0 || tokens_for_expert_1_from_rank_0 || tokens_for_expert_2_from_rank_0 || tokens_for_expert_0_from_rank_1 || tokens_for_expert_1_from_rank_1 || tokens_for_expert_2_from_rank_1
    # GPU 1: tokens_for_expert_3_from_rank_0 || tokens_for_expert_4_from_rank_0 || tokens_for_expert_5_from_rank_0 || tokens_for_expert_3_from_rank_1 || tokens_for_expert_4_from_rank_1 || tokens_for_expert_5_from_rank_1
    tokens_by_expert = rank_first_to_expert_first(
        tokens_by_rank, n_tokens_for_expert_from_rank
    )
    # GPU 0: tokens_for_expert_0_from_rank_0 || tokens_for_expert_0_from_rank_1 || tokens_for_expert_1_from_rank_0 || tokens_for_expert_1_from_rank_1 || tokens_for_expert_2_from_rank_0 || tokens_for_expert_2_from_rank_1
    # GPU 1: tokens_for_expert_3_from_rank_0 || tokens_for_expert_3_from_rank_1 || tokens_for_expert_4_from_rank_0 || tokens_for_expert_4_from_rank_1 || tokens_for_expert_5_from_rank_0 || tokens_for_expert_5_from_rank_1

    imag = torch.imag(tokens_by_expert)
    assert (imag >= num_experts_per_rank * rank).all()
    assert (imag < num_experts_per_rank * (rank + 1)).all()
    assert (imag[:-1] <= imag[1:]).all()

    # (Omitted) Process the tokens with the experts.
    processed_tokens_by_expert = tokens_by_expert

    processed_tokens_by_rank = expert_first_to_rank_first(
        processed_tokens_by_expert, n_tokens_for_expert_from_rank
    )
    # GPU 0: tokens_for_expert_0_from_rank_0 || tokens_for_expert_1_from_rank_0 || tokens_for_expert_2_from_rank_0 || tokens_for_expert_0_from_rank_1 || tokens_for_expert_1_from_rank_1 || tokens_for_expert_2_from_rank_1
    # GPU 1: tokens_for_expert_3_from_rank_0 || tokens_for_expert_4_from_rank_0 || tokens_for_expert_5_from_rank_0 || tokens_for_expert_3_from_rank_1 || tokens_for_expert_4_from_rank_1 || tokens_for_expert_5_from_rank_1

    processed_tokens = torch.empty(n_tokens, dtype=torch.complex64, device="cuda")
    dist.all_to_all_single(
        processed_tokens,
        processed_tokens_by_rank,
        n_tokens_to_rank.tolist(),
        n_tokens_from_rank.tolist(),
    )

    torch.testing.assert_close(processed_tokens, tokens, rtol=0, atol=0)
