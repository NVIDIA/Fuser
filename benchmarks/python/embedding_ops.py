# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import torch.nn.functional as F


def embedding(inputs: list):
    indices, embedding_table = inputs
    return F.embedding(indices, embedding_table)


# (vocab, hidden) configurations seen in models.
EMBEDDING_CONFIGS = [
    (152064, 3584),  # hf_qwen2
    (32064, 3072),  # hf_phi3
    (131072, 5120),  # hf_mistral_nemo
]

SEQ_LENGTHS = [
    1024,
    2048,
    3072,
    4096,
    8192,
    12288,
    16384,
    20480,
    24576,
    28672,
    32768,
]
