# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import pytest
import torch
import urllib.request
from contextlib import contextmanager


def download_as_module(url, module_name):
    urllib.request.urlretrieve(url, f"{module_name}.py")


@pytest.fixture(scope="module")
def model():
    download_as_module(
        "https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/inference/kernel.py",
        "kernel",
    )
    download_as_module(
        "https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/inference/model.py",
        "deepseek_v3_model",
    )
    import deepseek_v3_model

    yield deepseek_v3_model

    del deepseek_v3_model

    os.remove("deepseek_v3_model.py")
    os.remove("kernel.py")


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


def test_transformer_block(model):
    args = model.ModelArgs(dim=7168)

    dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
    with default_tensor_type(dtype=dtype, device="cuda"):
        transformer_block = model.Block(10, args)

        batch_size = 1
        assert batch_size <= args.max_batch_size
        seq_len = 4096
        assert seq_len <= args.max_seq_len
        inp = torch.randn(batch_size, seq_len, args.dim)
        start_pos = 0
        freq_cis = model.precompute_freqs_cis(args)
        mask = torch.full((seq_len, seq_len), float("-inf")).triu_(1)

    out = transformer_block(
        inp, start_pos, freq_cis[start_pos : start_pos + seq_len], mask
    )
    assert out.size() == (batch_size, seq_len, args.dim)
    assert out.dtype == torch.bfloat16
    assert out.is_cuda
