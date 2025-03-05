# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import transformers
import torch
from contextlib import contextmanager


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


def test_transformer_layer():
    config = transformers.AutoConfig.from_pretrained(
        "deepseek-ai/deepseek-v3", trust_remote_code=True
    )
    config.num_hidden_layers = 1
    config.first_k_dense_replace = 0
    delattr(config, "quantization_config")

    with default_tensor_type(dtype=config.torch_dtype, device="cuda"):
        model = transformers.AutoModel.from_config(config, trust_remote_code=True)
        # Training is unavailable (cf. https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py#L439)
        model.eval()

        transformer_layer = model.layers[0]

        batch_size = 1
        seq_len = 4096
        inp = torch.randn(batch_size, seq_len, config.hidden_size)
        mask = transformers.modeling_attn_mask_utils._prepare_4d_causal_attention_mask(
            None, [batch_size, seq_len], inp, past_key_values_length=0
        )
        (out,) = transformer_layer(inp, attention_mask=mask)

        assert out.size() == (batch_size, seq_len, config.hidden_size)
        assert out.dtype == torch.bfloat16
        assert out.is_cuda
