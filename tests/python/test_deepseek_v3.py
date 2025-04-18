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


# This test timed out once when downloading
# "/deepseek-ai/DeepSeek-V3/resolve/main/configuration_deepseek.py" (cf.
# http://nv/eCm). I consider this a one-off, but please let me know if this
# error becomes consistent.
def test_transformer_layer():
    config = transformers.AutoConfig.from_pretrained(
        "deepseek-ai/deepseek-v3", trust_remote_code=True
    )

    # Create only one layer which is sufficient for the test.
    config.num_hidden_layers = 1
    # Without this, the first and only layer will have a dense MLP instead of MoE.
    config.first_k_dense_replace = 0
    # Disable quantization so the test can run on A100 and is made easier for nvFuser.
    delattr(config, "quantization_config")

    with default_tensor_type(dtype=config.torch_dtype, device="cuda"):
        model = transformers.AutoModel.from_config(config, trust_remote_code=True)
        # Training is unavailable (cf. https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py#L439)
        model.eval()

        transformer_layer = model.layers[0]

        batch_size = 1
        seq_len = 2048
        inp = torch.randn(batch_size, seq_len, config.hidden_size)
        mask = transformers.modeling_attn_mask_utils._prepare_4d_causal_attention_mask(
            None, [batch_size, seq_len], inp, past_key_values_length=0
        )
        (out,) = transformer_layer(inp, attention_mask=mask)

        assert out.size() == (batch_size, seq_len, config.hidden_size)
        assert out.dtype == config.torch_dtype
        assert out.is_cuda
