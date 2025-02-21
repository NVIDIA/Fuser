# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import sys
import torch
from functools import partial


from .model_configs import configs


# Note this computes the bandwidth for weight gradient if the entire computation is done in a
# single kernel. We don't consider the specific segmentation happening yet, since we don't have
# a good clue how it should be segmented at this moment.
def embedding_weight_grad_iobytes(config, dtype):
    n_elements = 0
    # adding size of grad
    n_elements += config.batch_size * config.seq_len * config.hidden_size
    # adding size of index
    n_elements += config.batch_size * config.seq_len
    # adding size of output (weight.grad)
    n_elements += config.vocab_size * config.hidden_size
    # scale by dtype size
    return n_elements * dtype.itemsize


def hf_qwen2(dtype):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2PreTrainedModel

    class MyModel(Qwen2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.embed_tokens = torch.nn.Embedding(
                config.vocab_size, config.hidden_size, config.pad_token_id
            )
            # Initialize weights and apply final processing
            self.post_init()

        def forward(self, input_ids: torch.LongTensor):
            inputs_embeds = self.embed_tokens(input_ids)
            return (inputs_embeds,)

    config = configs["hf_qwen2"]()

    def inputs(dtype, batch_size=config.batch_size, seq_len=config.seq_len):
        input_ids = torch.randint(
            0,
            config.vocab_size,
            (batch_size, seq_len),
            device="cuda",
            requires_grad=False,
        )
        return {"input_ids": input_ids}

    def grads(dtype, batch_size=config.batch_size, seq_len=config.seq_len):
        grad = torch.randn(
            batch_size,
            seq_len,
            config.hidden_size,
            device="cuda",
            dtype=dtype,
            requires_grad=False,
        )
        return grad

    return (
        MyModel(config).cuda().to(dtype),
        partial(inputs, dtype),
        partial(grads, dtype),
        partial(embedding_weight_grad_iobytes, config, dtype),
    )


def hf_phi3(dtype):
    from transformers.models.phi3 import Phi3PreTrainedModel

    class MyModel(Phi3PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.padding_idx = config.pad_token_id
            self.embed_tokens = torch.nn.Embedding(
                config.vocab_size, config.hidden_size, self.padding_idx
            )

            # Initialize weights and apply final processing
            self.post_init()

        def forward(self, input_ids: torch.LongTensor):
            inputs_embeds = self.embed_tokens(input_ids)
            return (inputs_embeds,)

    config = configs["hf_phi3"]()

    def inputs(dtype, batch_size=config.batch_size, seq_len=config.seq_len):
        input_ids = torch.randint(
            0,
            config.vocab_size,
            (batch_size, seq_len),
            device="cuda",
            requires_grad=False,
        )
        return {"input_ids": input_ids}

    def grads(dtype, batch_size=config.batch_size, seq_len=config.seq_len):
        grad = torch.randn(
            batch_size,
            seq_len,
            config.hidden_size,
            device="cuda",
            dtype=dtype,
            requires_grad=False,
        )
        return grad

    return (
        MyModel(config).cuda().to(dtype),
        partial(inputs, dtype),
        partial(grads, dtype),
        partial(embedding_weight_grad_iobytes, config, dtype),
    )


def hf_mistral_nemo(dtype):
    from transformers.models.mistral import MistralPreTrainedModel

    class MyModel(MistralPreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.embed_tokens = torch.nn.Embedding(
                config.vocab_size, config.hidden_size, config.pad_token_id
            )
            # Initialize weights and apply final processing
            self.post_init()

        def forward(self, input_ids: torch.LongTensor):
            inputs_embeds = self.embed_tokens(input_ids)
            return (inputs_embeds,)

    config = configs["hf_mistral_nemo"]()

    def inputs(dtype, batch_size=config.batch_size, seq_len=config.seq_len):
        input_ids = torch.randint(
            0,
            config.vocab_size,
            (batch_size, seq_len),
            device="cuda",
            requires_grad=False,
        )
        return {"input_ids": input_ids}

    def grads(dtype, batch_size=config.batch_size, seq_len=config.seq_len):
        grad = torch.randn(
            batch_size,
            seq_len,
            config.hidden_size,
            device="cuda",
            dtype=dtype,
            requires_grad=False,
        )
        return grad

    return (
        MyModel(config).cuda().to(dtype),
        partial(inputs, dtype),
        partial(grads, dtype),
        partial(embedding_weight_grad_iobytes, config, dtype),
    )


embedding_setup = {
    "hf_qwen2": hf_qwen2,
    "hf_phi3": hf_phi3,
    "hf_mistral_nemo": hf_mistral_nemo,
}
