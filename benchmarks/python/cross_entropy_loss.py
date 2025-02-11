# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import sys
import torch
from functools import partial


from .model_configs import configs


def hf_qwen2(dtype):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2PreTrainedModel

    class MyModel(Qwen2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.lm_head = torch.nn.Linear(
                config.hidden_size, config.vocab_size, bias=False
            )
            # Initialize weights and apply final processing
            self.post_init()

        def forward(self, hidden_states: torch.Tensor, labels: torch.LongTensor):
            logits = self.lm_head(hidden_states)
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size
            )
            return (loss,)

    config = configs["hf_qwen2"]()

    def inputs(dtype, batch_size=config.batch_size, seq_len=config.seq_len):
        hidden_states = torch.randn(
            batch_size,
            seq_len,
            config.hidden_size,
            device="cuda",
            dtype=dtype,
            requires_grad=True,
        )
        labels = torch.randint(
            0,
            config.vocab_size,
            (batch_size, seq_len),
            device="cuda",
            requires_grad=False,
        )
        return {"hidden_states": hidden_states, "labels": labels}

    def grads(dtype, batch_size=config.batch_size, seq_len=config.seq_len):
        grad = torch.tensor(1, device="cuda", dtype=dtype, requires_grad=False)
        return grad

    def iobytes():
        return 1

    return (
        MyModel(config).cuda().to(dtype),
        partial(inputs, dtype),
        partial(grads, dtype),
        iobytes,
    )


def hf_phi3(dtype):
    from transformers.models.phi3 import Phi3PreTrainedModel

    class MyModel(Phi3PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.padding_idx = config.pad_token_id
            self.lm_head = torch.nn.Linear(
                config.hidden_size, config.vocab_size, bias=False
            )

            # Initialize weights and apply final processing
            self.post_init()

        def forward(self, hidden_states: torch.Tensor, labels: torch.LongTensor):
            logits = self.lm_head(hidden_states)
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size
            )
            return (loss,)

    config = configs["hf_phi3"]()

    def inputs(dtype, batch_size=config.batch_size, seq_len=config.seq_len):
        hidden_states = torch.randn(
            batch_size,
            seq_len,
            config.hidden_size,
            device="cuda",
            dtype=dtype,
            requires_grad=True,
        )
        labels = torch.randint(
            0,
            config.vocab_size,
            (batch_size, seq_len),
            device="cuda",
            requires_grad=False,
        )
        return {"hidden_states": hidden_states, "labels": labels}

    def grads(dtype, batch_size=config.batch_size, seq_len=config.seq_len):
        grad = torch.tensor(1, device="cuda", dtype=dtype, requires_grad=False)
        return grad

    def iobytes():
        return 1

    return (
        MyModel(config).cuda().to(dtype),
        partial(inputs, dtype),
        partial(grads, dtype),
        iobytes,
    )


def hf_mistral_nemo(dtype):
    from transformers.models.mistral import MistralPreTrainedModel

    class MyModel(MistralPreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.padding_idx = config.pad_token_id
            self.lm_head = torch.nn.Linear(
                config.hidden_size, config.vocab_size, bias=False
            )

            # Initialize weights and apply final processing
            self.post_init()

        def forward(self, hidden_states: torch.Tensor, labels: torch.LongTensor):
            logits = self.lm_head(hidden_states)
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size
            )
            return (loss,)

    config = configs["hf_mistral_nemo"]()

    def inputs(dtype, batch_size=config.batch_size, seq_len=config.seq_len):
        hidden_states = torch.randn(
            batch_size,
            seq_len,
            config.hidden_size,
            device="cuda",
            dtype=dtype,
            requires_grad=True,
        )
        labels = torch.randint(
            0,
            config.vocab_size,
            (batch_size, seq_len),
            device="cuda",
            requires_grad=False,
        )
        return {"hidden_states": hidden_states, "labels": labels}

    def grads(dtype, batch_size=config.batch_size, seq_len=config.seq_len):
        grad = torch.tensor(1, device="cuda", dtype=dtype, requires_grad=False)
        return grad

    def iobytes():
        return 1

    return (
        MyModel(config).cuda().to(dtype),
        partial(inputs, dtype),
        partial(grads, dtype),
        iobytes,
    )


cross_entropy_loss_setup = {
    "hf_qwen2": hf_qwen2,
    "hf_phi3": hf_phi3,
    "hf_mistral_nemo": hf_mistral_nemo,
}
