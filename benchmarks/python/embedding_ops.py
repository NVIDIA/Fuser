# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import torch

from .model_configs import configs


class EmbeddingBase:
    def __init__(self, model_name, dtype):
        self.config = configs[model_name]()
        self.dtype = dtype

    def model(self):
        pass

    def inputs(self):
        input_ids = torch.randint(
            0,
            self.config.vocab_size,
            (self.config.batch_size, self.config.seq_len),
            device="cuda",
            requires_grad=False,
        )
        return {"input_ids": input_ids}

    def grads(self):
        grad = torch.randn(
            self.config.batch_size,
            self.config.seq_len,
            self.config.hidden_size,
            device="cuda",
            dtype=self.dtype,
            requires_grad=False,
        )
        return grad

    # Note this computes the bandwidth for weight gradient if the entire computation is done in a
    # single kernel. We don't consider the specific segmentation happening yet, since we don't have
    # a good clue how it should be segmented at this moment.
    def grad_iobytes(self):
        n_elements = 0
        # adding size of grad
        n_elements += (
            self.config.batch_size * self.config.seq_len * self.config.hidden_size
        )
        # adding size of index
        n_elements += self.config.batch_size * self.config.seq_len
        # adding size of output (weight.grad)
        n_elements += self.config.vocab_size * self.config.hidden_size
        # scale by dtype size
        return n_elements * self.dtype.itemsize


class HfQwen2(EmbeddingBase):
    def __init__(self, dtype):
        super().__init__("hf_qwen2", dtype)

    def model(self):
        from transformers.models.qwen2.modeling_qwen2 import Qwen2PreTrainedModel

        class MyModel(Qwen2PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.embed_tokens = torch.nn.Embedding(
                    config.vocab_size,
                    config.hidden_size,
                    config.pad_token_id,
                )
                # Initialize weights and apply final processing
                self.post_init()

            def forward(self, input_ids: torch.LongTensor):
                inputs_embeds = self.embed_tokens(input_ids)
                return (inputs_embeds,)

        return MyModel(self.config).cuda().to(self.dtype)


class HfPhi3(EmbeddingBase):
    def __init__(self, dtype):
        super().__init__("hf_phi3", dtype)

    def model(self):
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

        return MyModel(self.config).cuda().to(self.dtype)


class HfMistralNemo(EmbeddingBase):
    def __init__(self, dtype):
        super().__init__("hf_mistral_nemo", dtype)

    def model(self):
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

        return MyModel(self.config).cuda().to(self.dtype)


embedding_setup = {
    "hf_qwen2": HfQwen2,
    "hf_phi3": HfPhi3,
    "hf_mistral_nemo": HfMistralNemo,
}
