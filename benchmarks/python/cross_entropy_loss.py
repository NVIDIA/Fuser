# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import torch

from .model_configs import configs


class CrossEntropyLossBase:
    def __init__(self, model_name, dtype):
        self.config = configs[model_name]()
        self.dtype = dtype

    def model(self):
        raise NotImplementedError

    def inputs(self):
        hidden_states = torch.randn(
            self.config.batch_size,
            self.config.seq_len,
            self.config.hidden_size,
            device="cuda",
            dtype=self.dtype,
            requires_grad=True,
        )
        labels = torch.randint(
            0,
            self.config.vocab_size,
            (self.config.batch_size, self.config.seq_len),
            device="cuda",
            requires_grad=False,
        )
        return {"hidden_states": hidden_states, "labels": labels}

    def grads(self):
        grad = torch.tensor(1, device="cuda", dtype=self.dtype, requires_grad=False)
        return grad

    # please note that this computes the IO bytes based on the graph/decomposition
    # ===== Backward graph 0 =====
    # <eval_with_key>.3 class GraphModule(torch.nn.Module):
    # def forward(self, primals_1: "i64[8192][1]cuda:0", primals_2: "f32[8192, 3024][3024, 1]cuda:0",
    #        amax: "f32[8192, 1][1, 1]cuda:0", log: "f32[8192, 1][1, 1]cuda:0", convert_element_type: "f32[][]cuda:0", tangents_1: "f32[][]cuda:0"):
    #     # File: /opt/pytorch/nvfuser/test.py:6 in fn, code: return torch.nn.functional.cross_entropy(input, target)
    #    div_1: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type);  tangents_1 = convert_element_type = None
    #    unsqueeze_1: "i64[8192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_1, 1);  primals_1 = None
    #    ne_3: "b8[8192, 1][1, 1]cuda:0" = torch.ops.aten.ne.Scalar(unsqueeze_1, -100)
    #    full_default: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    #    where_2: "i64[8192, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_3, unsqueeze_1, full_default);  unsqueeze_1 = full_default = None
    #    full_default_3: "f32[8192, 3024][3024, 1]cuda:0" = torch.ops.aten.full.default([8192, 3024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    #    scatter: "f32[8192, 3024][3024, 1]cuda:0" = torch.ops.aten.scatter.value(full_default_3, 1, where_2, -1.0);  full_default_3 = where_2 = None
    #    full_default_1: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    #    where_3: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_3, div_1, full_default_1);  ne_3 = div_1 = full_default_1 = None
    #    mul: "f32[8192, 3024][3024, 1]cuda:0" = torch.ops.aten.mul.Tensor(scatter, where_3);  scatter = where_3 = None
    #    sub: "f32[8192, 3024][3024, 1]cuda:0" = torch.ops.aten.sub.Tensor(primals_2, amax);  primals_2 = amax = None
    #    sub_1: "f32[8192, 3024][3024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
    #    exp_1: "f32[8192, 3024][3024, 1]cuda:0" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    #    sum_4: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul, [1], True)
    #    mul_1: "f32[8192, 3024][3024, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_1, sum_4);  exp_1 = sum_4 = None
    #    sub_2: "f32[8192, 3024][3024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul, mul_1);  mul = mul_1 = None
    #    return (None, sub_2)

    # We account for inputs primals_1, primals_2, amax, log
    # and for the output sub_2
    def grad_iobytes(self):
        n_elements = 0
        # adding size of primals_2 and the output
        n_elements += 2 * (
            self.config.batch_size * self.config.seq_len * self.config.vocab_size
        )
        # adding size of amax and log and primals_1
        n_elements += 3 * self.config.batch_size * self.config.seq_len
        # scale by dtype size
        return n_elements * self.dtype.itemsize


class HfQwen2(CrossEntropyLossBase):
    def __init__(self, dtype):
        super().__init__("hf_qwen2", dtype)

    def model(self):
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

        return MyModel(self.config).cuda().to(self.dtype)


class HfPhi3(CrossEntropyLossBase):
    def __init__(self, dtype):
        super().__init__("hf_phi3", dtype)

    def model(self):
        from transformers.models.phi3 import Phi3PreTrainedModel

        class MyModel(Phi3PreTrainedModel):
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

        return MyModel(self.config).cuda().to(self.dtype)


class HfMistralNemo(CrossEntropyLossBase):
    def __init__(self, dtype):
        super().__init__("hf_mistral_nemo", dtype)

    def model(self):
        from transformers.models.mistral import MistralPreTrainedModel

        class MyModel(MistralPreTrainedModel):
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

        return MyModel(self.config).cuda().to(self.dtype)


cross_entropy_loss_setup = {
    "hf_qwen2": HfQwen2,
    "hf_phi3": HfPhi3,
    "hf_mistral_nemo": HfMistralNemo,
}


class SyntheticMiniModel:
    @staticmethod
    def mini_model(logits, labels, cross_entropy_no_redu):
        if cross_entropy_no_redu:
          return torch.nn.functional.cross_entropy(logits, labels, reduction='none')
        else:
          labels = torch.nn.functional.pad(labels, (0, 1))
          labels = labels[1 : labels.shape[-1]]
          logits = logits.to(dtype=torch.float32)
          logits = logits.squeeze(dim=0)
          return torch.nn.functional.cross_entropy(logits, labels)

    @staticmethod
    def inputs(batch_size, vocab_size, cross_entropy_no_redu):
        if cross_entropy_no_redu:
          input = torch.randn(
              batch_size,
              vocab_size,
              device="cuda",
              dtype=torch.bfloat16,
          )
          labels = torch.randint(
              0,
              vocab_size,
              (batch_size,),
              device="cuda",
              dtype=torch.int64,
          )          
        else:
          input = torch.randn(
              1,
              batch_size,
              vocab_size,
              device="cuda",
              dtype=torch.bfloat16,
              requires_grad=True,
          )
          labels = torch.randint(
              0,
              vocab_size - 1,
              (batch_size,),
              device="cuda",
              requires_grad=False,
          )
        return (input, labels, cross_entropy_no_redu)

    @staticmethod
    def grads():
        grad = torch.tensor(1, device="cuda", dtype=torch.float32, requires_grad=False)
        return grad

    @staticmethod
    def generate_vocab_sizes():
        sizes_from_models = [
            49152,  # Starcoder
            129280,  # DeepSeek-R1
            128256,  # Llama3
            202048,  # Llama4
            256000,  # Gemma2
            131072,  # Mistral
            152064,  # Qwen2
            32064,  # Phi3.5
            100352,  # Phi4
            50264,  # GPT-2
        ]

        powers_of_2 = [2**i * 1024 for i in range(4, 9)]

        combined_set = sorted(set(sizes_from_models) | set(powers_of_2))

        # for each vocab size in the set we increment in steps 64 in +/- 5 directions
        # which gives the total number of vocab sizes to benchmark
        variations = set()
        step = 64
        for num in combined_set:
            for i in range(1, 6):
                variations.add(num + (i * step))
                variations.add(num - (i * step))

            variations.add(num)

        return sorted(variations)
