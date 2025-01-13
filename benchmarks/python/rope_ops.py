# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
from nvfuser import FusionDefinition, DataType
import torch

from torch import nn

from typing import Tuple
from functools import partial


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    if cos.dim() > 1:
        # batch dimensions must align
        # sin/cos are (B, T, hs) so we unsqeeze -3 for nh
        # we count from back because all of apply_rope does
        cos = cos.unsqueeze(-3)
        sin = sin.unsqueeze(-3)

    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)


def llama_hf_rope(config_str):
    class Config:
        def __init__(
            self, n_head, head_size, n_query_groups, rope_n_elem, batches, seq_length
        ):
            self.n_head = n_head
            self.head_size = head_size
            self.n_query_groups = n_query_groups
            self.rope_n_elem = rope_n_elem
            self.batches = batches
            self.seq_length = seq_length

    class LitGPTRope(torch.nn.Module):
        def __init__(self, config):
            super(LitGPTRope, self).__init__()
            self.config = config

        def forward(self, qkv, cos, sin):
            B, T, _ = qkv.size()
            # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
            q_per_kv = self.config.n_head // self.config.n_query_groups
            total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
            qkv = qkv.view(
                B, T, self.config.n_query_groups, total_qkv, self.config.head_size
            )
            qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

            # split batched computation into three
            q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)

            # maybe repeat k and v if for the non multi-head attention cases
            # training: flash attention requires it
            # inference: multi-query would require a full kv cache so avoid it to limit its memory usage
            # if self.config.n_query_groups != self.config.n_head and (input_pos is None or self.config.n_query_groups != 1):
            if self.config.n_query_groups != self.config.n_head and (
                self.config.n_query_groups != 1
            ):
                k = k.expand(
                    B, self.config.n_query_groups, q_per_kv, T, self.config.head_size
                )
                v = v.expand(
                    B, self.config.n_query_groups, q_per_kv, T, self.config.head_size
                )

            q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
            k = k.reshape(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
            v = v.reshape(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)

            q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
            k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
            q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
            k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)
            return q, k

    configs = {}
    configs["llama_2_7b_hf_rope"] = Config(
        n_head=32,
        head_size=128,
        n_query_groups=32,
        rope_n_elem=128,
        batches=2,
        seq_length=4096,
    )
    configs["llama_3_8B_rope"] = Config(
        n_head=32,
        head_size=128,
        n_query_groups=8,
        rope_n_elem=128,
        batches=2,
        seq_length=8192,
    )

    cfg = configs[config_str]

    def inputs():
        qkv = torch.randn(
            cfg.batches,
            cfg.seq_length,
            cfg.head_size * (cfg.n_head + 2 * cfg.n_query_groups),
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        cos = torch.randn(
            cfg.seq_length,
            cfg.rope_n_elem,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=False,
        )
        sin = torch.randn(
            cfg.seq_length,
            cfg.rope_n_elem,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=False,
        )
        return qkv, cos, sin

    def grads():
        grad = torch.randn(
            cfg.batches,
            cfg.n_head,
            cfg.seq_length,
            cfg.head_size,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=False,
        )
        return grad

    # Manual IOBytes computes the total bandwidth for thunder backward trace.
    def iobytes():
        n_elements = 0
        # adding size of q.grad + k.grad
        n_elements += 2 * cfg.batches * cfg.n_head * cfg.seq_length * cfg.head_size
        # adding size of cos, sin
        n_elements += 2 * cfg.seq_length * cfg.rope_n_elem
        # adding size of qkv.grad
        n_elements += (
            cfg.batches
            * cfg.seq_length
            * cfg.head_size
            * (cfg.n_head + 2 * cfg.n_query_groups)
        )
        # scale by dtype size
        return n_elements * torch.bfloat16.itemsize

    return LitGPTRope(cfg).cuda().bfloat16(), inputs, grads, iobytes


def hf_qwen2_rope():
    import json
    from transformers.models.qwen2 import Qwen2Config

    qwen_cfg_str = r"""{
      "_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
      "architectures": [
        "Qwen2ForCausalLM"
      ],
      "attention_dropout": 0.0,
      "bos_token_id": 151643,
      "eos_token_id": 151645,
      "hidden_act": "silu",
      "hidden_size": 3584,
      "initializer_range": 0.02,
      "intermediate_size": 18944,
      "max_position_embeddings": 32768,
      "max_window_layers": 28,
      "model_type": "qwen2",
      "num_attention_heads": 28,
      "num_hidden_layers": 28,
      "num_key_value_heads": 4,
      "rms_norm_eps": 1e-06,
      "rope_theta": 1000000.0,
      "sliding_window": null,
      "tie_word_embeddings": false,
      "torch_dtype": "bfloat16",
      "transformers_version": "4.43.3",
      "use_cache": true,
      "use_sliding_window": false,
      "vocab_size": 152064
    }
    """

    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`, *optional*):
                Deprecated and unused.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    class Qwen2Rope(nn.Module):
        def __init__(self, config: Qwen2Config):
            super().__init__()
            self.config = config

            self.hidden_size = config.hidden_size
            self.num_heads = config.num_attention_heads
            self.head_dim = self.hidden_size // self.num_heads
            self.num_key_value_heads = config.num_key_value_heads
            self.num_key_value_groups = self.num_heads // self.num_key_value_heads
            self.max_position_embeddings = config.max_position_embeddings
            self.rope_theta = config.rope_theta
            self.is_causal = True
            self.attention_dropout = config.attention_dropout

            if (self.head_dim * self.num_heads) != self.hidden_size:
                raise ValueError(
                    f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                    f" and `num_heads`: {self.num_heads})."
                )

        def forward(
            self,
            query_in_states: torch.Tensor,
            key_in_states: torch.Tensor,
            value_in_states: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            past_key_value = None
            bsz, q_len, _ = query_in_states.size()

            query_states = query_in_states.view(
                bsz, q_len, self.num_heads, self.head_dim
            ).transpose(1, 2)
            key_states = key_in_states.view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)
            value_states = value_in_states.view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)

            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

            if past_key_value is not None:
                assert False

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            return query_states, key_states, value_states

    cfg = Qwen2Config.from_dict(json.loads(qwen_cfg_str))
    cfg.batch_size = 1
    cfg.seq_len = 4096

    head_dim = cfg.hidden_size // cfg.num_attention_heads

    def inputs():
        q = torch.randn(
            cfg.batch_size,
            cfg.seq_len,
            cfg.num_attention_heads * head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        k = torch.randn(
            cfg.batch_size,
            cfg.seq_len,
            cfg.num_key_value_heads * head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        v = torch.randn(
            cfg.batch_size,
            cfg.seq_len,
            cfg.num_key_value_heads * head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        cos = torch.randn(
            cfg.batch_size,
            cfg.seq_len,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        sin = torch.randn(
            cfg.batch_size,
            cfg.seq_len,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        return q, k, v, cos, sin

    def grads():
        grad = torch.randn(
            cfg.batch_size,
            cfg.num_attention_heads,
            cfg.seq_len,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=False,
        )
        return grad

    # Manual IOBytes computes the total bandwidth for thunder backward trace.
    def iobytes():
        n_elements = 0
        # adding size of query_states.grad + key_states.grad + value_states.grad
        n_elements += (
            3 * cfg.batch_size * cfg.num_attention_heads * cfg.seq_len * head_dim
        )
        # adding size of query_states + key_states
        n_elements += (
            2 * cfg.batch_size * cfg.num_attention_heads * cfg.seq_len * head_dim
        )
        # adding size of cos, sin
        n_elements += 2 * cfg.batch_size * cfg.seq_len * head_dim
        # adding size of q.grad
        n_elements += cfg.batch_size * cfg.seq_len * cfg.num_attention_heads * head_dim
        # adding size of k.grad, v.grad
        n_elements += (
            2 * cfg.batch_size * cfg.seq_len * cfg.num_key_value_heads * head_dim
        )
        # adding size of cos.grad, sin.grad
        n_elements += 2 * cfg.batch_size * cfg.seq_len * head_dim
        # scale by dtype size
        return n_elements * torch.bfloat16.itemsize

    return Qwen2Rope(cfg).cuda().bfloat16(), inputs, grads, iobytes


def hf_phi3_rope():
    import json
    from transformers.models.phi3 import Phi3Config

    phi35_cfg_str = r"""{
      "_name_or_path": "microsoft/Phi-3.5-mini-instruct",
      "architectures": [
        "Phi3ForCausalLM"
      ],
      "attention_bias": false,
      "attention_dropout": 0.0,
      "auto_map": {
        "AutoConfig": "microsoft/Phi-3.5-mini-instruct--configuration_phi3.Phi3Config",
        "AutoModelForCausalLM": "microsoft/Phi-3.5-mini-instruct--modeling_phi3.Phi3ForCausalLM"
      },
      "bos_token_id": 1,
      "embd_pdrop": 0.0,
      "eos_token_id": 32000,
      "hidden_act": "silu",
      "hidden_size": 3072,
      "initializer_range": 0.02,
      "intermediate_size": 8192,
      "max_position_embeddings": 131072,
      "model_type": "phi3",
      "num_attention_heads": 32,
      "num_hidden_layers": 32,
      "num_key_value_heads": 32,
      "original_max_position_embeddings": 4096,
      "pad_token_id": 32000,
      "resid_pdrop": 0.0,
      "rms_norm_eps": 1e-05,
      "rope_scaling": {
        "long_factor": [
          1.0800000429153442,
          1.1100000143051147,
          1.1399999856948853,
          1.340000033378601,
          1.5899999141693115,
          1.600000023841858,
          1.6200000047683716,
          2.620000123977661,
          3.2300000190734863,
          3.2300000190734863,
          4.789999961853027,
          7.400000095367432,
          7.700000286102295,
          9.09000015258789,
          12.199999809265137,
          17.670000076293945,
          24.46000099182129,
          28.57000160217285,
          30.420001983642578,
          30.840002059936523,
          32.590003967285156,
          32.93000411987305,
          42.320003509521484,
          44.96000289916992,
          50.340003967285156,
          50.45000457763672,
          57.55000305175781,
          57.93000411987305,
          58.21000289916992,
          60.1400032043457,
          62.61000442504883,
          62.62000274658203,
          62.71000289916992,
          63.1400032043457,
          63.1400032043457,
          63.77000427246094,
          63.93000411987305,
          63.96000289916992,
          63.970001220703125,
          64.02999877929688,
          64.06999969482422,
          64.08000183105469,
          64.12000274658203,
          64.41000366210938,
          64.4800033569336,
          64.51000213623047,
          64.52999877929688,
          64.83999633789062
        ],
       "short_factor": [
          1.0,
          1.0199999809265137,
          1.0299999713897705,
          1.0299999713897705,
          1.0499999523162842,
          1.0499999523162842,
          1.0499999523162842,
          1.0499999523162842,
          1.0499999523162842,
          1.0699999332427979,
          1.0999999046325684,
          1.1099998950958252,
          1.1599998474121094,
          1.1599998474121094,
          1.1699998378753662,
          1.2899998426437378,
          1.339999794960022,
          1.679999828338623,
          1.7899998426437378,
          1.8199998140335083,
          1.8499997854232788,
          1.8799997568130493,
          1.9099997282028198,
          1.9399996995925903,
          1.9899996519088745,
          2.0199997425079346,
          2.0199997425079346,
          2.0199997425079346,
          2.0199997425079346,
          2.0199997425079346,
          2.0199997425079346,
          2.0299997329711914,
          2.0299997329711914,
          2.0299997329711914,
          2.0299997329711914,
          2.0299997329711914,
          2.0299997329711914,
          2.0299997329711914,
          2.0299997329711914,
          2.0299997329711914,
          2.0799996852874756,
          2.0899996757507324,
          2.189999580383301,
          2.2199995517730713,
          2.5899994373321533,
          2.729999542236328,
          2.749999523162842,
          2.8399994373321533
        ],
        "type": "longrope"
      },
      "rope_theta": 10000.0,
      "sliding_window": 262144,
      "tie_word_embeddings": false,
      "torch_dtype": "bfloat16",
      "transformers_version": "4.46.3",
      "use_cache": true,
      "vocab_size": 32064
    }"""

    class Phi3RotaryEmbedding(nn.Module):
        def __init__(
            self, dim, max_position_embeddings=2048, base=10000.0, device=None
        ):
            super().__init__()

            self.dim = dim
            self.max_position_embddings = max_position_embeddings
            self.base = base

            inv_freq = 1.0 / (
                self.base
                ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)
            )
            self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

        @torch.no_grad()
        def forward(self, x, position_ids, seq_len=None):
            # x: [bs, num_attention_heads, seq_len, head_size]
            self.inv_freq.to(x.device)
            inv_freq_expanded = (
                self.inv_freq[None, :, None]
                .float()
                .expand(position_ids.shape[0], -1, 1)
            )
            position_ids_expanded = position_ids[:, None, :].float()
            # Force float32 since bfloat16 loses precision on long contexts
            # See https://github.com/huggingface/transformers/pull/29285
            device_type = x.device.type
            device_type = (
                device_type
                if isinstance(device_type, str) and device_type != "mps"
                else "cpu"
            )
            with torch.autocast(device_type=device_type, enabled=False):
                freqs = (
                    inv_freq_expanded.float() @ position_ids_expanded.float()
                ).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos()
                sin = emb.sin()
            return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`, *optional*):
                Deprecated and unused.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    class HfPhi3Rope(nn.Module):
        """Multi-headed attention from 'Attention Is All You Need' paper"""

        def __init__(self, config: Phi3Config):
            super().__init__()
            self.config = config

            self.hidden_size = config.hidden_size
            self.num_heads = config.num_attention_heads
            self.head_dim = self.hidden_size // self.num_heads
            self.num_key_value_heads = config.num_key_value_heads
            self.num_key_value_groups = self.num_heads // self.num_key_value_heads
            self.max_position_embeddings = config.max_position_embeddings
            self.original_max_position_embeddings = (
                config.original_max_position_embeddings
            )
            self.rope_theta = config.rope_theta
            self.rope_scaling = config.rope_scaling
            self.is_causal = True

            if (self.head_dim * self.num_heads) != self.hidden_size:
                raise ValueError(
                    f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                    f" and `num_heads`: {self.num_heads})."
                )

            self.rotary_emb = Phi3RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )

        def forward(
            self, qkv: torch.Tensor, position_ids: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            past_key_value = None
            bsz, q_len, _ = qkv.size()

            query_pos = self.num_heads * self.head_dim
            query_states = qkv[..., :query_pos]
            key_states = qkv[
                ..., query_pos : query_pos + self.num_key_value_heads * self.head_dim
            ]
            value_states = qkv[
                ..., query_pos + self.num_key_value_heads * self.head_dim :
            ]

            query_states = query_states.view(
                bsz, q_len, self.num_heads, self.head_dim
            ).transpose(1, 2)
            key_states = key_states.view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)
            value_states = value_states.view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)

            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                assert False
            cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)

            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )

            if past_key_value is not None:
                assert False

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            return query_states, key_states, value_states

    cfg = Phi3Config.from_dict(json.loads(phi35_cfg_str))
    cfg.batch_size = 1
    cfg.seq_len = 8192
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    def inputs():
        qkv = torch.randn(
            cfg.batch_size,
            cfg.seq_len,
            cfg.num_attention_heads * head_dim
            + 2 * (cfg.num_key_value_heads * head_dim),
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        position_ids = torch.arange(0, cfg.seq_len, device="cuda").unsqueeze(0)
        return qkv, position_ids

    def grads():
        grad = torch.randn(
            cfg.batch_size,
            cfg.num_attention_heads,
            cfg.seq_len,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=False,
        )
        return grad

    # Manual IOBytes computes the total bandwidth for thunder backward trace.
    def iobytes():
        n_elements = 0
        # adding size of query_states.grad + key_states.grad +  value_states.grad
        n_elements += (
            3 * cfg.batch_size * cfg.num_attention_heads * cfg.seq_len * head_dim
        )
        # adding size of qkv.grad
        n_elements += (
            cfg.batch_size
            * cfg.seq_len
            * (
                cfg.num_attention_heads * head_dim
                + 2 * (cfg.num_key_value_heads * head_dim)
            )
        )
        # matmul output size
        n_elements_matmul_out = head_dim / 2 * cfg.seq_len
        # totoal io sizes
        return (
            n_elements * torch.bfloat16.itemsize
            + n_elements_matmul_out * torch.float32.itemsize
        )

    return HfPhi3Rope(cfg).cuda().bfloat16(), inputs, grads, iobytes


def hf_mistral_nemo_rope():
    import json
    from transformers.models.mistral import MistralConfig

    mistral_cfg_str = r"""{
      "_name_or_path": "mistralai/Mistral-Nemo-Base-2407",
      "architectures": [
        "MistralForCausalLM"
      ],
      "attention_dropout": 0.0,
      "bos_token_id": 1,
      "eos_token_id": 2,
      "head_dim": 128,
      "hidden_act": "silu",
      "hidden_size": 5120,
      "initializer_range": 0.02,
      "intermediate_size": 14336,
      "max_position_embeddings": 128000,
      "model_type": "mistral",
      "num_attention_heads": 32,
      "num_hidden_layers": 40,
      "num_key_value_heads": 8,
      "rms_norm_eps": 1e-05,
      "rope_theta": 1000000.0,
      "sliding_window": null,
      "tie_word_embeddings": false,
      "torch_dtype": "bfloat16",
      "transformers_version": "4.43.3",
      "use_cache": true,
      "vocab_size": 131072
    }
    """

    class MistralRotaryEmbedding(nn.Module):
        def __init__(
            self, dim, max_position_embeddings=2048, base=10000.0, device=None
        ):
            super().__init__()

            self.dim = dim
            self.max_position_embeddings = max_position_embeddings
            self.base = base
            inv_freq = 1.0 / (
                self.base
                ** (
                    torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                    / self.dim
                )
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        @torch.no_grad()
        # copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.forward
        # TODO(joao): add me back asap :)
        def forward(self, x, position_ids):
            # x: [bs, num_attention_heads, seq_len, head_size]
            inv_freq_expanded = (
                self.inv_freq[None, :, None]
                .float()
                .expand(position_ids.shape[0], -1, 1)
            )
            position_ids_expanded = position_ids[:, None, :].float()
            # Force float32 since bfloat16 loses precision on long contexts
            # See https://github.com/huggingface/transformers/pull/29285
            device_type = x.device.type
            device_type = (
                device_type
                if isinstance(device_type, str) and device_type != "mps"
                else "cpu"
            )
            with torch.autocast(device_type=device_type, enabled=False):
                freqs = (
                    inv_freq_expanded.float() @ position_ids_expanded.float()
                ).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos()
                sin = emb.sin()
            return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`, *optional*):
                Deprecated and unused.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    class MistralNemoRope(nn.Module):
        def __init__(self, config: MistralConfig):
            super().__init__()
            self.config = config

            self.attention_dropout = config.attention_dropout
            self.hidden_size = config.hidden_size
            self.num_heads = config.num_attention_heads
            self.head_dim = config.head_dim
            self.num_key_value_heads = config.num_key_value_heads
            self.num_key_value_groups = self.num_heads // self.num_key_value_heads
            self.max_position_embeddings = config.max_position_embeddings
            self.rope_theta = config.rope_theta
            self.is_causal = True

            self.rotary_emb = MistralRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )

        def forward(
            self,
            query_in_states: torch.Tensor,
            key_in_states: torch.Tensor,
            value_in_states: torch.Tensor,
            position_ids: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            past_key_value = None
            bsz, q_len, _ = query_in_states.size()

            query_states = query_in_states.view(
                bsz, q_len, self.num_heads, self.head_dim
            ).transpose(1, 2)
            key_states = key_in_states.view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)
            value_states = value_in_states.view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)

            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

            if past_key_value is not None:
                assert False

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            return query_states, key_states, value_states

    cfg = MistralConfig.from_dict(json.loads(mistral_cfg_str))
    cfg.batch_size = 1
    cfg.seq_len = 4096

    head_dim = cfg.hidden_size // cfg.num_attention_heads

    def inputs():
        q = torch.randn(
            cfg.batch_size,
            cfg.seq_len,
            cfg.num_attention_heads * cfg.head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        k = torch.randn(
            cfg.batch_size,
            cfg.seq_len,
            cfg.num_key_value_heads * cfg.head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        v = torch.randn(
            cfg.batch_size,
            cfg.seq_len,
            cfg.num_key_value_heads * cfg.head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        position_ids = torch.arange(0, cfg.seq_len, device="cuda").unsqueeze(0)
        return q, k, v, position_ids

    def grads():
        grad = torch.randn(
            cfg.batch_size,
            cfg.num_attention_heads,
            cfg.seq_len,
            cfg.head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=False,
        )
        return grad

    # Manual IOBytes computes the total bandwidth for thunder backward trace.
    def iobytes():
        n_elements = 0
        # adding size of query_states.grad + key_states.grad +  value_states.grad
        n_elements += (
            3 * cfg.batch_size * cfg.num_attention_heads * cfg.seq_len * cfg.head_dim
        )
        # adding size of q.grad
        n_elements += (
            cfg.batch_size * cfg.seq_len * cfg.num_attention_heads * cfg.head_dim
        )
        # adding size of k.grad, v.grad
        n_elements += (
            2 * cfg.batch_size * cfg.seq_len * cfg.num_key_value_heads * cfg.head_dim
        )
        # matmul output size
        n_elements_matmul_out = head_dim / 2 * cfg.seq_len
        # totoal io sizes
        return (
            n_elements * torch.bfloat16.itemsize
            + n_elements_matmul_out * torch.float32.itemsize
        )

    return MistralNemoRope(cfg).cuda().bfloat16(), inputs, grads, iobytes


# The setup returns a function that would setup benchmark by returning:
#    fwd_model, inputs_fn, grads_fn, iobytes_fn
rope_setup = {
    "llama_2_7b_hf_rope": partial(llama_hf_rope, config_str="llama_2_7b_hf_rope"),
    "llama_3_8B_rope": partial(llama_hf_rope, config_str="llama_3_8B_rope"),
    "hf_qwen2_rope": hf_qwen2_rope,
    "hf_phi3_rope": hf_phi3_rope,
    "hf_mistral_nemo_rope": hf_mistral_nemo_rope,
}
