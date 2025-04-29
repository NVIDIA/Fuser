# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
from functools import partial


def llama_hf_cfg(config_str):
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

    configs = {}
    configs["llama_2_7b_hf"] = Config(
        n_head=32,
        head_size=128,
        n_query_groups=32,
        rope_n_elem=128,
        batches=2,
        seq_length=4096,
    )
    configs["llama_3_8B"] = Config(
        n_head=32,
        head_size=128,
        n_query_groups=8,
        rope_n_elem=128,
        batches=2,
        seq_length=8192,
    )

    return configs[config_str]


def hf_qwen2_cfg():
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    config.batch_size = 1
    config.seq_len = 4096
    config._attn_implementation = "sdpa"
    return config


def hf_phi3_cfg():
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    config.batch_size = 1
    config.seq_len = 8192
    config._attn_implementation = "sdpa"
    return config


def hf_mistral_nemo_cfg():
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

    cfg = MistralConfig.from_dict(json.loads(mistral_cfg_str))
    cfg.batch_size = 1
    cfg.seq_len = 4096
    cfg._attn_implementation = "sdpa"

    return cfg


def litgpt_cfg(model_name):
    import litgpt

    cfg = litgpt.Config.from_name(model_name)
    cfg.batch_size = 1
    cfg.seq_len = 4096
    cfg.name_or_path = model_name

    return cfg


configs = {
    "llama_2_7b_hf": partial(llama_hf_cfg, config_str="llama_2_7b_hf"),
    "llama_3_8B": partial(llama_hf_cfg, config_str="llama_3_8B"),
    "hf_qwen2": hf_qwen2_cfg,
    "hf_phi3": hf_phi3_cfg,
    "hf_mistral_nemo": hf_mistral_nemo_cfg,
    "litgpt": litgpt_cfg,
}
