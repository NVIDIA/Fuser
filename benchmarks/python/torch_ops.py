# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn.functional as F


def dropout_layernorm(inputs: list):
    inp1, inp2, weights, bias, dropout_p = inputs
    return F.layer_norm(
        inp2 + torch.nn.functional.dropout(inp1, p=dropout_p),
        normalized_shape=inp1.shape[1:],
        weight=weights,
        bias=bias,
    )


def dropout_rmsnorm(inputs: list):
    inp1, inp2, weights, dropout_p = inputs
    x = inp2 + F.dropout(inp1, p=dropout_p)
    output = weights * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
    return output


def gelu(inputs: list):
    inp, bias = inputs
    return F.gelu(inp + bias, approximate="tanh")


def huggingface_attn(inputs: list):
    # Reference implementation in Thunder: https://github.com/Lightning-AI/lightning-thunder/blob/888b46324462fba70f93d5017bc0d99025f05091/thunder/tests/hf_bart_self_attn.py#L73-L83
    inp, attention_mask, size, dropout_p = inputs
    batch_size, seq_len, nh, n_embd = size
    attn = (inp + attention_mask).view(batch_size * nh, seq_len, seq_len)
    attn = F.softmax(attn, dim=-1)
    output = F.dropout(attn, p=dropout_p)
    return output


def layernorm(inputs: list):
    inp, weights, bias = inputs
    return F.layer_norm(
        inp,
        inp.shape[1:],
        weight=weights,
        bias=bias,
    )


def nanogpt_attn(inputs: list):
    # Reference implementation from Thunder: https://github.com/Lightning-AI/lightning-thunder/blob/d3da8517bff02a913fd149b4d6559f6b5a4c6c7f/thunder/tests/nanogpt_model.py#L102-L106
    inp, bias, size, dropout_p = inputs
    batch_size, seq_len, nh, n_embd = size
    hs = n_embd // nh
    attn = inp / (hs**0.5)
    attn = attn.masked_fill(bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))
    attn = F.softmax(attn, dim=-1)
    output = F.dropout(attn, p=dropout_p)
    return output


def rmsnorm(inputs: list):
    inp, weights = inputs
    squared_mean = (inp**2).mean(1, keepdim=True)
    rms_eps = torch.sqrt(squared_mean + 1e-5)
    output = weights * (inp / rms_eps)
    return output


def scale_bias_relu(inputs: list):
    inp, scale, bias = inputs
    return F.relu(inp * scale + bias)


def silu_mul(inputs: list):
    x, y = inputs
    return F.silu(x) * y


def softmax(inputs: list):
    inp, reduction_axis = inputs
    return F.softmax(inp, dim=reduction_axis)


def embedding(inputs: list):
    indices, embedding_table = inputs
    return F.embedding(indices, embedding_table)


def scatter_reduce(inputs: list):
    bmm_out: torch.Tensor  # [seq*top_k, hidden]
    idxs: torch.Tensor  # [seq*top_k]
    topk_weight: torch.Tensor  # [seq , top_k]]
    bmm_out, idxs, topk_weight = inputs
    out = bmm_out.index_put([idxs], bmm_out)  # [seq*top_k, hidden]
    out = out.reshape(*topk_weight.shape, -1)  # [seq, top_k, hidden]
    out = out * topk_weight.unsqueeze(-1)  # [seq, top_k, hidden]
    return out.sum(dim=1)  # [seq, hidden]
