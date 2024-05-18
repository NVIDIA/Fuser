# Model is based on LitGPT:
# https://github.com/Lightning-AI/litgpt/blob/main/litgpt/model.py
import math
import time
from dataclasses import dataclass

from typing import Optional, Tuple

import torch
from torch import nn

BATCH_SIZE = 2

import thunder
import nvfuser

# Config for Mistral 7B
# https://github.com/Lightning-AI/litgpt/blob/e60f21a49a435efd215cb858200396f7b24baf17/litgpt/config.py#L1375-L1389
@dataclass
class Config:
    name = "Mistral-7B-v0.1"
    n_layer = 1  # NOTE: Use just one transformer block! Should be 32 for real 7B model
    ### n_embd, intermediate_size, n_query_groups, n_head, head_size have impact on the matmul sizes
    n_embd = 4096
    intermediate_size = 14336
    n_query_groups = 8
    n_head = 32
    head_size = 128
    ###
    norm_eps = 1e-05
    bias = False
    lm_head_bias = False
    block_size = 4096
    padded_vocab_size = 32000
    rope_n_elem = head_size


class GPT(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.lm_head = nn.Linear(
            config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias
        )
        self.wte = nn.Embedding(config.padded_vocab_size, config.n_embd)
        self.transformer_blocks = nn.ModuleList(
            Block(config) for _ in range(config.n_layer)
        )
        self.ln_f = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.max_seq_length = self.config.block_size

        cos, sin = self.rope_cache()
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        T = idx.shape[1]
        if self.max_seq_length < T:
            raise ValueError(
                f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}."
            )

        cos = self.cos[:T]
        sin = self.sin[:T]

        x = self.wte(idx)  # token embeddings of shape (b, t, n_embd)

        for block in self.transformer_blocks:
            x = block(x, cos, sin)

        x = self.ln_f(x)
        return self.lm_head(x)  # (b, t, vocab_size)

    def rope_cache(
        self, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return build_rope_cache(
            seq_len=self.max_seq_length,
            n_elem=self.config.rope_n_elem,
            device=device,
        )

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    device: Optional[torch.device] = None,
    base: int = 10000,
    condense_ratio: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Enhanced Transformer with Rotary Position Embedding.
    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))
    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio
    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)
    return torch.cos(idx_theta), torch.sin(idx_theta)


class Block(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.norm_1 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.mlp = LLaMAMLP(config)

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x_normed = self.norm_1(x)
        attention_output = self.attn(x_normed, cos, sin)
        x = attention_output + x
        x = self.mlp(self.norm_2(x)) + x
        return x


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size

        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)

        # output projection
        # if `head_size` is explicitly specified in the config, `n_emd` might not be equal to `head_size * n_head`
        self.proj = nn.Linear(
            config.head_size * config.n_head, config.n_embd, bias=config.bias
        )

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x)

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
        if (
            self.config.n_query_groups != self.config.n_head
            and self.config.n_query_groups != 1
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

        y = self.scaled_dot_product_attention(q, k, v)

        y = y.reshape(
            B, T, self.config.head_size * self.config.n_head
        )  # re-assemble all head outputs side by side

        # output projection
        return self.proj(y)

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.config.head_size)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, scale=scale, is_causal=True
        )
        return y.transpose(1, 2)


class LLaMAMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.fc_2 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        return self.proj(x)


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization.
    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        x_normed = x_normed.to(dtype=dtype)
        return x_normed * self.weight


config = Config()
init_device = torch.device("cuda")
with init_device:
    model = GPT(config).to(dtype=torch.bfloat16)

input_shape = (BATCH_SIZE, config.block_size)
x = torch.randint(
    0, config.padded_vocab_size, input_shape, dtype=torch.int64, device="cuda"
)

# jit_model = thunder.jit(model, nv_enable_linear=True, nv_enable_matmul=True)

# out = jit_model(x)

import torch
from nvfuser import FusionDefinition, DataType

def nvfuser_fusion_id14(fd : FusionDefinition) -> None :
    T0 = fd.define_tensor(shape=[-1, -1, -1], contiguity=[None, None, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 1, 0])
    T1 = fd.define_tensor(shape=[-1, -1, -1], contiguity=[True, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 1, 0])
    T2 = fd.define_tensor(shape=[-1, -1, -1, -1], contiguity=[None, None, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[3, 2, 1, 0])
    T3 = fd.define_tensor(shape=[-1, -1, -1, -1], contiguity=[None, None, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[3, 2, 1, 0])
    T4 = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
    T5 = fd.ops.cast(T1, dtype=DataType.Float)
    T6 = fd.ops.mul(T5, T5)
    T7 = fd.ops.sum(T6, dims=[2], keepdim=False, dtype=DataType.Null)
    S8 = fd.define_scalar(2, dtype=DataType.Int)
    S9 = fd.define_scalar(4096, dtype=DataType.Int)
    S10 = fd.define_scalar(1, dtype=DataType.Int)
    V11 = fd.define_vector([S8, S9, S10], dtype=DataType.Int)
    T12 = fd.ops.broadcast_in_dim(T7, shape=V11, broadcast_dims=[0, 1])
    S13 = fd.define_scalar(4096.00, dtype=DataType.Double)
    S14 = fd.ops.reciprocal(S13)
    T15 = fd.ops.mul(T12, S14)
    S16 = fd.define_scalar(1.00000e-05, dtype=DataType.Double)
    T17 = fd.ops.add(T15, S16)
    T18 = fd.ops.rsqrt(T17)
    S19 = fd.define_scalar(2, dtype=DataType.Int)
    S20 = fd.define_scalar(4096, dtype=DataType.Int)
    S21 = fd.define_scalar(4096, dtype=DataType.Int)
    V22 = fd.define_vector([S19, S20, S21], dtype=DataType.Int)
    T23 = fd.ops.broadcast_in_dim(T18, shape=V22, broadcast_dims=[0, 1, 2])
    T24 = fd.ops.mul(T5, T23)
    T25 = fd.ops.cast(T0, dtype=DataType.Float)
    T26 = fd.ops.mul(T24, T25)
    T27 = fd.ops.cast(T26, dtype=DataType.BFloat16)
    T28 = fd.ops.linear(T27, T4)
    S29 = fd.define_scalar(2, dtype=DataType.Int)
    S30 = fd.define_scalar(4096, dtype=DataType.Int)
    S31 = fd.define_scalar(8, dtype=DataType.Int)
    S32 = fd.define_scalar(6, dtype=DataType.Int)
    S33 = fd.define_scalar(128, dtype=DataType.Int)
    V34 = fd.define_vector([S29, S30, S31, S32, S33], dtype=DataType.Int)
    T35 = fd.ops.reshape(T28, new_shape=V34)
    T36 = fd.ops.permute(T35, dims=[0, 2, 3, 1, 4])
    
    # Q, K, V
    T37 = fd.ops.slice(T36, start_indices=[0, 0, 0, 0, 0], end_indices=[2, 8, 4, 4096, 128], strides=[1, 1, 1, 1, 1])
    T38 = fd.ops.slice(T36, start_indices=[0, 0, 4, 0, 0], end_indices=[2, 8, 5, 4096, 128], strides=[1, 1, 1, 1, 1])
    T39 = fd.ops.slice(T36, start_indices=[0, 0, 5, 0, 0], end_indices=[2, 8, 6, 4096, 128], strides=[1, 1, 1, 1, 1])

    S40 = fd.define_scalar(2, dtype=DataType.Int)
    S41 = fd.define_scalar(8, dtype=DataType.Int)
    S42 = fd.define_scalar(4, dtype=DataType.Int)
    S43 = fd.define_scalar(4096, dtype=DataType.Int)
    S44 = fd.define_scalar(128, dtype=DataType.Int)
    V45 = fd.define_vector([S40, S41, S42, S43, S44], dtype=DataType.Int)
    T46 = fd.ops.broadcast_in_dim(T38, shape=V45, broadcast_dims=[0, 1, 2, 3, 4])
    S47 = fd.define_scalar(2, dtype=DataType.Int)
    S48 = fd.define_scalar(32, dtype=DataType.Int)
    S49 = fd.define_scalar(4096, dtype=DataType.Int)
    S50 = fd.define_scalar(128, dtype=DataType.Int)
    V51 = fd.define_vector([S47, S48, S49, S50], dtype=DataType.Int)
    T52 = fd.ops.reshape(T37, new_shape=V51)

    S53 = fd.define_scalar(2, dtype=DataType.Int)
    S54 = fd.define_scalar(32, dtype=DataType.Int)
    S55 = fd.define_scalar(4096, dtype=DataType.Int)
    S56 = fd.define_scalar(128, dtype=DataType.Int)
    V57 = fd.define_vector([S53, S54, S55, S56], dtype=DataType.Int)
    T58 = fd.ops.reshape(T46, new_shape=V57)

    T59 = fd.ops.slice(T52, start_indices=[0, 0, 0, 0], end_indices=[2, 32, 4096, 128], strides=[1, 1, 1, 1])

    # RoPE: Q
    T60 = fd.ops.slice(T59, start_indices=[0, 0, 0, 0], end_indices=[2, 32, 4096, 64], strides=[1, 1, 1, 1])
    T61 = fd.ops.slice(T59, start_indices=[0, 0, 0, 64], end_indices=[2, 32, 4096, 128], strides=[1, 1, 1, 1])
    T62 = fd.ops.cast(T61, dtype=DataType.Float)
    T63 = fd.ops.neg(T62)
    T64 = fd.ops.cast(T63, dtype=DataType.BFloat16)
    T65 = fd.ops.cat([T64, T60], dim=-1)
    T66 = fd.ops.cast(T59, dtype=DataType.Float)
    T67 = fd.ops.cast(T2, dtype=DataType.Float)
    T68 = fd.ops.mul(T66, T67)
    T69 = fd.ops.cast(T65, dtype=DataType.Float)
    T70 = fd.ops.cast(T3, dtype=DataType.Float)
    T71 = fd.ops.mul(T69, T70)
    T72 = fd.ops.add(T68, T71)
    T73 = fd.ops.cast(T72, dtype=DataType.BFloat16)

    #RoPE: K
    T74 = fd.ops.slice(T58, start_indices=[0, 0, 0, 0], end_indices=[2, 32, 4096, 128], strides=[1, 1, 1, 1])
    T75 = fd.ops.slice(T74, start_indices=[0, 0, 0, 0], end_indices=[2, 32, 4096, 64], strides=[1, 1, 1, 1])
    T76 = fd.ops.slice(T74, start_indices=[0, 0, 0, 64], end_indices=[2, 32, 4096, 128], strides=[1, 1, 1, 1])
    T77 = fd.ops.cast(T76, dtype=DataType.Float)
    T78 = fd.ops.neg(T77)
    T79 = fd.ops.cast(T78, dtype=DataType.BFloat16)
    T80 = fd.ops.cat([T79, T75], dim=-1)
    T81 = fd.ops.cast(T74, dtype=DataType.Float)
    T82 = fd.ops.mul(T81, T67)
    T83 = fd.ops.cast(T80, dtype=DataType.Float)
    T84 = fd.ops.mul(T83, T70)
    T85 = fd.ops.add(T82, T84)
    T86 = fd.ops.cast(T85, dtype=DataType.BFloat16)

    T87 = fd.ops.slice(T52, start_indices=[0, 0, 0, 0], end_indices=[2, 32, 4096, 1], strides=[1, 1, 1, 1])
    T88 = fd.ops.cat([T73, T87], dim=-1)

    # T89 = fd.ops.slice(T58, start_indices=[0, 0, 0, 0], end_indices=[2, 32, 4096, 128], strides=[1, 1, 1, 1])
    # T90 = fd.ops.cat([T86, T89], dim=-1)
    
    fd.add_output(T18)
    fd.add_output(T27)
    fd.add_output(T39)
    fd.add_output(T88)
    # fd.add_output(T90)

with FusionDefinition() as fd:
    nvfuser_fusion_id14(fd)

# def cat_repro(fd : FusionDefinition) -> None :
    # T52 = fd.define_tensor(shape=[-1, -1, -1, -1], contiguity=[True, True, True, True], dtype=DataType.BFloat16, is_cpu=False)
    # T86 = fd.define_tensor(shape=[-1, -1, -1, -1], contiguity=[True, True, True, True], dtype=DataType.BFloat16, is_cpu=False)
    # T87 = fd.ops.slice(T52, start_indices=[0, 0, 0, 0], end_indices=[2, 32, 4096, 0], strides=[1, 1, 1, 1])
    # T88 = fd.ops.cat([T73, T87], dim=-1)
    # T89 = fd.ops.slice(T58, start_indices=[0, 0, 0, 0], end_indices=[2, 32, 4096, 0], strides=[1, 1, 1, 1])

    # T90 = fd.ops.cat([T86, T89], dim=-1)

inputs = [
    torch.randn((4096,), dtype=torch.bfloat16, device='cuda:0').as_strided((2, 4096, 4096), (0, 0, 1)),
    torch.randn((33554432,), dtype=torch.bfloat16, device='cuda:0').as_strided((2, 4096, 4096), (16777216, 4096, 1)),
    torch.randn((524288,), dtype=torch.bfloat16, device='cuda:0').as_strided((2, 32, 4096, 128), (0, 0, 128, 1)),
    torch.randn((524288,), dtype=torch.bfloat16, device='cuda:0').as_strided((2, 32, 4096, 128), (0, 0, 128, 1)),
    torch.randn((25165824,), dtype=torch.bfloat16, device='cuda:0').as_strided((6144, 4096), (4096, 1)),
]
fd.execute(inputs)

