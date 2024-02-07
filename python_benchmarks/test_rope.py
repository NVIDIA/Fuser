import pytest
from nvfuser import FusionDefinition, DataType
from .core import run_benchmark, clear_cuda_cache
import torch


def rope_fusion(fd: FusionDefinition) -> None:
    q = fd.define_tensor(
        shape=[32, 4096, 32, 128],
        dtype=DataType.BFloat16,
    )
    cos = fd.define_tensor(
        shape=[4096, 128],
        dtype=DataType.BFloat16,
    )
    sin = fd.define_tensor(
        shape=[4096, 128],
        dtype=DataType.BFloat16,
    )

    q = fd.ops.permute(q, dims=[0, 2, 1, 3])
    q_left = fd.ops.slice(
        q,
        start_indices=[0, 0, 0, 0],
        end_indices=[32, 32, 4096, 64],
        strides=[1, 1, 1, 1],
    )
    q_right = fd.ops.slice(
        q,
        start_indices=[0, 0, 0, 64],
        end_indices=[32, 32, 4096, 128],
        strides=[1, 1, 1, 1],
    )

    q_right = fd.ops.cast(q_right, dtype=DataType.Float)
    q_right = -q_right
    q_right = fd.ops.cast(q_right, dtype=DataType.BFloat16)

    q_rotated = fd.ops.cat([q_right, q_left], dim=-1)

    cos = fd.ops.broadcast_in_dim(
        cos, shape=[1, 1, 4096, 128], broadcast_dims=[2, 3]
    )
    sin = fd.ops.broadcast_in_dim(
        sin, shape=[1, 1, 4096, 128], broadcast_dims=[2, 3]
    )
    fd.add_output(q * cos + q_rotated * sin)


# Idea from @nikitaved. We `cat` (or `stack` to be precise) the embeddings, which are constant, and don't `cat` in the fusion.
def rope_without_cat_fusion(fd: FusionDefinition) -> None:
    q = fd.define_tensor(
        shape=[32, 4096, 32, 128],
        dtype=DataType.BFloat16,
    )
    cos_sin_matrix = fd.define_tensor(
        shape=[4096, 2, 64, 2],
        dtype=DataType.BFloat16,
    )

    q = fd.ops.reshape(q, new_shape=[32, 4096, 32, 2, 64])
    q = fd.ops.permute(q, dims=[0, 2, 1, 4, 3])
    q = fd.ops.broadcast_in_dim(q, shape=[32, 32, 4096, 1, 64, 2], broadcast_dims=[0, 1, 2, 4, 5])

    cos_sin_matrix = fd.ops.broadcast_in_dim(cos_sin_matrix, shape=[32, 32, 4096, 2, 64, 2], broadcast_dims=[2, 3, 4, 5])

    out = fd.ops.sum(q * cos_sin_matrix, [-1])
    out = fd.ops.reshape(out, new_shape=[32, 32, 4096, 128])
    fd.add_output(out)


def test_rope_benchmark(benchmark, disable_validation: bool, disable_benchmarking: bool):
    clear_cuda_cache()

    with FusionDefinition() as fd:
        rope_fusion(fd)

    q = torch.randn(32, 4096, 32, 128, dtype=torch.bfloat16, device="cuda:0")
    freqs = torch.randn(4096, 64, dtype=torch.bfloat16, device="cuda:0")
    emb = torch.concat([freqs, freqs], dim=-1)
    inputs = [q, emb.cos(), emb.sin()]

    if not disable_validation:
        with FusionDefinition() as fd_ref:
            rope_without_cat_fusion(fd_ref)

        cos_and_minus_sin = torch.stack([freqs.cos(), -freqs.sin()], dim=-1)
        sin_and_cos = torch.stack([freqs.sin(), freqs.cos()], dim=-1)
        cos_sin_matrix = torch.stack([cos_and_minus_sin, sin_and_cos], dim=1)
        rope_without_cat_out = fd_ref.execute([q, cos_sin_matrix])
        fd.validate(inputs, rope_without_cat_out)

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, inputs)
