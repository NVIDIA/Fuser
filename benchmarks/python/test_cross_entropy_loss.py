# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch
from nvfuser import FusionDefinition, DataType

from .core import run_benchmark, with_executor, unary_bwd_torch, clear_dynamo_cache
from .cross_entropy_loss import (
    cross_entropy_loss_setup,
    SyntheticMiniModel,
)


@pytest.mark.parametrize(
    "variation",
    [
        "hf_qwen2",
        "hf_phi3",
        "hf_mistral_nemo",
    ],
)
@pytest.mark.parametrize(
    "executor", ["eager", "torchcompile", "thunder", "thunder-torchcompile"]
)
def test_cross_entropy_fwd_benchmark(
    benchmark,
    variation: str,
    executor: str,
):
    kwargs = {}
    if executor == "torchcompile":
        clear_dynamo_cache()

    test_case = cross_entropy_loss_setup[variation](dtype=torch.bfloat16)
    inputs = test_case.inputs()
    model = test_case.model()

    def fwd_call(inp):
        return model(**inp)

    # Compile the fwd fn for torchcompile
    benchmark_fn = with_executor(executor, fwd_call, **kwargs)
    run_benchmark(benchmark, benchmark_fn, inputs)


@pytest.mark.parametrize(
    "variation",
    [
        "hf_qwen2",
        "hf_phi3",
        "hf_mistral_nemo",
    ],
)
@pytest.mark.parametrize(
    "executor", ["eager", "torchcompile", "thunder", "thunder-torchcompile"]
)
def test_cross_entropy_bwd_benchmark(
    benchmark,
    variation: str,
    executor: str,
):
    kwargs = {}
    if executor == "torchcompile":
        clear_dynamo_cache()

    test_case = cross_entropy_loss_setup[variation](dtype=torch.bfloat16)
    fwd_inputs = test_case.inputs()
    model = test_case.model()

    def fwd_call(inp):
        return model(**inp)

    # execute the compiled fwd fn
    fwd_fn = with_executor(executor, fwd_call, **kwargs)
    outputs = fwd_fn(fwd_inputs)

    assert len(outputs) == 1

    run_benchmark(
        benchmark,
        unary_bwd_torch,
        [outputs[0], test_case.grads(), *fwd_inputs, *list(model.parameters())],
        iobytes=test_case.grad_iobytes(),
    )


@pytest.mark.parametrize(
    "executor", ["eager", "torchcompile", "thunder", "thunder-torchcompile"]
)
@pytest.mark.parametrize("vocab_size", SyntheticMiniModel.generate_vocab_sizes())
def test_cross_entropy_mini_benchmark_fwd(benchmark, executor: str, vocab_size: int):
    if executor == "torchcompile":
        clear_dynamo_cache()

    # picking a value that doesn't OOM for large vocab sizes
    batch_size = 4096

    def fwd_call(inp):
        return SyntheticMiniModel.mini_model(*inp)

    inputs = SyntheticMiniModel.inputs(int(batch_size), int(vocab_size))

    fwd_fn = with_executor(executor, fwd_call)
    run_benchmark(benchmark, fwd_fn, inputs)


@pytest.mark.parametrize(
    "executor", ["eager", "torchcompile", "thunder", "thunder-torchcompile"]
)
@pytest.mark.parametrize("vocab_size", SyntheticMiniModel.generate_vocab_sizes())
def test_cross_entropy_mini_benchmark_bwd(benchmark, executor: str, vocab_size: int):
    if executor == "torchcompile":
        clear_dynamo_cache()

    # picking a value that doesn't OOM for large vocab sizes
    batch_size = 4096

    def fwd_call(inp):
        return SyntheticMiniModel.mini_model(*inp)

    inputs = SyntheticMiniModel.inputs(batch_size, vocab_size)

    fwd_fn = with_executor(executor, fwd_call)
    outputs = fwd_fn(inputs)
    grads = SyntheticMiniModel.grads()
    run_benchmark(benchmark, unary_bwd_torch, [outputs, grads, *inputs])


def nvfuser_cross_entropy_fusion(fd: FusionDefinition) -> None:
    """
    NvFuser fusion definition for torch.nn.functional.cross_entropy(logits, labels, reduction='none')
    1. Compute LSE (log-sum-exp) for each row
    2. Access target logit directly: logits[i, labels[i]]
    3. Compute loss: loss = lse - target_logit
    """
    # Input tensors
    T0 = fd.define_tensor(
        shape=[-1, -1],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T1 = fd.define_tensor(
        shape=[-1],
        contiguity=[True],
        dtype=DataType.Int,
        is_cpu=False,
        stride_order=[0],
    )
    T2 = fd.ops.cast(T0, dtype=DataType.Float)
    T3 = fd.ops.max(T2, dims=[1], keepdim=False, dtype=DataType.Null)
    V4 = fd.define_vector([T0.size(0), 1], dtype=DataType.Int)
    T5 = fd.ops.broadcast_in_dim(T3, shape=V4, broadcast_dims=[0])
    V6 = fd.define_vector([T0.size(0), T0.size(1)], dtype=DataType.Int)
    T11 = fd.ops.broadcast_in_dim(T5, shape=V6, broadcast_dims=[0, 1])
    T12 = fd.ops.sub(T2, T11)
    T13 = fd.ops.exp(T12)
    T14 = fd.ops.sum(T13, dims=[1], keepdim=False, dtype=DataType.Null)
    T15 = fd.ops.log(T14)
    T16 = fd.ops.add(T15, T3)
    T17 = fd.ops.broadcast_in_dim(T1, shape=V4, broadcast_dims=[0])
    T18 = fd.ops.gather(T0, T17, dim=1)
    T19 = fd.ops.squeeze(T18, dims=[1])
    T20 = fd.ops.cast(T19, dtype=DataType.Float)
    T21 = fd.ops.sub(T16, T20)
    T22 = fd.ops.cast(T21, dtype=DataType.BFloat16)
    fd.add_output(T22)


def torch_cross_entropy(logits, labels):
    return torch.nn.functional.cross_entropy(logits, labels, reduction="none")


@pytest.mark.parametrize("vocab_size", SyntheticMiniModel.generate_vocab_sizes())
def test_cross_entropy_nvf_benchmark(
    benchmark,
    vocab_size: int,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    with FusionDefinition() as fd:
        nvfuser_cross_entropy_fusion(fd)

    batch_size = 4096
    logits = torch.randn(batch_size, vocab_size, device="cuda", dtype=torch.bfloat16)
    labels = torch.randint(
        0, vocab_size, (batch_size,), device="cuda", dtype=torch.int64
    )
    inputs = [logits, labels]
    if not disable_validation:
        fd.validate(inputs, [torch_cross_entropy(*inputs)])

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, inputs)
