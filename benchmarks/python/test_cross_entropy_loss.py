# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest

import torch
from nvfuser_direct import FusionDefinition, DataType
from .core import (
    run_benchmark,
    with_executor,
    unary_bwd_torch,
    clear_dynamo_cache,
    check_module_available,
)
from .cross_entropy_loss import (
    cross_entropy_loss_setup,
    SyntheticMiniModel,
)
from .torch_ops import cross_entropy as torch_cross_entropy_fwd


if check_module_available("quack"):
    from quack.cross_entropy import cross_entropy_fwd as quack_cross_entropy_fwd


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


# Simple test of F.cross_entropy without final reduction
# thunder won't use nvFuser for this fusion, here we manually define the fusion
def nvfuser_cross_entropy_fusion(
    fd: FusionDefinition, batch_size: int, vocab_size: int
) -> None:
    """
    NvFuser fusion definition for torch.nn.functional.cross_entropy(logits, labels, reduction='none')
    1. Compute LSE (log-sum-exp) for each row
    2. Access target logit directly: logits[i, labels[i]]
    3. Compute loss: loss = lse - target_logit
    """
    # Input tensors
    T0 = fd.define_tensor(
        shape=[batch_size, vocab_size],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T1 = fd.define_tensor(
        shape=[batch_size],
        contiguity=[True],
        dtype=DataType.Int,
        is_cpu=False,
        stride_order=[0],
    )
    T2 = fd.ops.cast(T0, dtype=DataType.Float)
    T3 = fd.ops.max(T2, dims=[1], keepdim=False, dtype=DataType.Null)
    T4 = fd.ops.broadcast_in_dim(T3, shape=[batch_size, 1], broadcast_dims=[0])
    T11 = fd.ops.broadcast_in_dim(
        T4, shape=[batch_size, vocab_size], broadcast_dims=[0, 1]
    )
    T12 = fd.ops.sub(T2, T11)
    T13 = fd.ops.exp(T12)
    T14 = fd.ops.sum(T13, dims=[1], keepdim=False, dtype=DataType.Null)
    T15 = fd.ops.log(T14)
    T16 = fd.ops.add(T15, T3)
    T17 = fd.ops.broadcast_in_dim(T1, shape=[batch_size, 1], broadcast_dims=[0])
    T18 = fd.ops.gather(T0, T17, dim=1)
    T19 = fd.ops.squeeze(T18, dims=[1])
    T20 = fd.ops.cast(T19, dtype=DataType.Float)
    T21 = fd.ops.sub(T16, T20)
    T22 = fd.ops.cast(T21, dtype=DataType.BFloat16)
    fd.add_output(T22)


@pytest.mark.parametrize("vocab_size", SyntheticMiniModel.sizes_from_models)
def test_function_cross_entropy_fwd_nvf_benchmark(
    benchmark, vocab_size: int, disable_validation: bool, disable_benchmarking: bool
):
    batch_size = 4096
    inputs = [
        0.1
        * torch.randn(
            batch_size,
            vocab_size,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=False,
        ),
        torch.randint(
            0,
            vocab_size,
            (batch_size,),
            device="cuda",
            dtype=torch.int64,
            requires_grad=False,
        ),
    ]
    with FusionDefinition() as fd:
        nvfuser_cross_entropy_fusion(fd, batch_size, vocab_size)

    if not disable_validation:
        fd.validate(inputs, [torch_cross_entropy_fwd(inputs)])

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, inputs)


def quack_cross_entropy_fwd_wrapper(inputs: list):
    return quack_cross_entropy_fwd(*inputs, return_dx=False)


# cross entropy in thunder is split into log_softmax and nll_loss without
# using nvFuser, the performance using nvFuser should be measured using
# test_function_cross_entropy_fwd_nvf_benchmark
@pytest.mark.parametrize(
    "executor",
    [
        "eager",
        "torchcompile",
        "thunder-torchcompile",
        pytest.param(
            "quack",
            marks=pytest.mark.skipif(
                not check_module_available("quack"),
                reason="quack executor is not available on this device",
            ),
        ),
    ],
)
@pytest.mark.parametrize("vocab_size", SyntheticMiniModel.sizes_from_models)
def test_function_cross_entropy_fwd_benchmark(
    benchmark, executor: str, vocab_size: int
):
    if executor == "torchcompile":
        clear_dynamo_cache()

    batch_size = 4096
    # vocab_size is large, scale by 0.1 to avoid overflow and represent realistic values, same used in quack test
    logits = 0.1 * torch.randn(
        batch_size, vocab_size, device="cuda", dtype=torch.bfloat16, requires_grad=False
    )
    labels = torch.randint(
        0,
        vocab_size,
        (batch_size,),
        device="cuda",
        dtype=torch.int64,
        requires_grad=False,
    )

    if executor == "quack":
        fwd_fn = quack_cross_entropy_fwd_wrapper
    else:
        fwd_fn = with_executor(executor, torch_cross_entropy_fwd)

    run_benchmark(benchmark, fwd_fn, [logits, labels])
