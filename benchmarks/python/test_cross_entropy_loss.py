# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest

import torch

from .core import run_benchmark, with_executor, unary_bwd_torch, clear_dynamo_cache
from .cross_entropy_loss import (
    cross_entropy_loss_setup,
    SyntheticMiniModel,
)

from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype


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


def nvfuser_fusion_id1(fd : FusionDefinition) -> None :
    T0 = fd.define_tensor(shape=[4096], contiguity=[True], dtype=DataType.Int, is_cpu=False, stride_order=[0])
    T1 = fd.define_tensor(shape=[1, 4096, -1], contiguity=[None, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 1, 0])
    V00 = fd.ops.shape(T0)
    Shape_T0 = fd.ops.at(V00, index=-1)
    V01 = fd.ops.shape(T1)
    Shape_T1_0 = fd.ops.at(V01, index=0)
    Shape_T1_1 = fd.ops.at(V01, index=1)
    Shape_T1_2 = fd.ops.at(V01, index=2)
    S1 = fd.define_scalar(1, dtype=DataType.Int)
    S1 = fd.ops.add(S1, Shape_T0)
    S2 = fd.define_scalar(0, dtype=DataType.Int)
    T6 = fd.ops.pad(T0, [0, 1], S2)
    T13 = fd.ops.slice(T6, start_indices=[1], end_indices=[S1], strides=[1], manual_normalization=0)
    T14 = fd.ops.cast(T1, dtype=DataType.Float)
    T15 = fd.ops.squeeze(T14, dims=[0], squeeze_expanded=False)
    S16 = fd.define_scalar(-100, dtype=DataType.Int)
    S17 = fd.define_scalar(0.00000, dtype=DataType.Float)
    T18 = fd.ops.ne(T13, S16)
    T19 = fd.ops.where(T18, T13, S17)
    V20 = fd.ops.shape(T13)
    S21 = fd.ops.at(V20, index=-1)
    T24 = fd.ops.broadcast_in_dim(T19, shape=[S21, 1], broadcast_dims=[0])
    T25 = fd.ops.take_along_axis(T15, T24, dim=1)
    V26 = fd.ops.shape(T13)
    S27 = fd.ops.at(V26, index=-1)
    T29 = fd.ops.reshape(T25, new_shape=[S27])
    T30 = fd.ops.max(T15, dims=[1], keepdim=False, dtype=DataType.Null)
    V31 = fd.ops.shape(T13)
    S32 = fd.ops.at(V31, index=-1)
    T35 = fd.ops.broadcast_in_dim(T30, shape=[S32, 1], broadcast_dims=[0])
    T15_ = fd.ops.segment_set(T1)
    T15_ = fd.ops.cast(T15_, dtype=DataType.Float)
    T15_ = fd.ops.squeeze(T15_, dims=[0], squeeze_expanded=False)
    T36 = fd.ops.sub(T15_, T35)
    T37 = fd.ops.exp(T36)
    T38 = fd.ops.sum(T37, dims=[1], keepdim=False, dtype=DataType.Null)
    T39 = fd.ops.log(T38)
    T40 = fd.ops.sub(T29, T30)
    T41 = fd.ops.sub(T40, T39)
    T42 = fd.ops.neg(T41)
    T43 = fd.ops.where(T18, T42, S17)
    T44 = fd.ops.sum(T18, dims=[0], keepdim=False, dtype=DataType.Null)
    T45 = fd.ops.cast(T44, dtype=DataType.Float)
    T46 = fd.ops.sum(T43, dims=[0], keepdim=False, dtype=DataType.Null)
    T47 = fd.ops.div(T46, T45)
    fd.add_output(T47)



def nvfuser_fusion_id0(fd: FusionDefinition, inputs) -> None:
    # T0 = fd.define_tensor(
    #     shape=[1, 4096, 152064],
    #     contiguity=[None, True, True],
    #     dtype=DataType.BFloat16,
    #     is_cpu=False,
    #     stride_order=[2, 1, 0],
    # )
    # T1 = fd.define_tensor(
    #     shape=[1, 4096],
    #     contiguity=[None, True],
    #     dtype=DataType.Int,
    #     is_cpu=False,
    #     stride_order=[1, 0],
    # )
    T0 = fd.from_pytorch(inputs[0])
    T1 = fd.from_pytorch(inputs[1])
    T2 = fd.ops.cast(T0, dtype=DataType.Float)
    S3 = fd.define_scalar(-100, dtype=DataType.Int)
    T9 = fd.ops.pad(T1, [0, 1, 0, 0], S3)
    T19 = fd.ops.slice(
        T9,
        start_indices=[0, 1],
        end_indices=[1, 4097],
        strides=[1, 1],
        manual_normalization=0,
    )
    T20 = fd.ops.stride_order(T19, stride_order=[1, 0])
    T24 = fd.ops.reshape(T2, new_shape=[4096, 152064])
    T27 = fd.ops.reshape(T20, new_shape=[4096])
    S28 = fd.define_scalar(-100, dtype=DataType.Int)
    S29 = fd.define_scalar(0.00000, dtype=DataType.Float)
    T30 = fd.ops.ne(T27, S28)
    T31 = fd.ops.where(T30, T27, S29)
    V32 = fd.ops.shape(T27)
    S33 = fd.ops.at(V32, index=-1)
    T36 = fd.ops.broadcast_in_dim(T31, shape=[S33, 1], broadcast_dims=[0])
    T37 = fd.ops.take_along_axis(T24, T36, dim=1)
    V38 = fd.ops.shape(T27)
    S39 = fd.ops.at(V38, index=-1)
    T41 = fd.ops.reshape(T37, new_shape=[S39])
    T42 = fd.ops.max(T24, dims=[1], keepdim=False, dtype=DataType.Null)
    V43 = fd.ops.shape(T27)
    S44 = fd.ops.at(V43, index=-1)
    T47 = fd.ops.broadcast_in_dim(T42, shape=[S44, 1], broadcast_dims=[0])
    T0_ = fd.ops.segment_set(T0)
    T0_ = fd.ops.cast(T0_, dtype=DataType.Float)
    T24_ = fd.ops.reshape(T0_, new_shape=[4096, 152064])
    T48 = fd.ops.sub(T24_, T47)
    T49 = fd.ops.exp(T48)
    T50 = fd.ops.sum(T49, dims=[1], keepdim=False, dtype=DataType.Null)
    T51 = fd.ops.log(T50)
    T52 = fd.ops.sub(T41, T42)
    T53 = fd.ops.sub(T52, T51)
    T54 = fd.ops.neg(T53)
    T55 = fd.ops.where(T30, T54, S29)
    T56 = fd.ops.sum(T30, dims=[0], keepdim=False, dtype=DataType.Null)
    T57 = fd.ops.cast(T56, dtype=DataType.Float)
    T58 = fd.ops.sum(T55, dims=[0], keepdim=False, dtype=DataType.Null)
    T59 = fd.ops.div(T58, T57)
    fd.add_output(T59)


@pytest.mark.parametrize("vocab_size", SyntheticMiniModel.generate_vocab_sizes())
def test_run_loss_benchmark(benchmark, vocab_size):

    inputs = [

        torch.randint(
            0,
            128,
            (
                # 1,
                4096,
            ),
            requires_grad=False,
            device="cuda",
        ),

        torch.randn(
            4096, vocab_size, requires_grad=False, device="cuda", dtype=torch.bfloat16
        ),
    ]

    inputs[1] = torch.broadcast_to(
        inputs[1], (1, *inputs[1].shape)
    )  # Add batch dimension for the model

    fun = nvfuser_fusion_id1

    with FusionDefinition() as fd:
        fun(fd)

    run_benchmark(benchmark, fd.execute, inputs)
