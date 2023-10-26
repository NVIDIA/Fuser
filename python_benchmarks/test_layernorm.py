import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES


def layernorm_fwd_fusion(
    fd: FusionDefinition,
    dtype: DataType,
    eps: float = 1e-5,
) -> None:
    T0 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )
    T1 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)
    T2 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)

    if dtype in PROMOTE_DTYPES:
        T0 = fd.ops.cast(T0, dtype=DataType.Float)
        T1 = fd.ops.cast(T1, dtype=DataType.Float)
        T2 = fd.ops.cast(T2, dtype=DataType.Float)

    T3, T4 = fd.ops.var_mean(T0, axes=[1], correction=0, keepdim=False)

    V6 = fd.define_vector([T0.size(0), 1], dtype=DataType.Int)
    T7 = fd.ops.broadcast_in_dim(T3, shape=V6, broadcast_dims=[0])
    T11 = fd.ops.broadcast_in_dim(T4, shape=V6, broadcast_dims=[0])

    S12 = fd.define_scalar(eps, dtype=DataType.Double)
    T13 = fd.ops.add(T7, S12)
    T14 = fd.ops.rsqrt(T13)

    V17 = T0.shape()
    T18 = fd.ops.broadcast_in_dim(T11, shape=V17, broadcast_dims=[0, 1])
    T19 = fd.ops.sub(T0, T18)
    T23 = fd.ops.broadcast_in_dim(T14, shape=V17, broadcast_dims=[0, 1])
    T24 = fd.ops.mul(T19, T23)

    T25 = fd.ops.broadcast_in_dim(T1, shape=V17, broadcast_dims=[1])
    T26 = fd.ops.mul(T24, T25)
    T27 = fd.ops.broadcast_in_dim(T2, shape=V17, broadcast_dims=[1])
    T28 = fd.ops.add(T26, T27)

    if dtype in PROMOTE_DTYPES:
        T28 = fd.ops.cast(T28, dtype=dtype)

    fd.add_output(T28)
    fd.add_output(T4)
    fd.add_output(T14)


def layernorm_bwd_fusion(
    fd: FusionDefinition,
    dtype: DataType,
) -> None:
    T0 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )
    T1 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )

    T2 = fd.define_tensor(
        shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False
    )
    T3 = fd.define_tensor(
        shape=[-1, 1], contiguity=[True, None], dtype=DataType.Float, is_cpu=False
    )

    T4 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)

    if dtype in PROMOTE_DTYPES:
        T0 = fd.ops.cast(T0, dtype=DataType.Float)
        T1 = fd.ops.cast(T1, dtype=DataType.Float)
        T4 = fd.ops.cast(T4, dtype=DataType.Float)

    V8 = fd.define_vector([T0.size(0), 1], dtype=DataType.Int)
    T9 = fd.ops.broadcast_in_dim(T2, shape=V8, broadcast_dims=[0])
    V12 = T0.shape()
    T13 = fd.ops.broadcast_in_dim(T9, shape=V12, broadcast_dims=[0, 1])
    T14 = fd.ops.sub(T0, T13)

    T18 = fd.ops.broadcast_in_dim(T3, shape=V12, broadcast_dims=[0, 1])
    T19 = fd.ops.mul(T14, T18)

    T23 = fd.ops.broadcast_in_dim(T4, shape=V12, broadcast_dims=[1])
    T28 = fd.ops.sum(T1, axes=[0], keepdim=False, dtype=DataType.Null)

    T30 = fd.ops.mul(T1, T23)
    T31 = fd.ops.mul(T1, T19)
    T32 = fd.ops.sum(T31, axes=[0], keepdim=False, dtype=DataType.Null)

    T34 = fd.ops.mul(T30, T18)
    T35 = fd.ops.mul(T30, T14)
    T36 = fd.ops.sum(T35, axes=[1], keepdim=False, dtype=DataType.Null)

    T40 = fd.ops.broadcast_in_dim(T36, shape=V8, broadcast_dims=[0])
    T41 = fd.ops.neg(T34)
    T42 = fd.ops.sum(T41, axes=[1], keepdim=False, dtype=DataType.Null)
    T46 = fd.ops.broadcast_in_dim(T42, shape=V8, broadcast_dims=[0])
    S47 = fd.define_scalar(-0.500000, dtype=DataType.Double)
    T48 = fd.ops.mul(S47, T40)
    S49 = fd.define_scalar(3.00000, dtype=DataType.Double)
    T50 = fd.ops.pow(T3, S49)
    T51 = fd.ops.mul(T48, T50)
    T54 = fd.ops.sum(T46, axes=[1], keepdim=False, dtype=DataType.Null)
    T55 = fd.ops.sum(T51, axes=[1], keepdim=False, dtype=DataType.Null)

    T59 = fd.ops.broadcast_in_dim(T55, shape=V8, broadcast_dims=[0])
    T63 = fd.ops.broadcast_in_dim(T59, shape=V12, broadcast_dims=[0, 1])
    T67 = fd.ops.broadcast_in_dim(T2, shape=V8, broadcast_dims=[0])
    T71 = fd.ops.broadcast_in_dim(T67, shape=V12, broadcast_dims=[0, 1])

    S72 = fd.define_scalar(2.00000, dtype=DataType.Double)
    T73 = fd.ops.mul(S72, T63)
    T74 = fd.ops.sub(T0, T71)
    T75 = fd.ops.mul(T73, T74)

    S77 = fd.ops.reciprocal(T0.size(1))
    T78 = fd.ops.mul(T75, S77)
    T82 = fd.ops.broadcast_in_dim(T54, shape=V8, broadcast_dims=[0])
    T86 = fd.ops.broadcast_in_dim(T82, shape=V12, broadcast_dims=[0, 1])
    T88 = fd.ops.mul(S77, T86)
    T89 = fd.ops.add(T78, T88)
    T90 = fd.ops.add(T34, T89)

    if dtype in PROMOTE_DTYPES:
        T28 = fd.ops.cast(T28, dtype=dtype)
        T90 = fd.ops.cast(T90, dtype=dtype)
        T32 = fd.ops.cast(T32, dtype=dtype)

    fd.add_output(T90)
    fd.add_output(T32)
    fd.add_output(T28)


@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_layernorm_fwd_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
    eps: float = 1e-5,
):
    inputs = [
        torch.randn(*size, device="cuda", dtype=dtype),
        torch.randn(size[1], device="cuda", dtype=dtype),
        torch.randn(size[1], device="cuda", dtype=dtype),
    ]

    with FusionDefinition() as fd:
        layernorm_fwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype))

    if not disable_validation:
        eager_output = torch.nn.functional.layer_norm(
            inputs[0], inputs[0].shape[1:], weight=inputs[1], bias=inputs[2]
        )

        mean = inputs.to(torch.float).mean(dim=-1)
        variance = inputs.to(torch.float).var(dim=-1, unbiased=False)
        invstd = (1.0 / torch.sqrt(variance + eps)).unsqueeze(1)

        fd.validate(inputs, [eager_output, mean, invstd])

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, inputs)

    torch.cuda.empty_cache()


@pytest.mark.parametrize("size", generate_input_sizes(dims=2))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_layernorm_bwd_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
    eps: float = 1e-5,
):
    inputs = torch.randn(*size, device="cuda", dtype=dtype, requires_grad=True)
    grads = torch.randn(*size, device="cuda", dtype=dtype)
    weights = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)
    bias = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)

    mean = inputs.to(torch.float).mean(dim=-1)
    variance = inputs.to(torch.float).var(dim=-1, unbiased=False)
    invstd = (1.0 / torch.sqrt(variance + eps)).unsqueeze(1)

    with FusionDefinition() as fd:
        layernorm_bwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype))

    if not disable_validation:
        eager_output = torch.nn.functional.layer_norm(
            inputs, inputs.shape[1:], weight=weights, bias=bias
        )
        eager_output.backward(grads)
        fd.validate([inputs, grads, mean, invstd, weights], [inputs.grad, weights.grad, bias.grad])


    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [inputs, grads, mean, invstd, weights])

    torch.cuda.empty_cache()
