import pytest
from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import run_benchmark
import torch
from .global_params import generate_input_sizes, FLOAT_DTYPES, PROMOTE_DTYPES

def rmsnorm_fwd_fusion(
    fd: FusionDefinition,
    dtype: DataType,
    eps:float = 1e-5,
):  
    inputs = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=dtype)
    weights = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype)

    if dtype in PROMOTE_DTYPES:
        inputs = fd.ops.cast(inputs, dtype=DataType.Float)
        weights = fd.ops.cast(weights, dtype=DataType.Float)
        
    inputs_sq = fd.ops.mul(inputs, inputs)
    squared_sum = fd.ops.sum(inputs_sq, axes=[1], keepdim=True)
    var = fd.ops.div(squared_sum, inputs.size(1))
    eps_const = fd.define_scalar(eps)
    var_eps = fd.ops.add(var, eps_const)
    invstd = fd.ops.rsqrt(var_eps)
    pre_scale = fd.ops.mul(inputs, invstd)
    weights_bcast = fd.ops.broadcast_in_dim(weights, shape=inputs.shape(), broadcast_dims=[1])
    out = fd.ops.mul(pre_scale, weights_bcast)
    if dtype in PROMOTE_DTYPES:
        out = fd.ops.cast(out, dtype=dtype)
    fd.add_output(out)
    fd.add_output(invstd)

def rmsnorm_bwd_fusion(
    fd: FusionDefinition,
    dtype: DataType,
):
    # inputs = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=dtype)
    # grads = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=dtype)
    # weights = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype)
    # invstd = fd.define_tensor(shape=[-1, 1], contiguity=[True, None], dtype=DataType.Float)

    # if dtype in PROMOTE_DTYPES:
    #     inputs = fd.ops.cast(inputs, dtype=DataType.Float)
    #     grads = fd.ops.cast(grads, dtype=DataType.Float)
    #     weights = fd.ops.cast(weights, dtype=DataType.Float)
    
    # invstd_bcast = fd.ops.broadcast_in_dim(invstd, shape=inputs.shape(), broadcast_dims=[0, 1])
    # x_norm = fd.ops.mul(inputs, invstd_bcast)
    # weights_bcast = fd.ops.broadcast_in_dim(weights, shape=inputs.shape(), broadcast_dims=[1])
    # grad_x_norm = fd.ops.mul(grads, weights_bcast)

    # T0 = fd.ops.mul(inputs.size(1), grad_x_norm)
    # T1 = fd.ops.sum(grad_x_norm, axes = [1], keepdim=True)
    # T2 = fd.ops.sum(fd.ops.mul(x_norm, grad_x_norm), axes=[1], keepdim=True)
    # T3 = fd.ops.mul(x_norm, T2)

    # inner = fd.ops.sub(fd.ops.sub(T0, T1), T3)
    # inverse_norm_size = fd.ops.reciprocal(inputs.size(1))
    # grad_inputs = fd.ops.mul(fd.ops.mul(inverse_norm_size, invstd_bcast), inner)
    # grad_weights = fd.ops.sum(fd.ops.mul(grads, x_norm), axes=[0], keepdim=False)

    # if dtype in PROMOTE_DTYPES:
    #     grad_inputs = fd.ops.cast(grad_inputs, dtype=dtype)
    #     grad_weights = fd.ops.cast(grad_weights, dtype=dtype)
    # fd.add_output(grad_inputs)
    # fd.add_output(grad_weights)
    T0 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )
    T1 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )

    # T2 = fd.define_tensor(
    #     shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False
    # )
    T4 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)
    T3 = fd.define_tensor(
        shape=[-1, 1], contiguity=[True, None], dtype=DataType.Float, is_cpu=False
    )

    

    if dtype in PROMOTE_DTYPES:
        T0 = fd.ops.cast(T0, dtype=DataType.Float)
        T1 = fd.ops.cast(T1, dtype=DataType.Float)
        T4 = fd.ops.cast(T4, dtype=DataType.Float)

    V8 = fd.define_vector([T0.size(0), 1], dtype=DataType.Int)
    # T9 = fd.ops.broadcast_in_dim(T2, shape=V8, broadcast_dims=[0])
    V12 = T0.shape()
    # T13 = fd.ops.broadcast_in_dim(T9, shape=V12, broadcast_dims=[0, 1])
    # T14 = fd.ops.sub(T0, T13)

    T18 = fd.ops.broadcast_in_dim(T3, shape=V12, broadcast_dims=[0, 1])
    T19 = fd.ops.mul(T0, T18)

    T23 = fd.ops.broadcast_in_dim(T4, shape=V12, broadcast_dims=[1])
    T28 = fd.ops.sum(T1, axes=[0], keepdim=False, dtype=DataType.Null)

    T30 = fd.ops.mul(T1, T23)
    T31 = fd.ops.mul(T1, T19)
    T32 = fd.ops.sum(T31, axes=[0], keepdim=False, dtype=DataType.Null)

    T34 = fd.ops.mul(T30, T18)
    T35 = fd.ops.mul(T30, T0)
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
    # T67 = fd.ops.broadcast_in_dim(T2, shape=V8, broadcast_dims=[0])
    # T71 = fd.ops.broadcast_in_dim(T67, shape=V12, broadcast_dims=[0, 1])

    S72 = fd.define_scalar(2.00000, dtype=DataType.Double)
    T73 = fd.ops.mul(S72, T63)
    # T74 = fd.ops.sub(T0, T71)
    T75 = fd.ops.mul(T73, T0)

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
def test_rmsnorm_fwd_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
    eps:float = 1e-5,
):
    inputs = torch.randn(*size, device="cuda", dtype=dtype)
    weights = torch.randn(size[1], device="cuda", dtype=dtype)

    with FusionDefinition() as fd:
        rmsnorm_fwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype))
    
    if not disable_validation:
        nvf_output = fd.execute([inputs, weights])
        ms = (inputs**2).mean(1, keepdim=True)
        eager_output = weights * (inputs / torch.sqrt(ms + eps))
        
        assert torch.allclose(nvf_output[0], eager_output, rtol=1e-3, atol=1e-3), \
            f"Max error: {torch.max(torch.abs(nvf_output[0] - eager_output))}"

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [inputs, weights])

    torch.cuda.empty_cache()

@pytest.mark.parametrize("size", [(128, 64)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_rmsnorm_bwd_benchmark(
    benchmark,
    size: tuple,
    dtype: torch.dtype,
    disable_validation: bool,
    disable_benchmarking: bool,
    eps:float = 1e-5,
):
    inputs = torch.randn(*size, device="cuda", dtype=dtype, requires_grad=True)
    grads = torch.randn(*size, device="cuda", dtype=dtype)
    weights = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)

    ms = (inputs.to(torch.float)**2).mean(1)
    invstd = 1.0 / torch.sqrt(ms + eps)

    with FusionDefinition() as fd:
        rmsnorm_bwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype))

    with FusionDefinition() as fd0:
        rmsnorm_fwd_fusion(fd0, torch_dtype_to_nvfuser_dtype(dtype))

    if not disable_validation:
        nvf_output_fwd = fd0.execute([inputs, weights])
        print(nvf_output_fwd[1].shape)
        nvf_output = fd.execute([inputs, grads, weights, nvf_output_fwd[1]])
        ms = (inputs**2).mean(1, keepdim=True)
        eager_output = weights * (inputs / torch.sqrt(ms + eps))
        eager_output.backward(grads)

        assert torch.allclose(nvf_output[0], inputs.grad, rtol=1e-3, atol=1e-3), \
            f"Max error: {torch.max(torch.abs(nvf_output[0] - inputs.grad))}"
        print(nvf_output[0].shape)
        print(nvf_output[1].shape)

    if not disable_benchmarking:
        run_benchmark(benchmark, fd.execute, [inputs, grads, weights, invstd])

    torch.cuda.empty_cache()