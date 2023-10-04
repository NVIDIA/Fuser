from nvfuser import FusionDefinition, DataType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import runBenchmark
import torch
from .global_params import pytestmark, RTOL, ATOL


def layernorm_fwd_fusion(
    fd: FusionDefinition,
    dtype: DataType,
    eps: float = 1e-5,
) -> None:
    T0 = fd.define_tensor(
        shape=[-1, -1], contiguity=[True, True], dtype=dtype, is_cpu=False
    )

    if dtype is not DataType.Float:
        T0 = fd.ops.cast(T0, dtype=DataType.Float)

    T2, T3 = fd.ops.var_mean(T0, axes=[1], correction=0, keepdim=False)

    V6 = fd.define_vector([T0.size(0), 1], dtype=DataType.Int)
    T7 = fd.ops.broadcast_in_dim(T2, shape=V6, broadcast_dims=[0])
    T11 = fd.ops.broadcast_in_dim(T3, shape=V6, broadcast_dims=[0])

    S12 = fd.define_scalar(eps, dtype=DataType.Double)
    T13 = fd.ops.add(T7, S12)
    T14 = fd.ops.rsqrt(T13)

    V17 = fd.define_vector([T0.size(0), T0.size(1)], dtype=DataType.Int)
    T18 = fd.ops.broadcast_in_dim(T11, shape=V17, broadcast_dims=[0, 1])
    T19 = fd.ops.sub(T0, T18)
    T23 = fd.ops.broadcast_in_dim(T14, shape=V17, broadcast_dims=[0, 1])
    T24 = fd.ops.mul(T19, T23)

    if dtype is not DataType.Float:
        T24 = fd.ops.cast(T24, dtype=dtype)
    fd.add_output(T24)
    fd.add_output(T3)
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

    if dtype is not DataType.Float:
        T0 = fd.ops.cast(T0, dtype=DataType.Float)
        T1 = fd.ops.cast(T1, dtype=DataType.Float)

    T2 = fd.define_tensor(
        shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False
    )
    T3 = fd.define_tensor(
        shape=[-1, 1], contiguity=[True, None], dtype=DataType.Float, is_cpu=False
    )

    V7 = fd.define_vector([T0.size(0), 1], dtype=DataType.Int)
    T8 = fd.ops.broadcast_in_dim(T2, shape=V7, broadcast_dims=[0])
    V11 = fd.define_vector([T0.size(0), T0.size(1)], dtype=DataType.Int)
    T12 = fd.ops.broadcast_in_dim(T8, shape=V11, broadcast_dims=[0, 1])

    T13 = fd.ops.sub(T0, T12)

    T17 = fd.ops.broadcast_in_dim(T3, shape=V11, broadcast_dims=[0, 1])
    T19 = fd.ops.mul(T1, T17)
    T20 = fd.ops.mul(T1, T13)
    T21 = fd.ops.sum(T20, axes=[1], keepdim=False, dtype=DataType.Null)

    T25 = fd.ops.broadcast_in_dim(T21, shape=V7, broadcast_dims=[0])
    T26 = fd.ops.neg(T19)
    T27 = fd.ops.sum(T26, axes=[1], keepdim=False, dtype=DataType.Null)

    T31 = fd.ops.broadcast_in_dim(T27, shape=V7, broadcast_dims=[0])
    S32 = fd.define_scalar(-0.500000, dtype=DataType.Double)
    T33 = fd.ops.mul(S32, T25)
    S34 = fd.define_scalar(3.00000, dtype=DataType.Double)
    T35 = fd.ops.pow(T3, S34)
    T36 = fd.ops.mul(T33, T35)
    T39 = fd.ops.sum(T31, axes=[1], keepdim=False, dtype=DataType.Null)
    T40 = fd.ops.sum(T36, axes=[1], keepdim=False, dtype=DataType.Null)

    T44 = fd.ops.broadcast_in_dim(T40, shape=V7, broadcast_dims=[0])
    T48 = fd.ops.broadcast_in_dim(T44, shape=V11, broadcast_dims=[0, 1])

    T52 = fd.ops.broadcast_in_dim(T2, shape=V7, broadcast_dims=[0])
    T56 = fd.ops.broadcast_in_dim(T52, shape=V11, broadcast_dims=[0, 1])

    S57 = fd.define_scalar(2.00000, dtype=DataType.Double)
    T58 = fd.ops.mul(S57, T48)
    T59 = fd.ops.sub(T0, T56)
    T60 = fd.ops.mul(T58, T59)

    S62 = fd.ops.reciprocal(T0.size(1))
    T63 = fd.ops.mul(T60, S62)

    T67 = fd.ops.broadcast_in_dim(T39, shape=V7, broadcast_dims=[0])
    T71 = fd.ops.broadcast_in_dim(T67, shape=V11, broadcast_dims=[0, 1])
    T73 = fd.ops.mul(S62, T71)
    T74 = fd.ops.add(T63, T73)
    T75 = fd.ops.add(T19, T74)

    if dtype is not DataType.Float:
        T75 = fd.ops.cast(T75, dtype=dtype)
    fd.add_output(T75)


def test_layernorm_fwd_benchmark(benchmark, size, dtype, test_correctness):
    inputs = [torch.randn(*size, device="cuda", dtype=dtype)]

    with FusionDefinition() as fd:
        layernorm_fwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype))

    if test_correctness:
        nvf_output = fd.execute(inputs)
        eager_output = torch.nn.functional.layer_norm(inputs[0], inputs[0].shape[1:])
        assert torch.allclose(nvf_output[0], eager_output, rtol=RTOL, atol=ATOL)

    runBenchmark(benchmark, fd.execute, inputs)


def test_layernorm_bwd_benchmark(benchmark, size, dtype, test_correctness, eps=1e-5):
    inputs = torch.randn(*size, device="cuda", dtype=dtype, requires_grad=True)
    grads = torch.randn(*size, device="cuda", dtype=dtype)

    mean = inputs.to(torch.float).mean(dim=-1)
    variance = inputs.to(torch.float).var(dim=-1, unbiased=False)
    invstd = (1.0 / torch.sqrt(variance + eps)).unsqueeze(1)

    with FusionDefinition() as fd:
        layernorm_bwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype))

    if test_correctness:
        eager_output = torch.nn.functional.layer_norm(inputs, inputs.shape[1:])
        eager_output.backward(grads)
        nvf_output = fd.execute([inputs, grads, mean, invstd])
        assert torch.allclose(
            nvf_output[0], inputs.grad, rtol=RTOL, atol=ATOL
        ), f"{torch.max(nvf_output[0] - inputs.grad)}"

    runBenchmark(benchmark, fd.execute, [inputs, grads, mean, invstd])
