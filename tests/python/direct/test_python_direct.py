# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

from nvfuser_direct import FusionDefinition, DataType, version
import torch
import pytest
import io
import re
from contextlib import redirect_stdout, redirect_stderr


def test_python_version_API():
    from nvfuser_direct.nvfuser_direct_version import Version

    assert version() > "0.0.0"
    assert version() > Version("0.0.0")


def test_fusion_not_defined():
    inputs = [
        torch.randn(4, 4, device="cpu"),
    ]

    # A FusionDefinition object is constructed but not defined, should trip an error
    try:
        fd = FusionDefinition()
        out = fd.execute(inputs)
        raise RuntimeError("Expecting an error for an empty FusionDefinition!")
    except NotImplementedError as e:
        assert (
            "Fusion does not exist! Use `with FusionDefinition() as fd: ...` to define a fusion."
            in str(e)
        )


def test_fusion_empty():
    inputs = [
        torch.randn(4, 4, device="cpu"),
    ]

    # A FusionDefinition object is constructed but not defined, should trip an error
    with pytest.raises(NotImplementedError, match="Fusion is empty!"):
        with FusionDefinition() as fd:
            pass
        out = fd.execute(inputs)


def test_from_pytorch_fails_on_cpu_tensor():
    inputs = [
        torch.randn(4, 4, device="cpu"),
    ]

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    error_msg = (
        "Found unsupported device cpu, only scalar CPU or CUDA tensors are supported"
    )
    with pytest.raises(ValueError, match=error_msg), redirect_stdout(
        stdout_capture
    ), redirect_stderr(stderr_capture):
        with FusionDefinition() as fd:
            t0 = fd.from_pytorch(inputs[0])
            t1 = fd.ops.relu(t0)
            fd.add_output(t1)


def test_fusion_definition_print():
    with FusionDefinition() as fd:
        tv0 = fd.define_tensor(
            shape=[2, 4, 8],
        )
        tv1 = fd.define_tensor(
            shape=[2, 4, 8],
        )
        tv2 = fd.ops.add(tv0, tv1)
        fd.add_output(tv2)

    # Test fusion math representation
    prescheduled_fusion_definition = """Inputs:
  T0_g_float[iS0{2}, iS1{4}, iS2{8}]
  T1_g_float[iS3{2}, iS4{4}, iS5{8}]
Outputs:
  T2_g_float[iS6{2}, iS7{4}, iS8{8}]

%kernel_math {
T2_g_float[iS6{2}, iS7{4}, iS8{8}]
   = T0_g_float[iS0{2}, iS1{4}, iS2{8}]
   + T1_g_float[iS3{2}, iS4{4}, iS5{8}];
} // %kernel_math \n\n"""
    assert fd.fusion.print_math() == prescheduled_fusion_definition

    # Test TensorView string representation
    assert str(tv0) == r"T0_g_float[iS0{2}, iS1{4}, iS2{8}]"
    assert str(tv1) == r"T1_g_float[iS3{2}, iS4{4}, iS5{8}]"
    assert str(tv2) == r"T2_g_float[iS6{2}, iS7{4}, iS8{8}]"

    # Test TensorDomain string representation
    assert str(tv0.domain()) == r"[iS0{2}, iS1{4}, iS2{8}]"
    assert str(tv1.domain()) == r"[iS3{2}, iS4{4}, iS5{8}]"
    assert str(tv2.domain()) == r"[iS6{2}, iS7{4}, iS8{8}]"

    # Test IterDomain string representation
    assert str(tv0.axis(0)) == r"iS0{2}"
    assert str(tv1.axis(0)) == r"iS3{2}"
    assert str(tv2.axis(0)) == r"iS6{2}"

    # Test axis extents
    assert str(tv0.axis(0).extent()) == r"2"
    assert str(tv1.axis(0).extent()) == r"2"
    assert str(tv2.axis(0).extent()) == r"2"


def test_fusion_execution_cache():
    with FusionDefinition() as fd:
        tv0 = fd.define_tensor(
            shape=[2, 4, 8],
        )
        tv1 = fd.define_tensor(
            shape=[2, 4, 8],
        )
        tv2 = fd.ops.add(tv0, tv1)
        fd.add_output(tv2)

    # Test fusion execution
    inputs = [
        torch.randn(2, 4, 8, device="cuda"),
        torch.randn(2, 4, 8, device="cuda"),
    ]
    results = fd.execute(inputs)
    assert torch.allclose(results[0], inputs[0] + inputs[1])

    # Test fusion math representation after compilation
    prescheduled_fusion_definition = """Inputs:
  T0_g_float[iS0{2}, iS1{4}, iS2{8}]
  T1_g_float[iS3{2}, iS4{4}, iS5{8}]
Outputs:
  T2_g_float[iS6{2}, iS7{4}, iS8{8}]

%kernel_math {
T2_g_float[iS6{2}, iS7{4}, iS8{8}]
   = T0_g_float[iS0{2}, iS1{4}, iS2{8}]
   + T1_g_float[iS3{2}, iS4{4}, iS5{8}];
} // %kernel_math \n\n"""
    assert fd.fusion.print_math() == prescheduled_fusion_definition

    # Test compilation status
    assert fd.fec.is_compiled(inputs)

    # Test scheduled IR representation
    scheduled_ir = """Inputs:
  T0_g_float[iS58{1}, iS59{1}, iS57{128}]
  T1_g_float[iS46{1}, iS47{1}, iS45{128}]
Outputs:
  T2_g_float[iblockIdx.x28{1}, iUS29{1}, ithreadIdx.x27{128}] ca_pos( 2 ) produce_pos( 3 )

%kernel {
T3_l_float[iblockIdx.x52{1}, iUS53{1}, ithreadIdx.x51{128}] ca_pos( 2 )
   = Set( T0_g_float[iS58{1}, iS59{1}, iS57{128}], cache_op=Streaming )
T4_l_float[iblockIdx.x40{1}, iUS41{1}, ithreadIdx.x39{128}] ca_pos( 2 )
   = Set( T1_g_float[iS46{1}, iS47{1}, iS45{128}], cache_op=Streaming )
T5_l_float[iblockIdx.x34{1}, iUS35{1}, ithreadIdx.x33{128}] ca_pos( 3 ) produce_pos( 2 )
   = T3_l_float[iblockIdx.x52{1}, iUS53{1}, ithreadIdx.x51{128}] ca_pos( 2 )
   + T4_l_float[iblockIdx.x40{1}, iUS41{1}, ithreadIdx.x39{128}] ca_pos( 2 );
T2_g_float[iblockIdx.x28{1}, iUS29{1}, ithreadIdx.x27{128}] ca_pos( 2 ) produce_pos( 3 )
   = Set( T5_l_float[iblockIdx.x34{1}, iUS35{1}, ithreadIdx.x33{128}] ca_pos( 3 ) produce_pos( 2 ), cache_op=Streaming )
} // %kernel\n"""
    assert fd.fec.get_scheduled_ir(inputs) == scheduled_ir
    assert fd.fec.get_most_recent_scheduled_ir() == scheduled_ir

    # Test CUDA kernel representation
    cuda_kernel = """// Codegen generated code
__global__ void nvfuser_pointwise_f0_c1_r0_g0(Tensor<float, 3, 3> T0, Tensor<float, 3, 3> T1, Tensor<float, 3, 3> T2) {
  nvfuser_index_t i0;
  i0 = ((nvfuser_index_t)threadIdx.x) % 32;
  nvfuser_index_t i1;
  i1 = ((nvfuser_index_t)threadIdx.x) / 32;
  nvfuser_index_t i2;
  i2 = i0 / 8;
  nvfuser_index_t i3;
  i3 = i0 % 8;
  nvfuser_index_t i4;
  i4 = ((nvfuser_index_t)threadIdx.x) + (128 * ((nvfuser_index_t)blockIdx.x));
  if ((i4 < 64)) {
    Array<float, 1, 1> T4;
    T4[0] = 0;
    T4[0]
       = T1[((((T1.alloc_stride[0LL] * i1) + (T1.alloc_stride[1LL] * i2)) + (T1.alloc_stride[2LL] * i3)) + ((4 * T1.alloc_stride[0LL]) * ((nvfuser_index_t)blockIdx.x)))];
    Array<float, 1, 1> T3;
    T3[0] = 0;
    T3[0]
       = T0[((((T0.alloc_stride[0LL] * i1) + (T0.alloc_stride[1LL] * i2)) + (T0.alloc_stride[2LL] * i3)) + ((4 * T0.alloc_stride[0LL]) * ((nvfuser_index_t)blockIdx.x)))];
    Array<float, 1, 1> T5;
    T5[0]
      = T3[0]
      + T4[0];
    T2[i4]
       = T5[0];
  }
}\n"""
    assert fd.fec.get_cuda_kernel(inputs) == cuda_kernel


def test_repro_script_for():
    inputs = [
        torch.ones(2, 4, 8, device="cuda"),
        torch.ones(2, 4, 8, device="cuda"),
    ]

    with FusionDefinition() as fd:
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.from_pytorch(inputs[1])
        c0 = fd.define_scalar(3.0)

        t2 = fd.ops.add(t0, t1)
        t3 = fd.ops.mul(t2, c0)
        t4 = fd.ops.sum(t3, [1], False, DataType.Float)

        fd.add_output(t4)

    expected_repro = """
import torch
from nvfuser_direct import FusionDefinition, DataType
def nvfuser_fusion(fd : FusionDefinition) -> None :
    tv0 = fd.define_tensor(shape=[-1, -1, -1], contiguity=[True, True, True], dtype=DataType.Float, is_cpu=False)
    tv1 = fd.define_tensor(shape=[-1, -1, -1], contiguity=[True, True, True], dtype=DataType.Float, is_cpu=False)
    tv2 = fd.ops.add(tv0, tv1)
    tv3 = fd.ops.mul(tv2, 3.00000)
    tv4 = fd.ops.sum(tv3, dims=[1], dtype=DataType.Float)
    fd.add_output(tv4)
with FusionDefinition() as fd:
    nvfuser_fusion(fd)
"""

    expected_inputs = """
inputs = [
    torch.testing.make_tensor((2, 4, 8), dtype=torch.float32, device='cuda:0'),
    torch.testing.make_tensor((2, 4, 8), dtype=torch.float32, device='cuda:0'),
]
fd.execute(inputs)\n"""

    # fd.repro_script_for(inputs) should have input arguments
    repro_with_inputs = fd.repro_script_for(inputs)
    assert expected_repro in repro_with_inputs
    assert expected_inputs in repro_with_inputs

    # fd.repro_script_for() should NOT have input arguments
    repro_without_inputs = fd.repro_script_for()
    assert expected_repro in repro_without_inputs
    assert expected_inputs not in repro_without_inputs

    # Check last_repro_script fails gracefully.
    with pytest.raises(
        AssertionError,
        match=r"fd.last_repro_script\(\) cannot provide a repro because fd.execute\(inputs, save_repro_state=True\) was not executed!",
    ):
        fd.last_repro_script()

    # Test fd.execute(inputs, save_repro_inputs=True) ; fd.last_repro_script()
    fd.execute(inputs, save_repro_inputs=True)
    last_repro = fd.last_repro_script()
    assert repro_with_inputs == last_repro


def test_define_tensor():
    with FusionDefinition() as fd:
        tv0 = fd.define_tensor(
            shape=[-1, -1, -1],
            contiguity=[True, True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        tv1 = fd.define_tensor(
            shape=[-1, -1, -1],
            contiguity=[True, True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        tv2 = fd.ops.add(tv0, tv1)
        fd.add_output(tv2)

    # Test fusion execution with dynamic shapes
    inputs = [
        torch.randn(2, 4, 8, device="cuda"),
        torch.randn(2, 4, 8, device="cuda"),
    ]
    outputs = fd.execute(inputs)
    assert torch.allclose(outputs[0], inputs[0] + inputs[1])


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="test_selected_device requires multiple GPUs",
)
def test_execute_with_different_device():
    inputs = [
        torch.ones(2, 4, 8, device="cuda:1"),
        torch.ones(2, 4, 8, device="cuda:1"),
    ]

    with FusionDefinition() as fd:
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.from_pytorch(inputs[1])
        c0 = fd.define_scalar(3.0)

        t2 = fd.ops.add(t0, t1)
        t3 = fd.ops.mul(t2, c0)
        t4 = fd.ops.sum(t3, [1], False, DataType.Float)

        fd.add_output(t4)

    outputs = fd.execute(inputs, device="cuda:1")
    assert len(outputs) == 1
    assert outputs[0].device.index == 1


def test_enable_disable_options():
    m = 24
    n = 16
    k = 8
    inps = [
        torch.randn(m, k, device="cuda", dtype=torch.float),
        torch.randn(k, n, device="cuda", dtype=torch.float),
    ]

    def fusion_func(fd: FusionDefinition, inps) -> None:
        t0 = fd.from_pytorch(inps[0])
        t1 = fd.from_pytorch(inps[1])
        t2 = fd.ops.matmul(t0, t1)
        fd.add_output(t2)

    with FusionDefinition() as fd:
        fusion_func(fd, inps=inps)

    # By default, matmul will be be run through expr_eval scheduler.
    # Through setting the enable and disable options as below,
    # we can execute it through matmul scheduler. The above fusion will not
    # be accepted by the matmul scheduler since the outputs are of type Float and raises a RuntimeError.
    # Note: We use this error-based test since for compatible dtypes (float16/bfloat16),
    # the matmul scheduler ran into a scheduling error on H100. This test might be more robust against
    # changes in matmul scheduler in the interim.

    with pytest.raises(
        RuntimeError, match="Can not find a scheduler to schedule fusion segment"
    ):
        fd.execute(
            inps, _enable_options=["fuse_matmul"], _disable_options=["matmul_expr_eval"]
        )


# Test that we properly raise an error when passing inputs with the wrong types
def test_mismatched_input_types():
    scalar_inp = 2.0
    tensor_inp = torch.rand((15,), dtype=torch.float32, device="cuda:0")

    def fusion_func(fd: FusionDefinition):
        T0 = fd.define_tensor(
            shape=[-1],
            contiguity=[True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[0],
        )
        s0 = fd.define_scalar()
        T1 = fd.ops.mul(T0, s0)
        fd.add_output(T1)

    with FusionDefinition() as fd:
        fusion_func(fd)

    with pytest.raises(
        Exception,
        match="Expected input 0, .*, to be an at::Tensor but got scalar 2",
    ):
        nvf_out = fd.execute([scalar_inp, scalar_inp])

    with pytest.raises(
        Exception,
        match=re.escape(
            "Expected input 1, d2, to be a scalar but got float tensor of rank 1"
        ),
    ):
        nvf_out = fd.execute([tensor_inp, tensor_inp])

    with pytest.raises(
        Exception,
        match="Expected input 0, .*, to be bound to a tensor of dtype float, but got a tensor of dtype __half",
    ):
        wrong_tensor_inp = torch.rand((15,), dtype=torch.float16, device="cuda:0")
        nvf_out = fd.execute([wrong_tensor_inp, 2.0])

    with pytest.raises(
        Exception,
        match=re.escape(
            "Scalar value (2,1) is not compatible with the expected data type: double."
        ),
    ):
        nvf_out = fd.execute([tensor_inp, 2.0 + 1.0j])
