# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
import torch._refs as refs
import torch._prims as prims

import io
from contextlib import redirect_stdout, redirect_stderr
import pytest

from nvfuser_direct import FusionDefinition, DataType
import random
import itertools
import math
from functools import partial


def test_broadcast_in_dim_with_dynamic_shapes(nvfuser_direct_test):
    inputs_1 = [
        torch.randn(2, 3, 4, device="cuda"),
        torch.randn(4, device="cuda"),
    ]
    inputs_2 = [
        torch.randn(2, 3, 1024, device="cuda"),
        torch.randn(1024, device="cuda"),
    ]

    def fusion_func_1(fd: FusionDefinition):
        t0 = fd.define_tensor(shape=[-1, -1, -1], contiguity=[True, True, True])
        t1 = fd.define_tensor(shape=[-1], contiguity=[True])

        t1_b = fd.ops.broadcast_in_dim(t1, t0.shape(), [2])
        t2 = fd.ops.add(t0, t1_b)

        fd.add_output(t2)

    def fusion_func_2(fd: FusionDefinition):
        t0 = fd.define_tensor(shape=[-1, -1, -1], contiguity=[True, True, True])
        t1 = fd.define_tensor(shape=[-1], contiguity=[True])

        t1_b = fd.ops.broadcast_in_dim(t1, inputs_1[0].size(), [2])
        t2 = fd.ops.add(t0, t1_b)

        fd.add_output(t2)

    def fusion_func_3(fd: FusionDefinition):
        t0 = fd.define_tensor(shape=[-1, -1, -1], contiguity=[True, True, True])
        t1 = fd.define_tensor(shape=[-1], contiguity=[True])

        t1_b = fd.ops.broadcast_in_dim(t1, inputs_2[0].size(), [2])
        t2 = fd.ops.add(t0, t1_b)

        fd.add_output(t2)

    # Func_1 uses tensor.shape() to propagate dynamic size, therefore, it is
    # expected that test 2 should be cached based on test 2

    # Test 1
    inputs = inputs_1
    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func_1, inputs)
    eager_out = refs.add(
        inputs[0], prims.broadcast_in_dim(inputs[1], inputs[0].size(), [2])
    )
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])

    # Test 2
    inputs = inputs_2
    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func_1, inputs)
    eager_out = refs.add(
        inputs[0], prims.broadcast_in_dim(inputs[1], inputs[0].size(), [2])
    )
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])

    # Func_2 and Func_3 are nearly identical except that have a different
    # concrete output shape for their broadcast_in_dim.  Therefore, test 4
    # should not be cached.
    # Note: It is assumed that definition will change with Tensor Size with
    # concrete shapes.

    # Test 3
    inputs = inputs_1
    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func_2, inputs)
    eager_out = refs.add(
        inputs[0], prims.broadcast_in_dim(inputs[1], inputs[0].size(), [2])
    )
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])

    # Test 4
    inputs = inputs_2
    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func_3, inputs)
    eager_out = refs.add(
        inputs[0], prims.broadcast_in_dim(inputs[1], inputs[0].size(), [2])
    )
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])


# Test that symbolic IterDomains can be concatenated
# https://github.com/NVIDIA/Fuser/issues/1554
def test_cat_symbolic(nvfuser_direct_test):
    inputs = [
        0.29730177875068026,
        0.29730177875068026,
        4,
        64,
        768,
        4,
        64,
        768,
        2,
        torch.randn([4, 6, 64, 128], dtype=torch.float32, device="cuda"),
        torch.randn([4, 6, 64, 128], dtype=torch.float32, device="cuda"),
        torch.randn([4, 64, 768], dtype=torch.float32, device="cuda"),
    ]

    def fusion_func(fd: FusionDefinition) -> None:
        S0 = fd.define_scalar(None, dtype=DataType.Double)
        S1 = fd.define_scalar(None, dtype=DataType.Double)
        S2 = fd.define_scalar(None, dtype=DataType.Int)
        S3 = fd.define_scalar(None, dtype=DataType.Int)
        S4 = fd.define_scalar(None, dtype=DataType.Int)
        S5 = fd.define_scalar(None, dtype=DataType.Int)
        S6 = fd.define_scalar(None, dtype=DataType.Int)
        S7 = fd.define_scalar(None, dtype=DataType.Int)
        S8 = fd.define_scalar(None, dtype=DataType.Int)
        T9 = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=[True, True, True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[3, 2, 1, 0],
        )
        T10 = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=[True, True, True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[3, 2, 1, 0],
        )
        T11 = fd.define_tensor(
            shape=[-1, -1, -1],
            contiguity=[True, True, True],
            dtype=DataType.Float,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T12 = fd.ops.mul(T10, S1)
        T13 = fd.ops.permute(T12, dims=[0, 1, 3, 2])
        T14 = fd.ops.mul(T9, S0)
        T15 = fd.ops.permute(T14, dims=[0, 2, 1, 3])
        S16 = fd.define_scalar(4, dtype=DataType.Int)
        S17 = fd.define_scalar(64, dtype=DataType.Int)
        S18 = fd.define_scalar(768, dtype=DataType.Int)
        T20 = fd.ops.reshape(T15, new_shape=[S16, S17, S18])
        T21 = fd.ops.permute(T13, dims=[0, 2, 1, 3])
        S22 = fd.define_scalar(4, dtype=DataType.Int)
        S23 = fd.define_scalar(64, dtype=DataType.Int)
        S24 = fd.define_scalar(768, dtype=DataType.Int)
        T26 = fd.ops.reshape(T21, new_shape=[S22, S23, S24])
        T27 = fd.ops.cat([T20, T26, T11], dim=2)
        T28 = fd.ops.sum(T27, [0, 1], keepdim=False, dtype=DataType.Null)
        fd.add_output(T27)
        fd.add_output(T28)

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)

    t12 = inputs[1] * inputs[-2]
    t13 = torch.permute(t12, [0, 1, 3, 2])
    t14 = inputs[0] * inputs[-3]
    t15 = torch.permute(t14, [0, 2, 1, 3])
    t20 = torch.reshape(t15, [4, 64, 768])
    t21 = torch.permute(t13, [0, 2, 1, 3])
    t26 = torch.reshape(t21, [4, 64, 768])
    t27 = torch.cat([t20, t26, inputs[-1]], dim=2)
    t28 = t27.sum([0, 1])

    nvfuser_direct_test.assertEqual(nvf_out[0], t27)
    nvfuser_direct_test.assertEqual(nvf_out[1], t28)


def test_slice_error_checks(nvfuser_direct_test):
    inputs = [
        [torch.randn(10, 10, device="cuda")],
        [torch.randn(5, 5, device="cuda")],
    ]

    def check_start_indices(fd: FusionDefinition, acts) -> None:
        T0 = fd.from_pytorch(acts[0])
        T1 = fd.ops.slice(
            T0, start_indices=[-1, -2], end_indices=[5, 5], strides=[7, 7]
        )
        fd.add_output(T1)

    def check_end_indices(fd: FusionDefinition, acts) -> None:
        T0 = fd.from_pytorch(acts[0])
        T1 = fd.ops.slice(T0, start_indices=[3, 4], end_indices=[1, 2], strides=[1, 1])
        fd.add_output(T1)

    def check_strides(fd: FusionDefinition, acts) -> None:
        T0 = fd.from_pytorch(acts[0])
        T1 = fd.ops.slice(T0, start_indices=[0, 0], end_indices=[5, 5], strides=[5, 5])
        fd.add_output(T1)

    def check_tensor_dims(fd: FusionDefinition, acts) -> None:
        T0 = fd.from_pytorch(acts[0])
        T1 = fd.ops.slice(
            T0, start_indices=[0, 0, 0], end_indices=[4, 4, 4], strides=[1, 1, 1]
        )
        fd.add_output(T1)

    def check_slice_dims_start(fd: FusionDefinition, acts) -> None:
        T0 = fd.from_pytorch(acts[0])
        T1 = fd.ops.slice(
            T0, start_indices=[0, 0, 0], end_indices=[4, 4], strides=[1, 1]
        )
        fd.add_output(T1)

    def check_slice_dims_end(fd: FusionDefinition, acts) -> None:
        T0 = fd.from_pytorch(acts[0])
        T1 = fd.ops.slice(
            T0, start_indices=[0, 0], end_indices=[4, 4, 4], strides=[1, 1]
        )
        fd.add_output(T1)

    def check_slice_dims_stride(fd: FusionDefinition, acts) -> None:
        T0 = fd.from_pytorch(acts[0])
        T1 = fd.ops.slice(
            T0, start_indices=[0, 0], end_indices=[4, 4], strides=[1, 1, 1]
        )
        fd.add_output(T1)

    def check_nostrides(fd: FusionDefinition, acts) -> None:
        T0 = fd.from_pytorch(acts[0])
        T1 = fd.ops.slice(T0, start_indices=[2, 2], end_indices=[4, 4])
        fd.add_output(T1)

    def legal(fd: FusionDefinition, acts) -> None:
        T0 = fd.from_pytorch(acts[0])
        T1 = fd.ops.slice(T0, start_indices=[6, 6], end_indices=[8, 8], strides=[1, 1])
        fd.add_output(T1)

    checks = [
        (
            check_start_indices,
            "Slice operation start_indices must be greater than or equal to 0. .*",
        ),
        (
            check_end_indices,
            "Slice operation end_indices must be greater than or equal to start_indices. .*",
        ),
        (
            check_strides,
            "nvFuser Limitation: All slice operation strides must be of const size 1.*",
        ),
        (
            check_tensor_dims,
            "Number of tensor dimensions does not match slice dimensions! .*",
        ),
        (
            check_slice_dims_start,
            "Slice start_indices and strides don't match! .*",
        ),
        (
            check_slice_dims_end,
            "Slice indexing attribute dimensions don't match! .*",
        ),
        (
            check_slice_dims_stride,
            "Slice start_indices and strides don't match! .*",
        ),
        (check_nostrides, None),
        (legal, None),
    ]

    # Redirect stdout and stderr messages to log to avoid false positives in
    # the CI.
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    for combination in itertools.product(inputs, checks):
        inputs, (check, error_msg) = combination
        if error_msg is None:
            out = nvfuser_direct_test.exec_nvfuser(
                partial(check, acts=inputs),
                inputs,
            )
        else:
            with pytest.raises(RuntimeError, match=error_msg), redirect_stdout(
                stdout_capture
            ), redirect_stderr(stderr_capture):
                nvfuser_direct_test.exec_nvfuser(
                    partial(check, acts=inputs),
                    inputs,
                )


# Test that deterministic random ops (uniform, normal) give same results as
# their stochastic versions
def test_deterministic_random(nvfuser_direct_test):
    inputs = [
        torch.randn([5, 9], device="cuda", dtype=torch.float32),
    ]

    for rand_op_name in ["uniform", "normal"]:

        def fusion_func(fd: FusionDefinition, *, deterministic) -> None:
            t1 = fd.from_pytorch(inputs[0])
            a = fd.define_scalar(0.3, DataType.Float)
            b = fd.define_scalar(1.7, DataType.Float)
            rand_op = getattr(fd.ops, rand_op_name)
            if deterministic:
                rng_seed = fd.define_scalar(DataType.Int)
                rng_offset = fd.define_scalar(DataType.Int)
                u = rand_op(
                    a, b, shape=[5, 9], rng_seed=rng_seed, rng_offset=rng_offset
                )
            else:
                u = rand_op(a, b, shape=[5, 9])
            t2 = fd.ops.mul(t1, u)
            fd.add_output(t2)

        # exec_nvfuser tests printing and serde, so run that for each definition first
        nvfuser_direct_test.exec_nvfuser(
            partial(fusion_func, deterministic=False), inputs
        )
        nvfuser_direct_test.exec_nvfuser(
            partial(fusion_func, deterministic=True), [inputs[0], 0, 0]
        )

        # Now instantiate FusionDefinitions in each mode
        with FusionDefinition() as fd_stoch:
            fusion_func(fd_stoch, deterministic=False)
        with FusionDefinition() as fd_det:
            fusion_func(fd_det, deterministic=True)

        # Get the current RNG state to restore after this test.
        state = torch.cuda.get_rng_state()
        # Test with three different random seeds
        for _ in range(3):
            max_seed = 2**63 - 1
            seed = random.randint(0, max_seed)
            torch.manual_seed(seed)

            stateful_sequence = [fd_stoch.execute(inputs) for _ in range(10)]
            # Each call to uniform with DataType::Float will advance the offset by one
            # See Note [Divide offset by 4] in rng.cpp for more information
            stateless_sequence = [
                fd_det.execute([inputs[0], seed, rng_offset])
                for rng_offset in range(10)
            ]

            for i, (sful, sless) in enumerate(
                zip(stateful_sequence, stateless_sequence)
            ):
                nvfuser_direct_test.assertEqual(sful[0], sless[0])
        # Restore the RNG state
        torch.cuda.set_rng_state(state)


# Test that the range of generated uniform values spans the proper range
# https://github.com/NVIDIA/Fuser/issues/1653
def test_uniform_range(nvfuser_direct_test):
    dtypes = [DataType.Double, DataType.Float, DataType.Half, DataType.BFloat16]

    def run_test(left: float, right: float, dtype: DataType):
        samples_per_run = 2**29

        def fusion_fn(fd: FusionDefinition):
            # Generate enough values to reasonably expect to sample the ends of the range
            S0 = fd.define_scalar(left, dtype=DataType.Double)
            S1 = fd.define_scalar(right, dtype=DataType.Double)
            output = fd.ops.uniform(S0, S1, shape=[samples_per_run], dtype=dtype)
            fd.add_output(output)

        with FusionDefinition() as fd:
            fusion_fn(fd)

        output = fd.execute([])[0]

        x = output.amax()
        m = output.amin()
        mu = output.type(torch.float64).mean()
        # Repeat to improve chances of sampling extreme values
        num_runs = 100
        num_samples = num_runs * samples_per_run
        for i in range(num_runs):
            u = fd.execute([])[0]
            x = torch.maximum(x, u.amax())
            m = torch.minimum(m, u.amin())
            mu = mu + (u.type(torch.float64).mean() - mu) / (i + 2)

        # round-trip cast to find expected min
        theomin = torch.tensor(left, dtype=output.dtype).item()
        theomu = 0.5 * (right + left)
        theomax = torch.nextafter(
            torch.tensor(right, dtype=output.dtype),
            torch.tensor(left, dtype=output.dtype),
        )

        assert (
            m.item() >= theomin
        ), f"{output.dtype} expected min generated value {theomin} but found {m.item()}"
        assert (
            m.item() <= theomax
        ), f"{output.dtype} expected max generated value {theomax} but found {x.item()}"

        # uniform distribution on [0, 1) has mean 0.5 and variance 1/12
        # The standard error of the mean is then 1/sqrt(12 *
        # num_samples). We use the precision at 1.0 as a surrogate for
        # the contribution of rounding to the standard error of the
        # finite-precision mean.
        assert abs(mu.item() - theomu) < (right - left) * max(
            right - x.item(), 3.0 / math.sqrt(12 * num_samples)
        ), f"{output.dtype} expected mean generated value {theomu} but found {mu.item()}"

        if dtype not in [DataType.Float, DataType.Double]:
            # For reduced precision types, check that we sample the extreme
            # values. We don't do this for full precision types since the
            # amount of samples required would be too large.
            assert (
                m.item() == theomin
            ), f"{output.dtype} expected min generated value {theomin} but found {m.item()}"
            assert (
                x.item() == theomax
            ), f"{output.dtype} expected max generated value {theomax} but found {x.item()}"

    # test standard and non-standard uniform
    for left, right in [[0.0, 1.0], [-1.5, 3.7]]:
        for dtype in dtypes:
            run_test(left, right, dtype)


def test_cat_qwen2_v2(nvfuser_direct_test):
    def qwen2_cat_fusion_2(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[2048, 512],
            contiguity=[True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[1, 0],
        )
        S1 = fd.define_scalar(None, dtype=DataType.Int)
        S2 = fd.define_scalar(None, dtype=DataType.Int)
        T3 = fd.define_tensor(
            shape=[1, 4, 2048, 128],
            contiguity=[None, True, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[3, 1, 2, 0],
        )
        T4 = fd.define_tensor(
            shape=[1, 2048, 128],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 0, 1],
        )
        T5 = fd.define_tensor(
            shape=[1, 2048, 128],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 0, 1],
        )
        T6 = fd.define_tensor(
            shape=[1, 28, 2048, 128],
            contiguity=[None, True, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[3, 1, 2, 0],
        )
        T7 = fd.define_tensor(
            shape=[1, 2048, 512],
            contiguity=[None, True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        T12 = fd.ops.reshape(T0, new_shape=[1, 2048, 512])
        T13 = fd.ops.cast(T12, dtype=DataType.Float)
        S14 = fd.define_scalar(0.00000, dtype=DataType.Double)
        S15 = fd.define_scalar(1.00000, dtype=DataType.Double)
        S16 = fd.define_scalar(1, dtype=DataType.Int)
        S17 = fd.define_scalar(2048, dtype=DataType.Int)
        S18 = fd.define_scalar(512, dtype=DataType.Int)
        T20 = fd.ops.uniform(
            S14,
            S15,
            shape=[S16, S17, S18],
            rng_seed=S2,
            rng_offset=S1,
            dtype=DataType.BFloat16,
        )
        S21 = fd.define_scalar(4.00000, dtype=DataType.Double)
        T22 = fd.ops.mul(T13, S21)
        S23 = fd.define_scalar(0.900000, dtype=DataType.Double)
        T24 = fd.ops.lt(T20, S23)
        T25 = fd.ops.cast(T24, dtype=DataType.Float)
        T41 = fd.ops.slice(
            T3,
            start_indices=[0, 0, 0, 64],
            end_indices=[1, 4, 2048, 128],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T42 = fd.ops.mul(T22, T25)
        T43 = fd.ops.cast(T41, dtype=DataType.Float)
        T44 = fd.ops.neg(T43)
        T50 = fd.ops.broadcast_in_dim(
            T4, shape=[1, 1, 2048, 128], broadcast_dims=[0, 2, 3]
        )
        T66 = fd.ops.slice(
            T3,
            start_indices=[0, 0, 0, 0],
            end_indices=[1, 4, 2048, 64],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T67 = fd.ops.cast(T44, dtype=DataType.BFloat16)
        T73 = fd.ops.broadcast_in_dim(
            T5, shape=[1, 1, 2048, 128], broadcast_dims=[0, 2, 3]
        )
        T89 = fd.ops.slice(
            T6,
            start_indices=[0, 0, 0, 64],
            end_indices=[1, 28, 2048, 128],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        S90 = fd.define_scalar(1.11111, dtype=DataType.Double)
        T91 = fd.ops.mul(T42, S90)
        T97 = fd.ops.broadcast_in_dim(
            T50, shape=[1, 4, 2048, 128], broadcast_dims=[0, 1, 2, 3]
        )
        T98 = fd.ops.cat([T67, T66], dim=-1, manual_padding=0)
        T104 = fd.ops.broadcast_in_dim(
            T73, shape=[1, 4, 2048, 128], broadcast_dims=[0, 1, 2, 3]
        )
        T105 = fd.ops.cast(T89, dtype=DataType.Float)
        T106 = fd.ops.cast(T97, dtype=DataType.Float)
        T107 = fd.ops.cast(T98, dtype=DataType.Float)
        T108 = fd.ops.cast(T104, dtype=DataType.Float)
        T109 = fd.ops.cast(T3, dtype=DataType.Float)
        T110 = fd.ops.neg(T105)
        T111 = fd.ops.cast(T7, dtype=DataType.Float)
        T112 = fd.ops.mul(T107, T106)
        T113 = fd.ops.mul(T109, T108)
        T129 = fd.ops.slice(
            T6,
            start_indices=[0, 0, 0, 0],
            end_indices=[1, 28, 2048, 64],
            strides=[1, 1, 1, 1],
            manual_normalization=0,
        )
        T130 = fd.ops.cast(T110, dtype=DataType.BFloat16)
        T131 = fd.ops.add(T111, T91)
        T137 = fd.ops.broadcast_in_dim(
            T50, shape=[1, 28, 2048, 128], broadcast_dims=[0, 1, 2, 3]
        )
        T138 = fd.ops.cat([T130, T129], dim=-1, manual_padding=0)
        T144 = fd.ops.broadcast_in_dim(
            T73, shape=[1, 28, 2048, 128], broadcast_dims=[0, 1, 2, 3]
        )
        T145 = fd.ops.cast(T131, dtype=DataType.BFloat16)
        T146 = fd.ops.cast(T137, dtype=DataType.Float)
        T147 = fd.ops.cast(T138, dtype=DataType.Float)
        T148 = fd.ops.cast(T144, dtype=DataType.Float)
        T149 = fd.ops.cast(T6, dtype=DataType.Float)
        T155 = fd.ops.reshape(T145, new_shape=[1, 2048, 4, 128])
        T156 = fd.ops.add(T113, T112)
        T157 = fd.ops.mul(T147, T146)
        T158 = fd.ops.mul(T149, T148)
        T159 = fd.ops.permute(T155, dims=[0, 2, 1, 3])
        T160 = fd.ops.cast(T156, dtype=DataType.BFloat16)
        T167 = fd.ops.broadcast_in_dim(
            T159, shape=[1, 4, 1, 2048, 128], broadcast_dims=[0, 1, 3, 4]
        )
        T174 = fd.ops.broadcast_in_dim(
            T160, shape=[1, 4, 1, 2048, 128], broadcast_dims=[0, 1, 3, 4]
        )
        T181 = fd.ops.broadcast_in_dim(
            T167, shape=[1, 4, 7, 2048, 128], broadcast_dims=[0, 1, 2, 3, 4]
        )
        T188 = fd.ops.broadcast_in_dim(
            T174, shape=[1, 4, 7, 2048, 128], broadcast_dims=[0, 1, 2, 3, 4]
        )
        T189 = fd.ops.add(T158, T157)
        T195 = fd.ops.reshape(T181, new_shape=[1, 28, 2048, 128])
        T201 = fd.ops.reshape(T188, new_shape=[1, 28, 2048, 128])
        T202 = fd.ops.cast(T189, dtype=DataType.BFloat16)
        T203 = fd.ops.stride_order(T195, stride_order=[3, 2, 1, 0])
        T204 = fd.ops.stride_order(T201, stride_order=[3, 2, 1, 0])
        T205 = fd.ops.stride_order(T202, stride_order=[3, 2, 1, 0])
        fd.add_output(T159)
        fd.add_output(T160)
        fd.add_output(T203)
        fd.add_output(T204)
        fd.add_output(T205)

    inputs = [
        torch.testing.make_tensor((2048, 512), dtype=torch.bfloat16, device="cuda:0"),
        25546,
        1400552702872758,
        torch.randn(1048576, dtype=torch.bfloat16, device="cuda:0").as_strided(
            (1, 4, 2048, 128), (1048576, 128, 512, 1)
        ),
        torch.randn(262144, dtype=torch.bfloat16, device="cuda:0").as_strided(
            (1, 2048, 128), (262144, 1, 2048)
        ),
        torch.randn(262144, dtype=torch.bfloat16, device="cuda:0").as_strided(
            (1, 2048, 128), (262144, 1, 2048)
        ),
        torch.randn(7340032, dtype=torch.bfloat16, device="cuda:0").as_strided(
            (1, 28, 2048, 128), (7340032, 128, 3584, 1)
        ),
        torch.testing.make_tensor(
            (1, 2048, 512), dtype=torch.bfloat16, device="cuda:0"
        ),
    ]

    nvfuser_direct_test.exec_nvfuser(qwen2_cat_fusion_2, inputs)


def test_nanogpt_mha_dpa(nvfuser_direct_test):
    inputs = [
        torch.randn(16, 16, 128, 128, device="cuda"),
        torch.randn(1, 1, 1024, 1024, device="cuda"),
    ]

    def nvfuser_fusion(fd: FusionDefinition, prob) -> None:
        T0 = fd.define_tensor(
            shape=[-1, -1, -1, -1],
            contiguity=[True, True, True, True],
            dtype=DataType.Float,
            is_cpu=False,
        )
        T1 = fd.define_tensor(
            shape=[1, 1, -1, -1],
            contiguity=[None, None, True, True],
            dtype=DataType.Float,
            is_cpu=False,
        )
        S2 = fd.define_scalar(0.125000, dtype=DataType.Double)
        T3 = fd.ops.mul(T0, S2)
        T4 = fd.ops.slice(
            T1,
            start_indices=[0, 0, 0, 0],
            end_indices=[1, 1, 128, 128],
            strides=[1, 1, 1, 1],
        )
        S5 = fd.define_scalar(0.00000, dtype=DataType.Double)
        T6 = fd.ops.eq(S5, T4)
        T7 = fd.ops.broadcast_in_dim(
            T6, shape=[16, 16, 128, 128], broadcast_dims=[0, 1, 2, 3]
        )
        S8 = fd.define_scalar(float("-inf"), dtype=DataType.Double)
        T9 = fd.ops.where(T7, S8, T3)
        S10 = fd.define_scalar(-1, dtype=DataType.Int)
        S11 = fd.define_scalar(4, dtype=DataType.Int)
        S12 = fd.ops.add(S10, S11)
        T13 = fd.ops.max(T9, dims=[3], keepdim=False, dtype=DataType.Null)
        T14 = fd.ops.broadcast_in_dim(
            T13, shape=[16, 16, 128, 1], broadcast_dims=[0, 1, 2]
        )
        T15 = fd.ops.broadcast_in_dim(
            T14, shape=[16, 16, 128, 128], broadcast_dims=[0, 1, 2, 3]
        )
        T16 = fd.ops.sub(T9, T15)
        T17 = fd.ops.exp(T16)
        S18 = fd.define_scalar(-1, dtype=DataType.Int)
        S19 = fd.define_scalar(4, dtype=DataType.Int)
        S20 = fd.ops.add(S18, S19)
        T21 = fd.ops.sum(T17, dims=[3], keepdim=False, dtype=DataType.Null)
        T22 = fd.ops.broadcast_in_dim(
            T21, shape=[16, 16, 128, 1], broadcast_dims=[0, 1, 2]
        )
        T23 = fd.ops.broadcast_in_dim(
            T22, shape=[16, 16, 128, 128], broadcast_dims=[0, 1, 2, 3]
        )
        T24 = fd.ops.div(T17, T23)
        S25 = fd.define_scalar(16, dtype=DataType.Int)
        S26 = fd.define_scalar(16, dtype=DataType.Int)
        S27 = fd.define_scalar(128, dtype=DataType.Int)
        S28 = fd.define_scalar(128, dtype=DataType.Int)
        S29 = fd.define_scalar(0.00000, dtype=DataType.Double)
        S30 = fd.define_scalar(1.00000, dtype=DataType.Double)
        T31 = fd.ops.uniform(S29, S30, shape=[S25, S26, S27, S28], dtype=DataType.Float)
        S32 = fd.define_scalar(1.0 - prob, dtype=DataType.Double)
        T33 = fd.ops.lt(T31, S32)
        T34 = fd.ops.cast(T33, dtype=DataType.Float)
        T35 = fd.ops.mul(T24, T34)
        S36 = fd.define_scalar(1.0 / (1.0 - prob), dtype=DataType.Double)
        T37 = fd.ops.mul(T35, S36)
        fd.add_output(T37)

    def torch_def(acts, bias, n_seq_len, n_head_dim, prob):
        att = acts * (1.0 / math.sqrt(n_head_dim))
        att = att.masked_fill(bias[:, :, :n_seq_len, :n_seq_len] == 0, float("-inf"))
        att = torch.nn.functional.softmax(att, dim=-1)
        att = torch.nn.functional.dropout(att, p=prob)
        return att

    # NOTE: The dropout probabilities need to be set to 0 elements zeroed out
    # in order to match implementations as eager and nvFuser do not have matching
    # blocking.
    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(
        partial(nvfuser_fusion, prob=0.0), inputs
    )
    eager_out = torch_def(inputs[0], inputs[1], 128, 64, 0.0)

    for idx in range(len(nvf_out)):
        nvfuser_direct_test.assertEqual(eager_out, nvf_out[idx])


def test_nanogpt_split_mha_linears(nvfuser_direct_test):
    inputs = [
        torch.randn(16, 128, 3072, device="cuda"),
    ]

    def nvfuser_fusion_0(fd: FusionDefinition) -> None:
        T0 = fd.from_pytorch(inputs[0])
        T0_slice1 = fd.ops.slice(T0, [0, 0, 0], [16, 128, 1024], [1, 1, 1])
        T0_slice2 = fd.ops.slice(T0, [0, 0, 1024], [16, 128, 2048], [1, 1, 1])
        T0_slice3 = fd.ops.slice(T0, [0, 0, 2048], [16, 128, 3072], [1, 1, 1])
        T1_slice1 = fd.ops.reshape(T0_slice1, [16, 128, 16, 64])
        T1_slice2 = fd.ops.reshape(T0_slice2, [16, 128, 16, 64])
        T1_slice3 = fd.ops.reshape(T0_slice3, [16, 128, 16, 64])
        T2_slice1 = fd.ops.permute(T1_slice1, [0, 2, 1, 3])
        T2_slice2 = fd.ops.permute(T1_slice2, [0, 2, 1, 3])
        T2_slice3 = fd.ops.permute(T1_slice3, [0, 2, 1, 3])
        fd.add_output(T2_slice1)
        fd.add_output(T2_slice2)
        fd.add_output(T2_slice3)

    def torch_def_0(acts, n_embd, n_head):
        B, T, C = acts.size()
        q, k, v = acts.split(n_embd, dim=2)
        k = k.view(B, T, n_head, (C // 3) // n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, n_head, (C // 3) // n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, n_head, (C // 3) // n_head).transpose(1, 2)  # (B, nh, T, hs)
        return (
            q,
            k,
            v,
        )

    def nvfuser_fusion_1(fd: FusionDefinition) -> None:
        T0 = fd.define_tensor(
            shape=[-1, -1, -1],
            contiguity=[True, True, True],
            dtype=DataType.Float,
            is_cpu=False,
        )
        T1 = fd.ops.slice(
            T0,
            start_indices=[0, 0, 0],
            end_indices=[16, 128, 1024],
            strides=[1, 1, 1],
        )
        T2 = fd.ops.slice(
            T0,
            start_indices=[0, 0, 1024],
            end_indices=[16, 128, 2048],
            strides=[1, 1, 1],
        )
        T3 = fd.ops.slice(
            T0,
            start_indices=[0, 0, 2048],
            end_indices=[16, 128, 3072],
            strides=[1, 1, 1],
        )
        fd.add_output(T1)
        fd.add_output(T2)
        fd.add_output(T3)

    def torch_def_1(acts, n_embd, n_head):
        B, T, C = acts.size()
        q, k, v = acts.split(n_embd, dim=2)
        return (
            q,
            k,
            v,
        )

    tests = [
        (nvfuser_fusion_0, torch_def_0),
        (nvfuser_fusion_1, torch_def_1),
    ]

    for nvf_func, torch_func in tests:
        nvf_out, _ = nvfuser_direct_test.exec_nvfuser(nvf_func, inputs)
        eager_out = torch_func(*inputs, 1024, 16)
        for idx in range(len(eager_out)):
            nvfuser_direct_test.assertEqual(eager_out[idx], nvf_out[idx])
