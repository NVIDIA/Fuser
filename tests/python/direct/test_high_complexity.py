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
