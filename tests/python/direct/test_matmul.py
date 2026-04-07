# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
import pytest
from python.direct_utils import verify_stride_order
from nvfuser_direct import FusionDefinition
from functools import partial
import itertools
import torch.nn.functional as F


def test_matmul(nvfuser_direct_test):
    m = 24
    n = 16
    k = 8
    inputs_tt = [
        torch.randn(m, k, device="cuda", dtype=torch.float16),
        torch.randn(k, n, device="cuda", dtype=torch.float16),
    ]

    inputs_tn = [
        inputs_tt[0].clone(),
        inputs_tt[1].clone().as_strided(size=[k, n], stride=[1, k]),
    ]

    inputs_nt = [
        inputs_tt[0].clone().as_strided(size=[m, k], stride=[1, m]),
        inputs_tt[1].clone(),
    ]

    inputs_tn = [inputs_tt[0].clone(), inputs_tn[1].clone()]

    inputs_nn = [inputs_nt[0].clone(), inputs_tn[1].clone()]

    def fusion_func(fd: FusionDefinition, inps) -> None:
        t0 = fd.from_pytorch(inps[0])
        t1 = fd.from_pytorch(inps[1])
        t2 = fd.ops.matmul(t0, t1)
        fd.add_output(t2)

    for inps in [inputs_tt, inputs_tn, inputs_nt, inputs_nn]:
        nvf_out, _ = nvfuser_direct_test.exec_nvfuser(
            partial(fusion_func, inps=inps), inps
        )
        eager_out = torch.matmul(inps[0], inps[1])
        fp16_nvf_out = nvf_out[0]
        nvfuser_direct_test.assertEqual(eager_out, fp16_nvf_out)


def test_linear_without_bias(nvfuser_direct_test):
    """
    Test linear without bias. Check that bias keyword argument does not appear in python representation.
    """
    m = 24
    n = 16
    k = 8
    inputs = [
        torch.randn(m, k, device="cuda", dtype=torch.bfloat16),
        torch.randn(n, k, device="cuda", dtype=torch.bfloat16),
    ]

    def fusion_func(fd: FusionDefinition) -> None:
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.from_pytorch(inputs[1])
        t2 = fd.ops.linear(t0, t1)
        fd.add_output(t2)

    # Check that bias is not included with linear
    fd_str = """def nvfuser_fusion(fd : FusionDefinition) -> None :
    tv0 = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False)
    tv1 = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False)
    tv2 = fd.ops.linear(tv0, tv1)
    fd.add_output(tv2)"""

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(
        fusion_func, inputs, expected_fd_str=fd_str
    )
    eager_out = torch.nn.functional.linear(inputs[0], inputs[1])
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])


def test_linear_with_bias(nvfuser_direct_test):
    """
    Test linear with bias. Check that bias keyword argument appears in python representation.
    """
    m = 24
    n = 16
    k = 8
    inputs = [
        torch.randn(m, k, device="cuda", dtype=torch.bfloat16),
        torch.randn(n, k, device="cuda", dtype=torch.bfloat16),
        torch.randn(n, device="cuda", dtype=torch.bfloat16),
    ]

    def fusion_func(fd: FusionDefinition) -> None:
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.from_pytorch(inputs[1])
        t2 = fd.from_pytorch(inputs[2])
        t3 = fd.ops.linear(t0, t1, t2)
        fd.add_output(t3)

    fd_str = """def nvfuser_fusion(fd : FusionDefinition) -> None :
    tv0 = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False)
    tv1 = fd.define_tensor(shape=[-1, -1], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False)
    tv2 = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False)
    tv3 = fd.ops.linear(tv0, tv1, bias=tv2)
    fd.add_output(tv3)"""

    nvf_out, _ = nvfuser_direct_test.exec_nvfuser(
        fusion_func, inputs, expected_fd_str=fd_str
    )
    eager_out = torch.nn.functional.linear(inputs[0], inputs[1], inputs[2])
    nvfuser_direct_test.assertEqual(eager_out, nvf_out[0])


@pytest.mark.parametrize("bias", [False, True])
def test_linear(nvfuser_direct_test, bias):
    m = 24
    n = 16
    k = 8
    bias1d = torch.randn(n, device="cuda", dtype=torch.float16) if bias else None

    inputs_mk_nk = [
        torch.randn(m, k, device="cuda", dtype=torch.float16),
        torch.randn(n, k, device="cuda", dtype=torch.float16),
    ]

    inputs_mk_kn = [
        inputs_mk_nk[0].clone(),
        inputs_mk_nk[1].clone().as_strided(size=[n, k], stride=[1, n]),
    ]

    inputs_km_nk = [
        inputs_mk_nk[0].clone().as_strided(size=[m, k], stride=[1, m]),
        inputs_mk_nk[1].clone(),
    ]

    inputs_km_kn = [
        inputs_km_nk[0].clone(),
        inputs_mk_kn[1].clone(),
    ]

    def fusion_func(
        fd: FusionDefinition,
        inp: torch.Tensor,
        wt: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> None:
        t0 = fd.from_pytorch(inp)
        t1 = fd.from_pytorch(wt)
        if bias is not None:
            t2 = fd.from_pytorch(bias)
            t_out = fd.ops.linear(t0, t1, t2)
        else:
            t_out = fd.ops.linear(t0, t1)
        fd.add_output(t_out)

    in_tensors = [inputs_mk_nk, inputs_mk_kn, inputs_km_nk, inputs_km_kn]
    for inp, wt in in_tensors:
        input_tensors = (inp, wt, bias1d) if bias1d is not None else (inp, wt)
        nvf_out, _ = nvfuser_direct_test.exec_nvfuser(
            partial(fusion_func, inp=inp, wt=wt, bias=bias1d),
            input_tensors,
        )
        eager_out = F.linear(inp, wt, bias1d)
        fp16_nvf_out = nvf_out[0]
        nvfuser_direct_test.assertEqual(fp16_nvf_out, eager_out, atol=1e-3, rtol=0)


def test_linear_slice(nvfuser_direct_test):
    def fusion_func(fd: FusionDefinition) -> None:
        a = fd.define_tensor([1, 2, 3])
        b = fd.define_tensor([4, 3])
        c = fd.ops.linear(a, b)
        d = fd.ops.slice(c, start_indices=[0, 0, 0], end_indices=[1, 2, 3])
        fd.add_output(d)

    inputs = [
        torch.randn(1, 2, 3, device="cuda:0"),
        torch.randn(4, 3, device="cuda:0"),
    ]
    nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)


def test_matmul_operandA_2d_operandB_3d(nvfuser_direct_test):
    def fusion_func(fd: FusionDefinition) -> None:
        a = fd.define_tensor([2, 3])
        b = fd.define_tensor([7, 3, 5])
        c = fd.ops.matmul(a, b)
        assert c.ndim == 3
        fd.add_output(c)

    inputs = [
        torch.testing.make_tensor(2, 3, dtype=torch.float32, device="cuda"),
        torch.testing.make_tensor(7, 3, 5, dtype=torch.float32, device="cuda"),
    ]
    outputs, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
    assert outputs[0].ndim == 3


def test_matmul_stride(nvfuser_direct_test):
    n, h, l, s, e = 4, 8, 16, 16, 8
    inputs = [
        torch.randn(n, h, l, e, device="cuda", dtype=torch.float16, requires_grad=True),
        torch.randn(n, h, s, e, device="cuda", dtype=torch.float16, requires_grad=True),
    ]
    for stride_order in itertools.permutations(range(4)):

        def fusion_func(fd: FusionDefinition) -> None:
            q = fd.from_pytorch(inputs[0])
            k = fd.from_pytorch(inputs[1])
            k_t = fd.ops.permute(k, [0, 1, 3, 2])
            out = fd.ops.matmul(q, k_t)
            strided_out = fd.ops.stride_order(out, stride_order)
            fd.add_output(strided_out)

        nvf_out, _ = nvfuser_direct_test.exec_nvfuser(fusion_func, inputs)
        eager_out = torch.matmul(inputs[0], torch.transpose(inputs[1], -2, -1))
        verify_stride_order(nvf_out[0].stride(), stride_order)
        nvfuser_direct_test.assertEqual(nvf_out[0], eager_out)
