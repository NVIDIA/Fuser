# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
from python.utils import NVFuserTest, is_pre_volta, verify_stride_order
from nvfuser import FusionDefinition, DataType
import pytest
from functools import partial
import itertools
import torch.nn.functional as F


@pytest.mark.skipif(is_pre_volta(), reason="Only supported on Volta and newer devices.")
class TestMatmul(NVFuserTest):
    def test_matmul(self):
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
            nvf_out, _ = self.exec_nvfuser(partial(fusion_func, inps=inps), inps)
            eager_out = torch.matmul(inps[0], inps[1])
            fp16_nvf_out = nvf_out[0]
            self.assertEqual(eager_out, fp16_nvf_out)

    def test_linear(self):
        m = 24
        n = 16
        k = 8
        bias1d = torch.randn(n, device="cuda", dtype=torch.float16)

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
        bias = [None, bias1d]
        for [inp, wt], bias in list(itertools.product(in_tensors, bias)):
            with self.subTest(inp=inp, wt=wt, bias=bias):
                input_tensors = (inp, wt, bias) if bias is not None else (inp, wt)
                nvf_out, _ = self.exec_nvfuser(
                    partial(fusion_func, inp=inp, wt=wt, bias=bias),
                    input_tensors,
                )
                eager_out = F.linear(inp, wt, bias)
                fp16_nvf_out = nvf_out[0]
                torch.testing.assert_close(fp16_nvf_out, eager_out, atol=1e-3, rtol=0)

    def test_matmul_issue_2354(self):
        inputs = [
            torch.randn((8, 4), dtype=torch.float32, device="cuda:0"),
            torch.randn(
                (
                    6,
                    2,
                    4,
                ),
                dtype=torch.float32,
                device="cuda:0",
            ),
        ]

        def fusion_func(fd: FusionDefinition):
            T0 = fd.define_tensor(
                shape=[-1, -1],
                contiguity=[True, True],
                dtype=DataType.Float,
                is_cpu=False,
                stride_order=[1, 0],
            )
            T1 = fd.define_tensor(
                shape=[-1, -1, -1],
                contiguity=[True, True, True],
                dtype=DataType.Float,
                is_cpu=False,
                stride_order=[2, 1, 0],
            )
            T2 = fd.ops.linear(T1, T0)
            S3 = fd.define_scalar(1.41421, dtype=DataType.Double)
            T4 = fd.ops.mul(T2, S3)
            fd.add_output(T2)
            fd.add_output(T4)

        self.exec_nvfuser(fusion_func, inputs)

    # Tests broadcast reduction axis in matmul: Issue #2532.
    def test_repro_issue2532(self):
        def fusion_func(fd: FusionDefinition) -> None:
            T0 = fd.define_tensor(
                shape=[-1, -1, 1],
                contiguity=[True, None, True],
                dtype=DataType.Float,
                is_cpu=False,
                stride_order=[2, 0, 1],
            )
            T1 = fd.define_tensor(
                shape=[-1, 1, -1],
                contiguity=[True, None, True],
                dtype=DataType.Float,
                is_cpu=False,
                stride_order=[2, 1, 0],
            )
            T2 = fd.ops.sum(T1, dims=[0, 1], keepdim=False, dtype=DataType.Null)
            T3 = fd.ops.matmul(T0, T1)
            T4 = fd.ops.sum(T3, dims=[0], keepdim=False, dtype=DataType.Null)
            fd.add_output(T2)
            fd.add_output(T4)

        inputs = [
            torch.randn((2 * 32,), dtype=torch.float32, device="cuda:0").as_strided(
                (2, 32, 1), (32, 1, 32)
            ),
            torch.randn((2 * 16,), dtype=torch.float32, device="cuda:0").as_strided(
                (2, 1, 16), (16, 16, 1)
            ),
        ]
        self.exec_nvfuser(fusion_func, inputs)

    def test_linear_slice(self):
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
        self.exec_nvfuser(fusion_func, inputs)

    def test_2d_x_3d(self):
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
        outputs, _ = self.exec_nvfuser(fusion_func, inputs)
        assert outputs[0].ndim == 3

    def test_matmul_stride(self):
        n, h, l, s, e = 4, 8, 16, 16, 8
        inputs = [
            torch.randn(
                n, h, l, e, device="cuda", dtype=torch.float16, requires_grad=True
            ),
            torch.randn(
                n, h, s, e, device="cuda", dtype=torch.float16, requires_grad=True
            ),
        ]
        for perm in itertools.permutations(range(4), 4):

            def fusion_func(fd: FusionDefinition) -> None:
                q = fd.from_pytorch(inputs[0])
                k = fd.from_pytorch(inputs[1])
                k_t = fd.ops.permute(k, [0, 1, 3, 2])
                out = fd.ops.matmul(q, k_t)
                fd.add_output(out, stride_order=perm)

            with FusionDefinition() as fd:
                fusion_func(fd)
            nvf_out = fd.execute(inputs)
            eager_out = torch.matmul(inputs[0], torch.transpose(inputs[1], -2, -1))
            verify_stride_order(nvf_out[0].stride(), perm)
            torch.testing.assert_close(nvf_out[0], eager_out)
