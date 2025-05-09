# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import nvfuser_next
from nvfuser_next import FusionDefinition


def test_import_correct():
    try:
        import nvfuser_next  # noqa: F401
    except Exception as e:
        raise RuntimeError("Failed to import nvfuser_next.")


def test_import_conflict_next_then_nvfuser():
    try:
        import nvfuser_next  # noqa: F401
        import nvfuser  # noqa: F401
    except AssertionError as e:
        expected_msg = (
            "Cannot import nvfuser if nvfuser_next module is already imported."
        )
        assert expected_msg in str(e)
        return
    raise AssertionError("Expected AssertionError from imports.")


def test_fusion_definition():
    fd = FusionDefinition()

    tv0 = fd.define_tensor(
        shape=[2, 4, 8],
    )
    tv1 = fd.define_tensor(
        shape=[2, 4, 8],
    )

    fd.add_input(tv0)
    fd.add_input(tv1)
    tv2 = fd.ops.add(tv0, tv1)
    fd.add_output(tv2)

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


def test_define_tensor():
    with FusionDefinition() as fd:
        tv0 = fd.define_tensor(
            shape=[-1, -1, -1],
            contiguity=[True, True, True],
            is_cpu=False,
            stride_order=[2, 1, 0],
        )
        tv1 = fd.define_tensor(
            shape=[-1, -1, -1],
            contiguity=[True, True, True],
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
