# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

from nvfuser_direct import FusionDefinition


def test_fusion_definition():
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
