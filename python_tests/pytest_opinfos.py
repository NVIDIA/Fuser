# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
import jax
from pytest_core import (
    elementwise_unary_generator,
    _elementwise_unary_torch,
    OpInfo,
    ReferenceType,
    slice_sample_generator,
    slice_error_sample_generator,
    define_tensor_sample_generator,
    define_tensor_error_sample_generator,
)

eps = 1e-2

opinfos = []

""" Start Fusion Input Operations """
fusion_input_ops = []

define_tensor_opinfo = OpInfo(
    lambda fd: fd.define_tensor,
    "define_tensor",
    sample_input_generator=define_tensor_sample_generator,
    error_input_generator=define_tensor_error_sample_generator,
    is_fusion_input_op=True,
)
fusion_input_ops.append(define_tensor_opinfo)

""" End Fusion Input Operations """

""" Start Unary-Float Operations """
elementwise_unary_ops = []

acos_opinfo = OpInfo(
    lambda fd: fd.ops.acos,
    "acos",
    domain=(-1, 1),
    sample_input_generator=elementwise_unary_generator,
    reference=_elementwise_unary_torch(torch.acos),
)
elementwise_unary_ops.append(acos_opinfo)

""" End Unary-Float Operations """

""" Start Shape Operations """

shape_ops = []

slice_opinfo = OpInfo(
    lambda fd: fd.ops.slice,
    "slice",
    sample_input_generator=slice_sample_generator,
    error_input_generator=slice_error_sample_generator,
    reference=jax.lax.slice,
    reference_type=ReferenceType.Jax,
)
shape_ops.append(slice_opinfo)

""" End Shape Operations """

# Puts all opinfos into the "opinfos" list
opinfos.extend(elementwise_unary_ops)
opinfos.extend(shape_ops)
opinfos.extend(fusion_input_ops)
