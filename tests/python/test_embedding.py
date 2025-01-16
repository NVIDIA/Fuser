# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
from nvfuser import FusionDefinition, DataType
import pytest
import torch.nn.functional as F


@pytest.mark.parametrize("padding_idx", [None, -2])
@pytest.mark.parametrize("max_norm", [None, 1e-5])
@pytest.mark.parametrize("norm_type", [None, 1.0])
@pytest.mark.parametrize("scale_grad_by_freq", [None, True])
@pytest.mark.parametrize("sparse", [None, True])
def test_embedding(
    padding_idx: None | int,
    max_norm: None | float,
    norm_type: None | float,
    scale_grad_by_freq: None | bool,
    sparse: None | bool,
):
    def fusion_func(
        fd: FusionDefinition,
        has_optional_inputs: list[bool],
        optional_inputs_dtypes: list[DataType],
    ):
        input = fd.define_tensor(
            shape=[-1],
            contiguity=[True],
            dtype=DataType.Int,
            is_cpu=False,
        )
        weight = fd.define_tensor(
            shape=[-1, -1],
            contiguity=[True, True],
            dtype=DataType.BFloat16,
            is_cpu=False,
        )
        # padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse
        optional_inputs = [None] * 5
        for idx in range(len(optional_inputs)):
            if has_optional_inputs[idx]:
                optional_inputs[idx] = fd.define_scalar(
                    value=None, dtype=optional_inputs_dtypes[idx]
                )
        out = fd.ops.embedding_fwd(input, weight, *optional_inputs)
        fd.add_output(out)

    N, S = 10, 3
    input = torch.randint(
        N, (S,), dtype=torch.int64, device="cuda", requires_grad=False
    )
    weight = torch.randn(N, S, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    optional_inputs_dtypes = [
        DataType.Int,
        DataType.Float,
        DataType.Float,
        DataType.Bool,
        DataType.Bool,
    ]

    # This is not in pytest_ops.py since the torch API does not accept None values for some arguments.
    # Different inputs for nvfuser and torch API cannot be handled within OpInfo
    optional_inputs = [padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse]
    has_optional_inputs = [None] * 5
    inputs = [input, weight]
    for idx, param in enumerate(optional_inputs):
        if param is not None:
            has_optional_inputs[idx] = True
            inputs.append(param)

    with FusionDefinition() as fd:
        fusion_func(
            fd,
            has_optional_inputs=has_optional_inputs,
            optional_inputs_dtypes=optional_inputs_dtypes,
        )
    nvf_out = fd.execute(inputs)

    norm_type = 2.0 if norm_type is None else norm_type
    scale_grad_by_freq = False if scale_grad_by_freq is None else scale_grad_by_freq
    sparse = False if sparse is None else sparse
    ref_out = F.embedding(
        input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse
    )
    torch.testing.assert_close(nvf_out[0], ref_out)
