# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
from utils import NVFuserTest, is_pre_ampere
from nvfuser import FusionDefinition, DataType, FusionCache
import pytest
import itertools
from functools import partial
import torch.nn.functional as F

class TestEmbedding(NVFuserTest):
    def test_embedding(self):
        def fusion_func(
            fd: FusionDefinition, 
            has_optional_inputs: list[bool],
            optional_inputs_dtypes: list[DataType]
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
                optional_inputs[idx] = fd.define_scalar(value=None, dtype=optional_inputs_dtypes[idx])
            out = fd.ops.embedding(input, weight, *optional_inputs)
            fd.add_output(out)

        N, S = 5, 2
        input = torch.randint(N, (S,), dtype=torch.int64, device='cuda', requires_grad=False)
        weight = torch.randn(N, S, dtype=torch.bfloat16, device='cuda', requires_grad=True)
        
        # padding_idx_vals = [None, -1]
        # max_norm_vals = [None, 1e-5]
        # norm_type_vals = [None, 2, 1]
        # scale_grad_by_freq = [None, True]
        # sparse = [None, False]
        padding_idx_vals = [None]
        max_norm_vals = [1e-5]
        norm_type_vals = [None]
        scale_grad_by_freq = [None]
        sparse = [None]
        optional_inputs_dtypes = [DataType.Int, DataType.Float, DataType.Float, DataType.Bool, DataType.Bool]

        
        # TODO: Try to move this to pytest_ops.py. Currently, it does not work since the API between nvFuser and torch differs.
        for padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse in itertools.product(
            padding_idx_vals, max_norm_vals, norm_type_vals, scale_grad_by_freq, sparse
        ):
            with self.subTest(padding_idx=padding_idx, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse):
                # Reset the FusionCache or the fusion would not recompile for all subtests, failing checks in exec_nvfuser.
                FusionCache.reset()
                optional_inputs = [padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse]
                has_optional_inputs = [None] * 5
                inputs = [input, weight]
                for idx, param in enumerate(optional_inputs):
                    if param is not None:
                        has_optional_inputs[idx] = True
                        inputs.append(param)
                
                with FusionDefinition() as fd:
                  fusion_func(fd, 
                              has_optional_inputs=has_optional_inputs, 
                              optional_inputs_dtypes = optional_inputs_dtypes)    
                nvf_out = fd.execute(inputs) 

                torch.manual_seed(0)
                norm_type = 2.0 if norm_type is None else norm_type
                scale_grad_by_freq = False if scale_grad_by_freq is None else scale_grad_by_freq
                sparse = False if sparse is None else sparse
                ref_out = F.embedding(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
                print (nvf_out[0])
                print (ref_out)
                torch.testing.assert_close(nvf_out[0], ref_out)