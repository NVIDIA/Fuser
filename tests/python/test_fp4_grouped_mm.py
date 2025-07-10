# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from nvfuser import FusionDefinition, DataType
from nvfuser.testing.utils import NVFuserTest

FLOAT8_E4M3_MAX = 448.0
FLOAT4_E2M1_MAX = 6.0

BLOCK_SIZE = 16

def round_up(x, y):
    return (x + y - 1) // y * y

# NOTE: This is from pytorch nvfp4 gemm tests.
def to_fp4(x): 
    def down_size(size):
        assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
        return (*size[:-1], size[-1] // 2)
    def pack_uint4(uint8_data) -> torch.Tensor:
        # converting to uint8 for operations
        shape = uint8_data.shape 
        assert shape[-1] % 2 == 0
        uint8_data = uint8_data.contiguous().view(-1)
        return (uint8_data[1::2] << 4 | uint8_data[::2]).view(down_size(shape))
    
    from torch.testing._internal.common_quantized import _f32_to_floatx_unpacked
    x = _f32_to_floatx_unpacked(x.float(), 2, 1)
    x = pack_uint4(x)
    x = x.view(torch.float4_e2m1fn_x2)
    return x
    
def scale_to_nvfp4(x, g_sf):
    x_blocked = x.view(*x.shape[:-1], -1, BLOCK_SIZE)
    x_blocked_g_scaled = x_blocked / g_sf
    block_sf = x_blocked_g_scaled.abs().amax(-1)
    
    block_sf = block_sf.clamp(max=FLOAT8_E4M3_MAX)
    x_blocked_scaled = x_blocked_g_scaled / block_sf.unsqueeze(-1)
    x_blocked_scaled = x_blocked_scaled.view(x.shape)
    return to_fp4(x_blocked_scaled), block_sf.to(dtype=torch.float8_e4m3fn)

def activation_scale_to_nvfp4(x, g_sf, offsets, blockscale_offsets):
    m = x.size(0)
    k = x.size(1)
    g = g_sf.size(0)
    padded_m_size = blockscale_offsets[g-1] + round_up(m - offsets[g-1], 128)
    block_scale = torch.empty((padded_m_size, k // BLOCK_SIZE), dtype=torch.float8_e4m3fn, device='cuda:0')
    v_scaled = torch.empty((m, k // 2), dtype=torch.float4_e2m1fn_x2, device='cuda:0')
    for i in range(len(g_sf)):
        l = offsets[i]
        if i == g-1:
            r = m
        else:
            r = offsets[i+1]
        l_sf = blockscale_offsets[i]
        r_sf = l_sf + r - l
        v_scaled[l:r], block_scale[l_sf:r_sf] = scale_to_nvfp4(x[l:r], g_sf[i])
    return v_scaled, block_scale

class TestAlias(NVFuserTest):
    def test_cutlass_nvfp4_grouped_mm(self):
        INPUT_DTYPE = torch.uint8

        # k dimension is multiple of 128 to avoid padding
        m = 1024
        n = 128
        k = 256
        tokens_per_expert = [115, 144, 8]
        tokens_per_expert.append(m - sum(tokens_per_expert))
        g = len(tokens_per_expert)
        
        mat1 = torch.testing.make_tensor((m, k), dtype=torch.float32, device='cuda:0')
        # format is g, n, k instead of g, k, n
        mat2 = torch.testing.make_tensor((g, n, k), dtype=torch.float32, device='cuda:0')
        
        offsets = torch.empty((g,), dtype=torch.int32, device='cuda:0')
        blockscale_offsets = torch.empty((g,), dtype=torch.int32, device='cuda:0')
        problem_sizes = torch.empty((g, 3), dtype=torch.int32, device='cuda:0')
        
        # prepare quantization for mat2 
        mat2_gs = torch.empty((g,), dtype=torch.float32, device='cuda:0')
        scale2 = torch.empty((g, n, k // BLOCK_SIZE), dtype=torch.float8_e4m3fn, device='cuda:0')

        acc_tokens = 0
        rounded_acc_tokens = 0
        mat2_scaled = torch.empty((g, n, k // 2), dtype=torch.float4_e2m1fn_x2, device='cuda:0')
            
        for i in range(g):
            mat2_gs[i] = FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX / mat2[i].max()
            offsets[i] = acc_tokens
            blockscale_offsets[i] = rounded_acc_tokens
            acc_tokens += tokens_per_expert[i]
            # Note: we technically don't need to round up, since k is perfectly sized.
            rounded_acc_tokens += round_up(tokens_per_expert[i], 128)
            
            problem_sizes[i][0] = tokens_per_expert[i]
            problem_sizes[i][1] = n
            problem_sizes[i][2] = k
            
            #mat2[i], scale2[i] = scale_to_nvfp4(mat2[i], mat2_gs[i])
            scaled_mat2_i, bs_mat2_i = scale_to_nvfp4(mat2[i], mat2_gs[i])
            mat2_scaled[i] = scaled_mat2_i
            scale2[i] = bs_mat2_i
        
        # prepare quantization for mat1
        # note: following sglang implementation, not computing global scaling factor for mat1
        #       similarly, we don't need to apply mat1_gs to alpha
        mat1_gs = torch.ones((g,), dtype=torch.float32, device='cuda:0')
        mat1, scale1 = activation_scale_to_nvfp4(mat1, mat1_gs, offsets, blockscale_offsets)

        
        
        ab_strides = torch.full(
            (g,), k, dtype=torch.int64, device='cuda:0' 
        )
        c_strides = torch.full(
            (g,), n, dtype=torch.int64, device='cuda:0' 
        )
        
        
        def nvfuser_fusion_id0(fd : FusionDefinition) -> None :
            mat1 = fd.define_tensor(shape=[-1, -1], contiguity=True, dtype=DataType.Float4_e2m1fn, is_cpu=False)
            mat2 = fd.define_tensor(shape=[-1, -1, -1], contiguity=True, dtype=DataType.Float4_e2m1fn, is_cpu=False, stride_order=[2, 0, 1])
            scale1 = fd.define_tensor(shape=[-1, -1], contiguity=True, dtype=DataType.Float8_e4m3fn, is_cpu=False)
            scale2 = fd.define_tensor(shape=[-1, -1, -1], contiguity=True, dtype=DataType.Float8_e4m3fn, is_cpu=False)
            alpha = fd.define_tensor(shape=[-1], contiguity=True, dtype=DataType.Float, is_cpu=False)
            problem_sizes = fd.define_tensor(shape=[-1, -1], contiguity=True, dtype=DataType.Int32, is_cpu=False)
            offsets = fd.define_tensor(shape=[-1], contiguity=True, dtype=DataType.Int32, is_cpu=False)
            blockscale_offsets = fd.define_tensor(shape=[-1], contiguity=True, dtype=DataType.Int32, is_cpu=False)
            out = fd.ops.cutlass_nvfp4_grouped_mm(mat1, mat2, scale1, scale2, alpha, problem_sizes, offsets, blockscale_offsets, DataType.BFloat16)
            fd.add_output(out)
        
        with FusionDefinition() as fd:
            nvfuser_fusion_id0(fd)
        
        inputs = [
          mat1.view(torch.float4_e2m1fn_x2), mat2_scaled.view(torch.float4_e2m1fn_x2).transpose(-1, -2), scale1, scale2, mat2_gs, problem_sizes, offsets, blockscale_offsets
        ]
        
        o = fd.execute(inputs)[0]
        
        
        o_decomposed_ref = torch.empty(m, n, dtype=torch.bfloat16, device='cuda:0')
        for i in range(g):
            l = offsets[i]
            l_sf = blockscale_offsets[i]
            if i == g-1:
                r = m
            else:
                r = offsets[i+1]
            r_sf = round_up(tokens_per_expert[i], 128) + l_sf
            # for some reason I cannot feed mat2_gs[i] as alpha in the torch kernel. this triggers a cublas invalid value error
            o_decomposed_ref[l:r] = torch._scaled_mm(mat1[l:r], mat2_scaled[i].transpose(-1, -2), scale1[l_sf:r_sf], scale2[i], None, None, torch.bfloat16) * mat2_gs[i]
        
        # I think we have higher error because we are not fusing the scaling factor
        assert torch.allclose(o_decomposed_ref, o, atol=1e-2, rtol=1e-2)


        # TODO: remove this, it's not relevant here.
        o_ref = torch.empty(m, n, dtype=torch.bfloat16, device='cuda:0')
        import nvfuser_direct
        nvfuser_direct.nvf_cutlass.nvfp4_scaled_grouped_mm(o_ref, mat1.view(INPUT_DTYPE), mat2_scaled.view(INPUT_DTYPE), scale1, scale2, mat2_gs, ab_strides, c_strides, problem_sizes, offsets, blockscale_offsets)
        assert torch.allclose(o_ref, o_decomposed_ref, atol=1e-3, rtol=1e-3)
