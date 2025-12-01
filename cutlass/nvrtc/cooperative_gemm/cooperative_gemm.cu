// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <cute/tensor.hpp>
#include "cute.cuh"

template <
    uint32_t ThreadBlockSize,
    uint32_t CopyMaxVecBits,
    class GMemALayout,
    class GMemBLayout,
    class GMemCLayout,
    class SMemALayout,
    class SMemBLayout,
    class SMemCLayout,
    class TA,
    class TB,
    class TC,
    class Alpha,
    class Beta,
    class TiledMma,
    class ALoadTransform,
    class BLoadTransform,
    class CLoadTransform,
    class CStoreTransform,
    class SMemCopyOpA,
    class SMemCopyOpB,
    class SMemCopyLdOpC,
    class SMemCopyStOpC>
__launch_bounds__(ThreadBlockSize) __global__ void cooperative_gemm_kernel(
    GMemALayout gmem_a_layout,
    GMemBLayout gmem_b_layout,
    GMemCLayout gmem_c_layout,
    SMemALayout smem_a_layout,
    SMemBLayout smem_b_layout,
    SMemCLayout smem_c_layout,
    TA const* a,
    TB const* b,
    TC const* c,
    TC* c_out,
    Alpha const alpha,
    Beta const beta,
    TiledMma tiled_mma,
    ALoadTransform a_load_transform,
    BLoadTransform b_load_transform,
    CLoadTransform c_load_transform,
    CStoreTransform c_store_transform,
    SMemCopyOpA a_copy_op,
    SMemCopyOpB b_copy_op,
    SMemCopyLdOpC c_copy_ld_op,
    SMemCopyStOpC c_copy_st_op) {
  using namespace cute;

  Tensor g_a_tensor = make_tensor(make_gmem_ptr(a), gmem_a_layout);
  Tensor g_b_tensor = make_tensor(make_gmem_ptr(b), gmem_b_layout);
  Tensor g_c_tensor = make_tensor(make_gmem_ptr(c), gmem_c_layout);
  Tensor g_c_out_tensor = make_tensor(make_gmem_ptr(c_out), gmem_c_layout);

  constexpr uint32_t copy_max_vec_bytes = CopyMaxVecBits / 8;

  extern __shared__ float4 smem_buf[];
  auto* smem_ptr = reinterpret_cast<unsigned char*>(smem_buf);
  auto* smem_ptr_a = smem_ptr;
  auto* smem_ptr_b = smem_ptr_a +
      round_up((sizeof(TA) * cosize(smem_a_layout)), copy_max_vec_bytes);
  auto* smem_ptr_c = smem_ptr_b +
      round_up((sizeof(TB) * cosize(smem_b_layout)), copy_max_vec_bytes);

  Tensor s_a_tensor = make_tensor(make_smem_ptr<TA>(smem_ptr_a), smem_a_layout);
  Tensor s_b_tensor = make_tensor(make_smem_ptr<TB>(smem_ptr_b), smem_b_layout);
  Tensor s_c_tensor = make_tensor(make_smem_ptr<TC>(smem_ptr_c), smem_c_layout);

  cooperative_copy<ThreadBlockSize, CopyMaxVecBits>(
      threadIdx.x, g_a_tensor, s_a_tensor);
  cooperative_copy<ThreadBlockSize, CopyMaxVecBits>(
      threadIdx.x, g_b_tensor, s_b_tensor);
  cooperative_copy<ThreadBlockSize, CopyMaxVecBits>(
      threadIdx.x, g_c_tensor, s_c_tensor);

  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();

  cooperative_gemm(
      threadIdx.x,
      tiled_mma,
      alpha,
      s_a_tensor,
      s_b_tensor,
      beta,
      s_c_tensor,
      a_load_transform,
      b_load_transform,
      c_load_transform,
      c_store_transform,
      a_copy_op,
      b_copy_op,
      c_copy_ld_op,
      c_copy_st_op);
  __syncthreads();

  cooperative_copy<ThreadBlockSize, CopyMaxVecBits>(
      threadIdx.x, s_c_tensor, g_c_out_tensor);
}
