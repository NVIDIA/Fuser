// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ir/all_nodes.h>
#include <tma.h>

namespace nvfuser::tma {

#if (__CUDACC_VER_MAJOR__ >= 12)

inline CUtensorMapDataType getCUtensorMapDataType(DataType dtype) {
  switch (std::get<PrimDataType>(dtype.type)) {
    case PrimDataType::Double:
      return CU_TENSOR_MAP_DATA_TYPE_FLOAT64;
    case PrimDataType::Float:
      return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    case PrimDataType::Half:
      return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    case PrimDataType::BFloat16:
      return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    case PrimDataType::Int:
      return CU_TENSOR_MAP_DATA_TYPE_INT64;
    case PrimDataType::Int32:
      return CU_TENSOR_MAP_DATA_TYPE_INT32;
    default:
      TORCH_INTERNAL_ASSERT(false, "Unknown tensor map data type!");
  }
}

inline CUtensorMapSwizzle getCUtensorMapSwizzle(TensorMapSwizzleType swizzle) {
  switch (swizzle) {
    case TensorMapSwizzleType::NoSwizzle:
      return CU_TENSOR_MAP_SWIZZLE_NONE;
    case TensorMapSwizzleType::B32:
      return CU_TENSOR_MAP_SWIZZLE_32B;
    case TensorMapSwizzleType::B64:
      return CU_TENSOR_MAP_SWIZZLE_64B;
    case TensorMapSwizzleType::B128:
      return CU_TENSOR_MAP_SWIZZLE_128B;
    default:
      TORCH_INTERNAL_ASSERT(false, "Unknown tensor map swizzle type!");
  }
}
#endif

TensorMap TensorMapInfo::operator()(
    void* gmem_base_ptr,
    ExpressionEvaluator& ee) const {
  TensorMap tmap;
#if (__CUDACC_VER_MAJOR__ >= 12)
  TORCH_INTERNAL_ASSERT(tv->getDataType().has_value());

  auto dim = gmem_shape.size();
  TORCH_INTERNAL_ASSERT(
      dim == gmem_strides.size() + 1 && // gmem_strides[0] implicitly 1
          dim == box_shape.size() && dim == box_strides.size(),
      "Tensor dimensionality mismatch");
  TORCH_INTERNAL_ASSERT(dim <= 5, "TMA only supports up to 5 dimensions");

  std::vector<uint64_t> evaluated_gmem_shape;
  evaluated_gmem_shape.reserve(dim);
  for (auto v : gmem_shape) {
    auto opt = ee.evaluate(v);
    TORCH_INTERNAL_ASSERT(
        opt.has_value(),
        "Failed to inference TMA tensor map. Unknown gmem_shape value: ",
        v->toInlineString());
    evaluated_gmem_shape.push_back(opt->as<int64_t>());
  }

  std::vector<uint64_t> evaluated_gmem_strides;
  evaluated_gmem_strides.reserve(dim);
  for (auto v : gmem_strides) {
    auto opt = ee.evaluate(v);
    TORCH_INTERNAL_ASSERT(
        opt.has_value(),
        "Failed to inference TMA tensor map. Unknown gmem_strides value: ",
        v->toInlineString());
    evaluated_gmem_strides.push_back(opt->as<int64_t>());
  }
  TORCH_INTERNAL_ASSERT(
      evaluated_gmem_strides.at(0) == 1,
      "The stride of the starting dimension must be 1");

  std::vector<uint32_t> evaluated_box_shape;
  evaluated_box_shape.reserve(dim);
  for (auto v : box_shape) {
    auto opt = ee.evaluate(v);
    TORCH_INTERNAL_ASSERT(
        opt.has_value(),
        "Failed to inference TMA tensor map. Unknown box_shape value: ",
        v->toInlineString());
    evaluated_box_shape.push_back(opt->as<int64_t>());
  }

  std::vector<uint32_t> evaluated_box_strides;
  evaluated_box_strides.reserve(dim);
  for (auto v : box_strides) {
    auto opt = ee.evaluate(v);
    TORCH_INTERNAL_ASSERT(
        opt.has_value(),
        "Failed to inference TMA tensor map. Unknown box_strides value: ",
        v->toInlineString());
    evaluated_box_strides.push_back(opt->as<int64_t>());
  }

  // TODO: check all requirements for TMA tensor map as described in driver API
  // stride, non-overlapping, etc.

  CUresult result = cuTensorMapEncodeTiled(
      &tmap,
      getCUtensorMapDataType(tv->getDataType().value()),
      dim,
      gmem_base_ptr, // TODO: check alignment
      evaluated_gmem_shape.data(),
      evaluated_gmem_strides.data() + 1, // gmem_strides[0] implicitly 1
      evaluated_box_shape.data(),
      evaluated_box_strides.data(),
      CU_TENSOR_MAP_INTERLEAVE_NONE,  // TODO: determine interleaving
      getCUtensorMapSwizzle(
          TensorMapSwizzleType::NoSwizzle), // TODO: support swizzle
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE); // TODO: what is this?

  TORCH_INTERNAL_ASSERT(result == CUDA_SUCCESS, "Failed to create tensor map!");
#endif
  return tmap;
}

} // namespace nvfuser::tma