// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <tma.h>

#include <cuda_utils.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <utils.h>

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>

namespace nvfuser {

namespace tma {

#if (CUDA_VERSION >= 12000)

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
      NVF_ERROR(false, "Unknown tensor map data type!");
  }
}

inline CUtensorMapInterleave getCUtensorMapInterleave(
    TensorMapInterleave interleave) {
  switch (interleave) {
    case TensorMapInterleave::NoInterleave:
      return CU_TENSOR_MAP_INTERLEAVE_NONE;
    case TensorMapInterleave::B16:
      return CU_TENSOR_MAP_INTERLEAVE_16B;
    case TensorMapInterleave::B32:
      return CU_TENSOR_MAP_INTERLEAVE_32B;
    default:
      NVF_ERROR(false, "Unknown tensor map interleave type!");
  }
}

inline CUtensorMapSwizzle getCUtensorMapSwizzle(TensorMapSwizzle swizzle) {
  switch (swizzle) {
    case TensorMapSwizzle::NoSwizzle:
      return CU_TENSOR_MAP_SWIZZLE_NONE;
    case TensorMapSwizzle::B32:
      return CU_TENSOR_MAP_SWIZZLE_32B;
    case TensorMapSwizzle::B64:
      return CU_TENSOR_MAP_SWIZZLE_64B;
    case TensorMapSwizzle::B128:
      return CU_TENSOR_MAP_SWIZZLE_128B;
    default:
      NVF_ERROR(false, "Unknown tensor map swizzle type!");
  }
}

inline CUtensorMapL2promotion getCUtensorMapL2Promotion(
    TensorMapL2Promotion l2_promotion) {
  switch (l2_promotion) {
    case TensorMapL2Promotion::NoL2Promotion:
      return CU_TENSOR_MAP_L2_PROMOTION_NONE;
    case TensorMapL2Promotion::B64:
      return CU_TENSOR_MAP_L2_PROMOTION_L2_64B;
    case TensorMapL2Promotion::B128:
      return CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    case TensorMapL2Promotion::B256:
      return CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
    default:
      NVF_ERROR(false, "Unknown tensor map L2 promotion type!");
  }
}

inline CUtensorMapFloatOOBfill getCUtensorMapFloatOOBfill(
    TensorMapFloatOOBFill oob_fill) {
  switch (oob_fill) {
    case TensorMapFloatOOBFill::NoOOBFill:
      return CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    case TensorMapFloatOOBFill::NaN_Request_Zero_FMA:
      return CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA;
    default:
      NVF_ERROR(false, "Unknown tensor map OOB fill type!");
  }
}

using TensorMap = CUtensorMap;

#endif

std::ostream& operator<<(std::ostream& os, TensorMapInterleave interleave) {
  switch (interleave) {
    case TensorMapInterleave::NoInterleave:
      os << "NoInterleave";
      break;
    case TensorMapInterleave::B16:
      os << "16B";
      break;
    case TensorMapInterleave::B32:
      os << "32B";
      break;
    default:
      NVF_CHECK(false, "Unknown tensor map interleave type!");
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, TensorMapSwizzle swizzle) {
  switch (swizzle) {
    case TensorMapSwizzle::NoSwizzle:
      os << "NoSwizzle";
      break;
    case TensorMapSwizzle::B32:
      os << "32B";
      break;
    case TensorMapSwizzle::B64:
      os << "64B";
      break;
    case TensorMapSwizzle::B128:
      os << "128B";
      break;
    default:
      NVF_CHECK(false, "Unknown tensor map swizzle type!");
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, TensorMapL2Promotion l2_promotion) {
  switch (l2_promotion) {
    case TensorMapL2Promotion::NoL2Promotion:
      os << "NoL2Promotion";
      break;
    case TensorMapL2Promotion::B64:
      os << "64B";
      break;
    case TensorMapL2Promotion::B128:
      os << "128B";
      break;
    case TensorMapL2Promotion::B256:
      os << "256B";
      break;
    default:
      NVF_CHECK(false, "Unknown tensor map L2 promotion type!");
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, TensorMapFloatOOBFill oob_fill) {
  switch (oob_fill) {
    case TensorMapFloatOOBFill::NoOOBFill:
      os << "NoOOBFill";
      break;
    case TensorMapFloatOOBFill::NaN_Request_Zero_FMA:
      os << "NaN_Request_Zero_FMA";
      break;
    default:
      NVF_CHECK(false, "Unknown tensor map float OOB fill type!");
      break;
  }
  return os;
}

Val* encodeTensorMapTiled(
    DataType data_type,
    Val* global_address,
    Val* global_dim,
    Val* global_strides,
    Val* box_dim,
    Val* element_strides,
    TensorMapInterleave interleave,
    TensorMapSwizzle swizzle,
    TensorMapL2Promotion l2_promotion,
    TensorMapFloatOOBFill oob_fill) {
  auto output = IrBuilder::create<Val>(
      OpaqueType::make<TensorMap>("const __grid_constant__ TensorMap"));
  IrBuilder::create<kir::EncodeTensorMapTiled>(
      output,
      data_type,
      global_address,
      global_dim,
      global_strides,
      box_dim,
      element_strides,
      interleave,
      swizzle,
      l2_promotion,
      oob_fill);
  return output;
}

} // namespace tma

std::vector<PolymorphicValue> kir::EncodeTensorMapTiled::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  using namespace tma;
#if (CUDA_VERSION >= 12000)
  cuuint32_t tensor_rank = tensorRank();
  NVF_ERROR(
      inputs.size() == 5,
      "Incorrect number of inputs to EncodeTensorMapTiled!");

  NVF_ERROR(inputs.at(0).is<Pointer>());
  void* global_address = (void*)inputs.at(0);

  NVF_ERROR(inputs.at(1).is<std::vector>());
  auto global_dim = (std::vector<cuuint64_t>)inputs.at(1);

  NVF_ERROR(inputs.at(2).is<std::vector>());
  auto global_strides = (std::vector<cuuint64_t>)inputs.at(2);

  NVF_ERROR(inputs.at(3).is<std::vector>());
  auto box_dim = (std::vector<cuuint32_t>)inputs.at(3);

  NVF_ERROR(inputs.at(4).is<std::vector>());
  auto element_strides = (std::vector<cuuint32_t>)inputs.at(4);

  CUtensorMapDataType data_type = getCUtensorMapDataType(dataType());
  CUtensorMapInterleave interleave =
      getCUtensorMapInterleave(this->interleave());
  CUtensorMapSwizzle swizzle = getCUtensorMapSwizzle(this->swizzle());
  CUtensorMapL2promotion l2_promotion =
      getCUtensorMapL2Promotion(l2Promotion());
  CUtensorMapFloatOOBfill oob_fill = getCUtensorMapFloatOOBfill(oobFill());

  // Checks based on the documentation of cuTensorMapEncodeTiled, error messages
  // are mostly directly copied from the doc

  NVF_ERROR(
      tensor_rank != 0 && tensor_rank <= 5,
      "tensorRank must be non-zero and less than or equal to the maximum supported dimensionality of 5.",
      " tensor_rank = ",
      tensor_rank);
  if (interleave != CU_TENSOR_MAP_INTERLEAVE_NONE) {
    NVF_ERROR(
        tensor_rank >= 3,
        "If interleave is not CU_TENSOR_MAP_INTERLEAVE_NONE, then tensorRank must additionally be greater than or equal to 3.",
        " tensor_rank = ",
        tensor_rank);
  }

  size_t global_address_int = reinterpret_cast<size_t>(global_address);
  if (interleave == CU_TENSOR_MAP_INTERLEAVE_32B) {
    NVF_ERROR(
        global_address_int % 32 == 0,
        "globalAddress, which specifies the starting address of the memory region described, must be 32 byte aligned when interleave is CU_TENSOR_MAP_INTERLEAVE_32B and 16 byte aligned otherwise.",
        " global_address = ",
        global_address_int,
        ", interleave mode = ",
        interleave);
  } else {
    NVF_ERROR(
        global_address_int % 16 == 0,
        "globalAddress, which specifies the starting address of the memory region described, must be 32 byte aligned when interleave is CU_TENSOR_MAP_INTERLEAVE_32B and 16 byte aligned otherwise.",
        " global_address = ",
        global_address_int,
        ", interleave mode = ",
        interleave);
  }

  for (auto global_dim_val : global_dim) {
    constexpr cuuint64_t max_size = (cuuint64_t)1 << 32;
    NVF_ERROR(
        global_dim_val != 0 && global_dim_val <= max_size,
        "globalDim array, which specifies tensor size of each of the tensorRank dimensions, must be non-zero and less than or equal to 2^32.",
        " global_dim_val = ",
        global_dim_val);
  }

  for (auto global_stride_val : global_strides) {
    constexpr cuuint64_t max_stride = (cuuint64_t)1 << 40;
    NVF_ERROR(
        global_stride_val % 16 == 0 && global_stride_val <= max_stride,
        "globalStrides array, which specifies tensor stride of each of the lower tensorRank - 1 dimensions in bytes, must be a multiple of 16 and less than 2^40.",
        " global_stride_val = ",
        global_stride_val);
    if (interleave == CU_TENSOR_MAP_INTERLEAVE_32B) {
      NVF_ERROR(
          global_stride_val % 32 == 0,
          "The stride must be a multiple of 32 when interleave is CU_TENSOR_MAP_INTERLEAVE_32B.",
          " global_stride_val = ",
          global_stride_val);
    }
  }
  int64_t elem_size = (int64_t)dataTypeSize(dataType());
  if (tensor_rank > 1) {
    int64_t padding0 =
        (int64_t)global_strides.at(0) - (int64_t)global_dim.at(0) * elem_size;
    NVF_ERROR(
        padding0 >= 0,
        "Negative pad0 for: globalStrides[0] = globalDim[0] * elementSizeInBytes(tensorDataType) + padding[0];"
        " padding0 = ",
        padding0);
    for (int i = 1; i < (int64_t)tensor_rank - 1; i++) {
      int64_t stride_mul_pad_i = (int64_t)global_strides.at(i) -
          (int64_t)global_dim.at(i) * (int64_t)global_strides.at(i - 1);
      NVF_ERROR(
          stride_mul_pad_i >= 0,
          "Negative globalStrides[i – 1] * padding[i] for: globalStrides[i] = globalStrides[i – 1] * (globalDim[i] + padding[i]);",
          " stride_mul_pad_i = ",
          stride_mul_pad_i);
      // TODO: the check below is copied from the official doc, but does it
      // really make sense? Strides are in the unit of bytes, but global_dim is
      // in the unit of elements, how can they compare with each other?
      NVF_ERROR(global_strides.at(i) >= global_dim.at(i));
    }
  }

  for (auto box_dim_val : box_dim) {
    NVF_ERROR(
        box_dim_val != 0 && box_dim_val <= 256,
        "boxDim array, which specifies number of elements to be traversed along each of the tensorRank dimensions, must be non-zero and less than or equal to 256.",
        " box_dim_val = ",
        box_dim_val);
  }
  if (interleave == CU_TENSOR_MAP_INTERLEAVE_NONE) {
    NVF_ERROR(
        (box_dim.at(0) * elem_size) % 16 == 0,
        "When interleave is CU_TENSOR_MAP_INTERLEAVE_NONE, { boxDim[0] * elementSizeInBytes( tensorDataType ) } must be a multiple of 16 bytes.",
        " box_dim[0] = ",
        box_dim.at(0),
        " elem_size = ",
        elem_size);
  }

  for (auto element_stride_val : element_strides) {
    NVF_ERROR(
        element_stride_val != 0 && element_stride_val <= 8,
        "elementStrides array, which specifies the iteration step along each of the tensorRank dimensions, must be non-zero and less than or equal to 8.",
        " element_stride_val = ",
        element_stride_val);
  }

  if (interleave == CU_TENSOR_MAP_INTERLEAVE_NONE) {
    auto bounding_box_inner_dim = box_dim.at(0) * elem_size;
    switch (swizzle) {
      case CU_TENSOR_MAP_SWIZZLE_32B:
        NVF_ERROR(
            bounding_box_inner_dim <= 32,
            "CU_TENSOR_MAP_SWIZZLE_32B implies the bounding box inner dimension will be <= 32.",
            " bounding_box_inner_dim = ",
            bounding_box_inner_dim);
        break;
      case CU_TENSOR_MAP_SWIZZLE_64B:
        NVF_ERROR(
            bounding_box_inner_dim <= 64,
            "CU_TENSOR_MAP_SWIZZLE_64B implies the bounding box inner dimension will be <= 64.",
            " bounding_box_inner_dim = ",
            bounding_box_inner_dim);
        break;
      case CU_TENSOR_MAP_SWIZZLE_128B:
        NVF_ERROR(
            bounding_box_inner_dim <= 128,
            "CU_TENSOR_MAP_SWIZZLE_128B implies the bounding box inner dimension will be <= 128.",
            " bounding_box_inner_dim = ",
            bounding_box_inner_dim);
        break;
      default:;
    }
  }

  if (interleave == CU_TENSOR_MAP_INTERLEAVE_32B) {
    NVF_ERROR(
        swizzle == CU_TENSOR_MAP_SWIZZLE_32B,
        "When interleave is CU_TENSOR_MAP_INTERLEAVE_32B, swizzle must be CU_TENSOR_MAP_SWIZZLE_32B.");
  }

  if (oob_fill == CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA) {
    NVF_ERROR(
        isFloatingPointType(dataType()),
        "CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA can only be used when tensorDataType represents a floating-point data type.",
        " dataType = ",
        dataType());
  }
  // All check passes

  // For the case where tensor_rank == 1, global_strides is not used, however,
  // we still need to pass a valid pointer to cuTensorMapEncodeTiled...
  cuuint64_t useless_data = 0;

  TensorMap tensor_map;
  NVFUSER_CUDA_SAFE_CALL(cuTensorMapEncodeTiled(
      &tensor_map,
      data_type,
      tensor_rank,
      global_address,
      global_dim.data(),
      global_strides.empty() ? &useless_data : global_strides.data(),
      box_dim.data(),
      element_strides.data(),
      interleave,
      swizzle,
      l2_promotion,
      oob_fill));

  return {Opaque{tensor_map}};
#else
  NVF_ERROR(false, "TMA is only supported on CUDA 12 and above!");
#endif
}

} // namespace nvfuser
