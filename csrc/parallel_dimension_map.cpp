// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <parallel_dimension_map.h>

#include <ATen/cuda/CUDAContext.h>
#include <expr_simplifier.h>
#include <ir_utils.h>
#include <iter_visitor.h>
#include <lower2device.h>

#include <sstream>

namespace nvfuser {

void ParallelDimensionMap::build(Fusion* fusion) {
  // Scan all TVs to build dim_map_
  auto all_vals = fusion->usedMathVals();
  for (auto tv : ir_utils::filterByType<TensorView>(all_vals)) {
    for (auto id : tv->domain()->domain()) {
      auto ptype = id->getParallelType();
      if (!isParallelTypeThread(ptype)) {
        continue;
      }
      exact_types_.insert(ptype); // insert now and cleanup later

      auto concrete_id = GpuLower::current()->caMap()->getConcreteMappedID(
          id, IdMappingMode::EXACT);
      if (concrete_id->isBroadcast()) {
        // Broadcasted concrete id's don't specify anything about shape
        continue;
      }

      if (dim_map_.count(ptype) == 0) {
        dim_map_[ptype] = concrete_id->extent();
      } else {
        dim_map_.at(ptype) = SimplifyingIrBuilder::maxExpr(
            dim_map_.at(ptype), concrete_id->extent());
      }
    }
  }

  // Simplify dim_map_
  for (auto& [k, v] : dim_map_) {
    v = simplifyExpr(v);
  }

  // Compute exact_types_
  for (auto tv : ir_utils::filterByType<TensorView>(all_vals)) {
    for (auto id : tv->domain()->domain()) {
      auto ptype = id->getParallelType();
      if (exact_types_.count(ptype) == 0) {
        continue;
      }
      auto concrete_id = GpuLower::current()->caMap()->getConcreteMappedID(
          id, IdMappingMode::EXACT);
      if (concrete_id->isBroadcast()) {
        // Broadcasted concrete id's don't specify anything about shape
        continue;
      }
      if (simplifyExpr(SimplifyingIrBuilder::eqExpr(
                           dim_map_.at(ptype), concrete_id->extent()))
              ->getBool() != true) {
        exact_types_.erase(ptype);
      }
    }
  }

  adjustMappingsForWarpPadding();
}

void ParallelDimensionMap::adjustMappingsForWarpPadding() {
  const auto gpu_lower = GpuLower::current();

  // If TIDx is padded to a multiple of the warp size, mark it as
  // non-exact.

  auto& warp_info = gpu_lower->getWarpPaddedParallelInfo();
  // TIDx isn't really padded if there isn't a warp reduction (this could
  // change)
  if (!(warp_info.is_tidx_padded && warp_info.has_warp_reduction)) {
    return;
  }

  const auto tidx_pt = ParallelType::TIDx;
  auto warp_size = 32;

  // If the dimension of TIDx is actually a multple of the warp size
  // before padding, it can be left as exact
  if (isExact(tidx_pt)) {
    auto tidx_dim = dynamic_cast<Int*>(getRaw(tidx_pt));
    if (tidx_dim) {
      if (tidx_dim->isConst()) {
        auto tidx_dim_val = tidx_dim->value().value();
        if (tidx_dim_val % warp_size == 0) {
          // Dimension of TIDx is a multiple of the warp size
          return;
        }
      }
      // If tidx is strictly defined as blockDim.x then it must be set to a
      // multiple of the warp and can be considered exact
      if (tidx_dim->sameAs(NamedScalar::getParallelDim(tidx_pt))) {
        return;
      }
    }
  }

  // TIDx is padded to a multiple of warp. If it's known to be a
  // single warp, use the constant warp size as the dimension of
  // TIDx. Otherwise, just use blockDim.x.
  if (warp_info.is_tidx_single_warp) {
    dim_map_.at(ParallelType::TIDx) = IrBuilder::create<Int>(warp_size);
  } else {
    dim_map_.at(ParallelType::TIDx) =
        NamedScalar::getParallelDim(ParallelType::TIDx);
  }

  // TIDx is no longer exact
  exact_types_.erase(ParallelType::TIDx);
}

Val* ParallelDimensionMap::getRaw(ParallelType pt) const {
  TORCH_INTERNAL_ASSERT(isParallelTypeThread(pt), "Invalid ParallelType: ", pt);
  auto it = dim_map_.find(pt);
  if (it == dim_map_.end()) {
    return nullptr;
  } else {
    return it->second;
  }
}

Val* ParallelDimensionMap::get(ParallelType pt) const {
  auto raw = getRaw(pt);
  if (raw != nullptr && !raw->isConstInt()) {
    return NamedScalar::getParallelDim(pt);
  }
  return raw;
}

bool ParallelDimensionMap::isExact(ParallelType pt) const {
  return exact_types_.find(pt) != exact_types_.end();
}

std::string ParallelDimensionMap::toString() const {
  std::stringstream ss;
  for (auto pt : kParallelTypeThreads) {
    ss << pt << ": ";
    auto dim = getRaw(pt);
    if (dim != nullptr) {
      ss << dim->toInlineString();
      if (isExact(pt)) {
        ss << ", exact";
      } else {
        ss << ", non-exact";
      }
    } else {
      ss << "unused";
    }
    ss << "\n";
  }

  return ss.str();
}

} // namespace nvfuser
