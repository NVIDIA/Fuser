// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <parallel_dimension_map.h>

#include <ATen/cuda/CUDAContext.h>
#include <device_lower/lower2device.h>
#include <disjoint_set.h>
#include <expr_simplifier.h>
#include <ir/utils.h>
#include <iter_visitor.h>

#include <functional>
#include <sstream>
#include <string>
#include <utility>

using PAndID = std::pair<nvfuser::ParallelType, nvfuser::IterDomain*>;

namespace std {

template <>
struct hash<PAndID> {
  std::size_t operator()(const PAndID& data) const noexcept {
    size_t ptype = static_cast<size_t>(data.first);
    size_t address = reinterpret_cast<size_t>(data.second);
    size_t combined = (address << 8) | ptype;
    return std::hash<size_t>()(combined);
  }
};

} // namespace std

namespace nvfuser {

void ParallelDimensionMap::build(Fusion* fusion) {
  VectorOfUniqueEntries<PAndID> all_concrete_ids;
  auto all_vals = fusion->usedMathVals();
  for (auto tv : ir_utils::filterByType<TensorView>(all_vals)) {
    if (tv->isCircularBuffered() &&
        std::holds_alternative<WarpSpecialized>(
            tv->circularBufferOptions().type)) {
      const auto& warp_specialized =
          std::get<WarpSpecialized>(tv->circularBufferOptions().type);
      warp_specialized_types_.insert(warp_specialized.on);
      if (warp_specialized.num_registers.has_value()) {
        ws_with_register_sharing_.insert(warp_specialized.on);
      }
    }
    for (auto id : tv->domain()->allIDs()) {
      auto ptype = id->getParallelType();
      if (!isParallelTypeThread(ptype)) {
        continue;
      }
      auto concrete_id = GpuLower::current()->caMap()->getConcreteMappedID(
          id, IdMappingMode::EXACT);
      if (concrete_id->isBroadcast()) {
        // Broadcasted concrete id's don't specify anything about shape
        continue;
      }
      all_concrete_ids.pushBack(std::make_pair(ptype, concrete_id));
    }
  }

  // Scan all TVs to build dim_map_
  for (auto [ptype, concrete_id] : all_concrete_ids) {
    exact_types_.insert(ptype); // insert now and cleanup later
    if (dim_map_.count(ptype) == 0) {
      dim_map_[ptype] = concrete_id->extent();
    } else {
      dim_map_.at(ptype) = SimplifyingIrBuilder::maxExpr(
          dim_map_.at(ptype), concrete_id->extent());
    }
  }

  // Simplify dim_map_
  for (auto& [k, v] : dim_map_) {
    v = simplifyExpr(v);
  }

  // Compute exact_types_
  for (auto [ptype, concrete_id] : all_concrete_ids) {
    auto expr_val = simplifyExpr(SimplifyingIrBuilder::eqExpr(
                                     dim_map_.at(ptype), concrete_id->extent()))
                        ->value();
    if (!expr_val.hasValue() || !expr_val.as<bool>()) {
      exact_types_.erase(ptype);
    }
  }

  adjustMappingsForWarpPadding();
  std::cout << "afterWarp Specialization Pad " << std::endl;
  adjustMappingsForWarpSpecilization();
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
  auto warp_size_val = IrBuilder::create<Val>(32L, DataType::Index);
  auto tidx_dim = getRaw(tidx_pt);

  NVF_ERROR(tidx_dim != nullptr);

  // If tidx is strictly defined as blockDim.x then it must be set to a
  // multiple of the warp, there is nothing to do
  if (tidx_dim->sameAs(NamedScalar::getParallelDim(tidx_pt))) {
    return;
  }

  auto expr_val =
      simplifyExpr(SimplifyingIrBuilder::eqExpr(
                       SimplifyingIrBuilder::modExpr(tidx_dim, warp_size_val),
                       tidx_dim->container()->zeroVal()))
          ->value();

  // If already multiple of warp, nothing to do
  if (expr_val.is<bool>() && expr_val.as<bool>()) {
    return;
  }

  // TIDx is padded to a multiple of warp. If it's known to be a
  // single warp, use the constant warp size as the dimension of
  // TIDx. Otherwise, just use blockDim.x.
  if (warp_info.is_tidx_single_warp) {
    dim_map_.at(ParallelType::TIDx) = warp_size_val;
  } else {
    dim_map_.at(ParallelType::TIDx) =
        simplifyExpr(SimplifyingIrBuilder::mulExpr(
            SimplifyingIrBuilder::ceilDivExpr(tidx_dim, warp_size_val),
            warp_size_val));
  }

  // TIDx is no longer exact
  exact_types_.erase(ParallelType::TIDx);
}

void ParallelDimensionMap::adjustMappingsForWarpSpecilization() {
  NVF_ERROR(
      ws_with_register_sharing_.size() <= 1,
      "Warp specilization with register sharing is only supported on one parallel type.");
  // shortcut for case without register sharing
  if (ws_with_register_sharing_.empty()) {
    for (auto pt : warp_specialized_types_) {
      auto dim_it = dim_map_.find(pt);
      if (dim_it == dim_map_.end()) {
        dim_map_[pt] = IrBuilder::create<Val>(2, DataType::Index);
      } else {
        // Intentionally not using SimplifyingIrBuilder::addExpr here so that
        // we still have access to the pointer to the original IR node.
        // We need the pointer to the original IR node because we want
        // getRawCompute to be callable in an environment without FusionGuard,
        // that is, when the IR container is read-only. In such an environment,
        // we can't create new IR nodes for (x - 1). By using
        // IrBuilder::addExpr, we can always create IR nodes like addExpr(x, 1),
        // and SimplifyingIrBuilder::addExpr in getRawCompute will be able to
        // simplify find the x when we do addExpr(addExpr(x, 1) - 1).
        dim_map_[pt] = IrBuilder::addExpr(
            dim_it->second, IrBuilder::create<Val>(1, DataType::Index));
      }
      exact_types_.erase(pt);
    }
    return;
  }
  // For register sharing, require contiguous 128 threads calling the same
  // setreg instruction
  auto pt = *ws_with_register_sharing_.begin();
  auto dim_it = dim_map_.find(pt);

  auto checkAndPadDim = [&](ParallelType pt_checked, ParallelType pt_padded) {
    Val* bdim = dim_map_.at(pt_checked);
    NVF_ERROR(
        bdim->isConstScalar(),
        "bdim must be a constant scalar for register sharing, bdim= ",
        bdim->toInlineString());
    int64_t bdim_val = bdim->value().as<int64_t>();
    NVF_ERROR(
        (128 % bdim_val == 0 || bdim_val % 128 == 0),
        "For register sharing bdim_val must can evenly divide or divide by 128, bdim= ",
        bdim_val);
    if (dim_it == dim_map_.end()) {
      NVF_ERROR(
          bdim_val >= 128 && bdim_val % 128 == 0,
          "Before warp specilization padding, there must be 128 * N threads, bdim= ",
          bdim_val);
    } else {
      if (dim_it->second->isConstScalar()) {
        int64_t threads_bofore_pad =
            dim_it->second->value().as<int64_t>() * bdim_val;
        NVF_ERROR(
            threads_bofore_pad >= 128 && threads_bofore_pad % 128 == 0,
            "Before warp specilization padding, there must be 128 * N threads, bdim= ",
            threads_bofore_pad);
      } else {
        // we can't verify wheter the block size is multiple of 128. Defer check
        // to executor.
      }
    }

    int64_t pad_val = bdim_val > 128 ? 1 : 128 / bdim_val;
    auto off_set = IrBuilder::create<Val>(pad_val, DataType::Index);
    auto current_val = dim_it == dim_map_.end()
        ? IrBuilder::create<Val>(1, DataType::Index)
        : dim_it->second;
    dim_map_[pt_padded] = IrBuilder::addExpr(current_val, off_set);
    return pad_val;
  };

  auto pad128 = [&](ParallelType pt_pad) {
    auto off_set = IrBuilder::create<Val>(128, DataType::Index);
    NVF_ERROR(
        dim_it != dim_map_.end(),
        "Padding 128 threads to a non-existing dim leads to 129 threads, which is not a multiply of 128.");
    if (dim_it->second->isConstScalar()) {
      NVF_ERROR(
          dim_it->second->value().as<int64_t>() % 128 == 0,
          "Padding 128 threads to a dim that is not a multiply of 128 leads to a dim can't be divised by 128.");
    }
    dim_map_[pt_pad] = IrBuilder::addExpr(dim_it->second, off_set);
    return 128;
  };
  switch (pt) {
    // threadIdx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z *
    // blockDim.x * blockDim.y If on TIDx, we need to patch additional 128
    // threads executor should ensure there are 128 * N threads before padding
    case ParallelType::TIDx:
      warp_specialization_padded_vals_[pt] = pad128(pt);
      break;
    case ParallelType::TIDy:
      if (!dim_map_.contains(ParallelType::TIDx)) {
        // If TIDx is not used, pad 128 threads to TIDy
        warp_specialization_padded_vals_[pt] = pad128(pt);
      } else {
        // If TIDx is used, pad [128 / bdimx] or 1 (when bdimx > 128 &
        // bdimx%128==0) threads to TIDy
        warp_specialization_padded_vals_[pt] =
            checkAndPadDim(ParallelType::TIDx, pt);
      }
      break;
    case ParallelType::TIDz:
      if (!dim_map_.contains(ParallelType::TIDx)) {
        if (!dim_map_.contains(ParallelType::TIDy)) {
          // If TIDx & TIDy are not used, pad 128 threads to TIDz
          warp_specialization_padded_vals_[pt] = pad128(pt);
        } else {
          // If TIDx is not used, TIDy is used, pad 128 / bdimx threads or 1 to
          // TIDz
          warp_specialization_padded_vals_[pt] =
              checkAndPadDim(ParallelType::TIDy, pt);
        }
      } else {
        // If TIDx is used, TIDy is not used, pad 128 / bdimx threads or 1 to
        // TIDz
        if (!dim_map_.contains(ParallelType::TIDy)) {
          warp_specialization_padded_vals_[pt] =
              checkAndPadDim(ParallelType::TIDx, pt);
        } else {
          // If TIDx & TIDy are used, pad 128/(bdimx*bdimy) threads or 1 to TIDz
          Val* bdimx = dim_map_.at(ParallelType::TIDx);
          Val* bdimy = dim_map_.at(ParallelType::TIDy);
          NVF_ERROR(
              bdimx->isConstScalar() && bdimy->isConstScalar(),
              "bdimx and bdimy must be constant scalars for register sharing on bdimz, bdimx= ",
              bdimx->toString(),
              ", bdimy= ",
              bdimy->toString());
          int64_t bdim_xy_val =
              bdimx->value().as<int64_t>() * bdimy->value().as<int64_t>();
          NVF_ERROR(
              (128 % bdim_xy_val == 0 || bdim_xy_val % 128 == 0),
              "For register sharing on TIDz, TIDx*TIDy must can evenly divide or divide by 128, bdim_xy_val= ",
              bdim_xy_val);
          int64_t pad_val = bdim_xy_val > 128 ? 1 : 128 / bdim_xy_val;
          auto off_set = IrBuilder::create<Val>(pad_val, DataType::Index);
          dim_map_[pt] = dim_it == dim_map_.end()
              ? off_set
              : IrBuilder::addExpr(dim_it->second, off_set);
          warp_specialization_padded_vals_[pt] = pad_val;
        }
      }
      break;
    default:
      NVF_THROW("Unsupported parallel type for register sharing: ", pt);
      break;
  }
  exact_types_.erase(pt);
}

Val* ParallelDimensionMap::getRaw(ParallelType pt) const {
  NVF_ERROR(isParallelTypeThread(pt), "Invalid ParallelType: ", pt);
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

Val* ParallelDimensionMap::getRawCompute(ParallelType pt) const {
  Val* raw = getRaw(pt);
  if (warp_specialized_types_.count(pt)) {
    auto padded_val = getWarpSpecilizationPaddedVal(pt);
    return SimplifyingIrBuilder::addExpr(raw, -padded_val);
  }
  return raw;
}

Val* ParallelDimensionMap::getNumComputeThreadsEachBlock() const {
  Val* num_threads = FusionGuard::getCurFusion()->oneVal();
  for (auto pt : kParallelTypeTIDs) {
    auto dim = getRawCompute(pt);
    if (dim == nullptr) {
      continue;
    }
    num_threads = SimplifyingIrBuilder::mulExpr(num_threads, dim);
  }
  return num_threads;
}

int64_t ParallelDimensionMap::getWarpSpecilizationPaddedVal(
    ParallelType pt) const {
  NVF_ERROR(
      warp_specialized_types_.contains(pt), "Can't find ParallelType: ", pt);
  if (!ws_with_register_sharing_.contains(pt)) {
    return 1;
  }
  NVF_ERROR(
      warp_specialization_padded_vals_.contains(pt),
      "Can't find padded val for: ",
      pt);
  return warp_specialization_padded_vals_.at(pt);
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
