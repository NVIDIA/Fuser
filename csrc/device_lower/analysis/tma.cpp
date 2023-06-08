// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <device_lower/analysis/tma.h>
#include <device_lower/lower2device.h>
#include <ir/all_nodes.h>
#include <kernel_ir_dispatch.h>
#include <tma.h>

namespace nvfuser {

// This is a super hacky function to just feed the right index to make the
// test pass. The main purpose of this function is to make the initial bring-up
// PR small, but still able to test the infrastructure built in this PR. This
// function will be removed and replaced with a real implementation in a
// follow-up PR.
class CollectTMATensorMapInfo : public kir::IrVisitor {
 public:
  using kir::IrVisitor::handle;
  CollectTMATensorMapInfo(const std::vector<Expr*>& exprs) {
    handle(exprs);
  }
  // Assuming the following schedule:
  // [crd0, bulk, crd1, bulk, crd2, bulk, ...]
  // where crd0 and the bulk next to it is split from a single ID in allocation
  // domain, same for other crds and bulks. Tensors are assumed to be
  // contiguous. There is no check for the assumptions above.
  void handle(Expr* expr) override {
    if (expr->isA<LoadStoreOp>()) {
      auto ldst = expr->as<LoadStoreOp>();
      if (ldst->opType() == LoadStoreOpType::CpAsyncBulkTensorTile) {
        auto& tensor_map = GpuLower::current()->tmaTensorMaps().emplace_back(
            tma::TensorMapInfo{});
        auto out = ldst->out()->as<kir::TensorIndex>();
        auto out_tv = out->view();
        tensor_map.dtype = std::get<PrimDataType>(out_tv->dtype().type);
        tensor_map.swizzle = tma::TensorMapSwizzleType::NoSwizzle;
        tensor_map.tv = out_tv;
        auto dim = (int64_t)out_tv->getMaybeAllocationDomain().size();
        for (int64_t i = 0; i < dim; ++i) {
          auto ii = dim - i - 1;
          tensor_map.gmem_shape.emplace_back(
              out_tv->getMaybeAllocationDomain().at(ii)->extent());
          tensor_map.box_shape.emplace_back(
              out_tv->getLeafDomain().at(2 * ii + 1)->extent());

          Val* stride = ldst->container()->oneVal();
          for (int64_t j : c10::irange(ii + 1, dim)) {
            stride = SimplifyingIrBuilder::mulExpr(
                stride, out_tv->getMaybeAllocationDomain().at(j)->extent());
          }
          tensor_map.gmem_strides.emplace_back(stride);
          // TODO: support discontig
          tensor_map.box_strides.emplace_back(expr->container()->oneVal());
        }
        GpuLower::current()->tmaTensorMapsMap()[ldst] =
            GpuLower::current()->tmaTensorMaps().size() - 1;
      }
    } else {
      kir::IrVisitor::handle(expr);
    }
  }
};

void collectTMATensorMapInfo(const std::vector<Expr*>& exprs) {
  CollectTMATensorMapInfo _(exprs);
}

} // namespace nvfuser
