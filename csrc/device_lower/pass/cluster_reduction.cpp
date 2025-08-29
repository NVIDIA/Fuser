// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/pass/cluster_reduction.h>

#include <device_lower/lower2device.h>
#include <ir/utils.h>
#include <kernel_ir.h>
#include <kernel_ir_dispatch.h>

#include <vector>

namespace nvfuser {

namespace {

// Converts ReductionOp to ClusterReductionOp for cluster-level reductions
class ClusterReductionConverter : public kir::ExprMutator {
 public:
  static std::vector<Expr*> convert(const std::vector<Expr*>& exprs) {
    ClusterReductionConverter converter(exprs);
    return converter.exprs_;
  }

 private:
  explicit ClusterReductionConverter(const std::vector<Expr*>& exprs) {
    kir::ExprMutator::traverseAndInsert(exprs);
  }

  using kir::ExprMutator::handle;

  void handle(ReductionOp* rop) final {
    // Only convert single-output reductions marked for cluster reduction
    if (ir_utils::getTvOutput(rop)->domain()->hasClusterReduction()) {
      // Extract reduction parameters
      auto out = rop->out();
      auto in = rop->in();
      auto reduction_op_type = rop->getReductionOpType();
      auto init_val = rop->init();

      // Get mbarrier allocated during allocation pass
      auto cluster_mbarrier_tv =
          GpuLower::current()->clusterReductionMBarrier();
      NVF_CHECK(
          cluster_mbarrier_tv != nullptr,
          "Cluster mbarrier must be allocated for cluster reductions");
      // Index into mbarrier array for this reduction
      auto mbarrier = IrBuilder::create<kir::TensorIndex>(
          cluster_mbarrier_tv,
          IrBuilder::create<Val>(current_cluster_index_, DataType::Index));
      current_cluster_index_++;
      // Replace ReductionOp with ClusterReductionOp
      auto cluster_reduction = IrBuilder::create<kir::ClusterReductionOp>(
          out, in, reduction_op_type, init_val, mbarrier);
      registerReplace(rop, cluster_reduction);
    }
  }

  int64_t current_cluster_index_ = 0; // Track mbarrier index assignment
};

} // namespace

std::vector<Expr*> convertToClusterReduction(const std::vector<Expr*>& exprs) {
  if (GpuLower::current()->clusterReductionCount() < 1) {
    return exprs;
  }
  return ClusterReductionConverter::convert(exprs);
}

} // namespace nvfuser
