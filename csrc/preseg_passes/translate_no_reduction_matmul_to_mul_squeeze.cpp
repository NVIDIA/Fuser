// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ir/utils.h>
#include <iter_visitor.h>
#include <logical_domain_map.h>
#include <ops/all_ops.h>
#include <preseg_passes/translate_no_reduction_matmul_to_mul_squeeze.h>

#include <vector>

namespace nvfuser::preseg_passes {

namespace {

// Translation algorithm overview:
//
// Step 1: Inspection. Traverses the given fusion and looks for a
// sequence of MatmulOps with K=1.
//
// Step 2: Apply the translation.
class NoReductionMatmulToMulSqueezeTranslator {
 public:
  NoReductionMatmulToMulSqueezeTranslator(Fusion* fusion) : fusion_(fusion) {}

  void run() {
    inspect();
    translate();
  }

 private:
  // Traverse through the fusion and gather all MatmulOps with K=1.
  void inspect() {
    const auto exprs = fusion_->exprs();

    for (auto matmul : ir_utils::filterByType<MatmulOp>(exprs)) {
      // Find the K dimension
      auto in_a = matmul->inA();
      auto in_b = matmul->inB();

      auto domain_a = TensorDomain::noReductions(in_a->getLogicalDomain());
      auto domain_b = TensorDomain::noReductions(in_b->getLogicalDomain());

      auto ndims_a = domain_a.size();
      auto ndims_b = domain_b.size();

      if (ndims_a == 1 && ndims_b == 1) {
        // Dot product. Should this be translated to a reduction as
        // well?
        continue;
      }

      IterDomain* k_id_a = domain_a.back();

      if (!k_id_a->isBroadcast() || k_id_a->hasExpandedExtent()) {
        continue;
      }

      // There should be no case where this extent is not one.
      NVF_ERROR(
          k_id_a->extent()->isOneInt(),
          "Unexpected broadcast dimension: ",
          k_id_a->toString());

      no_reduction_matmul_.push_back(matmul);
    }
  }

  // For each detected MatmulOp, apply the translation of mul+squeeze.
  void translate() {
    for (auto matmul : no_reduction_matmul_) {
      auto in_a = matmul->inA();
      auto in_b = matmul->inB();
      auto dtype = matmul->out()->dtype();

      // Given:
      //
      // out = MatmulOp(in_a, in_b)
      //
      // We broadcast in_a and in_b as necessary and do a pointwise
      // multiplication and squeeze the K dimension. If there's no
      // batch dimensions, either A or B would need to be
      // broadcasted. Batch dimensions make things a little more
      // complicated, but we align the batch dimensions of A and B by
      // following the steps of torch.matmul
      // (https://pytorch.org/docs/stable/generated/torch.matmul.html).

      const auto ndims_a =
          (int64_t)(TensorDomain::noReductions(in_a->getLogicalDomain())
                        .size());
      const auto ndims_b =
          (int64_t)(TensorDomain::noReductions(in_b->getLogicalDomain())
                        .size());
      const auto batch_ndims_a = std::max(ndims_a - 2, (int64_t)0);
      const auto matrix_ndims_a = std::min(ndims_a, (int64_t)2);
      const auto batch_ndims_b = std::max(ndims_b - 2, (int64_t)0);
      const auto matrix_ndims_b = std::min(ndims_b, (int64_t)2);

      // Broadcast flags used to broadcast in_a. The number of false
      // flags should be equal to batch_ndims_a + matrix_ndims_a. The
      // vector is prepended with true if in_a misses some of the
      // batch dimensions.
      std::vector<bool> bc_flags_a;

      // Broadcast flags used to broadcast in_b.
      std::vector<bool> bc_flags_b;

      // Align the batch dimensions
      auto missing_batch_ndims = std::abs(batch_ndims_a - batch_ndims_b);
      if (missing_batch_ndims) {
        if (batch_ndims_a < batch_ndims_b) {
          for ([[maybe_unused]] const auto i : arange(missing_batch_ndims)) {
            bc_flags_a.push_back(true);
          }
        } else {
          for ([[maybe_unused]] const auto i : arange(missing_batch_ndims)) {
            bc_flags_b.push_back(true);
          }
        }
      }

      // Fill the false flags for the existing IDs
      for ([[maybe_unused]] const auto i :
           arange(batch_ndims_a + matrix_ndims_a)) {
        bc_flags_a.push_back(false);
      }

      for ([[maybe_unused]] const auto i :
           arange(batch_ndims_b + matrix_ndims_b)) {
        bc_flags_b.push_back(false);
      }

      // Add true flags if necessary for the matrix dimensions and
      // apply the mul+squeeze pattern. Do this based on three
      // different cases.
      //
      // Case 1: in_a is a 2-dimension matrix and in_b is
      // 1-dimensional (ignoring batch dimensions). In this case, the
      // output is just a matrix-vector multiply, i.e.,
      //
      // [M, b1] * [b1] => [M, b1] * [b2, b1] => [M, b1] => [M]
      //
      // Here, we insert one broadcast ID to in_b and apply the
      // mul-squeeze pattern.
      //
      // Case 2: in_a is 1-dimensional and b is 2-dimensional, i.e.:
      //
      // [b1] * [b1, N] => [b1, b2] * [b1, N] => [b1, N] => [N]
      //
      // Here, we insert one broadcast ID to in_a and apply the
      // mul-squeeze pattern.
      //
      //
      // Case 3: Both in_a and in_b are 2-dimensional. In this case,
      // we insert a broadcast ID to each of the inputs.
      //
      // [M, b1] * [b1, N] => [M, b2, b1] * [b3, N, b1] => [M, N, b1]
      // => [M, N]
      //
      // It may be possible to combine these three cases to reduce
      // repeated code, but having three separated cases seems to make
      // it easier to understand the logic.

      if (matrix_ndims_a == 2 && matrix_ndims_b == 1) {
        // Case 1
        bc_flags_b.push_back(false);
        *(bc_flags_b.rbegin() + 1) = true;
        in_b = broadcast(in_b, bc_flags_b);
        auto out = maybeCastOp(dtype, mul(in_a, in_b));
        std::vector<bool> squeeze_flags(out->nDims(), false);
        squeeze_flags.back() = true;
        IrBuilder::create<SqueezeOp>(matmul->out(), out, squeeze_flags);
      } else if (matrix_ndims_a == 1 && matrix_ndims_b == 2) {
        // Case 2
        bc_flags_a.push_back(true);
        in_a = broadcast(in_a, bc_flags_a);
        auto out = maybeCastOp(dtype, mul(in_a, in_b));
        std::vector<bool> squeeze_flags(out->nDims(), false);
        *(squeeze_flags.rbegin() + 1) = true;
        IrBuilder::create<SqueezeOp>(matmul->out(), out, squeeze_flags);
      } else {
        // Case 3
        bc_flags_a.push_back(false);
        *(bc_flags_a.rbegin() + 1) = true;
        auto in_a_bc = broadcast(in_a, bc_flags_a);

        auto in_b_t = transpose(in_b, -2, -1);
        bc_flags_b.push_back(false);
        *(bc_flags_b.rbegin() + 2) = true;
        auto in_b_bc = broadcast(in_b_t, bc_flags_b);

        auto out = maybeCastOp(dtype, mul(in_a_bc, in_b_bc));
        std::vector<bool> to_squeeze(out->nDims(), false);
        to_squeeze.back() = true;
        IrBuilder::create<SqueezeOp>(matmul->out(), out, to_squeeze);
      }
    }
  }

 private:
  Fusion* fusion_ = nullptr;
  // Map of concat exprs to their info about repetition
  std::vector<MatmulOp*> no_reduction_matmul_;
};

} // namespace

void TranslateNoReductionMatmulToMulSqueeze::runPass(Fusion* fusion) {
  FusionGuard fg(fusion);
  NoReductionMatmulToMulSqueezeTranslator translator(fusion);
  translator.run();
}

} // namespace nvfuser::preseg_passes
