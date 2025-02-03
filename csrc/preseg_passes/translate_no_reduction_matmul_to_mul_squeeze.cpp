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

#include <unordered_map>
#include <vector>

namespace nvfuser::preseg_passes {

namespace {

// Translation algorithm overview:
//
// Step 1: Inspection. Traverses the given fusion and looks for a
// sequence of ops that correspond to a repeatition. See
// RepeatToExpandTranslator::inspect() for more details.
//
// Step 2: Apply the translation in a reverse topologial order. See
// RepeatToExpandTranslator::translate() for more details.
class NoReductionMatmulToMulSqueezeTranslator {
 public:
  NoReductionMatmulToMulSqueezeTranslator(Fusion* fusion) : fusion_(fusion) {}

  void run() {
    inspect();
    translate();
  }

 private:
  // Traverse through the fusion and gather all patterns of a pad
  // followed by a concat. If a single concat op has multiple pad
  // inputs that resize the same iter domain of the same input tensor,
  // that must correspond to a repetition.
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

      std::cerr << "No-reduction matmul detected: " << matmul->toString();
      no_reduction_matmul_.push_back(matmul);
    }
  }

  // For each detected repetition, replace the output with a repeat
  // output.
  void translate() {
    for (auto matmul : no_reduction_matmul_) {
      std::cerr << "Translating: " << matmul->toString();
      auto in_a = matmul->inA();
      auto in_b = matmul->inB();

      auto ndims_a =
          (int64_t)(TensorDomain::noReductions(in_a->getLogicalDomain())
                        .size());
      auto ndims_b =
          (int64_t)(TensorDomain::noReductions(in_b->getLogicalDomain())
                        .size());
      auto batch_ndims_a = std::max(ndims_a - 2, (int64_t)0);
      auto matrix_ndims_a = std::min(ndims_a, (int64_t)2);
      auto batch_ndims_b = std::max(ndims_b - 2, (int64_t)0);
      auto matrix_ndims_b = std::min(ndims_b, (int64_t)2);

      // std::vector<bool> bc_flags_a(ndims_a, false);
      // std::vector<bool> bc_flags_b(ndims_b, false);
      std::vector<bool> bc_flags_a;
      std::vector<bool> bc_flags_b;

      // Align the batch dimensions
      auto missing_batch_ndims = std::abs(batch_ndims_a - batch_ndims_b);
      if (missing_batch_ndims) {
        if (batch_ndims_a < batch_ndims_b) {
          for ([[maybe_unused]] const auto i :
               c10::irange(missing_batch_ndims)) {
            bc_flags_a.push_back(true);
          }
        } else {
          for ([[maybe_unused]] const auto i :
               c10::irange(missing_batch_ndims)) {
            bc_flags_b.push_back(true);
          }
        }
      }

      for ([[maybe_unused]] const auto i :
           c10::irange(batch_ndims_a + matrix_ndims_a)) {
        bc_flags_a.push_back(false);
      }

      for ([[maybe_unused]] const auto i :
           c10::irange(batch_ndims_b + matrix_ndims_b)) {
        bc_flags_b.push_back(false);
      }

      // If a is 2-dimensional and b is 1-dimensional, the output is a
      // mat-vec multiply, i.e.:
      //
      // [M, b1] * [b1] => [M, b1] * [b2, b1] => [M, b1] => [M]
      if (matrix_ndims_a == 2 && matrix_ndims_b == 1) {
        if (std::any_of(bc_flags_a.begin(), bc_flags_a.end(), [](bool flag) {
              return flag;
            })) {
          in_a = broadcast(in_a, bc_flags_a);
        }
        bc_flags_b.push_back(false);
        *(bc_flags_b.rbegin() + 1) = true;
        in_b = broadcast(in_b, bc_flags_b);
        auto out = mul(in_a, in_b);
        std::vector<bool> squeeze_flags(out->nDims(), false);
        squeeze_flags.back() = true;
        IrBuilder::create<SqueezeOp>(matmul->out(), out, squeeze_flags);
      } else if (matrix_ndims_a == 1 && matrix_ndims_b == 2) {
        // a is 1-dimensional and b is 2-dimensional, i.e.:
        //
        // [b1] * [b1, N] => [b1, b2] * [b1, N] => [b1, N] => [N]
        bc_flags_a.push_back(true);
        in_a = broadcast(in_a, bc_flags_a);
        if (std::any_of(bc_flags_b.begin(), bc_flags_b.end(), [](bool flag) {
              return flag;
            })) {
          in_b = broadcast(in_a, bc_flags_b);
        }
        auto out = mul(in_a, in_b);
        std::vector<bool> squeeze_flags(out->nDims(), false);
        *(squeeze_flags.rbegin() + 1) = true;
        IrBuilder::create<SqueezeOp>(matmul->out(), out, squeeze_flags);
        continue;
      } else {
        // [M, b1] * [b1, N] => [M, b2, b1] * [b3, N, b1] => [M, N,
        // b1] => [M, N]

        bc_flags_a.push_back(false);
        *(bc_flags_a.rbegin() + 1) = true;
        std::cerr << "Broadcasting a: " << in_a->toString() << bc_flags_a
                  << "\n";
        auto in_a_bc = broadcast(in_a, bc_flags_a);

        auto in_b_t = transpose(in_b, -2, -1);
        bc_flags_b.push_back(false);
        *(bc_flags_b.rbegin() + 2) = true;
        std::cerr << "Broadcasting b: " << in_b->toString() << bc_flags_b
                  << "\n";
        auto in_b_bc = broadcast(in_b_t, bc_flags_b);

        auto out = mul(in_a_bc, in_b_bc);
        std::vector<bool> to_squeeze(out->nDims(), false);
        to_squeeze.back() = true;
        std::cerr << "Squeezing " << out->toString() << " with "
                  << toDelimitedString(to_squeeze) << "\n";
        auto squeeze =
            IrBuilder::create<SqueezeOp>(matmul->out(), out, to_squeeze);
        std::cerr << "Replaced with: " << squeeze->toString();
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
