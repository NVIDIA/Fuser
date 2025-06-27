// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/translate_repeat_to_expand.h>

#include <ir/utils.h>
#include <logical_domain_map.h>
#include <ops/all_ops.h>

#include <unordered_map>
#include <vector>

namespace nvfuser::preseg_passes {

namespace {

struct RepetitionInfo {
  // Input tensor that is repeated
  TensorView* input_tv = nullptr;
  // Repeated logical ID of the input tensor
  IterDomain* repeated_id = nullptr;
  // Tensors fed into the concat op
  std::vector<TensorView*> cat_inp_tvs;
};

// Translation algorithm overview:
//
// Step 1: Inspection. Traverses the given fusion and looks for a
// sequence of ops that correspond to a repeatition. See
// RepeatToExpandTranslator::inspect() for more details.
//
// Step 2: Apply the translation in a reverse topologial order. See
// RepeatToExpandTranslator::translate() for more details.
class RepeatToExpandTranslator {
 public:
  RepeatToExpandTranslator(Fusion* fusion) : fusion_(fusion) {}

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

    for (auto pad : ir_utils::filterByType<PadOp>(exprs)) {
      auto pad_inp = pad->input(0)->as<TensorView>();
      auto pad_out = pad->output(0)->as<TensorView>();

      // Not supported if there are multiple expanded logical IDs
      IterDomain* out_padded_root_id = nullptr;
      bool multiple_resizes_found = false;
      for (const auto i : arange(pad_out->getLogicalDomain().size())) {
        auto out_logical_id = pad_out->getLogicalDomain().at(i);
        auto resize = dynamic_cast<Resize*>(out_logical_id->definition());
        if (resize == nullptr) {
          continue;
        }
        if (out_padded_root_id != nullptr) {
          // Multiple IDs are resized. Not supported.
          multiple_resizes_found = true;
          break;
        }
        out_padded_root_id = resize->in();
      }

      if (multiple_resizes_found || out_padded_root_id == nullptr) {
        // Unsupported pattern
        break;
      }

      auto inp_padded_id = PairwiseLogicalDomainMap(pad_inp, pad_out)
                               .mapConsumerToProducer()
                               .at(out_padded_root_id);

      // The padded tensor must be immediately used by a concat only
      if (pad_out->uses().size() != 1 || !pad_out->uses().at(0)->isA<CatOp>()) {
        continue;
      }

      auto cat_op = pad_out->uses().at(0);

      // If other inputs to the same concat op are already found, make
      // sure this path from the pad op is compatible with the known
      // ops.
      if (auto it = repeat_info_map_.find(cat_op);
          it == repeat_info_map_.end()) {
        RepetitionInfo info;
        info.input_tv = pad_inp;
        info.repeated_id = inp_padded_id;
        info.cat_inp_tvs.push_back(pad_out);
        repeat_info_map_.emplace(cat_op, info);
      } else {
        auto& info = repeat_info_map_.at(cat_op);
        if (info.input_tv != pad_inp || info.repeated_id != inp_padded_id) {
          // Invalid
          repeat_info_map_.erase(cat_op);
          continue;
        }
        info.cat_inp_tvs.push_back(pad_out);
      }
    }

    // Remove invalid entries
    for (auto it = repeat_info_map_.begin(); it != repeat_info_map_.end();) {
      Expr* concatenating_expr = it->first;
      const RepetitionInfo& info = it->second;
      // Make sure all inputs to concatenating_expr are detected
      if (concatenating_expr->inputs().size() != info.cat_inp_tvs.size()) {
        // Invalid
        it = repeat_info_map_.erase(it);
        continue;
      }
      ++it;
    }
  }

  // For each detected repetition, replace the output with a repeat
  // output.
  void translate() {
    FusionGuard fg(fusion_);

    const auto exprs = fusion_->exprs();
    // Apply the translation in a reverse topological order. Since the
    // output of the repetition is replaced, the use exprs of the
    // output are replaced too, which may invalidate the inspected
    // info invalid.
    for (auto exprs_it = exprs.rbegin(); exprs_it != exprs.rend(); ++exprs_it) {
      Expr* expr = *exprs_it;
      auto repeat_info_map_it = repeat_info_map_.find(expr);
      if (repeat_info_map_it == repeat_info_map_.end()) {
        continue;
      }

      const auto& info = repeat_info_map_it->second;

      const auto num_repetitions = (int64_t)info.cat_inp_tvs.size();

      const auto inp_domain =
          TensorDomain::noReductions(info.input_tv->getLogicalDomain());

      std::vector<int64_t> repeated_times(inp_domain.size(), 1);
      auto repeated_id_it =
          std::find(inp_domain.begin(), inp_domain.end(), info.repeated_id);
      NVF_ERROR(repeated_id_it != inp_domain.end());
      auto repeated_dim = std::distance(inp_domain.begin(), repeated_id_it);
      repeated_times.at(repeated_dim) = num_repetitions;

      TensorView* replacement_tv = repeat(info.input_tv, repeated_times);

      ir_utils::replaceValInAllExprInputsAndFusionOutputs(
          expr->output(0), replacement_tv);
    }
  }

 private:
  Fusion* fusion_ = nullptr;
  // Map of concat exprs to their info about repetition
  std::unordered_map<Expr*, RepetitionInfo> repeat_info_map_;
};

} // namespace

void TranslateRepeatToExpand::runPass(Fusion* fusion) {
  FusionGuard fg(fusion);
  RepeatToExpandTranslator translator(fusion);
  translator.run();
}

} // namespace nvfuser::preseg_passes
