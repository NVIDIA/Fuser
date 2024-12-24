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
  TensorView* input_tv = nullptr;
  IterDomain* repeated_id = nullptr;
  std::vector<TensorView*> cat_inp_tvs;
  TensorView* output_tv = nullptr;
};

class RepeatToExpandTranslator {
 public:
  RepeatToExpandTranslator(Fusion* fusion) : fusion_(fusion) {}

  void run() {
    inspect();
    translate();
  }

  // TODO: Consider translating the addition-based concat to an actual
  // CatOp. By doing so, this pass would just need to find concat ops.
  void inspect() {
    const auto exprs = fusion_->exprs();

    auto get_cat_inp = [](TensorView* tv) -> TensorView* {
      if (tv->uses().size() != 1) {
        return nullptr;
      }

      // Skip cast
      if (auto uop = dynamic_cast<UnaryOp*>(tv->uses().at(0));
          uop != nullptr && uop->getUnaryOpType() == UnaryOpType::Cast) {
        tv = uop->output(0)->as<TensorView>();

        if (tv->uses().size() != 1) {
          return nullptr;
        }
      }

      if (tv->uses().size() != 1) {
        return nullptr;
      }

      auto use_expr = tv->uses().at(0);
      if (use_expr->isA<CatOp>() ||
          (use_expr->isA<BinaryOp>() &&
           use_expr->as<BinaryOp>()->getBinaryOpType() == BinaryOpType::Add)) {
        return tv;
      } else {
        return nullptr;
      }
    };

    for (auto pad : ir_utils::filterByType<PadOp>(exprs)) {
      auto repeat_inp = pad->input(0)->as<TensorView>();
      auto pad_out = pad->output(0)->as<TensorView>();

      // There must be just one logical ID expanded by this pad op
      IterDomain* out_padded_root_id = nullptr;
      bool multiple_resizes_found = false;
      for (const auto i : c10::irange(pad_out->getLogicalDomain().size())) {
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

      if (multiple_resizes_found) {
        break;
      }

      auto inp_padded_id = PairwiseLogicalDomainMap(repeat_inp, pad_out)
                               .mapConsumerToProducer()
                               .at(out_padded_root_id);

      auto cat_inp = get_cat_inp(pad_out);
      if (cat_inp == nullptr) {
        continue;
      }

      // Note that this can be a CatOp or an addition
      auto cat_op = cat_inp->uses().at(0);

      if (auto it = repeat_info_map.find(cat_op); it == repeat_info_map.end()) {
        RepetitionInfo info;
        info.input_tv = repeat_inp;
        info.repeated_id = inp_padded_id;
        info.cat_inp_tvs.push_back(cat_inp);
        repeat_info_map.emplace(cat_op, info);
      } else {
        auto& info = repeat_info_map.at(cat_op);
        if (info.input_tv != repeat_inp || info.repeated_id != inp_padded_id) {
          // Invalid
          repeat_info_map.erase(cat_op);
          continue;
        }
        info.cat_inp_tvs.push_back(cat_inp);
      }
    }

    // Remove invalid entries
    for (auto it = repeat_info_map.begin(); it != repeat_info_map.end();) {
      Expr* concatenating_expr = it->first;
      const RepetitionInfo& info = it->second;
      // Make sure all inputs to concatenating_expr are detected
      if (concatenating_expr->inputs().size() != info.cat_inp_tvs.size()) {
        // Invalid
        it = repeat_info_map.erase(it);
        continue;
      }
      ++it;
    }
  }

  void translate() {
    const auto exprs = fusion_->exprs();
    // Apply the translation in a reverse topological order. Since the
    // output of the repetition is replaced, the use exprs of the
    // output are replaced too, which may invalidate the inspected
    // info invalid.
    for (auto exprs_it = exprs.rbegin(); exprs_it != exprs.rend(); ++exprs_it) {
      Expr* expr = *exprs_it;
      auto repeat_info_map_it = repeat_info_map.find(expr);
      if (repeat_info_map_it == repeat_info_map.end()) {
        continue;
      }

      const auto& info = repeat_info_map_it->second;

      if (info.cat_inp_tvs.size() < 2) {
        continue;
      }

      // Step 1. Insert a broadcast ID immediately outside of the
      // repeated ID
      // Step 2. Expand the broadcast ID by the repetition factor
      // Step 3. Flatten the expanded ID and the repeated ID
      // Step 4. Cast the flattened tensor if necessary. If the
      // concatenation is done by addition and the inputs are fp16,
      // there must be casting to fp32 before the addition.

      // Step 1
      auto inp_domain =
          TensorDomain::noReductions(info.input_tv->getLogicalDomain());
      std::vector<bool> bcast_flags(inp_domain.size() + 1, false);
      auto repeated_id_offset = std::distance(
          inp_domain.begin(),
          std::find(inp_domain.begin(), inp_domain.end(), info.repeated_id));
      bcast_flags.at(repeated_id_offset) = true;
      auto broadcast_tv = broadcast(info.input_tv, bcast_flags);
      NVF_ERROR(broadcast_tv->nDims() == inp_domain.size() + 1);

      // Step 2
      std::vector<Val*> expanded_sizes(
          bcast_flags.size(), IrBuilder::create<Val>(-1L));
      expanded_sizes.at(repeated_id_offset) = IrBuilder::create<Val>(2L);
      auto expanded_tv = expand(broadcast_tv, expanded_sizes);

      // Step 3
      auto flattened_tv =
          flatten(expanded_tv, repeated_id_offset, repeated_id_offset + 1);

      // Step 4
      TensorView* new_out_tv = nullptr;
      auto origin_out_tv = expr->output(0)->as<TensorView>();

      if (info.input_tv->dtype() != origin_out_tv->dtype()) {
        // Input should be either Half or BFloat16
        NVF_ERROR(
            info.input_tv->dtype() == DataType::Half ||
                info.input_tv->dtype() == DataType::BFloat16,
            "Unexpected input type: ",
            info.input_tv->toString());
        // Output should be either Float
        NVF_ERROR(
            origin_out_tv->dtype() == DataType::Float,
            "Unexpected output type: ",
            origin_out_tv->toString());
        new_out_tv = castOp(DataType::Float, flattened_tv);
      } else {
        new_out_tv = flattened_tv;
      }

      ir_utils::replaceValInAllExprInputsAndFusionOutputs(
          origin_out_tv, new_out_tv);
    }
  }

 private:
  Fusion* fusion_ = nullptr;
  // Map of concatenating expr to its infoi
  std::unordered_map<Expr*, RepetitionInfo> repeat_info_map;
};

} // namespace

void TranslateRepeatToExpand::runPass(Fusion* fusion) {
  FusionGuard fg(fusion);
  RepeatToExpandTranslator translator(fusion);
  translator.run();
}

} // namespace nvfuser::preseg_passes
