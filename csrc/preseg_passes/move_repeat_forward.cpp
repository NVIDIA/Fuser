// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/move_repeat_forward.h>

#include <device_lower/utils.h>
#include <dispatch.h>
#include <ir/utils.h>
#include <logical_domain_map.h>
#include <ops/all_ops.h>
#include <scheduler/tools/static_repeat.h>

#include <unordered_map>
#include <vector>

namespace nvfuser::preseg_passes {

namespace {

class CanMovePast : public OptOutConstDispatch {
 public:
  CanMovePast(
      std::unordered_map<TensorView*, IterDomain*>& repeat_id_map,
      Expr* expr)
      : repeat_id_map_(repeat_id_map) {
    std::cerr << "CanMovePast: " << expr->toString();

    // If no input has info on repeat IDs, this expr should not matter
    if (std::ranges::none_of(expr->inputs(), [&](Val* inp) {
          return inp->isA<TensorView>() &&
              repeat_id_map_.contains(inp->as<TensorView>());
        })) {
      std::cerr << "Not possible\n";
      return;
    }

    can_move_ = false;
    dispatch(expr);
    std::cerr << "Can move: " << can_move_ << "\n";
    if (!can_move_) {
      for (auto out : ir_utils::filterByType<TensorView>(expr->outputs())) {
        repeat_id_map[out] = nullptr;
      }
    }
  }

  void handle(const UnaryOp* uop) override {
    handleNoSideEffectOp(uop);
  }

  void handle(const BinaryOp* bop) override {
    handleNoSideEffectOp(bop);
  }

  void handle(const TernaryOp* top) override {
    handleNoSideEffectOp(top);
  }

  void handleNoSideEffectOp(const Expr* op) {
    if (op->outputs().size() != 1) {
      // Not considered
      return;
    }

    auto out_tv = ir_utils::getTvOutput(op);

    auto inp_tvs = ir_utils::filterByType<TensorView>(op->inputs()).vector();

    auto inp_it = std::ranges::find_if(inp_tvs, [&](TensorView* inp) {
      return repeat_id_map_.find(inp) != repeat_id_map_.end();
    });
    NVF_ERROR(inp_it != inp_tvs.end());

    TensorView* first_repeat_inp_tv = *inp_it;
    auto first_repeat_inp_id = repeat_id_map_.at(first_repeat_inp_tv);
    if (first_repeat_inp_id == nullptr) {
      return;
    }

    const auto inp_map = PairwiseLogicalDomainMap(first_repeat_inp_tv, out_tv)
                             .mapProducerToConsumer();
    auto inp_map_it = inp_map.find(repeat_id_map_.at(first_repeat_inp_tv));
    NVF_ERROR(
        inp_map_it != inp_map.end(),
        "Cannot find a p2c mapping for ",
        repeat_id_map_.at(first_repeat_inp_tv)->toString());
    IterDomain* out_repeat_id = inp_map_it->second;

    for (auto inp_tv : ir_utils::filterByType<TensorView>(op->inputs())) {
      if (inp_tv == first_repeat_inp_tv) {
        continue;
      }

      const auto c2p_map = PairwiseLogicalDomainMap(inp_tv, out_tv)
                               .mapBroadcast(true)
                               .mapConsumerToProducer();
      auto c2p_map_it = c2p_map.find(out_repeat_id);
      NVF_ERROR(c2p_map_it != c2p_map.end());
      IterDomain* producer_id = c2p_map_it->second;

      if (auto repeat_id_map_it = repeat_id_map_.find(inp_tv);
          repeat_id_map_it != repeat_id_map_.end()) {
        if (repeat_id_map_it->second == nullptr) {
          // This input is already marked as invalid to cross
          return;
        }
        // This input has also a repeat ID.
        if (producer_id != repeat_id_map_it->second) {
          return;
        }
      } else {
        // Only possible to move when the other inputs are
        // broadcast since this argument should have the
        // expanded extent otherwise.
        if (!producer_id->isBroadcast()) {
          return;
        }
      }
    }

    repeat_id_map_[out_tv] = out_repeat_id;

    can_move_ = true;
  }

  void handleResizeBasedOp(const Expr* op) {
    std::cerr << "handleResizeBasedOp: " << op->toString();
    auto inp_tv = op->input(0)->as<TensorView>();
    auto out_tv = op->output(0)->as<TensorView>();
    auto inp_repeat_id = repeat_id_map_.at(inp_tv);
    if (inp_repeat_id == nullptr) {
      std::cerr << "Input tensor has a null repeat ID: " << inp_tv->toString()
                << "\n";
      return;
    }

    auto it = std::ranges::find(inp_tv->getLogicalDomain(), inp_repeat_id);
    NVF_ERROR(
        it != inp_tv->getLogicalDomain().end(),
        "Repeat ID not found in logical domain. ",
        inp_tv->toString(),
        ", logical: ",
        toDelimitedString(inp_tv->getLogicalDomain()),
        ", repeat ID: ",
        inp_repeat_id->toString());

    auto repeat_pos = std::distance(inp_tv->getLogicalDomain().begin(), it);

    std::cerr << "Repeat pos: " << repeat_pos << "\n";

    for (const auto i : arange(out_tv->getLogicalDomain().size())) {
      auto def = out_tv->getLogicalDomain().at(i)->definition();
      NVF_ERROR(def == nullptr || def->isA<Resize>());
      if (def->isA<Resize>() && i == repeat_pos) {
        return;
      }
    }

    repeat_id_map_[out_tv] = out_tv->getLogicalDomain().at(repeat_pos);

    can_move_ = true;
  }

  void handle(const SliceOp* slice_op) override {
    handleResizeBasedOp(slice_op);
  }

  void handle(const PadOp* pad_op) override {
    handleResizeBasedOp(pad_op);
  }

  void handle(const CatOp* cat_op) override {
    int64_t common_repeat_id_pos = -1;
    for (auto inp : cat_op->inputs()) {
      auto inp_tv = dynamic_cast<TensorView*>(inp);
      NVF_ERROR(inp_tv != nullptr);
      auto repeat_id_map_it = repeat_id_map_.find(inp_tv);
      if (repeat_id_map_it == repeat_id_map_.end()) {
        continue;
      }

      auto repeat_id = repeat_id_map_it->second;
      if (repeat_id == nullptr) {
        return;
      }

      auto logical = TensorDomain::noReductions(inp_tv->getLogicalDomain());
      auto repeat_id_pos = std::distance(
          logical.begin(),
          std::ranges::find(logical, repeat_id_map_it->second));
      if (repeat_id_pos == cat_op->concatenatedDim()) {
        return;
      }

      if (common_repeat_id_pos == -1) {
        common_repeat_id_pos = repeat_id_pos;
      } else if (common_repeat_id_pos != repeat_id_pos) {
        // Position must be uniform
        return;
      }
    }

    auto out_tv = cat_op->output(0)->as<TensorView>();
    repeat_id_map_[out_tv] =
        out_tv->getLogicalDomain().at(common_repeat_id_pos);

    can_move_ = true;
  }

 private:
  // This map needs to be cleaned up. Currently, if it has an ID
  // entry, it means that's the corresponding repeat ID of the
  // tensor. If it's nullptr, the tensor is no longer possible to move
  // over. If no entry exists, the tensor doesn't matter as it has no
  // dependency. Use a different data structure
  std::unordered_map<TensorView*, IterDomain*>& repeat_id_map_;
  bool can_move_ = false;
};

class MoveRepeatForward {
  using StaticRepeatInfo = scheduler_tools::StaticRepeatInfo;

 public:
  MoveRepeatForward(Fusion* fusion) : fusion_(fusion) {}

  void run() {
    // TODO: Repeat until done
    while (true) {
      auto info = getMaybeStaticRepeat();
      if (!info.has_value()) {
        return;
      }

      std::cerr << "Static repeat detected: "
                << info->repeat_input_tv->toString() << "\n";

      excluded_tvs_.insert(info->repeat_output_tv);

      auto [target_tv, repeat_id, repeat_id_map] = findMoveTarget(*info);
      if (target_tv != nullptr) {
        moveTo(*info, target_tv, repeat_id, repeat_id_map);
      } else {
        std::cerr << "Move target not found\n";
      }
    }
  }

 private:
  std::optional<scheduler_tools::StaticRepeatInfo> getMaybeStaticRepeat() {
    auto getStraightUseExprs = [](Val* inp,
                                  int num_exprs) -> std::vector<Expr*> {
      std::vector<Expr*> exprs;
      exprs.reserve(num_exprs);

      Val* cur_val = inp;
      for (auto i : arange(num_exprs)) {
        (void)i;
        if (cur_val->uses().size() != 1) {
          return {};
        }
        auto use_expr = cur_val->uses().at(0);
        if (use_expr->outputs().size() != 1) {
          return {};
        }
        exprs.push_back(use_expr);
        cur_val = use_expr->output(0);
      }

      return exprs;
    };

    scheduler_tools::StaticRepeatInfo info;

    for (auto tv : fusion_->allTvs()) {
      // Quick filtering before using getMaybeStaticRepeatInfo
      auto use_exprs = getStraightUseExprs(tv, 3);
      if (use_exprs.empty()) {
        continue;
      }

      if (!use_exprs.at(0)->isA<BroadcastOp>() ||
          !use_exprs.at(1)->isA<ExpandOp>() ||
          !use_exprs.at(2)->isA<ViewOp>()) {
        continue;
      }

      auto repeat_out = use_exprs.at(2)->output(0)->as<TensorView>();
      if (excluded_tvs_.contains(repeat_out)) {
        continue;
      }

      auto static_repeat_info =
          scheduler_tools::getMaybeStaticRepeatInfo(repeat_out);
      if (!static_repeat_info.has_value()) {
        continue;
      }

      return *static_repeat_info;
    }

    return std::nullopt;
  }

  // Reconsider returning the unordered_map
  std::tuple<
      TensorView*,
      IterDomain*,
      std::unordered_map<TensorView*, IterDomain*>>
  findMoveTarget(const scheduler_tools::StaticRepeatInfo& info) {
    Fusion* fusion = info.repeat_output_tv->fusion();
    auto all_exprs = DependencyCheck::getAllExprsBetween(
        {info.repeat_output_tv}, fusion->outputs());

    std::unordered_map<TensorView*, IterDomain*> repeat_id_map;

    IterDomain* repeat_out_id = nullptr;
    for (auto logical_id : info.reshape_output_tv->getLogicalDomain()) {
      auto merge = dynamic_cast<Merge*>(logical_id->definition());
      if (merge == nullptr) {
        continue;
      }
      if (merge->inner() == info.reshape_repeat_id ||
          merge->outer() == info.reshape_repeat_id) {
        repeat_out_id = logical_id;
        break;
      }
    }

    repeat_id_map.emplace(info.reshape_output_tv, repeat_out_id);

    for (auto expr : all_exprs) {
      CanMovePast(repeat_id_map, expr);
    }

    for (auto [tv, id] : repeat_id_map) {
      std::cerr << "Repeat tv: " << tv->toString()
                << ", id: " << (id == nullptr ? "<null>" : id->toString())
                << "\n";
    }

    const auto post_dominators = getAllPostDominators(info.reshape_output_tv);

    std::cerr << "Post dominators: " << toDelimitedString(post_dominators)
              << "\n";

    TensorView* target_tv = nullptr;
    IterDomain* repeat_id = nullptr;
    for (const auto& post_dominator : post_dominators | std::views::reverse) {
      NVF_ERROR(
          post_dominator->isA<TensorView>(),
          "Expected to be a tensor: ",
          post_dominator->toString());

      if (auto it = repeat_id_map.find(post_dominator->as<TensorView>());
          it != repeat_id_map.end() && it->second != nullptr) {
        target_tv = post_dominator->as<TensorView>();
        repeat_id = it->second;
        break;
      }
    }

    if (target_tv == nullptr) {
      return {nullptr, nullptr, {}};
    }

    std::cerr << "Target found: " << target_tv->toString() << ", "
              << repeat_id->toString() << "\n";

    return {target_tv, repeat_id, repeat_id_map};
  }

  std::vector<Val*> getAllPostDominators(TensorView* tv) const {
    std::deque<std::deque<Val*>> all_use_chains =
        DependencyCheck::getAllUseChains(tv);

    if (all_use_chains.empty()) {
      return {};
    }

    if (all_use_chains.size() == 1) {
      // Skip the first val as it's the given tensor
      NVF_ERROR_EQ(all_use_chains.at(0).at(0), tv);
      return {all_use_chains.at(0).begin() + 1, all_use_chains.at(0).end()};
    }

    std::vector<std::unordered_set<Val*>> all_use_val_sets;
    all_use_val_sets.reserve(all_use_chains.size());

    for (const auto& use_chain : all_use_chains) {
      all_use_val_sets.emplace_back(use_chain.begin(), use_chain.end());
    }

    const auto& use_chain = all_use_chains.at(0);
    std::vector<Val*> post_dominators;
    for (const auto& val : use_chain) {
      if (std::ranges::all_of(arange(all_use_chains.size()), [&](int i) {
            return all_use_val_sets.at(i).contains(val);
          })) {
        post_dominators.push_back(val);
      }
    }

    return post_dominators;
  }

  void moveTo(
      const StaticRepeatInfo& info,
      TensorView* target_tv,
      IterDomain* repeat_id,
      const std::unordered_map<TensorView*, IterDomain*>& repeat_id_map) {
    std::cerr << "Move repeat to " << target_tv->toString() << "\n";
    // Setting up a repeat op
    auto repeat_factor =
        info.reshape_repeat_id->extent()->evaluate().as<int64_t>();
    auto target_logical =
        TensorDomain::noReductions(target_tv->getLogicalDomain());
    auto logical_it = std::ranges::find(target_logical, repeat_id);
    NVF_ERROR(logical_it != target_logical.end());

    auto id_pos = std::distance(target_logical.begin(), logical_it);

    std::vector<int64_t> repeat_times(target_logical.size(), 1);
    repeat_times.at(id_pos) = repeat_factor;

    // Replace the repeated extent with the original pre-repeat extent
    std::unordered_map<Val*, Val*> replacement_map;
    for (auto val_to_update : DependencyCheck::getAllValsBetween(
             {info.reshape_output_tv}, {target_tv})) {
      if (val_to_update == info.reshape_output_tv) {
        continue;
      }

      std::cerr << "Updating " << val_to_update->toString() << "\n";

      auto tv_to_update = dynamic_cast<TensorView*>(val_to_update);
      NVF_ERROR(
          tv_to_update != nullptr,
          "Unexpected val: ",
          val_to_update->toString());

      IterDomain* repeat_id = repeat_id_map.at(tv_to_update);
      NVF_ERROR(repeat_id != nullptr);
      auto new_id = IterDomainBuilder(repeat_id)
                        .extent(info.repeat_input_id->extent())
                        .build();
      std::cerr << "Replacing " << repeat_id->toString() << " with "
                << new_id->toString() << "\n";
      replacement_map.emplace(repeat_id, new_id);
    }

    ir_utils::replaceValue(fusion_, replacement_map);

    std::cout << "Extent replaced\n";
    fusion_->printMath();
    std::cout << std::endl;

    // Insert a new repeat expr sequence after target_tv
    auto repeated_tv = repeat(target_tv, repeat_times);

    excluded_tvs_.insert(repeated_tv);

    ir_utils::replaceValInAllExprInputsAndFusionOutputs(target_tv, repeated_tv);

    // Remove the original repeat exprs
    ir_utils::replaceValInAllExprInputsAndFusionOutputs(
        info.repeat_output_tv, info.repeat_input_tv);
    std::cout << std::endl;
    std::cerr << "Moved\n";
    fusion_->printMath();
    std::cout << std::endl;
  }

 private:
  Fusion* fusion_ = nullptr;
  std::unordered_set<TensorView*> excluded_tvs_;
};

} // namespace

void MoveRepeatForwardPass::runPass(Fusion* fusion) {
  FusionGuard fg(fusion);
  MoveRepeatForward(fusion).run();
}

} // namespace nvfuser::preseg_passes
