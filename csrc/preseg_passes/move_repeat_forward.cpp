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

std::vector<Val*> getAllPostDominators(TensorView* tv) {
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
    // Skip the first val
    all_use_val_sets.emplace_back(use_chain.begin() + 1, use_chain.end());
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

// Run through a given list of exprs to analyze if the repetition of a
// given iter domain can be moved to after each expr. The analysis is
// currently incomplete. Needs to be expanded to support more ops.
class CanMoveOver : public OptOutConstDispatch {
 public:
  // reshape_output_tv The output tensor of a repeating reshape
  // repeated_logical_id The logical ID of reshape_output_tv that is repeated
  // sorted_exprs Exprs to analyze if the repetition can be moved over
  CanMoveOver(
      TensorView* reshape_output_tv,
      IterDomain* repeated_logical_id,
      const std::vector<Expr*>& sorted_exprs) {
    // repeat_id_map_ keeps track of repeated IDs of all tensors that
    // the repetition can be moved over. Start the analysis by
    // populating the map with the mapping for the reshape output tensor.
    repeat_id_map_.emplace(reshape_output_tv, repeated_logical_id);

    for (auto expr : sorted_exprs) {
      // If no input has info on repeat IDs, this expr should not
      // matter.
      if (std::ranges::none_of(expr->inputs(), [&](Val* inp) {
            return inp->isA<TensorView>() &&
                repeat_id_map_.contains(inp->as<TensorView>());
          })) {
        continue;
      }

      // If any of the op inputs is invalid to move over, it's also
      // invalid to move over this op
      bool is_invalid_to_move_over =
          std::ranges::any_of(expr->inputs(), [&](Val* inp) {
            return inp->isA<TensorView>() &&
                no_move_tvs_.contains(inp->as<TensorView>());
          });

      if (!is_invalid_to_move_over) {
        // Each handler is responsible for populating repeat_id_map_
        // for the outputs if it's valid to move the repeat over this
        // op.
        dispatch(expr);
      }

      // If it has an invalid input or there's no repeat ID for any
      // output tensor is found, mark the outputs as invalid to move
      // over
      if (is_invalid_to_move_over ||
          std::ranges::any_of(expr->outputs(), [&](Val* output) {
            return output->isA<TensorView>() &&
                !repeat_id_map_.contains(output->as<TensorView>());
          })) {
        for (auto out : ir_utils::filterByType<TensorView>(expr->outputs())) {
          no_move_tvs_.insert(out);
        }
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

  // If the op is valid to move the repeat move over, create a new
  // mapping in repeat_id_map. Otherwise, do nothing and just
  // return. It is assumed that none of the inputs are marked as
  // invalid to move over.
  void handleNoSideEffectOp(const Expr* op) {
    if (op->outputs().size() != 1) {
      // Not considered
      return;
    }

    auto out_tv = ir_utils::getTvOutput(op);

    auto inp_tvs = ir_utils::filterByType<TensorView>(op->inputs()).vector();

    // Find the first input tensor that has a mapping
    auto inp_it = std::ranges::find_if(inp_tvs, [&](TensorView* inp) {
      return repeat_id_map_.find(inp) != repeat_id_map_.end();
    });
    // This case should have been taken care before this handler is called
    NVF_ERROR(inp_it != inp_tvs.end());

    TensorView* first_repeat_inp_tv = *inp_it;

    const auto inp_map = PairwiseLogicalDomainMap(first_repeat_inp_tv, out_tv)
                             .mapProducerToConsumer();
    auto inp_map_it = inp_map.find(repeat_id_map_.at(first_repeat_inp_tv));
    NVF_ERROR(
        inp_map_it != inp_map.end(),
        "Cannot find a p2c mapping for ",
        repeat_id_map_.at(first_repeat_inp_tv)->toString());
    IterDomain* out_repeat_id = inp_map_it->second;

    // Check the other inputs. If all of the corresponding IDs of
    // other inputs are also repeated or a broadcast, it should be
    // valid to move.
    for (auto inp_tv : ir_utils::filterByType<TensorView>(op->inputs())) {
      if (inp_tv == first_repeat_inp_tv) {
        continue;
      }

      // Find the corresponding producer ID of out_repeat_id.
      const auto c2p_map = PairwiseLogicalDomainMap(inp_tv, out_tv)
                               .mapBroadcast(true)
                               .mapConsumerToProducer();
      auto c2p_map_it = c2p_map.find(out_repeat_id);
      NVF_ERROR(c2p_map_it != c2p_map.end());
      IterDomain* producer_id = c2p_map_it->second;

      // The only case we can move the repeat over this op is when the
      // correponding producer ID is also a repeated ID or a
      // broadcast. Otherwise, we can't say it's safe to move.
      if (producer_id->isBroadcast()) {
        continue;
      }

      if (auto repeat_id_map_it = repeat_id_map_.find(inp_tv);
          repeat_id_map_it != repeat_id_map_.end() &&
          producer_id == repeat_id_map_it->second) {
        continue;
      }

      // Not proven to be safe to move over this op
      return;
    }

    // Moving is confirmed to be valid
    repeat_id_map_[out_tv] = out_repeat_id;
  }

  // In the case of resize-based ops like slice, as long as the
  // repeated ID is not resized, it should be valid to move the repeat
  void handleResizeBasedOp(const Expr* op) {
    auto inp_tv = op->input(0)->as<TensorView>();
    auto out_tv = op->output(0)->as<TensorView>();
    auto inp_repeat_id = repeat_id_map_.at(inp_tv);

    auto it = std::ranges::find(inp_tv->getLogicalDomain(), inp_repeat_id);
    NVF_ERROR(
        it != inp_tv->getLogicalDomain().end(),
        "Repeat ID not found in logical domain. ",
        inp_tv->toString(),
        ", logical: ",
        toDelimitedString(inp_tv->getLogicalDomain()),
        ", repeat ID: ",
        inp_repeat_id->toString());

    auto out_repeat_id = PairwiseLogicalDomainMap(inp_tv, out_tv)
                             .mapProducerToConsumer()
                             .at(inp_repeat_id);

    // Check if there's a Resize op that takes out_repeat_id as the input
    for (const auto i : arange(out_tv->getLogicalDomain().size())) {
      auto def = out_tv->getLogicalDomain().at(i)->definition();
      NVF_ERROR(def == nullptr || def->isA<Resize>());
      if (def->isA<Resize>() && def->input(0) == out_repeat_id) {
        // This is invalid to move over
        return;
      }
    }

    repeat_id_map_[out_tv] = out_repeat_id;
  }

  void handle(const SliceOp* slice_op) override {
    handleResizeBasedOp(slice_op);
  }

  void handle(const PadOp* pad_op) override {
    handleResizeBasedOp(pad_op);
  }

  // CatOp inputs are already padded, which should be be already
  // analyzed by the PadOp handler. For CatOp, there should be nothing
  // special beyond the checks required for normal arithmetic ops
  void handle(const CatOp* cat_op) override {
    handleNoSideEffectOp(cat_op);
  }

  const std::unordered_map<TensorView*, IterDomain*>& repeatIdMap() const {
    return repeat_id_map_;
  }

 private:
  // Mappings of tensors to their repeated logical IDs
  std::unordered_map<TensorView*, IterDomain*> repeat_id_map_;
  // Tvs that the repeat cannot be moved past
  std::unordered_set<TensorView*> no_move_tvs_;
};

class MoveRepeatForward {
  using StaticRepeatInfo = scheduler_tools::StaticRepeatInfo;

 public:
  MoveRepeatForward(Fusion* fusion) : fusion_(fusion) {}

  void run() {
    auto reshape_ops = ir_utils::getOpsOfType<ViewOp>(fusion_);
    std::vector<TensorView*> reshape_output_tvs;
    reshape_output_tvs.reserve(reshape_ops.size());
    std::ranges::transform(
        reshape_ops,
        std::back_inserter(reshape_output_tvs),
        [](ViewOp* reshape) { return reshape->out(); });

    // For each reshape output, if it's a repeating reshape and a
    // valid move target is found, try moving the repetition after the
    // target
    for (auto reshape_output_tv : reshape_output_tvs) {
      auto static_repeat_info =
          scheduler_tools::getMaybeStaticRepeatInfo(reshape_output_tv);
      if (!static_repeat_info.has_value()) {
        continue;
      }

      auto move_target_info = findMoveTarget(*static_repeat_info);
      if (!move_target_info.has_value()) {
        continue;
      }

      moveTo(
          *static_repeat_info,
          move_target_info->first,
          move_target_info->second);
    }
  }

 private:
  // Try to find the furthest tensor that a given repetition can be
  // moved after. Returns a tensor where the repetition should be
  // moved if found. The map of repeat IDs is also returned as it is
  // necessary for the later move step.
  std::optional<
      std::pair<TensorView*, std::unordered_map<TensorView*, IterDomain*>>>
  findMoveTarget(const StaticRepeatInfo& info) {
    auto reshape_output_tv = info.reshape_output_tv;
    auto all_exprs = DependencyCheck::getAllExprsBetween(
        {reshape_output_tv}, fusion_->outputs());

    // Analyze all exprs after the repeating reshape. The returned map
    // should have mappings if it's valid to move the repeat after the
    // tensor.
    const auto repeat_id_map =
        CanMoveOver(reshape_output_tv, info.output_id, all_exprs).repeatIdMap();

    // To find the furthest possible position, check the furthest post
    // dominator and see if that's a valid position
    const auto post_dominators = getAllPostDominators(reshape_output_tv);

    TensorView* target_tv = nullptr;
    for (const auto& post_dominator : post_dominators | std::views::reverse) {
      NVF_ERROR(
          post_dominator->isA<TensorView>(),
          "Expected to be a tensor: ",
          post_dominator->toString());

      if (auto it = repeat_id_map.find(post_dominator->as<TensorView>());
          it != repeat_id_map.end()) {
        NVF_ERROR(it->second != nullptr);
        target_tv = post_dominator->as<TensorView>();
        break;
      }
    }

    // No valid position found
    if (target_tv == nullptr) {
      return std::nullopt;
    }

    return std::make_pair(target_tv, repeat_id_map);
  }

  // Move the repeat right after target_tv. All tensors between the
  // reshape output tensor and the target tensor are updated by
  // cancelling the repetition, i.e., shrinking the extent of the
  // repeated ID back to the original extent.
  //
  // More concretely, consider a simple case like below (2x repetition):
  //
  // t0: [i0, b1(2)]
  // t1 = reshape(t0); // [i0*2]
  // t2 = op1(t1); // [i0*2]
  // t3 = op2(t2); // [i0*2]
  //
  // Suppose we want to move the repeating reshape after t2. This
  // pattern is going to be transformed to:
  //
  // t0: [i0, b1(2)]
  // t4 = squeeze(t0, {1}); // [i0]
  // t2 = op1(t4); // [i0]
  // t5 = broadcast(t2); // [i0, b2(1)]
  // t6 = expand(t5); // [i0, b2(2)]
  // t7 = reshape(t6); // [i0*2]
  // t3 = op2(t7); // [i0*2]
  //
  // Notice that op1 now operates on an ID with the pre-repeat
  // extent.
  //
  // This transformation involves 1) squeezing the broadcast ID
  // representing the repetition factor; 1) changing the extent of t2 from
  // i0*2 to i0; 2) inserting the new repeat expr sequence after t2;
  void moveTo(
      const StaticRepeatInfo& info,
      TensorView* target_tv,
      const std::unordered_map<TensorView*, IterDomain*>& repeat_id_map) {
    auto target_tv_repeat_id = repeat_id_map.at(target_tv);

    auto target_logical =
        TensorDomain::noReductions(target_tv->getLogicalDomain());
    auto logical_it = std::ranges::find(target_logical, target_tv_repeat_id);
    NVF_ERROR(logical_it != target_logical.end());
    auto target_repeat_id_pos =
        std::distance(target_logical.begin(), logical_it);

    // Squeeze the broadcast of the tensor that was the input
    // to the original repeat pattern
    TensorView* squeeze_out = squeezeRepeatFactorBroadcast(info);

    // Replace the repeated extent with the original
    // pre-repeat extent
    replaceRepeatedExtents(info, target_tv, repeat_id_map);

    // Redirect the use of the original repeated tensor to the
    // squeezed tensor
    ir_utils::replaceValInAllExprInputsAndFusionOutputs(
        info.reshape_output_tv, squeeze_out);

    // At this point, the fusion should look like:
    //
    // t0: [i0, b1(2)]
    // t4 = squeeze(t0, {1}); // [i0]
    // t2 = op1(t4); // [i0]
    // t3 = op2(t7); // [i0*2]
    //
    // This fusion is still inconsistent as it lacks the repeatition
    // of i0 before op2.

    // Insert a new repeat expr sequence after target_tv. In the above
    // example, this corresponds to insertion of t5, t6 and t7
    // following t2.
    auto repeated_tv = repeatTensor(target_tv, target_repeat_id_pos, info);

    // Redirect the use of target_tv to repeated_tv
    ir_utils::replaceValInAllExprInputsAndFusionOutputs(target_tv, repeated_tv);
  }

  void replaceRepeatedExtents(
      const StaticRepeatInfo& info,
      TensorView* target_tv,
      const std::unordered_map<TensorView*, IterDomain*>& repeat_id_map) const {
    std::unordered_map<Val*, Val*> replacement_map;
    for (auto val_to_update : DependencyCheck::getAllValsBetween(
             {info.reshape_output_tv}, {target_tv})) {
      if (val_to_update == info.reshape_output_tv) {
        continue;
      }

      auto tv_to_update = dynamic_cast<TensorView*>(val_to_update);
      NVF_ERROR(
          tv_to_update != nullptr,
          "Unexpected val: ",
          val_to_update->toString());

      IterDomain* repeat_id = repeat_id_map.at(tv_to_update);
      NVF_ERROR(repeat_id != nullptr);

      auto new_id =
          IterDomainBuilder(repeat_id).extent(info.input_id->extent()).build();
      replacement_map.emplace(repeat_id, new_id);
    }
    ir_utils::replaceValue(fusion_, replacement_map);
  }

  TensorView* repeatTensor(
      TensorView* tv,
      int64_t repeat_id_pos,
      const StaticRepeatInfo& info) const {
    const auto logical_domain =
        TensorDomain::noReductions(tv->getLogicalDomain());
    std::vector<int64_t> repeat_times(logical_domain.size(), 1);
    const auto repeat_factor =
        info.factor_id->extent()->evaluate().as<int64_t>();
    repeat_times.at(repeat_id_pos) = repeat_factor;
    return repeat(tv, repeat_times);
  }

  TensorView* squeezeRepeatFactorBroadcast(const StaticRepeatInfo& info) const {
    auto repeat_output_tv = info.reshape_output_tv;
    auto broadcast_it =
        std::ranges::find(repeat_output_tv->getRootDomain(), info.factor_id);
    NVF_ERROR(broadcast_it != repeat_output_tv->getRootDomain().end());
    auto broadcast_pos =
        std::distance(repeat_output_tv->getRootDomain().begin(), broadcast_it);

    auto repeat_input_tv = repeat_output_tv->definition()
                               ->as<ViewOp>()
                               ->input(0)
                               ->as<TensorView>();

    return squeeze(
        repeat_input_tv,
        std::vector<int64_t>{broadcast_pos},
        /*squeeze_expanded=*/true);
  }

 private:
  Fusion* fusion_ = nullptr;
};

} // namespace

void MoveRepeatForwardPass::runPass(Fusion* fusion) {
  FusionGuard fg(fusion);
  MoveRepeatForward(fusion).run();
}

} // namespace nvfuser::preseg_passes
