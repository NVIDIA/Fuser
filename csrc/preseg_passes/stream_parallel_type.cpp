// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <id_model/id_model.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <ir/internal_base_nodes.h>
#include <ir/builder.h>
#include <preseg_passes/stream_parallel_type.h>
#include <ir/utils.h>


namespace nvfuser::preseg_passes {

bool haveDifferentShardings(
    const TensorView* producer,
    const TensorView* consumer,
    const IdModel& id_model) {
  // cpu scalars are not required to have a mesh
  if (producer->isCpuScalar() || consumer->isCpuScalar()) {
    return false;
  }

  // The rest of this function tries to do the following: for each pair of
  // logical-domain-mapped IterDomains (i.e. those mapped by
  // PairwiseLogicalDomainMap), check if they are sharded consistently. If not,
  // returns true. For example,
  //
  //   a: iDIDx{M}, iK
  //   b: iK, iDIDy{N}
  //   c = matmul(a, b): iDIDx{M}, iDIDy{N}
  //
  // haveDifferentShardings(a, c) only cares about iM, which is
  // logical-domain-mapped, but not iK or iN, which are not
  // logical-domain-mapped.
  //
  // One challenge is that DID parallelization doesn't always
  // happen on the root/logical IterDomains. For example, a root/logical
  // IterDomain may be outer-split by the number of devices, and only the outer
  // split gets parallelized on DID.
  //
  //   logical: iM
  //   loop: iDIDx{D}, iM/D
  //
  // Therefore, we collect all the loop IterDomains that depend on the
  // logical-domain-mapped IterDomains, and check if they are DID-parallelized
  // consistently.
  const std::unordered_map<IterDomain*, IterDomain*>& p2c =
      PairwiseLogicalDomainMap(producer, consumer).mapProducerToConsumer();
  std::unordered_set<IterDomain*> mapped_p_logical_ids;
  mapped_p_logical_ids.reserve(p2c.size());
  std::unordered_set<IterDomain*> mapped_c_root_ids;
  mapped_c_root_ids.reserve(p2c.size());
  for (IterDomain* p_logical_id : producer->getLogicalDomain()) {
    const auto i = p2c.find(p_logical_id);
    if (i == p2c.end()) {
      // This happens e.g. when `p_logical_id` is squeezed or is a product of a
      // reduction. Even if `p_logical_id` is parallelized on DID, the
      // dimension is size-1 and doesn't trigger resharding.
      continue;
    }
    mapped_p_logical_ids.insert(p_logical_id);
    mapped_c_root_ids.insert(i->second);
  }

  // In practice, only loop IterDomains can be parallelized, and no two loop
  // IterDomains in a TensorView can have the same parallel type. Therefore, we
  // do the check in reverse order for efficiency and simplicity:
  // 1. For each DID parallel type, find the loop IterDomain in producer and the
  // one in consumer that have the type.
  // 2. Find what IterDomains they come from in producer's logical or
  // consumer's root domain. If that input IterDomain is not
  // logical-domain-mapped, treat the loop IterDomain as not existing -- it is
  // parallelized but just not a concern for this producer-consumer pair.
  // 3. Check if the two loop IterDomains are almost-exactly mapped in the
  // IdModel.
  std::unordered_map<ParallelType, IterDomain*> p_parallel_type_to_id =
      mapDeviceParallelTypeToId(producer->getLoopDomain());
  std::unordered_map<ParallelType, IterDomain*> c_parallel_type_to_id =
      mapDeviceParallelTypeToId(consumer->getLoopDomain());

  for (const auto parallel_type : kParallelTypeDIDs) {
    IterDomain* p_loop_id = getOrDefault(p_parallel_type_to_id, parallel_type);
    if (p_loop_id != nullptr) {
      auto p_inputs =
          getInputsInTargetDomain(p_loop_id, producer->getLogicalDomain());
      if (!overlaps(p_inputs, mapped_p_logical_ids)) {
        p_loop_id = nullptr;
      }
    }

    IterDomain* c_loop_id = getOrDefault(c_parallel_type_to_id, parallel_type);
    if (c_loop_id != nullptr) {
      auto c_inputs =
          getInputsInTargetDomain(c_loop_id, consumer->getMaybeRootDomain());
      if (!overlaps(c_inputs, mapped_c_root_ids)) {
        c_loop_id = nullptr;
      }
    }

    auto is_mapped_in_id_model =
        [](IterDomain* a, IterDomain* b, const IdModel& id_model) -> bool {
      if (a == nullptr && b == nullptr) {
        return true;
      }

      if (a == nullptr || b == nullptr) {
        return false;
      }

      // Going between bDIDx{1} and iDIDx{N} doesn't trigger resharding, but
      // would be flagged by ALMOSTEXACT as a false positive.
      if (id_model.idGraph(IdMappingMode::BROADCAST)
              .disjointValSets()
              .strictAreMapped(a, b)) {
        return true;
      }

      // Check ALMOSTEXACT so iDIDx{N}*b{1} and iDIDx{N} are mapped.
      return id_model.idGraph(IdMappingMode::ALMOSTEXACT)
          .disjointValSets()
          .strictAreMapped(a, b);
    };

    if (!is_mapped_in_id_model(p_loop_id, c_loop_id, id_model)) {
      return true;
    }
  }

  return false;
}

class StreamParallelTypeHelper {
 public:
  StreamParallelTypeHelper(Fusion* fusion): container_(dynamic_cast<hir::HostIrContainer*>(fusion)), id_model_(fusion) {
    NVF_CHECK(host_ir_container, "Expected HostIrContainer");
    id_model.buildAlmostExactGraph();
    id_model.buildBroadcastGraph();
  }

  void runPass() {
    for (auto* expr : fusion_->topLevelExprs()) {
      if (isChangingStreamParallelType(expr)) {
        segments_.push_back({expr});
      } else {
        segments_.back().push_back(expr);
      }
    }
  }

 private:
  void findSegments () {

  }

  bool isChangingStreamParallelType(Expr* expr) {
    if (!ir_utils::isTvOp(expr)) {
      return false;
    }

    for (auto* input : ir_utils::filterByType<TensorView>(expr->inputs())) {
      for (auto* output : ir_utils::filterByType<TensorView>(expr->outputs())) {
        // exit early in the unsharded case for performance
        if (haveDifferentShardings(input, output, id_model)) {
          return true;
        }
      }
    }

    return false;
  }

 private:
  hir::HostIrContainer* container_;
  std::vector<std::vector<Expr*>> segments_;
  IdModel id_model_;
}

// returns the first stream axis in the domain, or nullptr if there is none. Throws if two axis are stream parallelized
IterDomain* getStreamAxisIndex(const std::vector<IterDomain*>& domain) {
  IterDomain* ret = nullptr
  for (auto id : domain) {
    if (id->getParallelType() == ParallelType::Stream) {
      NVF_CHECK(ret == nullptr, "Expected at most one stream axis in the domain, but found ", id, " and ", ret);
      ret = id;
    }
  }
  return ret;
}


// TODO: ideally we should look at the dag and use the segmenter. Here we take advantage of the linear structure of HostIrContainer::topLevelExprs to greedily merge the adjacent compatible stream for-loop bodies
void StreamParallelType::runPass(Fusion* fusion) {
  // check that there are no stream axes in the inputs
  NVF_CHECK(std::all_of(fusion->inputs().begin(), fusion->inputs().end(), [](Val* input) {
    auto input_tv = dynamic_cast<TensorView*>(input);
    return input_tv == nullptr || getStreamAxis(input_tv->getLoopDomain()) == nullptr;
  }), "Expected no stream axis in the TensorView inputs.");

  FusionGuard fg(fusion); // set as current container to register the newly created for-loops
  hir::HostIrContainer* container = dynamic_cast<hir::HostIrContainer*>(fusion);
  NVF_CHECK(container, "Expected HostIrContainer");
  // needed ?
  IdModel id_model_;
  id_model.buildAlmostExactGraph();

  const std::vector<Expr*>& exprs = ;
  auto new_top_level_exprs;
  // Step 1. Find the segments of expressions that can be merged into a single stream for-loop
  // At the end of this step, new_top_level_exprs contains a list of expressions including newly created for-loops that will represent the stream parallelization, and the relevant expressions grouped inside the for-loops bodies.
  for (auto expr : container->topLevelExprs()) {
    // we only support exprs having 0 or 1 output for now
    if (expr->outputs().size() == 0) {
      // If the expr has no output, we try to merge it with the previous for loop
      Expr* previous_expr = new_top_level_exprs.back();
      if (previous_expr->isA<ForLoop>()) {
        previous_expr->as<ForLoop>()->body().push_back(expr);
      } else {
        new_top_level_exprs.push_back(expr);
      }
    }
    NVF_CHECK(expr->outputs().size() == 1, "Each expr should have at most one output.");
    TensorView* output = expr->outputs().at(0)->as<TensorView>();
    // retrieves the Loop IterDomain that is stream parallelized, if any
    IterDomain* stream_axis = getStreamAxis(output->getLoopDomain());
    if (stream_axis == nullptr) {
      // if the consumer is not stream parallelized, it means the expr need not be inside a stream for-loop
      new_top_level_exprs.push_back(expr);
      continue;
    }
    // find the corresponding stream axis but in the Logical (and not Loop Domain)
    auto it_logical_stream_axis = std::find(output->getLogicalDomain().begin(), output->getLogicalDomain().end(), stream_axis);
    // for now we do not support split/merge stream axis
    NVF_ERROR(
      it_logical_stream_axis !=
          output->getLogicalDomain().end(),
      "Cannot stream parallelize on a split/merge axis ",
      stream_axis);
    // we don't support reducing or broadcasting a stream axis
    NVF_CHECK(stream_axis->getIterType() == IterType::Iteration, "Stream axis ", stream_axis, " should be an iteration axis.");
    // we don't support stream axis in the inputs nor the first expression
    NVF_CHECK(new_top_level_exprs.empty() == false, "Expected the first expr to not have a stream axis.");
    // We consider the previous expression to check whether the expr should create a new stream for-loop or be integrated into the previous one
    auto prev_expr = new_top_level_exprs.back();
    // check if the current expr can be merged with the previous stream for-loop
    if (auto prev_for_loop = dynamic_cast<ForLoop*>(prev_expr); prev_for_loop && id_model.idGraph(IdMappingMode::ALMOSTEXACT).disjointValSets().strictAreMapped(stream_axis, prev_for_loop->id())) {
      // merge with previous for-loop
      prev_for_loop->body().push_back(expr);
    } else {
      // create a new for-loop
      auto* j =
          IrBuilder::create<Val>(DataType::Index); // running index of the for-loop
      auto* start = hic->zeroVal();
      auto* stop = stream_axis->extent();
      auto* step = hic->oneVal();
      auto* for_loop = IrBuilder::create<ForLoop>(
          stream_axis,
          /*index=*/j,
          start,
          stop,
          step,
          /*vectorize=*/false,
          /*vectorize_shift=*/nullptr,
          /*unroll_required=*/false,
          CircularBufferLoopStage::NotApplicable,
          /*circular_buffer_loop_stage_depth=*/0);
      for_loop->body().push_back(expr);
      // replace the current expr by the for-loop containing it
      new_top_level_exprs.push_back(for_loop);
    }
  }

  // Step 2. Setup each for loop's body
  for (auto* top_level_expr : new_top_level_exprs) {
    if (!top_level_expr->isA<ForLoop>()) {
      continue;
    }
    auto* for_loop = top_level_expr->as<ForLoop>();
    // this will contain the new body of the current for-loop
    std::vector<Expr*> new_loop_body;

    auto* j = for_loop->index();
    auto* number_of_streams =
        IrBuilder::create<NamedScalar>("numberOfStreams", DataType::Int);
    auto* stream_index = mod(j, number_of_streams);
    auto* stream = IrBuilder::create<hir::Stream>(stream_index);
    auto* set_stream = IrBuilder::create<hir::SetCurrentStream>(stream);
    auto* initial_sync_stream =
        IrBuilder::create<hir::Synchronize>(original_stream);

    new_loop_body.push_back(set_stream);
    new_loop_body.push_back(initial_sync_stream);

    std::vector<TensorView*> tvs;
    for (auto expr: for_loop->body()) {
      for (auto* input : ir_utils::filterByType<TensorView>(expr->inputs())) {
        tvs.push_back(input);
      }
      for (auto* output : ir_utils::filterByType<TensorView>(expr->outputs())) {
        tvs.push_back(output);
      }
    }


    Expr* current_expr = expr;
    for (auto input: ir_utils::filterByType<TensorView>(expr->inputs())) {
      int64_t input_stream_id_logical_index = -1;
      for (auto id : input->getLoopDomain()) {
        if (id_model.idGraph(IdMappingMode::ALMOSTEXACT).disjointValSets().strictAreMapped(stream_axis, id)) {
          NVF_CHECK(input_stream_id_logical_index == -1, "Expected at most one axis mapping to the stream axis ", stream_axis, " in the tensor ", input, " loop's domain ", expr->getLoopDomain());
          auto it2 = std::find(input->getLogicalDomain().begin(), input->getLogicalDomain().end(), id);
          NVF_CHECK(it2 != input->getLogicalDomain().end(), "Expected to find ", id, " in ", input, "'s logical domain ", input->getLogicalDomain());
          input_stream_id_logical_index = std::distance(input->getLogicalDomain().begin(), it2);
        }
      if (input_stream_id_logical_index == -1) {
        continue;
      }
      TensorView* input_j = select(input, input_stream_id_logical_index, j);
      loop_body.push_back(input_j->definition());
      current_expr = ir_utils::replaceValInExprInputs(current_expr, input, input_j);
      }
    }

    int64_t output_stream_id_logical_index = -1;

    auto it2 = std::find(input->getLogicalDomain().begin(), input->getLogicalDomain().end(), id);
    NVF_CHECK(it2 != input->getLogicalDomain().end(), "Expected to find ", id, " in ", input, "'s logical domain ", input->getLogicalDomain());
    index = std::distance(input->getLogicalDomain().begin(), it2);

    if (index == -1) {
      continue;
    }
    TensorView* input_j = select(input, index, j);
    loop_body.push_back(input_j->definition());
    current_expr = ir_utils::replaceValInExprInputs(current_expr, input, input_j);



    auto* set_back_original_stream =
        IrBuilder::create<hir::SetCurrentStream>(original_stream);
    auto* sync_stream = IrBuilder::create<hir::Synchronize>(stream);

    new_loop_body.push_back(set_back_original_stream);
    new_loop_body.push_back(sync_stream);
  }
}

} // namespace nvfuser::preseg_passes
