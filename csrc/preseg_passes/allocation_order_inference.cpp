// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ir/all_nodes.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <preseg_passes/allocation_order_inference.h>
#include <root_domain_map.h>

namespace nvfuser::preseg_passes {

namespace {

// performs permutation by `alloc_order` on `tv`'s rfactor_domain.
std::vector<IterDomain*> constructAllocationDomain(
    TensorView* tv,
    const AllocationOrder& alloc_order) {
  auto rfactor_dom = tv->getMaybeRFactorDomain();
  auto rank = rfactor_dom.size();

  std::vector<IterDomain*> allocation_domain(rank, nullptr);
  // specify allocation domain with dimension per allocation order.
  for (auto i : c10::irange(rank)) {
    allocation_domain[i] = rfactor_dom.at(alloc_order.at(i));
  }

  return allocation_domain;
}

// NOTE: [Allocation Order Inference]
//
// AllocationOrderInferencer ctor takes a map of allocation order for inputs as
// `unordered_map<const TensorView*, AllocationOrder>`. It propagates
// AllocationOrder on a fusion and updates the the map with allocation order for
// other TensorView in the fusion.
//
// e.g.
//   std::unordered_map<const TensorView*, AllocationOrder> alloc_order_map;
//   // ... update alloc_order_map with AllocationOrder for tensors
//   //     (i.e. usually inputs)
//
//   // create AllocationOrderInferencer
//   AllocationOrderInferencer infer(alloc_order_map);
//   // propagates AllocationOrder from entries already in alloc_order_map
//   infer.traverse(fusion);
//   // all tensor that's propagated successfully will have their allocation
//   // order in alloc_order_map
//
// The protocol for AllocationOrder in alloc_order_map_ has three states. For
// each `tv`, its corresponding allocation order `alloc_order_map_[tv]`:
// 1. The allocation order has the same size as the `tv`'s rfactor domain;
//    This means it has a preferred allocation order and the entry should
//    participate in propagation.
// 2. The allocation order is an empty array;
//    This means it's a wild card and shouldn't dictate output allocation
//    order. But it marks that propagation is successful for `tv`.
//    i.e. This currently happens for TensorViews that's created by factory
//    methods and its consumers.
// 3. alloc_order_map_ does not have an entry for `tv`.
//    This is the case where propagation has not reach the `tv`, likely due to
//    lack of allocation order on inputs or certain operation not yet supported
//    by propagation rule.
//
// Identify the difference between case 2. and 3. above allows us to better
// handle `resolveAllocationOrder` among multiple candidates.
// i. We do not want to ignore candidates where propagation has failed and
// aggressively propagates allocatoin order through unresolved candidates. So we
// would want to identify case 3. ii. Tensors created by factory methods should
// carry a wild-card and should not actively participate propagation. Because
// those tensors are not going to affect vectorization. Hence we need to
// identify case 2.
class AllocationOrderInferencer : public IterVisitor {
 public:
  // Note: alloc_order_map_ is a reference to the ground truth of
  // alloc_order_map. The pass here tries to propagate the allocation order from
  // the ground truth.
  AllocationOrderInferencer(
      std::unordered_map<const TensorView*, AllocationOrder>& alloc_order_map)
      : alloc_order_map_(alloc_order_map) {}

 protected:
  using IterVisitor::handle;

  void handle(FullOp*) override;
  void handle(UnaryOp*) override;
  void handle(BroadcastOp*) override;
  void handle(BinaryOp*) override;
  void handle(TernaryOp*) override;
  void handle(PadOp*) override;
  void handle(ReductionOp*) override;
  // TODO: Add more propagation rules
  // void handle(LoadStoreOp*) override;
  // void handle(SqueezeOp*) override;
  // void handle(ExpandOp*) override;

 private:
  // mapping allocation domain from producer to consumer without reduction
  //
  // e.g.
  //   producer rfactor dom [r0', i0', i1', i2'] @ allocation order {0, 1, 3, 2}
  //    |       alloc dom [r0', i0', i2', i1']
  //    |
  //    Operation
  //    |
  //    v
  //   consumer rfactor dom [..., i0, ..., i1, ..., i2, ...]
  //
  // we construct allocation domain on producer, filtering out reduction, apply
  // root domain map from producer to consumer.
  //   [r0', i0', i2', i1'] -> [i0', i2', i1'] -> [i0, i2, i1]
  // so the function would return [i0, i2, i1]
  std::vector<IterDomain*> propagateAllocationDomain(
      TensorView* producer,
      TensorView* consumer) {
    // constructing alloc_domain for producer from its root domain, while
    // filtering out reduction because they won't appear in consumer's domain.
    std::vector<IterDomain*> alloc_domain = TensorDomain::noReductions(
        constructAllocationDomain(producer, alloc_order_map_.at(producer)));
    // creating producer to consumer root domain map
    std::unordered_map<IterDomain*, IterDomain*> p2c_map =
        PairwiseRootDomainMap(producer, consumer).mapProducerToConsumer();
    // map alloc_domain to consumer
    std::transform(
        alloc_domain.cbegin(),
        alloc_domain.cend(),
        alloc_domain.begin(),
        [&p2c_map](IterDomain* id) { return p2c_map.at(id); });
    return alloc_domain;
  }

  // Propagate allocation order from producer to consumer via:
  // 1. Constructs producer allocation_domain with its allocation order;
  // 2. Mapping it to consumer's root domain to create alloc_domain;
  // 3. Compute allocation order of consumer as the permutation between
  //    alloc_domain and `permutation_ref`.
  //
  // Returns true when producer has a recorded allocation order, false
  // otherwise. This function assumes that all root domain in consumer can be
  // mapped to producer.
  bool propagateAllocationOrder(
      TensorView* producer,
      TensorView* consumer,
      const std::vector<IterDomain*>& permutation_ref) {
    auto iter = alloc_order_map_.find(producer);
    // early return is producer doesn't have an entry in alloc_order_map_
    if (iter == alloc_order_map_.end()) {
      return false;
    }

    // early termination to propagate empty allocation order
    if (iter->second.empty()) {
      alloc_order_map_[consumer] = {};
      return true;
    }

    std::vector<IterDomain*> alloc_domain =
        propagateAllocationDomain(producer, consumer);
    // compute allocation order
    std::optional<AllocationOrder> permutation =
        ir_utils::computePermutation(permutation_ref, alloc_domain);

    NVF_ERROR(
        permutation.has_value(),
        "allocation order propagation from ",
        producer->toString(0),
        " to ",
        consumer->toString(0),
        " failed!");
    alloc_order_map_[consumer] = permutation.value();
    return true;
  }

  // Propagate allocation order from producer to consumer's rfactor_domain
  bool propagateAllocationOrder(TensorView* producer, TensorView* consumer) {
    return propagateAllocationOrder(
        producer, consumer, consumer->getMaybeRFactorDomain());
  }

  // Returns the candidate operand that dominates the allocation order.
  //
  // It scans through each candidate to find the first one that:
  //   1. is a TensorView
  //   2. has the most non_broadcast IterDomains
  //
  // The function returns a nullptr when it encounters a TensorView that does
  // not have an entry in alloc_order_map_, since this means we failed to
  // propagate memory format for an entry, we do NOT want to aggressively insert
  // output memory format.
  //
  // The function is used to resolve allocation order propagation for operator
  // with multiple operands. The operand with the most number of
  // non-broadcast IterDomain will be dominating the output allocation order.
  // The motivation behind it to avoid breaking allocation order propagation
  // from operands produced by broadcast. e.g. When a binary operator could take
  // in a channels_last 4d tensor and an unsqueezed bias vector. We'll want to
  // propagate the channels_last allocation order to output.
  //
  // Pre-condition: `candidates` must be the input operands of the same Expr.
  TensorView* resolveAllocationOrder(const std::vector<Val*>& candidates);

  // alloc_order_map_ records the allocation order of each TensorView.
  // Since it only handles permutation from a rfactor domain to allocation
  // domain, it can be interpreted as:
  //
  // e.g. TV0 rfactor domain [i0, i1, i2]
  //            alloc domain [i0, i2, i1]
  //        allocation order   0,  2,  1
  std::unordered_map<const TensorView*, AllocationOrder>& alloc_order_map_;
};

TensorView* AllocationOrderInferencer::resolveAllocationOrder(
    const std::vector<Val*>& candidates) {
  TensorView* src = nullptr;
  size_t non_bc_high_water_mark = 0;

  // helper utils to count the number of non broadcast / non reduction
  // iterdomain
  auto countLoopIterDomains = [](const TensorView* tv) -> size_t {
    return std::count_if(
        tv->getMaybeRFactorDomain().begin(),
        tv->getMaybeRFactorDomain().end(),
        [&](auto ptr_id) {
          return !ptr_id->isBroadcast() && !ptr_id->isReduction();
        });
  };

  for (auto* val : candidates) {
    auto* tv = dynamic_cast<TensorView*>(val);
    // skip non TensorView entry
    if (tv == nullptr) {
      continue;
    }

    auto iter = alloc_order_map_.find(tv);
    // stopping propagation when we encounter an entry that does not have an
    // allocation order. See NOTE: [Allocation Order Inference]
    if (iter == alloc_order_map_.end()) {
      return nullptr;
    }

    // skip entry that has an empty allocation order
    if (iter->second.empty()) {
      // We still want to ensure that we propagate empty allocation order if
      // there's no candidate with a non-empty allocation order
      if (src == nullptr) {
        src = tv;
      }

      // skip if unspecified
      continue;
    }

    // check if current entry sets new record for num of non broadcast / non
    // reduction iterdomain
    if (size_t non_bc_count = countLoopIterDomains(tv);
        non_bc_count > non_bc_high_water_mark) {
      non_bc_high_water_mark = non_bc_count;
      src = tv;
    }
  }

  return src;
}

// FullOp set empty allocation order to output
void AllocationOrderInferencer::handle(FullOp* op) {
  auto* out = static_cast<TensorView*>(op->output(0));
  alloc_order_map_[out] = {};
}

// UnaryOp propagation forward allocation order from input to output
void AllocationOrderInferencer::handle(UnaryOp* op) {
  auto* out = dynamic_cast<TensorView*>(op->out());
  if (out == nullptr) {
    return;
  }
  auto* in = op->in()->as<TensorView>();
  propagateAllocationOrder(in, out);
}

// BroadcastOp propagation:
//   1. preserves all allocation order of input iterdomain;
//   2. stacks all added broadcast iter domain on outputs as outer dimensions in
//   their natural position
//
// e.g.
//   TV0 rfactor dom [i0', i1', i2'] @ allocation order {0, 2, 1}
//    |    alloc dom [i0', i2', i1']
//    |
//    |
//    BroadcastOp
//    |
//    v
//   TV1 rfactor dom [i0, b3, i1, i2, b4]
//
//   step 0:
//       scan through all iterdomain in output TV1's rfactor domain
//       insert all broadcast domain to alloc_domain[b3, b4];
//
//   step 1:
//       computing iterdomain mapping from input to output;
//       [i0', i2', i1'] -> [i0, i2, i1]
//
//   step 2:
//       follow allocation order on input, insert the mapped iter domain on
//       output to alloc_domain[b3, b4, i0, i2, i1];
//
//   step 3:
//       compute permutation from alloc_domain to TV1's rfactor domain;
//       so output TV1 will have allocation order {1, 4, 0, 3, 2}
void AllocationOrderInferencer::handle(BroadcastOp* op) {
  auto* out = dynamic_cast<TensorView*>(op->out());
  if (out == nullptr) {
    return;
  }
  auto* in = op->in()->as<TensorView>();

  auto iter = alloc_order_map_.find(in);
  // early return when there's no recorded allocation order for `in`
  if (iter == alloc_order_map_.end()) {
    return;
  }

  // propagate empty allocation order;
  if (iter->second.empty()) {
    alloc_order_map_[out] = {};
    return;
  }

  size_t out_rank = out->nDims();
  std::vector<IterDomain*> alloc_domain;
  alloc_domain.reserve(out_rank);

  // step 0: insert all broadcast iterdomain in output
  for (auto i : c10::irange(out_rank)) {
    if (op->isBroadcastDim(i)) {
      alloc_domain.push_back(out->getMaybeRFactorDomain()[i]);
    }
  }

  // step 1: computing iterdomain mapping from input to output
  std::vector<IterDomain*> mapped_alloc_dom =
      propagateAllocationDomain(in, out);

  // step 2: push each mapped iterdomain
  std::copy(
      mapped_alloc_dom.begin(),
      mapped_alloc_dom.end(),
      std::back_inserter(alloc_domain));

  // step 3: compute permutation
  std::optional<AllocationOrder> permutation =
      ir_utils::computePermutation(out->getMaybeRFactorDomain(), alloc_domain);

  NVF_ERROR(
      permutation.has_value(),
      "allocation order propagation on broadcast op failed to compute valid permutation");
  alloc_order_map_[out] = permutation.value();
}

void AllocationOrderInferencer::handle(BinaryOp* op) {
  auto* out = dynamic_cast<TensorView*>(op->out());
  if (out == nullptr) {
    return;
  }
  propagateAllocationOrder(resolveAllocationOrder(op->inputs()), out);
}

void AllocationOrderInferencer::handle(TernaryOp* op) {
  auto* out = dynamic_cast<TensorView*>(op->out());
  if (out == nullptr) {
    return;
  }
  propagateAllocationOrder(resolveAllocationOrder(op->inputs()), out);
}

void AllocationOrderInferencer::handle(PadOp* op) {
  auto* out = dynamic_cast<TensorView*>(op->out());
  auto* in = dynamic_cast<TensorView*>(op->in());
  // Note: `out` from pad has rfactor domain that cannot be mapped back to
  // `in`'s root domain. Hence we use `out`'s root domain to match permutation.
  propagateAllocationOrder(in, out, out->getRootDomain());
}

void AllocationOrderInferencer::handle(ReductionOp* op) {
  auto* out = dynamic_cast<TensorView*>(op->out());
  auto* in = dynamic_cast<TensorView*>(op->in());
  propagateAllocationOrder(in, out);
}

} // namespace

// Note [ Allocation Order Propagation ]
//
// The propagation tries to propagate allocation order from inputs to the entire
// fusion:
//   1. Iterates through all inputs, looking for TensorView with allocation
//   domain that's a permutation of its corresponding rfactor domain and record
//   it as the allocation order of the tensor;
//   2. Traverse the fusion IR, propagate allocation order and record results in
//   alloc_order_map.
std::unordered_map<const TensorView*, AllocationOrder> inferenceAllocationOrder(
    Fusion* fusion) {
  std::unordered_map<const TensorView*, AllocationOrder> alloc_order_map;

  // // Note: we only consider simple permutation of allocation domain to rfactor
  // // domain.
  // for (auto tv : ir_utils::filterByType<TensorView>(fusion->inputs())) {
  //   std::optional<AllocationOrder> permutation = ir_utils::computePermutation(
  //       TensorDomain::noReductions(tv->getMaybeRFactorDomain()),
  //       TensorDomain::noReductions(tv->getMaybeAllocationDomain()));
  //   if (permutation.has_value()) {
  //     alloc_order_map[tv] = permutation.value();
  //   }
  // }

  // // Initialize AllocationOrderInferencer with allocation order of input tensor
  // // views
  // AllocationOrderInferencer infer(alloc_order_map);
  // infer.traverse(fusion);

  auto id_model = IdModel(fusion, /*build_graphs=*/false);
  const DisjointSets<Val*>& val_sets = id_model.idGraph(IdMappingMode::EXACT).disjointValSets();

  TensorView* ref = nullptr;
  // picking a candidate for propagation.
  for (auto tv : ir_utils::filterByType<TensorView>(fusion->inputs())) {
  }

  // propagating the allocation order through graph
  // option1: a vanilla mapping with `val_sets.strictAreMapped` and only manipulate things that is mapped.
  // option2: wondering if there's something for us to replay a partial map?! i.e. we can replay ref->rfactor --> ref->allocation to tv->rfactor

  // return the propagated map
  return alloc_order_map;
}

void AllocationDomainPass::runPass(Fusion* fusion) {
  std::unordered_map<const TensorView*, AllocationOrder> stride_mapping =
      inferenceAllocationOrder(fusion);

  for (Val* out_val : fusion->outputs()) {
    auto* out_tv = dynamic_cast<TensorView*>(out_val);
    // skip:
    //   1. non-tensor output;
    //   2. tensor output with allocation specified, assuming everything is
    //   semantical
    //   3. tensor output that's aliasing (Does aliased src matter?)
    if (out_tv == nullptr || out_tv->hasAllocation() ||
        fusion->getOutputAlias(out_val).type != AllocationType::New) {
      continue;
    }

    auto mapped_entry = stride_mapping.find(out_tv);
    if (mapped_entry == stride_mapping.end() || mapped_entry->second.empty()) {
      continue;
    }

    out_tv->setAllocationDomain(
        constructAllocationDomain(out_tv, mapped_entry->second), true);
  }
}

} // namespace nvfuser::preseg_passes
