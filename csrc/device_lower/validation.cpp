// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/validation.h>

#include <contiguity.h>
#include <device_lower/lower2device.h>
#include <device_lower/utils.h>
#include <instrumentation.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <scheduler/mma_utils.h>
#include <transform_iter.h>
#include <transform_replay.h>
#include <type.h>

#include <ATen/cuda/CUDAContext.h>
#include <limits>

namespace nvfuser {

namespace {

//! Validate multiple output tensors of the same expression, i.e.,
//! siblings, have valid domains and parallel types. Since siblings
//! are placed in the same loop nest, they must be parallelized the
//! same way. Will infer and modify serial parallel types if other
//! output/s are parallelized, so that user wouldn't have to specify
//! the same parallelization 3 times. Will throw if conflicts are
//! detected, i.e. TIDx vs BIDx etc.
class ValidateSiblings : public IterVisitor {
 public:
  static void validate(Fusion* fusion) {
    ValidateSiblings validator;
    validator.traverse(fusion);
  }

 private:
  using IterVisitor::handle;

  void dispatch(Expr* expr) final {
    if (!ir_utils::isTvOp(expr) || expr->outputs().size() < 2) {
      IterVisitor::dispatch(expr);
      return;
    }

    auto ref_output = expr->outputs().at(0)->as<TensorView>();
    auto ref_ndims = ref_output->nDims();
    const auto& ref_root = ref_output->getMaybeRootDomain();
    std::unordered_map<IterDomain*, IterDomain*> id_map;

    for (const auto sibling :
         ir_utils::filterByType<TensorView>(expr->outputs())) {
      if (ref_output == sibling) {
        continue;
      }

      NVF_ERROR(
          sibling->nDims() == ref_ndims,
          "Mismatched dimensionality detected. Expr: ",
          expr->toString(),
          "Ref output: ",
          ref_output->toString(),
          ". Sibling: ",
          sibling->toString());

      for (const auto i : c10::irange(ref_ndims)) {
        validateParallelTypes(ref_output->axis(i), sibling->axis(i));
      }

      for (const auto i : c10::irange(ref_root.size())) {
        id_map[ref_root[i]] = sibling->getMaybeRootDomain().at(i);
      }

      auto replay =
          BestEffortReplay(
              sibling->getLoopDomain(), ref_output->getLoopDomain(), id_map)
              .getIterDomainEquivalence();

      for (const auto i : c10::irange(ref_ndims)) {
        NVF_ERROR(
            replay.strictAreMapped(ref_output->axis(i), sibling->axis(i)),
            "Matching sibling ID not found. Expr: ",
            expr->toString(),
            "Ref ID: ",
            ref_output->axis(i)->toString(),
            "Sibling ID: ",
            sibling->axis(i)->toString());
      }
    }
  }

  // Parallelize id1 and id0 consistently if one is serial and the other isn't
  void validateParallelTypes(IterDomain* id0, IterDomain* id1) {
    const auto ptype0 = id0->getParallelType();
    const auto ptype1 = id1->getParallelType();

    if (ptype0 == ParallelType::Vectorize ||
        ptype1 == ParallelType::Vectorize) {
      auto other_type = ptype0 == ParallelType::Vectorize ? ptype1 : ptype0;
      NVF_ERROR(
          other_type == ParallelType::Vectorize ||
              (!isParallelTypeThreadDim(other_type) &&
               !isParallelTypeBlockDim(other_type)),
          "Vectorize type was parallelized inconsistently in. ",
          "Detected during promoting parallel types.");
      return;
    }

    if (ptype0 != ptype1) {
      NVF_CHECK(
          ptype0 == ParallelType::Serial || ptype1 == ParallelType::Serial,
          "Error promoting parallel types: ",
          ptype0,
          " and ",
          ptype1);
      if (ptype0 == ParallelType::Serial) {
        id0->parallelize(ptype1);
      }
      if (ptype1 == ParallelType::Serial) {
        id1->parallelize(ptype0);
      }
    }
  }
};

// Make sure all IterDomains are only used for a unique
// TensorView. Several mappings from IterDomains are
// created during lowering, which relies on the unique usage of
// IterDomains.
void validateIterDomainUsage(Fusion* fusion) {
  FUSER_PERF_SCOPE("GpuLower::Lower::validateIterDomainUse");
  FusionGuard fg(fusion);

  auto used_vals = fusion->usedMathVals();
  std::unordered_map<IterDomain*, TensorView*> domain_use_map;

  for (auto tv : ir_utils::filterByType<TensorView>(used_vals)) {
    for (auto id : tv->domain()->allIDs()) {
      auto it = domain_use_map.find(id);
      NVF_ERROR(
          it == domain_use_map.end(),
          "Multiple use of ",
          id,
          " detected.",
          " Used in both TV",
          tv->name(),
          " and TV",
          it->second->name());
      domain_use_map.insert({id, tv});
    }
  }
}

void validateCpAsyncBulk(const std::vector<TensorView*>& tvs) {
  for (auto tv : tvs) {
    for (auto id : tv->getLoopDomain()) {
      if (id->getParallelType() == ParallelType::Bulk) {
        NVF_ERROR(
            ir_utils::isCpAsyncBulk(tv->definition()),
            "ParallelType::Bulk is only supported for cp.async.bulk.");
      }
    }
  }
}

} // namespace

void validateIr(Fusion* fusion) {
  FUSER_PERF_SCOPE("GpuLower::Lower::validateIr");

  FusionGuard fg(fusion);

  fusion->validateInputs();

  // Validate Parallelization
  ValidateSiblings::validate(fusion);

  validateIterDomainUsage(fusion);

  auto dynamic_tvs = ir_utils::getTVsWithDynamicTransform(fusion);
  NVF_ERROR(
      dynamic_tvs.empty(),
      "Tensor with dynamic transform must be concretized before lowering: ",
      toDelimitedString(dynamic_tvs.begin(), dynamic_tvs.end()));

  auto all_tvs = fusion->allTvs();
  validateCpAsyncBulk(all_tvs);
}

namespace {

// Check contiguity for all allocation domains associated with Misaligned
// Vectorize ParallelType
void checkContiguity(
    const std::unordered_set<IterDomain*>& domains,
    TensorView* tv) {
  NVF_ERROR(tv->getMemoryType() == MemoryType::Global);

  for (const auto idx : c10::irange(tv->getMaybeAllocationDomain().size())) {
    auto alloc = tv->getMaybeAllocationDomain()[idx];
    if (domains.find(alloc) != domains.end()) {
      NVF_ERROR(
          !alloc->isBroadcast(),
          "Misaligned vectorization prohibits merging broadcast domains.",
          "Issue found in, ",
          tv);
      NVF_ERROR(
          tv->domain()->contiguity().at(idx).value_or(false),
          "Cannot merge non-contiguous allocation domains with misaligned vectorization.",
          "Issue found in, ",
          tv);
    }
  }
}

// Check all allocation iter domains in consumer that are present in domain,
// making sure they're contiguous. Map these domains to producer and make sure
// they are also contiguous in producer. Producer-consumer relationship is
// assumed to be through a set operation.
void checkContiguity(
    const std::unordered_set<IterDomain*>& domains,
    TensorView* consumer,
    TensorView* producer) {
  // This seems not quite right, shouldn't we be able to reverse this?
  NVF_ERROR(consumer->getMemoryType() == MemoryType::Local);
  NVF_ERROR(producer->getMemoryType() == MemoryType::Global);

  // TODO: we should use BestEffortReplay to find the correct c2p map for
  // allocation domain when it is different from logical domain.
  NVF_ERROR(
      !consumer->hasAllocation() && !producer->hasAllocation(),
      "Misaligned vectorization for allocation domain is not supported.");
  auto alloc_c2p =
      PairwiseLogicalDomainMap(producer, consumer).mapConsumerToProducer();

  std::unordered_map<IterDomain*, std::optional<bool>>
      producer_domain_contiguity;
  for (const auto idx :
       c10::irange(producer->getMaybeAllocationDomain().size())) {
    auto alloc = producer->getMaybeAllocationDomain().at(idx);
    auto contiguity = producer->domain()->contiguity().at(idx);
    producer_domain_contiguity.insert({alloc, contiguity});
  }

  for (auto consumer_alloc : consumer->getMaybeAllocationDomain()) {
    if (domains.find(consumer_alloc) != domains.end()) {
      auto producer_alloc = alloc_c2p.at(consumer_alloc);
      NVF_ERROR(
          producer_domain_contiguity.find(producer_alloc) !=
          producer_domain_contiguity.end());

      NVF_ERROR(
          !consumer_alloc->isBroadcast() || !producer_alloc->isBroadcast(),
          "Misaligned vectorization prohibits merging broadcast domains.",
          "Issue found in, ",
          consumer);

      NVF_ERROR(alloc_c2p.find(consumer_alloc) != alloc_c2p.end());

      NVF_ERROR(
          producer_domain_contiguity.at(producer_alloc).value_or(false),
          "Cannot merge non-contiguous allocation domains with misaligned vectorization.",
          "Issue found in, ",
          consumer);
    }
  }
}

class VectorizeValidator : public OptInDispatch {
 private:
  // Initially, vectorized_id is the IterDomain with Vectorize ParallelType
  // After processing all merge and split operations,
  // vectorized_id is the corresponding allocation domain
  VectorizeValidator(IterDomain* vectorized_id)
      : vectorized_id_(vectorized_id) {}

  using OptInDispatch::handle;

  void handle(Split* s) final {
    if (s->outer() == vectorized_id_) {
      is_valid = false;
    } else if (s->inner() == vectorized_id_) {
      vectorized_id_ = s->in();
    }
    domains_.insert(s->outer());
    domains_.insert(s->inner());
  }

  void handle(Merge* m) final {
    if (m->out() == vectorized_id_) {
      if (m->inner()->isBroadcast() && !m->outer()->isBroadcast()) {
        vectorized_id_ = m->outer();
      } else {
        vectorized_id_ = m->inner();
      }
    }
    domains_.insert(m->outer());
    domains_.insert(m->inner());
  }

  void handle(Resize* r) final {
    if (r->out() == vectorized_id_) {
      vectorized_id_ = r->in();
    }
    domains_.insert(r->in());
  }

  void handle(Swizzle* swizzle) final {
    if (swizzle->outX() == vectorized_id_ || swizzle->inX() == vectorized_id_ ||
        swizzle->outY() == vectorized_id_ || swizzle->inY() == vectorized_id_) {
      // Do not (yet) allow vectorization across any swizzled id.
      is_valid = false;
    }
  }

  void handle(Swizzle2D* swizzle) final {
    if (swizzle->outX() == vectorized_id_ || swizzle->inX() == vectorized_id_ ||
        swizzle->outY() == vectorized_id_ || swizzle->inY() == vectorized_id_) {
      // Do not (yet) allow vectorization across any swizzled id.
      is_valid = false;
    }
  }

  // Given the vectorized loop ID in a tensor, find its innermost ancestors in
  // the allocation domain.
  static IterDomain* getVectorizedIdInAllocationDomain(
      IterDomain* v_id,
      TensorView* tv,
      std::string name) {
    auto replay_exprs = DependencyCheck::getAllExprsBetween(
        {tv->getMaybeAllocationDomain().begin(),
         tv->getMaybeAllocationDomain().end()},
        {v_id});

    VectorizeValidator validator(v_id);

    for (auto expr_it = replay_exprs.rbegin(); expr_it != replay_exprs.rend();
         ++expr_it) {
      auto expr = *expr_it;
      validator.dispatch(expr);
    }

    NVF_CHECK(
        validator.is_valid,
        "Invalid vectorized pattern found, vectorization iter domains must be descendants of inner-most dimension.",
        "Issue found in, ",
        tv,
        "\n");

    if (v_id->getParallelType() == ParallelType::MisalignedVectorize) {
      if (tv->getMemoryType() == MemoryType::Global) {
        checkContiguity(validator.domains_, tv);
      } else if (tv->definition()->isA<LoadStoreOp>()) {
        auto input = tv->definition()->input(0);
        NVF_ERROR(input->isA<TensorView>());
        auto input_tv = input->as<TensorView>();
        checkContiguity(validator.domains_, tv, input_tv);
      }
    }

    NVF_ERROR(validator.vectorized_id_ != nullptr);

    // Contiguity is based on logical domain.
    IterDomain* last_alloc_dim = nullptr;
    size_t last_alloc_dim_pos = 0;
    for (size_t i = tv->getMaybeAllocationDomain().size(); i > 0; i--) {
      auto r_id = tv->getMaybeAllocationDomain()[i - 1];
      if (r_id->isReduction() || r_id->isBroadcast()) {
        continue;
      }
      if ((tv->getMemoryType() == MemoryType::Shared ||
           tv->getMemoryType() == MemoryType::Local) &&
          r_id->isBlockDim()) {
        // Inner-most parallelized dimensions don't count in allocation of
        // shared and local tensors.
        continue;
      }
      if (tv->getMemoryType() == MemoryType::Local && r_id->isThreadDim()) {
        continue;
      }
      last_alloc_dim = r_id;
      last_alloc_dim_pos = i - 1;
      break;
    }

    if (last_alloc_dim == nullptr) {
      // Should never get here, but that would mean there are no concrete
      // dims, so we should be fine.
      return nullptr;
    }

    auto ldst = dynamic_cast<LoadStoreOp*>(tv->definition());
    bool is_ldmatrix_trans =
        ldst != nullptr && mma_utils::isLdMatrixTranspose(ldst);
    if (!is_ldmatrix_trans) {
      // ldmatrix.trans is a hardware transpose instruction that can do
      // "vectorized" read from discontiguous memory
      NVF_CHECK(
          last_alloc_dim == validator.vectorized_id_,
          "Vectorized dim for ",
          name,
          " has to be from an inner most position. tv: ",
          tv,
          ", allocation domain: ",
          tv->getMaybeAllocationDomain(),
          ", vectorized id: ",
          validator.vectorized_id_,
          ", innermost id: ",
          last_alloc_dim);

      auto contiguity = tv->domain()->contiguity().at(last_alloc_dim_pos);
      NVF_CHECK(
          contiguity.value_or(false),
          "The innermost position has to be contiguous. tv: ",
          tv,
          ", allocation domain: ",
          tv->getMaybeAllocationDomain(),
          ", innermost id: ",
          last_alloc_dim->toString(),
          ", contiguity: ",
          contiguity.has_value() ? (*contiguity ? "t" : "f") : "n");
    }
    return validator.vectorized_id_;
  }

 private:
  std::unordered_set<IterDomain*> domains_;
  IterDomain* vectorized_id_ = nullptr;
  bool is_valid = true;

 public:
  static void validate(TensorView* tv) {
    // Make sure there's only one vectorized ID
    IterDomain* v_id = nullptr;
    for (auto id : tv->getLoopDomain()) {
      if (isParallelTypeVectorize(id->getParallelType())) {
        NVF_ERROR(
            v_id == nullptr,
            "Found two vectorized domains in ",
            tv,
            " only one is allowed.");
        v_id = id;
      }
    }

    // If no vectorized ids found simply return. If vectorized access is
    // broadcast, it won't generate an actual vector instruction, so can safely
    // be ignore
    if (v_id == nullptr || v_id->isBroadcast()) {
      return;
    }

    NVF_CHECK(
        v_id->extent()->isConstInt(),
        "Vectorizing a domain requires a constant integer size.");

    auto vector_word_size = v_id->extent()->evaluate().as<int64_t>();
    auto vector_size =
        dataTypeSize(
            tv->getDataType().value(), GpuLower::current()->indexType()) *
        vector_word_size;

    // Allow half2, float2, float4 and same sized vtypes.
    std::array<int64_t, 4> allowed_vector_sizes = {2, 4, 8, 16}; // NOLINT

    NVF_CHECK(
        std::find(
            allowed_vector_sizes.begin(),
            allowed_vector_sizes.end(),
            vector_size) != allowed_vector_sizes.end(),
        "Tried to vectorize a dim resulting in a word size of ",
        vector_size,
        " however, vector sizes only upto and including 16 bytes are supported.");

    auto consumer_vectorized_id =
        getVectorizedIdInAllocationDomain(v_id, tv, "consumer");
    if (consumer_vectorized_id == nullptr) {
      return;
    }

    // Save info required to lowering and runtime validation
    auto consumer_word_size_it =
        GpuLower::current()->vectorizedAccesses().find(tv);
    if (consumer_word_size_it !=
        GpuLower::current()->vectorizedAccesses().end()) {
      consumer_word_size_it->second =
          std::max(vector_word_size, consumer_word_size_it->second);
    } else {
      GpuLower::current()->vectorizedAccesses().emplace(tv, vector_word_size);
    }

    auto tv_def = tv->definition();
    NVF_ERROR(
        tv_def != nullptr,
        "Tv has no definition, cannot validate vectorization:",
        tv);
    auto producer_tv = tv_def->inputs().at(0)->as<TensorView>();
    auto producer_word_size_it =
        GpuLower::current()->vectorizedAccesses().find(producer_tv);
    if (producer_word_size_it !=
        GpuLower::current()->vectorizedAccesses().end()) {
      producer_word_size_it->second =
          std::max(vector_word_size, producer_word_size_it->second);
    } else {
      GpuLower::current()->vectorizedAccesses().emplace(
          producer_tv, vector_word_size);
    }

    VectorizedSetInfo vectorized_set_info;
    vectorized_set_info.consumer_tv = tv;
    vectorized_set_info.producer_tv = producer_tv;
    // Note that VectorizedSetInfo is about each instance of
    // vectorized set operations, so the word size is the size of this
    // specific vectorized set.
    vectorized_set_info.word_size = vector_word_size;
    vectorized_set_info.vectorized_loop_id = v_id;
    vectorized_set_info.vectorized_consumer_alloc_id = consumer_vectorized_id;

    // Validate producer
    auto pairwise_map = PairwiseLogicalDomainMap(producer_tv, tv);
    auto producer_replayed_as_consumer =
        TransformReplay::replayPasC(
            producer_tv,
            tv,
            -1,
            pairwise_map,
            TransformReplayOptions().replayResize())
            .first;
    ir_utils::TVDomainGuard domain_guard(
        producer_tv, producer_replayed_as_consumer);
    auto c2p_map =
        BestEffortReplay::replayPasC(producer_tv, tv, -1, pairwise_map)
            .getReplay();
    vectorized_set_info.vectorized_producer_alloc_id =
        getVectorizedIdInAllocationDomain(
            c2p_map.at(v_id), producer_tv, "producer");

    // For aligned vectorize, the extent of a vectorized domain must
    // be divisible by the vector word size. The domain is usually
    // just one of the allocation domains, but can be a merged domain of
    // contiguous domains. Those domains are saved in
    // VectorizedSetInfo.contig_alloc_ids in
    // fillConsumerVectorizedContigAllocationDomains called in
    // lower_index_compute.
    GpuLower::current()->vectorizedSetInfo().emplace_back(vectorized_set_info);
  }
};

} // namespace

// Uses ContigIDs to find allocation contig domains that a vectorized domain
// depends on. As ContigIDs depends on HaloInfo, this must be done
// after HaloInfo is created.
void validateAndCollectVectorizeInfo(Fusion* fusion) {
  FUSER_PERF_SCOPE("GpuLower::Lower::validateVectorize");
  FusionGuard fg(fusion);

  std::vector<Val*> used_vals = fusion->usedMathVals();
  for (auto* tv : ir_utils::filterByType<TensorView>(used_vals)) {
    bool has_vectorize_dim = false;
    bool has_misaligned_vectorize_dim = false;

    for (const auto i : c10::irange(tv->nDims())) {
      IterDomain* id = tv->axis(i);
      IterDomain* concrete_id = lower_utils::getConcreteLoopID(id);

      auto ptype = concrete_id->getParallelType();

      if (ptype == ParallelType::Vectorize) {
        // If we want to do this check up front we would have to do 2 things:
        // (1) Check that the tensor view with vectorize being set on it is
        // getting set outside the local compute at position
        // (2) Check any producers of the tensor view with vectorize being set
        // on it to make sure their compute at position isn't to the right of
        // the vectorize dim.
        NVF_ERROR(
            i >= tv->getMaxComputePosition(),
            "IterDomains to the left of the compute at point cannot be vectorized: ",
            tv,
            "\n");
        has_vectorize_dim = true;
      }

      if (concrete_id->getParallelType() == ParallelType::MisalignedVectorize) {
        NVF_ERROR(
            tv->getMaxComputePosition() == 0 ||
                tv->getMaxComputePosition() == tv->nDims() - 1,
            "Only allow misaligned vectorization in the -2 computeAt position.");
        NVF_ERROR(
            tv->getMemoryType() == MemoryType::Local ||
                tv->getMemoryType() == MemoryType::Global,
            "Only allow misaligned vectorization between global and local memory.");
        has_misaligned_vectorize_dim = true;
      }
    }
    if (has_vectorize_dim) {
      Expr* def = tv->definition();
      NVF_ERROR(
          def == nullptr || def->isA<LoadStoreOp>() || def->isA<SliceOp>() ||
              (def->isA<ReductionOp>() &&
               def->as<ReductionOp>()->serialGridReductionRequested()),
          "Vectorized accesses cannot be inline with computation: ",
          (def == nullptr ? tv->toString() : def->toString()));
    }
    // Validate the vectorized domain maps to the innermost domain of
    // tv. Note that we don't need to validate its producer tv as
    // both Vectorize and MisalignedVectorize can only be used with
    // UnaryOp::Set.
    if (has_vectorize_dim || has_misaligned_vectorize_dim) {
      VectorizeValidator::validate(tv);
    }
  }
}

namespace {

void fillVectorizedContigAllocationDomains(
    const TensorView* tv,
    const ContigIDs& contig_finder,
    IterDomain* vectorized_alloc_id,
    VectorizedSetInfo& info) {
  const auto& alloc_dom = tv->getMaybeAllocationDomain();

  // Find the allocation domains that are dependency of the merged contig
  // domain.

  auto consumer_indexed_it =
      contig_finder.allocToIndexedID().find(vectorized_alloc_id);
  NVF_ERROR(
      consumer_indexed_it != contig_finder.allocToIndexedID().end(),
      "Contiguity information not found for allocation domain: ",
      vectorized_alloc_id->toString());
  auto consumer_indexed_id = consumer_indexed_it->second;

  // Actual indexed allocation domains for this allocation domain. If
  // contig merge is done, multiple allocation domains are included.
  std::unordered_set<IterDomain*> indexed_alloc_ids;

  if (consumer_indexed_id == vectorized_alloc_id) {
    // Indexed domain is equal to the allocation domain, meaning no contig
    // merge is involved.
    indexed_alloc_ids.insert(vectorized_alloc_id);
  } else {
    auto consumer_within_contig_it =
        contig_finder.withinContigIDs().find(consumer_indexed_id);
    NVF_ERROR(
        consumer_within_contig_it != contig_finder.withinContigIDs().end());
    const auto& within_ids = consumer_within_contig_it->second;
    std::copy_if(
        alloc_dom.begin(),
        alloc_dom.end(),
        std::inserter(indexed_alloc_ids, indexed_alloc_ids.end()),
        [&](IterDomain* alloc_id) {
          return within_ids.find(alloc_id) != within_ids.end();
        });
  }

  // Store the contig merged allocation domains. If it is already set, pick
  // the smaller one as it is used for validating divisibility of the
  // merged extent.
  if (info.contig_alloc_ids.empty() ||
      indexed_alloc_ids.size() < info.contig_alloc_ids.size()) {
    info.contig_alloc_ids = indexed_alloc_ids;
  }
}

} // namespace

void fillConsumerVectorizedContigAllocationDomains(
    const TensorView* consumer_tv,
    const ContigIDs& contig_finder) {
  auto& info_vector = GpuLower::current()->vectorizedSetInfo();
  auto it = std::find_if(
      info_vector.begin(), info_vector.end(), [&consumer_tv](auto& info) {
        return info.consumer_tv == consumer_tv;
      });
  if (it == info_vector.end()) {
    return;
  }

  VectorizedSetInfo& info = *it;

  // info.vectorized_consumer_alloc_id is validated at this point to be the
  // last concrete allocation domain in consumer.
  auto consumer_alloc_id = info.vectorized_consumer_alloc_id;

  fillVectorizedContigAllocationDomains(
      consumer_tv, contig_finder, consumer_alloc_id, info);
}

void fillProducerVectorizedContigAllocationDomains(
    const TensorView* producer_tv,
    const TensorView* consumer_tv,
    const ContigIDs& contig_finder) {
  auto& info_vector = GpuLower::current()->vectorizedSetInfo();
  auto it = std::find_if(
      info_vector.begin(),
      info_vector.end(),
      [&producer_tv, &consumer_tv](auto& info) {
        return info.consumer_tv == consumer_tv &&
            info.producer_tv == producer_tv;
      });
  if (it == info_vector.end()) {
    return;
  }

  VectorizedSetInfo& info = *it;

  fillVectorizedContigAllocationDomains(
      producer_tv, contig_finder, info.vectorized_producer_alloc_id, info);
}

namespace {

//! Validates that the operand and result tensors
//!  of mma ops are swizzled and also validates
//!  specialization of tidx as lane id.
void validateMmaTensors(MmaOp* mma) {
  bool tidx_validated = false;
  std::vector<TensorView*> to_validate = {mma->out()->as<TensorView>()};

  if (ir_utils::isLdMatrixOp(mma->inA()->definition())) {
    to_validate.push_back(mma->inA()->as<TensorView>());
  }
  if (ir_utils::isLdMatrixOp(mma->inB()->definition())) {
    to_validate.push_back(mma->inB()->as<TensorView>());
  }

  for (auto tv : to_validate) {
    for (auto id : tv->getLoopDomain()) {
      auto ptype = id->getParallelType();
      if (ptype == ParallelType::TIDx) {
        if (!tidx_validated) {
          // Check that TIDx is exact lane_id
          const auto& paralel_dim_map =
              GpuLower::current()->parallelDimensionMap();
          NVF_ERROR(
              lower_utils::isExtentEqualToMaxParallelTypeExtent(id) &&
                  paralel_dim_map.get(ptype)->isConstInt(),
              "TIDx is reserved for lane id in mma kernels");
          if (mma->isHopper()) {
            NVF_ERROR(
                paralel_dim_map.get(ptype)->evaluate() ==
                    at::cuda::warp_size() * 4,
                "TIDx must be exactly a warp group for Hopper");
          } else {
            NVF_ERROR(
                paralel_dim_map.get(ptype)->evaluate() == at::cuda::warp_size(),
                "TIDx must be exactly a warp for Turing/Ampere");
          }
          tidx_validated = true;
        }
      }
    }
  }

  // Note: this check will be relaxed in a follow up.
  auto validate_operand = [mma](const TensorView* tv, MmaOperand operand) {
    if (mma->isHopper()) {
      if (operand == MmaOperand::B) {
        NVF_ERROR(
            tv->getMemoryType() == MemoryType::Shared,
            "Only supporting smem input for Hopper mma input B");
      } else {
        NVF_ERROR(
            tv->getMemoryType() == MemoryType::Local ||
                tv->getMemoryType() == MemoryType::Shared,
            "Only supporting register or shared memory input for Hopper mma input A");
      }
    } else {
      NVF_ERROR(
          tv->getMemoryType() == MemoryType::Local,
          "Only supporting register input for mma input on Ampere/Turing");
    }
  };

  validate_operand(mma->inA()->as<TensorView>(), MmaOperand::A);
  validate_operand(mma->inB()->as<TensorView>(), MmaOperand::B);
}

void validateSizeMemoryOp(LoadStoreOp* ldst) {
  if (!ldst->out()->isA<TensorView>()) {
    return;
  }

  if (ldst->opType() != LoadStoreOpType::CpAsync) {
    return;
  }

  int64_t byte_size = 1;
  auto output = ldst->out()->as<TensorView>();
  for (auto id : output->getLoopDomain()) {
    if (id->getParallelType() == ParallelType::Vectorize) {
      byte_size = id->extent()->evaluate().as<int64_t>();
      break;
    }
  }
  byte_size *=
      dataTypeSize(*output->getDataType(), GpuLower::current()->indexType());

  switch (ldst->cacheOp()) {
    case CacheOp::Global:
      NVF_CHECK(byte_size == 16, "Not supported byte size for cp.async.cg");
      break;
    case CacheOp::AllLevels:
      NVF_CHECK(
          byte_size == 4 || byte_size == 8 || byte_size == 16,
          "Not supported byte size for cp.async.ca");
      return;
    default:
      return;
  }
}

} // namespace

//! Validate data format and GPU arch compatibility of scheduled
//!  mma operators on the fusion.
void validateMma(Fusion* fusion) {
  auto exprs = StmtSort::getExprs(fusion);

  for (auto expr : exprs) {
    if (auto mma = dynamic_cast<MmaOp*>(expr)) {
      validateMmaTensors(mma);
    }
    if (auto ldst = dynamic_cast<LoadStoreOp*>(expr)) {
      validateSizeMemoryOp(ldst);
    }
  }
}

namespace {

// Utility function to validate a loop swizzle:
//  1. Throws an error if any output of the swizzle is not in loop_domain set.
//  2. Warns if any output of the swizzle is not the concrete id of the loop
//  map.
// The second case would make the codegen ignore this swizzle, as if it was not
// there at all.
void validateLoopSwizzle(
    Expr* swizzle_expr,
    std::unordered_set<IterDomain*>& loop_domains) {
  for (auto out_id :
       ir_utils::filterByType<IterDomain>(swizzle_expr->outputs())) {
    NVF_ERROR(
        loop_domains.count(out_id),
        "Loop swizzle can only be direct producer of loop domains.");
    if (lower_utils::getConcreteLoopID(out_id) != out_id) {
      TORCH_WARN_ONCE("Ignored loop swizzle :", swizzle_expr->toString());
    }
  }
}

} // namespace

void validateSwizzle(Fusion* fusion) {
  auto used_vals = fusion->usedMathVals();
  for (auto tv : ir_utils::filterByType<TensorView>(used_vals)) {
    if (tv->hasSwizzleOp()) {
      std::unordered_set<IterDomain*> tv_loop_domain_set(
          tv->getLoopDomain().begin(), tv->getLoopDomain().end());

      // Make sure no swizzle op is inlined:
      auto inlined_swizzles = ir_utils::getAllSwizzlesBetween(
          tv->getLogicalDomain(),
          {tv->getLoopDomain().begin(),
           tv->getLoopDomain().begin() + tv->getMaxComputePosition()});

      auto not_inlined_swizzles = ir_utils::getAllSwizzlesBetween(
          tv->getLogicalDomain(),
          {tv->getLoopDomain().begin() + tv->getMaxComputePosition(),
           tv->getLoopDomain().end()});

      // Check inlined swizzles: only loop swizzles can be inlined currently
      //  as inlining data swizzles would require addtional support of unswizzle
      //  operator, which currently doesn't have important use cases.
      for (auto swizzle_expr : inlined_swizzles) {
        NVF_ERROR(
            swizzle_expr->as<Swizzle2D>()->swizzleMode() == SwizzleMode::Loop,
            "Only support inlining loop swizzles");
        validateLoopSwizzle(swizzle_expr, tv_loop_domain_set);
      }

      std::unordered_set<Expr*> inlined_swizzle_set(
          inlined_swizzles.begin(), inlined_swizzles.end());

      // Check not inlined swizzles:
      //  Apply the loop swizzle check when it applies, and
      // also make sure that the no swizzle is also in inlined_swizzle set.
      // The latter would mean that one output of the swizzle is inlined while
      //  the other is not. Such case will not be supported.
      for (auto swizzle_expr : not_inlined_swizzles) {
        NVF_ERROR(
            !inlined_swizzle_set.count(swizzle_expr),
            "Cannot partially inline across swizzle domains.",
            swizzle_expr->toString());
        if (swizzle_expr->as<Swizzle2D>()->swizzleMode() == SwizzleMode::Loop) {
          validateLoopSwizzle(swizzle_expr, tv_loop_domain_set);
        }
      }
    }
  }
}

void validateAndConvertIterDomainGrouping(Fusion* fusion) {
  for (auto tv : fusion->allTvs()) {
    bool is_grouped = false;
    for (const auto id_idx : c10::irange(tv->nDims())) {
      const auto id = tv->axis(id_idx);
      auto ptype = lower_utils::getConcreteLoopID(id)->getParallelType();
      if (ptype != ParallelType::Group) {
        // Not a grouped ID
        continue;
      }

      // Remember if a grouped ID is found
      is_grouped = true;

      // Grouping only makes sense for the normal iteration or gather scatter
      // type
      NVF_CHECK(
          id->getIterType() == IterType::Iteration ||
              id->getIterType() == IterType::GatherScatter,
          "Invalid use of ParallelType::Group.",
          " Grouping of ",
          id->getIterType(),
          " is not allowed. ",
          tv->toString());

      // Extent must be static
      NVF_CHECK(
          id->extent()->value().is<int64_t>(),
          "Invalid use of ParallelType::Group.",
          " IterDomain must have a static extent: ",
          id->toString());

      // The CA position must be left of any grouped ID
      NVF_CHECK(
          tv->getMaxComputePosition() <= id_idx,
          "Invalid use of ParallelType::Group.",
          " ComputeAt position must be left of grouped IDs: ",
          tv->toString());

      // Similarly, the produce-at position must be left of any grouped ID
      NVF_CHECK(
          tv->getMaxProducerPosition() <= id_idx,
          "Invalid use of ParallelType::Group.",
          " ProduceAt position must be left of grouped IDs: ",
          tv->toString());
    }

    if (!is_grouped) {
      continue;
    }

    // Must be defined by ReductionOp
    auto def = tv->definition();
    NVF_CHECK(
        def != nullptr,
        "Invalid use of ParallelType::Group.",
        " Definition of tv with ParallelType::Group not found. ",
        tv->toString());

    NVF_CHECK(
        tv->definition()->isA<ReductionOp>() ||
            tv->definition()->isA<GroupedReductionOp>() ||
            tv->definition()->isA<WelfordOp>() ||
            tv->definition()->isA<GroupedWelfordOp>(),
        "Invalid use of ParallelType::Group. Only ReductionOp, GroupedReductionOp, WelfordOp and GroupedWelfordOp are allowed. ",
        tv->definition()->toString());

    // Convert ReductionOp to GroupedReductionOp
    if (tv->definition()->isA<ReductionOp>()) {
      auto rop = def->as<ReductionOp>();
      auto is_allreduce = rop->isAllreduce();

      std::vector<BinaryOpType> op_types({rop->getReductionOpType()});
      std::vector<Val*> init_vals({rop->init()});
      std::vector<Val*> outputs({rop->out()});
      std::vector<Val*> inputs({rop->in()});

      fusion->removeExpr(rop);
      IrBuilder::createInContainer<GroupedReductionOp>(
          fusion, op_types, init_vals, outputs, inputs, is_allreduce);
    } else if (tv->definition()->isA<WelfordOp>()) {
      // Convert WelfordOp to GroupedWelfordOp
      auto wop = def->as<WelfordOp>();
      auto is_allreduce = wop->isAllreduce();

      NVF_CHECK(
          is_allreduce,
          "Invalid use of ParallelType::Group.",
          " Only enabled for allreduce reductions: ",
          wop->toString());

      NVF_CHECK(
          tv->domain()->hasGridReduction(),
          "Invalid use of ParallelType::Group.",
          " Only enabled for grid reductions: ",
          wop->toString());

      std::vector<WelfordTriplet> output_vals(
          {{wop->outAvg(), wop->outVar(), wop->outN()}});
      std::vector<WelfordTriplet> input_vals(
          {{wop->inAvg(), wop->inVar(), wop->inN()}});
      std::vector<WelfordTriplet> init_vals(
          {{wop->initAvg(), wop->initVar(), wop->initN()}});
      fusion->removeExpr(wop);
      IrBuilder::createInContainer<GroupedWelfordOp>(
          fusion, output_vals, input_vals, init_vals, is_allreduce);
    }
  }
}

void validateGroupedReductions(Fusion* fusion) {
  for (auto expr : StmtSort::getExprs(fusion)) {
    if (auto grouped_reduction_op = dynamic_cast<GroupedReductionOp*>(expr)) {
      const auto num_exprs =
          grouped_reduction_op->numHorizontallyGroupedExprs();
      int64_t num_grouped_iterations = 1;
      auto out_tv = ir_utils::getTvOutput(grouped_reduction_op);
      for (auto axis : out_tv->getLoopDomain()) {
        if (axis->getParallelType() == ParallelType::Group) {
          num_grouped_iterations *= axis->extent()->value().as<int64_t>();
        }
      }
      NVF_CHECK(
          num_exprs * num_grouped_iterations <= kMaxNumGroupedReductions,
          "Too many grouped reductions: ",
          grouped_reduction_op->toString(),
          ". Up to ",
          kMaxNumGroupedReductions,
          " reductions are allowed.");
    }
  }
}

void validateLookupTV(Fusion* fusion) {
  for (auto expr : fusion->exprs()) {
    if (expr->isA<SelectOp>() || expr->isA<IndexSelectOp>()) {
      NVF_CHECK(
          expr->input(0)->isFusionInput(),
          "Lookup input must be a fusion input: ",
          expr->toString());
    }
  }
}

void validateReductions(Fusion* fusion) {
  for (auto rop : ir_utils::getOpsOfType<ReductionOp>(fusion)) {
    auto in = rop->in()->as<TensorView>();
    auto out = rop->out()->as<TensorView>();
    PairwiseLogicalDomainMap c2p_map(in, out);
    c2p_map.mapBroadcast(true);
    auto c2p = c2p_map.mapConsumerToProducer();
    for (auto out_id : out->getMaybeRootDomain()) {
      if (out_id->isReduction()) {
        auto in_it = c2p.find(out_id);
        NVF_ERROR(
            in_it != c2p.end(),
            "Could not find producer IterDomain mapped to ",
            out_id->toString());
        IterDomain* in_id = in_it->second;
        NVF_ERROR(
            !in_id->isBroadcast() || in_id->hasExpandedExtent(),
            "Reductions of unexpanded broadcast domains should be ",
            "converted to squeeze before lowering.");
      }
    }
  }
}

} // namespace nvfuser
