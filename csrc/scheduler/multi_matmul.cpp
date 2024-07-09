// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <abstract_tensor.h>
#include <abstract_tensor_schedule.h>
#include <disjoint_set.h>
#include <inlining.h>
#include <instrumentation.h>
#include <ir/utils.h>
#include <multidevice/utils.h>
#include <scheduler/debug_utils.h>
#include <scheduler/matmul.h>
#include <scheduler/matmul_utils.h>
#include <scheduler/mma_utils.h>
#include <scheduler/utils.h>
#include <val_graph.h>
#include <val_graph_visitor.h>

// NOTE: included to avoid compilation error caused by missing destructor in
// 'SchedulerRuntimeInfo'
#include <executor_utils.h>
#include "mma_type.h"

namespace nvfuser {

namespace {

class MultipleMatmulScheduler {
 public:
  MultipleMatmulScheduler(Fusion* fusion, const MatmulParams& params)
      : fusion_(fusion),
        params_(params),
        id_model_(fusion, /*build_graphs=*/false) {}

  void run() {
    // Make sure we don't have global memory set on intermediate tensors from
    // fusion segmentation
    scheduler_utils::clearMemorySpace(fusion_);

    // Cache inputs
    scheduler_utils::cacheInputs(fusion_, /*unroll=*/true);

    // Cache and fork outputs
    auto cached_outputs =
        scheduler_utils::cacheAndForkOutputs(fusion_, /*unroll=*/true);

    findPatterns();

    translatePatterns();

    findRoles();

    // This also collects mma_results_
    cacheOperands();

    // Unswizzle mma result in shared memory
    // Note that if we are using split-K, we will set up this buffer after
    // rfactoring the matmul, between the MmaOp and the ReductionOp, in order to
    // take advantage of unswizzling during the grid reduction
    smem_epilogues_ = mma_results_;

    makeAllTiles();

    doSplitKRFactor();

    fusion_->printMath();
  }

 private:
  void findPatterns() {
    patterns_ = mma_utils::findMatmulPatterns(fusion_);
    NVF_ERROR(!patterns_.empty(), "No matmul patterns were found");
  }

  void translatePatterns() {
    mma_ops_.reserve(patterns_.size());
    for (mma_utils::MatmulPattern& pattern : patterns_) {
      mma_ops_.push_back(pattern.translateToMmaOp());
    }

    // Build a new IdModel since translateToMmaOp creates new TVs
    updateIdModel();
  }

  // Get tensor roles and id roles
  // When there are multiple matmul patterns, we can have conflicting roles.
  // For now we throw an error if this is the case.
  // TODO: This should be checked in canScheduleCompileTime
  void findRoles() {
    const auto roles_opt = mma_utils::allPatternRoles(id_model_, patterns_);
    NVF_ERROR(
        roles_opt.has_value(),
        "Incompatible roles found between matmul patterns");
    std::tie(id_roles_, tensor_roles_) = roles_opt.value();

    mma_utils::MatmulOperandInnerDimsOpt inner_dims_opt =
        mma_utils::getOperandInnerDims(id_model_, id_roles_, tensor_roles_);
    NVF_ERROR(inner_dims_opt.isValid(), inner_dims_opt.getErrorMsg());
    inner_dims_ = inner_dims_opt.getData();

    as_ = tensor_roles_.at(MatmulTensorRole::OPERAND_A);
    bs_ = tensor_roles_.at(MatmulTensorRole::OPERAND_B);

    const bool has_epilogue =
        std::any_of(mma_ops_.begin(), mma_ops_.end(), [](MmaOp* mma) {
          return !mma->out()->isFusionOutput();
        });
    const bool has_fusion_c_roles =
        (0 != tensor_roles_.count(MatmulTensorRole::EPILOGUE_INPUT));
    has_non_mma_input_tvs_ = has_epilogue && has_fusion_c_roles;
  }

  // Including current tensor naming convention for reference,
  //  this is very temporary and will change over time and
  //  in fact the whole body of this function will
  //  eventually be a set of utility functions for different
  //  sections of matmul(fusion) kernels, with
  //  each having its own build out to do.
  //
  // Current naming convention is based on the following formula:
  //
  //  d = alpha * (a x b) + beta * c
  //
  // and is defined in the following way:
  //
  //  operands assumed in global memory : a, b, c
  //
  //  registers staging global load : ar, br (short for a/b read)
  //
  //  shared mem cache of operands : acw_smem, bcw_smem (short for a/b
  //  cache_write smem)
  //
  //  registers at shared memory load output : acr, bcr (short for a/b cache
  //  read)
  //
  //  register tensor input to the actual mma op: ab, bb (short for a/b
  //  broadcasted)
  //
  //  accumulator register: mma_result
  //   - mma_result is MmaOp output if there is epilogue
  //   - mma_result is dc (short for d cache) if there is no epilogue
  //
  //  result in global memory: d

  // Currently the support is for a, b, c and d as fusion inputs/outputs
  //  aka. no prolog fusion yet.
  void cacheOperands() {
    mma_results_.reserve(mma_ops_.size());
    for (MmaOp* mma : mma_ops_) {
      mma->setMacro(params_.mma_macro);

      // Setup register and shared memory stages:
      //   TODO: this section goes to a separate matmul util,
      //   and needs more configurability.

      // Setup accumulator register.
      mma_results_.push_back(mma->out()->as<TensorView>());
    }

    // TODO:
    //  Significant build out needed here
    //   for more flexibility and data type support.
    // Shared memory
    acw_smems_.resize(as_.size(), nullptr);
    bcw_smems_.resize(bs_.size(), nullptr);
    // Shared memory read
    acrs_.resize(as_.size(), nullptr);
    bcrs_.resize(bs_.size(), nullptr);

    // Use cp.async as requested in scheduler params.
    LoadStoreOpType load_op = LoadStoreOpType::Set;
    CacheOp cache_op_a = CacheOp::Unspecified;
    CacheOp cache_op_b = CacheOp::Unspecified;
    if (params_.async_gmem_load_operands) {
      load_op = LoadStoreOpType::CpAsync;
      auto getCacheOp = [](int64_t vec_size, TensorView* operand) -> CacheOp {
        int64_t vec_bytes = vec_size * dataTypeSize(operand->dtype());
        NVF_CHECK(
            vec_bytes == 4LL || vec_bytes == 8LL || vec_bytes == 16LL,
            "Unsupported async vectorization size ",
            vec_size,
            " = ",
            vec_bytes,
            " bytes for operand ",
            operand->toString(),
            " which has data type ",
            operand->dtype(),
            ". Size must be 4, 8, or 16 bytes. ",
            "MatmulParams::async_gmem_load_operands should be set to false in this case.");
        return vec_bytes == 16LL ? CacheOp::Global : CacheOp::AllLevels;
      };
      for (TensorView* a : as_) {
        cache_op_a = getCacheOp(params_.supported_vec_size.a, a);
      }
      for (TensorView* b : bs_) {
        cache_op_b = getCacheOp(params_.supported_vec_size.b, b);
      }
    }

    for (size_t i : c10::irange(as_.size())) {
      TensorView* a = as_[i];
      NVF_ERROR(a->uses().size() == 1);
      acw_smems_[i] = ir_utils::consumerTvsOf(a).at(0);
      acw_smems_[i]->definition()->as<LoadStoreOp>()->setOpType(load_op);
      acw_smems_[i]->definition()->as<LoadStoreOp>()->setCacheOp(cache_op_a);
      if (acw_smems_[i]->uses().size() > 1) {
        // There can be multiple uses for example if we have A @ B1 + A @ B2
        // then A will be cached to smem then it might be loaded into two
        // separate register buffers, one for each mma. Instead, we will load it
        // once into registers then re-use the register buffer for both mmas.
        acw_smems_[i]->cacheAfter();
      }
      NVF_ERROR(acw_smems_[i]->uses().size() == 1);
    }
    for (size_t i : c10::irange(bs_.size())) {
      TensorView* b = bs_[i];
      NVF_ERROR(b->uses().size() == 1);
      bcw_smems_[i] = ir_utils::consumerTvsOf(b).at(0);
      bcw_smems_[i]->definition()->as<LoadStoreOp>()->setOpType(load_op);
      bcw_smems_[i]->definition()->as<LoadStoreOp>()->setCacheOp(cache_op_b);
      if (bcw_smems_[i]->uses().size() > 1) {
        bcw_smems_[i]->cacheAfter();
      }
      NVF_ERROR(bcw_smems_[i]->uses().size() == 1);
    }

    // We add two LoadStore operators to the inputs of our fusions. The first
    // one is for a read from global memory and the second one (below) is for a
    // cache read. As an optimizaton, we avoid adding an operator if there's an
    // existing LoadStoreOp present. Please note that for the second LoadStore
    // we don't propagte the allocation domain, since the scheduler sets the
    // allocation domain in the registers.
    auto addSetForCacheRead = [](TensorView* tv_smem, TensorView** tv_r) {
      if (auto ldst = dynamic_cast<LoadStoreOp*>(tv_smem->uses().at(0))) {
        *tv_r = ldst->out()->as<TensorView>();
        ldst->setOpType(LoadStoreOpType::LdMatrix);
      } else {
        *tv_r = tv_smem->cacheAfter(
            LoadStoreOpType::LdMatrix,
            CacheOp::Unspecified,
            /*propagate_allocation_domain=*/false);
      }
    };

    for (size_t i : c10::irange(as_.size())) {
      addSetForCacheRead(acw_smems_[i], &acrs_[i]);
    }
    for (size_t i : c10::irange(bs_.size())) {
      addSetForCacheRead(bcw_smems_[i], &bcrs_[i]);
    }

    // Build a new IdModel since we used cacheAfter
    updateIdModel();
  }

  //! Rebuilds IdModel, then updates all ValGroups in abstract tensors to refer
  //! to the new IdModel. This is necessary whenever we perform an operation
  //! that creates a new TensorView, such as caching or rFactor
  void updateIdModel() {
    // Build new IdModel
    IdModel new_id_model(fusion_);

    // Get new permissive graph
    ValGraph& new_graph = new_id_model.idGraph(IdMappingMode::PERMISSIVE);

    // Update AbstractTensors
    for (mma_utils::AbstractMatmulTensor& abten : {std::ref(at_tiled_)}) {
      for (AbstractId& abs_id : abten.domain) {
        ValGroupAndItsGraph& vgg = abs_id.as<ValGroupAndItsGraph>();
        vgg.group = new_graph.toGroup(vgg.group->front());
        vgg.graph = &new_graph;
      }
    }

    // Update id_roles_
    std::unordered_map<ValGroup, MatmulDimRole> new_id_roles;
    for (auto& [k, v] : id_roles_) {
      const ValGroup& new_group = new_graph.toGroup(k->front());
      new_id_roles.emplace(new_group, v);
    }
    id_roles_ = new_id_roles;

    // Set id_model_ after we are done using the old one
    id_model_ = std::move(new_id_model);
  }

  // Gets canonical dim ordering then uses it to canonicalize each tensor in the
  // fusion, then create tiles and swizzle their ordering.
  void makeAllTiles() {
    ValGraph& graph = id_model_.idGraph(IdMappingMode::PERMISSIVE);
    std::vector<ValGroup> canonical_dim_ordering =
        mma_utils::canonicalDimOrdering(tensor_roles_, id_roles_, graph);

    at_tiled_.domain.reserve(canonical_dim_ordering.size());
    for (const ValGroup& vg : canonical_dim_ordering) {
      at_tiled_.domain.push_back(ValGroupAndItsGraph{vg, &graph});
      // Tag each dimension with a MatmulDimRole
      auto it = id_roles_.find(vg);
      NVF_ERROR(it != id_roles_.end());
      at_tiled_.tags.push_back({it->second});
    }

    mma_utils::mergeCanonicalAbstractTensor(at_tiled_);

    mma_utils::makeTile(at_tiled_, params_.tile_sizes.cta_tile.toVector());

    applyAbstractTransforms(at_tiled_, ir_utils::allTvs(fusion_), &graph);
  }

  void doSplitKRFactor() {
    for (TensorView*& mma_result : mma_results_) {
      if (params_.splitk_factor != 1) {
        // rFactor converts
        //   mma_result = mma(A, B, {/*Kf*/-5, /*Kg*/-4, /*Ki*/-1});
        // to
        //   intermediate = mma(A, B, {-4, -1});
        //   final_sum = sum(intermediate, {/*Kf*/-3});
        // and the method returns "intermediate". We need mma_result to refer to
        // the actual MmaOp output, so here we reassign that to the
        // intermediate.
        TensorView* splitk_sum = mma_result;
        mma_result = splitk_sum->rFactor({-4, -1});
        splitk_sums_.push_back(splitk_sum);
      }

      // At this point we have the following schedule:
      //   No split-K
      //     mma_result      [..., iMo, iNo, rKo, iMi, iNi, rKi]
      //   Split-K
      //     mma_result      [..., iMo, iNo, iKf, rKg, iMi, iNi, rKi]
      //     splitk_sum      [..., iMo, iNo, rKf, iMi, iNi]

      if (params_.use_smem_epilogue) {
        // Note that for split-K
        //   splitk_sum = sum(mma_result)
        // becomes
        //   smem_epilogue = set(mma_result)
        //   splitk_sum = sum(smem_epilogue)
        smem_epilogues_.push_back(mma_result->cacheAfter());
        // smem_epilogue = [..., iMo, iNo, iKf, iMi, iNi]
      }
    }
  }

 private:
  Fusion* fusion_;
  const MatmulParams& params_;
  IdModel id_model_;
  std::vector<mma_utils::MatmulPattern> patterns_;
  std::vector<MmaOp*> mma_ops_;
  mma_utils::DimRolesMap id_roles_;
  mma_utils::TensorRolesMap tensor_roles_;
  mma_utils::MatmulOperandInnerDims inner_dims_;
  std::vector<TensorView*> as_, bs_, acw_smems_, bcw_smems_, acrs_, bcrs_, abs_,
      bbs_, mma_results_, splitk_sums_, smem_epilogues_;
  bool has_non_mma_input_tvs_;

  // Track the role of each axis for each tensor in the Fusion
  std::unordered_map<TensorView*, std::vector<MatmulDimRole>> all_tv_dims_;

  mma_utils::AbstractMatmulTensor at_tiled_;
};

} // namespace

void scheduleMultipleMatmuls(Fusion* fusion, const MatmulParams& params) {
  FusionGuard fg(fusion);

  MultipleMatmulScheduler(fusion, params).run();

  // TODO: translate starting from matmul.cpp:1027
  // if (params.use_smem_epilogue) {
}

} // namespace nvfuser
