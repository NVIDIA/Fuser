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
#include <id_model/schedule.h>
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

// A matmul kernel might perform multiple matmuls; i.e. there can be multiple
// MmaOps in the scheduled tensor. Each one outputs a TensorView* which call an
// mma_result. Each MmaOp will also have two input TensorViews which we call
// "ab" and "bb". Again there can be multiple abs and multiple bbs in one
// fusion. These TensorViews are loaded from global memory tensors that we call
// "a" and "b" into shared memory tensors called acw_smem and bcw_smem. They are
// loaded from shared memory to register buffers we call "acr" and "bcr" ("cr"
// meaning "cache read" in this context).
//
// Putting this all together we have the following order for a simple matmul
//
//   a -> acw_smem -> acr -> ... -> ab
//                                    \
//                                      mma_result ->  ... -> dc -> d
//                                    /
//   b -> bcw_smem -> bcr -> ... -> bb
//
// The ... indicate that there might be other tensors involved in a prologue or
// epilogue section at that location.
//
// In this example there are two matmuls both using the same "a" operand:
//
//   b1 -> bcw_smem1 -> bcr1 -> ... -> bb1
//                                        \
//                                          mma_result1
//                                        /             \
//       a -> acw_smem -> acr -> ... -> ab                ... -> dc -> d
//                                        \             /
//                                          mma_result2
//                                        /
//   b2 -> bcw_smem2 -> bcr2 -> ... -> bb2
//
// Note that there can be more than one output d and each one will have its own
// register cache dc.
//
// Split-K and smem epilogue unswizzling add two additional tensors for each
// mma in the fusion: splitk_sum and smem_epilogue.
//
//   // No split-K, no smem epilogue unswizzling:
//     mma_result ->  ... -> dc -> d
//   // split-K, no smem epilogue unswizzling:
//     mma_result -> splitk_sum -> ... -> dc -> d
//   // smem epilogue unswizzling, no split-K:
//     mma_result -> smem_epilogue -> ... -> dc -> d
//   // split-K and smem epilogue unswizzling:
//     mma_result -> smem_epilogue -> splitk_sum -> ... -> dc -> d
//
// These additional tensors are added to each mma_result in the fusion.
//
// Each of the named tensors above is scheduled differently. We schedule them
// by building AbstractTensors for each tensor category; these are held in
// MultipleMatmulScheduler::schedules_.
class MultipleMatmulScheduler {
 public:
  MultipleMatmulScheduler(Fusion* fusion, const MatmulParams& params)
      : fusion_(fusion),
        params_(params),
        id_model_(fusion, /*build_graphs=*/false) {}

  void run() {
    // Clears memory spaces on intermediate tensors, calls
    // cache{After,Before,Fork} on inputs and outputs
    cacheInputsAndOutputs();

    // Finds matmul patterns and translates them to MmaOps, then finds tensor
    // and dimension roles for all tensors in the fusion
    findPatterns();
    translatePatterns();
    // translatePatterns changes the TensorView graph, so we build the IdModel
    // afterward
    buildIdModel();
    findRoles();

    // Defines acw_smem/bcw_smem and acr/bcr by possibly calling cacheAfter.
    // This also collects mma_results_
    defineOperandCaches();

    // TODO: Remove this as the methods below are implemented
    return;

    // Schedules:
    //   - global->smem (cp.async)
    //   - smem->register (ldmatrix)
    //   - prologue computation in registers, including broadcast to e.g.
    //   ab=[iM, bN, iK]
    schedulePrologues();

    // schedule mma instruction output (mma_result)
    scheduleMmaResults();

    // schedule epilogue
    scheduleEpilogue();

    // schedule splitk_sum
    scheduleSplitKSum();

    setUpInlining();

    // set up circular buffering. This must come after everything up to
    // mma_result is scheduled, since everything in the main loop will need to
    // be rotated
    setUpCircularBuffering();
  }

 private:
  void cacheInputsAndOutputs() {
    // Make sure we don't have global memory set on intermediate tensors from
    // fusion segmentation
    scheduler_utils::clearMemorySpace(fusion_);

    // Cache inputs
    scheduler_utils::cacheInputs(fusion_, /*unroll=*/true);

    // Cache and fork outputs
    cached_outputs_ =
        scheduler_utils::cacheAndForkOutputs(fusion_, /*unroll=*/true);
  }

  void findPatterns() {
    patterns_ = mma_utils::findMatmulPatterns(fusion_);
    NVF_ERROR(!patterns_.empty(), "No matmul patterns were found");
  }

  void countDims() {
    NVF_ERROR(!patterns_.empty());
    TensorView* mma_result = patterns_.front().output;
    num_device_dims_ = numDeviceDims(mma_result);
    for (const auto& it : id_roles_) {
      if (it.second == MatmulDimRole::Batch) {
        // All batch dims will be merged into one, if any exist
        num_local_batch_dims_ = 1;
      }
    }
    num_splitk_dims_ = params_.splitk_factor > 1 ? 1 : 0;
    // Subtract 6 for the [Mo, No, Ko, Mi, Ni, Ki]
    num_device_and_batch_dims_ = num_device_dims_ + num_local_batch_dims_;
  }

  void translatePatterns() {
    mma_ops_.reserve(patterns_.size());
    for (mma_utils::MatmulPattern& pattern : patterns_) {
      mma_ops_.push_back(pattern.translateToMmaOp());
    }
  }

  void buildIdModel() {
    id_model_.buildPermissiveGraph();
    ValGraph& new_graph = id_model_.idGraph(IdMappingMode::PERMISSIVE);
    graph_ = &id_model_.idGraph(IdMappingMode::PERMISSIVE);
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

    countDims();
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
  void defineOperandCaches() {
    mma_results_.reserve(mma_ops_.size());
    for (MmaOp* mma : mma_ops_) {
      mma->setMacro(params_.mma_macro);

      // Setup accumulator register.
      mma_results_.push_back(mma->out()->as<TensorView>());
    }

    LoadStoreOpType load_op = params_.async_gmem_load_operands
        ? LoadStoreOpType::CpAsync
        : LoadStoreOpType::Set;

    auto cacheOperandsToSmem = [&](const std::vector<TensorView*>& operands,
                                   std::vector<TensorView*>& smem_operands,
                                   int64_t vec_size) {
      // Use cp.async as requested in scheduler params.
      smem_operands.resize(operands.size(), nullptr);
      for (size_t i : c10::irange(operands.size())) {
        TensorView* operand = operands[i];
        CacheOp cache_op = CacheOp::Unspecified;
        if (params_.async_gmem_load_operands) {
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
          cache_op = vec_bytes == 16LL ? CacheOp::Global : CacheOp::AllLevels;
        };

        NVF_ERROR(operand->uses().size() == 1);
        smem_operands[i] = ir_utils::consumerTvsOf(operand).at(0);
        smem_operands[i]->definition()->as<LoadStoreOp>()->setOpType(load_op);
        smem_operands[i]->definition()->as<LoadStoreOp>()->setCacheOp(cache_op);
        if (smem_operands[i]->uses().size() > 1) {
          // There can be multiple uses for example if we have A @ B1 + A @ B2
          // then A will be cached to smem then it might be loaded into two
          // separate register buffers, one for each mma. Instead, we will load
          // it once into registers then re-use the register buffer for both
          // mmas.
          cacheAfter(smem_operands[i]);
        }
        NVF_ERROR(smem_operands[i]->uses().size() == 1);
        smem_operands[i]->setMemoryType(MemoryType::Shared);
      }
    };
    cacheOperandsToSmem(as_, acw_smems_, params_.supported_vec_size.a);
    cacheOperandsToSmem(bs_, bcw_smems_, params_.supported_vec_size.b);

    // We add two LoadStore operators to the inputs of our fusions. The first
    // one is for a read from global memory and the second one (below) is for a
    // cache read. As an optimizaton, we avoid adding an operator if there's an
    // existing LoadStoreOp present. Please note that for the second LoadStore
    // we don't propagate the allocation domain, since the scheduler sets the
    // allocation domain in the registers.
    auto addSetsForCacheReads = [&](const std::vector<TensorView*>& tv_smems,
                                    std::vector<TensorView*>& tv_rs) {
      tv_rs.resize(tv_smems.size(), nullptr);
      for (size_t i : c10::irange(tv_smems.size())) {
        TensorView* tv_smem = tv_smems[i];
        TensorView*& tv_r = tv_rs[i];

        if (auto ldst = dynamic_cast<LoadStoreOp*>(tv_smem->uses().at(0))) {
          tv_r = ldst->out()->as<TensorView>();
          ldst->setOpType(LoadStoreOpType::LdMatrix);
        } else {
          tv_r = cacheAfter(
              tv_smem,
              LoadStoreOpType::LdMatrix,
              CacheOp::Unspecified,
              /*propagate_allocation_domain=*/false);
        }
      }
    };
    // Shared memory read
    addSetsForCacheReads(acw_smems_, acrs_);
    addSetsForCacheReads(bcw_smems_, bcrs_);
  }

  void schedulePrologues() {
    NVF_ERROR(false, "schedulePrologues is not yet implemented");
  }

  void scheduleEpilogue() {
    NVF_ERROR(false, "scheduleEpilogue is not yet implemented");
  }

  void scheduleSplitKSum() {
    NVF_ERROR(false, "scheduleSplitKSum is not yet implemented");
  }

  void setUpInlining() {
    NVF_ERROR(false, "setUpInlining is not yet implemented");
  }

  // NOTE: this should be called after acw_smem, acr, ..., ab, and mma_result
  // transforms have been applied and inlining
  void setUpCircularBuffering() {
    NVF_ERROR(false, "setUpCircularBuffering is not yet implemented");
  }

 private:
  Fusion* fusion_;
  const MatmulParams& params_;
  IdModel id_model_;
  // Permissive graph of id_model_, which we modify at times using e.g.
  // AbstractTensor.split or by mapping vals in cacheAfter and rFactor
  ValGraph* graph_ = nullptr;
  std::vector<mma_utils::MatmulPattern> patterns_;
  std::vector<MmaOp*> mma_ops_;
  mma_utils::DimRolesMap id_roles_;
  mma_utils::TensorRolesMap tensor_roles_;
  mma_utils::MatmulOperandInnerDims inner_dims_;

  int64_t num_splitk_dims_ = 0, num_device_dims_ = 0, num_local_batch_dims_ = 0,
          num_device_and_batch_dims_ = 0;

  std::vector<std::pair<TensorView*, TensorView*>> cached_outputs_;

  std::vector<TensorView*> as_, bs_, acw_smems_, bcw_smems_, acrs_, bcrs_, abs_,
      bbs_, mma_results_, splitk_sums_, smem_epilogues_;
};

} // namespace

void scheduleMultipleMatmuls(Fusion* fusion, const MatmulParams& params) {
  FusionGuard fg(fusion);

  MultipleMatmulScheduler(fusion, params).run();
}

} // namespace nvfuser
