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

// Returns true if given number is power of 2
constexpr bool isPowOf2(int64_t x) {
  return x > 1 && (x & (x - 1)) == 0;
}

// Utility to check concrete static size:
inline void checkConcreteStaticDim(const AbstractId& abs_id) {
  IterDomain* id = nullptr;
  // TODO: it might make sense to create an IterDomain*
  // AbstractId::candidateId() function
  if (abs_id.is<IterDomain*>()) {
    id = abs_id.as<IterDomain*>();
  } else if (abs_id.is<ValGroupAndItsGraph>()) {
    auto vgg = abs_id.as<ValGroupAndItsGraph>();
    id = vgg.group->front()->as<IterDomain>();
  } else {
    NVF_ERROR(false, "Could not convert AbstractId to concrete IterDomain");
  }

  NVF_ERROR(
      !id->isBroadcast() && !id->isReduction(),
      "no support for reduction or broadcast domains, but got ",
      id->toString());
  NVF_ERROR(
      id->extent()->isConstInt(),
      "swizzled dimension's extend must be known during scheduling, got ",
      id->toString());
}

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

    swizzleBlockTiles();

    doSplitKRFactor();

    // Creates at_acw_smem_, at_bcw_smem_, and at_smem_epilogue_ all of which
    // swizzle the shared memory writes
    swizzleAllSharedMemory();

    scheduleWarpTileWithReduction();

    applyFinalTransforms();

    // swizzleSharedMemory(mma_result)

    /*
    mma_utils::scheduleWarpTileWithReduction(mma_result, gemm_tile);
    scheduler_utils::BoundedDirectionalTransformPropagator::bothWays(
        mma_result, -1, {acw_smem, bcw_smem}, {smem_epilogue});

    scheduleProlog(acw_smem, params.supported_vec_size.a, params);
    scheduleProlog(bcw_smem, params.supported_vec_size.b, params);

    mma = mma_result->definition()->as<MmaOp>();
    auto ab = mma->inA()->as<TensorView>();
    auto bb = mma->inB()->as<TensorView>();

    if (isTuring(params.mma_macro) || isAmpere(params.mma_macro)) {
      moveInnerBroadcastLeft(ab);
      moveInnerBroadcastLeft(bb);
    }

    ab->applyMmaSwizzle(MmaOperand::A);
    bb->applyMmaSwizzle(MmaOperand::B);

    propagate_mma_input_schedule_to(acw_smem, bcw_smem);
    mma_result->applyMmaSwizzle(MmaOperand::Accumulator);

    if (acr != ab) {
      //  -5  -4   -3   -2   -1
      //[8mi, 4k, 2ko, 2mo, 2ki]
      acr->setAllocationDomain(acr->getLoopDomain(), true);
      mma_utils::WarpMmaSwizzler::scheduleLdMatrix(acr, MmaOperand::A);
      ab->merge(-5);
      ab->axis(-4)->parallelize(ParallelType::TIDx);
      propagate_mma_input_schedule_to(acr, nullptr);
    }
    if (bcr != bb) {
      //   -5  -4   -3   -2   -1
      // [8ni, 4k, 2ko, 1no, 2ki]
      bcr->setAllocationDomain(bcr->getLoopDomain(), true);
      mma_utils::WarpMmaSwizzler::scheduleLdMatrix(bcr, MmaOperand::B);
      bb->merge(-5);
      bb->axis(-4)->parallelize(ParallelType::TIDx);
      propagate_mma_input_schedule_to(nullptr, bcr);
    }

    if (num_splitk_dims != 0) {
      mma_result->axis(num_device_and_batch_dims + 2)
          ->parallelize(ParallelType::BIDz);
    } else if (num_local_batch_dims > 0) {
      mma_result->axis(num_device_dims)->parallelize(ParallelType::BIDz);
    }
    switch (params.cta_order) {
      case MatmulParams::TileRasterizationOrder::RowMajor:
        mma_result->axis(num_device_and_batch_dims)
            ->parallelize(ParallelType::BIDx);
        mma_result->axis(num_device_and_batch_dims + 1)
            ->parallelize(ParallelType::BIDy);
        break;
      case MatmulParams::TileRasterizationOrder::ColumnMajor:
        mma_result->axis(num_device_and_batch_dims)
            ->parallelize(ParallelType::BIDy);
        mma_result->axis(num_device_and_batch_dims + 1)
            ->parallelize(ParallelType::BIDx);
        break;
      default:
        NVF_ERROR(
            false, "Invalid TileRasterizationOrder passed to Matmul scheduler");
    }

    // parallelize Mwo, Nwo by thread
    mma_result->axis(num_device_and_batch_dims + 4 + num_splitk_dims)
        ->parallelize(ParallelType::TIDz);
    mma_result->axis(num_device_and_batch_dims + 5 + num_splitk_dims)
        ->parallelize(ParallelType::TIDy);

    scheduler_utils::parallelizeAllLike(
        mma_result,
        -1,
        {acr, bcr, ab, bb},
        {ParallelType::TIDy, ParallelType::TIDz});

    // handle epilogue and always vectorize Ki
    if (params.use_smem_epilogue) {
      smem_epilogue->setMemoryType(MemoryType::Shared);
      auto swizzled_dom = swizzleSharedMemory(smem_epilogue);
      smem_epilogue->setLoopDomain(swizzled_dom.as<IterDomain*>());
      smem_epilogue->setHasSwizzleOp();
      scheduler_utils::BoundedDirectionalTransformPropagator::forward(
          mma_result,
          -1,
          {smem_epilogue},
          scheduler_utils::BoundedDirectionalTransformPropagator::Options()
              .propagateParallelType()
              .propagateToBoundary());
      smem_epilogue->axis(-1)->parallelize(ParallelType::Vectorize);

      for (auto [dc, d] : cached_outputs) {
        // Schedule output tensor differently for better global memory access
        // pattern.
        scheduleOutputTensor(
            mma_result, d, gemm_tile, params.supported_vec_size.epilogue);
        d->axis(-1)->parallelize(ParallelType::Vectorize);

        // Propagate output tensor transformations back to smem_epilogue
        scheduler_utils::BoundedDirectionalTransformPropagator::backward(
            d, -1, {smem_epilogue});
      }
    } else {
      for (auto [dc, d] : cached_outputs) {
        scheduler_utils::BoundedDirectionalTransformPropagator::forward(
            mma_result,
            -1,
            {d},
            scheduler_utils::BoundedDirectionalTransformPropagator::Options()
                .propagateParallelType()
                .propagateToBoundary());
        // We might propagate an inner dimension that is not compatible with the
        // output or bias-like inputs. In those cases, we will further split
    this
        // dimension with an outer unrolled loop to achieve the proper
        // vectorization as specified by params.supported_vec_size.epilogue.
        NVF_ERROR(d->axis(-1)->extent()->isConst());
        int64_t d_extent = d->axis(-1)->extent()->value().as<int64_t>();
        if (d_extent > params.supported_vec_size.epilogue) {
          // Should always be a divisible split
          NVF_ERROR(d_extent % params.supported_vec_size.epilogue == 0);
          d->split(-1, params.supported_vec_size.epilogue, true);
          d->axis(-2)->parallelize(ParallelType::Unroll);
        }
        d->axis(-1)->parallelize(ParallelType::Vectorize);
      }
    }
    // propagate output transformations to all inputs that are part of epilogue
    //  operations, input tvs with non-core roles
    //  core roles: essential for matmul, for example mma inputs' producers
    if (has_non_mma_input_tvs) {
      scheduleFusionInputsForEpilogue(tensor_roles, params.use_smem_epilogue);
    }

    scheduleSplitKSum(
        splitk_sum, num_device_and_batch_dims, params.use_smem_epilogue);

    // auto inline for all tensors except register tensors
    inlineMost(ir_utils::allTvsExcept(fusion, {acr, bcr, ab, bb}));

    // if auto inline, will inline to position-7, leads to performance
    regression inlineSelectedAt( {acr, bcr, ab, bb}, mma_result,
        num_device_and_batch_dims + 6 + num_splitk_dims);

    // Propagate mma output swizzle and parallelization down the DAG
    if (params.circular_buffer_options.circular_buffer_smem_write) {
      NVF_ERROR(
          params.circular_buffer_options.smem_circular_buffer_stage > 1,
          "Invalid buffer stage config")
      if (params.circular_buffer_options.smem_circular_buffer_stage > 2) {
        NVF_ERROR(
            params.async_gmem_load_operands,
            "Circular buffer only supports async load");
      }

      acw_smem->circularBuffer(
          params.circular_buffer_options.smem_circular_buffer_stage);
      bcw_smem->circularBuffer(
          params.circular_buffer_options.smem_circular_buffer_stage);
    }

    if (params.circular_buffer_options.circular_buffer_smem_read) {
      acr->circularBuffer(2);
      bcr->circularBuffer(2);
    }

    if (params.circular_buffer_options.circular_buffer_smem_read &&
        params.circular_buffer_options.circular_buffer_smem_write) {
      // rotate Kg loop
      scheduler_utils::rotateLoop(
          mma_result,
          num_device_and_batch_dims + 2 + num_splitk_dims,
          {acr, bcr});
    }

    NVF_ERROR(!cached_outputs.empty());
    mma_utils::MmaDataTypes data_types = {
        a->dtype(), b->dtype(), mma_result->dtype()};
    // NOTE: Batch split-K matmuls cannot currently re-use smem due to outer
    // batch loop
    bool guaranteed_operand_reuse =
        num_local_batch_dims == 0 || num_splitk_dims == 0;
    int64_t estimated_smem = mma_utils::computeExpectedSharedMemoryUsage(
        params,
        data_types,
        guaranteed_operand_reuse,
        guaranteed_operand_reuse);
    fusion->setExpectedDynamicSmemBytes(estimated_smem);
    */

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

      // Setup accumulator register.
      mma_results_.push_back(mma->out()->as<TensorView>());
    }

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

    auto cacheOperandsToSmem = [&](const std::vector<TensorView*>& operands,
                                   std::vector<TensorView*>& smem_operands,
                                   CacheOp cache_op) {
      smem_operands.resize(operands.size(), nullptr);
      for (size_t i : c10::irange(operands.size())) {
        TensorView* operand = operands[i];
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
    cacheOperandsToSmem(as_, acw_smems_, cache_op_a);
    cacheOperandsToSmem(bs_, bcw_smems_, cache_op_b);

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
        bool replaced_group = false;
        for (Val* v : *vgg.group) {
          try {
            vgg.group = new_graph.toGroup(v);
          } catch (...) {
            // new_graph.toGroup() might not be able to find v. This happens
            // when we replace a domain using rFactor for example. In such
            // cases, we move on and try other IDs in the group.
            continue;
          }
          replaced_group = true;
          break;
        }
        NVF_ERROR(
            replaced_group,
            "Failed to replace group used in AbstractTensor containing ",
            vgg.group->front()->toString());
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

    graph_ = &new_id_model.idGraph(IdMappingMode::PERMISSIVE);

    // Set id_model_ after we are done using the old one
    id_model_ = std::move(new_id_model);
  }

  // Gets canonical dim ordering then uses it to canonicalize each tensor in the
  // fusion, then create tiles and swizzle their ordering.
  void makeAllTiles() {
    std::vector<ValGroup> canonical_dim_ordering =
        mma_utils::canonicalDimOrdering(tensor_roles_, id_roles_, *graph_);

    at_tiled_.domain.reserve(canonical_dim_ordering.size());
    for (const ValGroup& vg : canonical_dim_ordering) {
      at_tiled_.domain.push_back(ValGroupAndItsGraph{vg, graph_});
      // Tag each dimension with a MatmulDimRole
      auto it = id_roles_.find(vg);
      NVF_ERROR(it != id_roles_.end());
      at_tiled_.tags.push_back({it->second});
    }

    mma_utils::mergeCanonicalAbstractTensor(at_tiled_);

    mma_utils::makeTile(at_tiled_, params_.tile_sizes.cta_tile.toVector());
  }

  void swizzleBlockTiles() {
    if (params_.grid_swizzle_factor != 1) {
      // Find position of outer M and N dims in at_tiled_
      int64_t Mo_pos = -1, No_pos = -1;
      for (size_t i : c10::irange(3)) {
        if (at_tiled_.getTag((int64_t)i) == MatmulDimRole::M) {
          Mo_pos = (int64_t)i;
        } else if (at_tiled_.getTag((int64_t)i) == MatmulDimRole::N) {
          No_pos = (int64_t)i;
        }
      }
      NVF_ERROR(
          Mo_pos != -1 && No_pos != -1,
          "Could not determine outer M and N dimensions");

      int factor = std::max(1, params_.grid_swizzle_factor); // must be >=1
      switch (params_.cta_order) {
        case MatmulParams::TileRasterizationOrder::RowMajor:
          // split   [I1, I2/factor, factor]
          // reorder [I1, factor, I2/factor]
          // merge   [I1*factor, I2/factor]
          // where I1 and I2 are the outer M and N dimensions, respectively
          at_tiled_.split(No_pos, factor);
          // If No_pos < Mo_pos, then the split above shifts Mo_pos by one
          if (No_pos < Mo_pos) {
            Mo_pos++;
          }
          at_tiled_.reorder({{No_pos, No_pos + 1}});
          at_tiled_.merge(Mo_pos, No_pos);
          break;

        case MatmulParams::TileRasterizationOrder::ColumnMajor:
          // split   [I1/factor, factor, I2]
          // reorder [I1/factor, I2, factor]
          // merge   [I1/factor, I2*factor]
          // where I1 and I2 are the outer M and N dimensions, respectively
          at_tiled_.split(Mo_pos, factor);
          // If No_pos < Mo_pos, then the split above shifts Mo_pos by one
          if (No_pos > Mo_pos) {
            No_pos++;
          }
          at_tiled_.reorder({{Mo_pos + 1, No_pos}});
          at_tiled_.merge(Mo_pos + 1, No_pos);
      }
    }
  }

  void doSplitKRFactor() {
    // Find Ko dimension in at_tiled_ by looking at tags
    int64_t Ko_dim = -1;
    int64_t Ki_dim = -1;
    for (size_t dim : c10::irange(at_tiled_.size())) {
      if (at_tiled_.getTag((int64_t)dim) == MatmulDimRole::K) {
        if (Ko_dim == -1) {
          Ko_dim = (int64_t)dim;
        } else {
          NVF_ERROR(Ki_dim == -1, "Expected exactly two K dimensions");
          Ki_dim = (int64_t)dim;
        }
      }
    }
    NVF_ERROR(Ko_dim != -1, "Could not find outer K dimension");

    // Split Ko -> [rKf, rKg]
    at_tiled_.split(Ko_dim, params_.splitk_factor, /*inner*/ false);
    // After splitting Ko we have Kf_dim = Ko_dim and Kg_dim = Kf_dim + 1
    int64_t Kf_dim = Ko_dim;
    Ki_dim++;

    // We need to apply the transforms here so that we can perform rFactor
    if (params_.splitk_factor != 1) {
      applyAbstractTransforms(at_tiled_, mma_results_);
    }

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
        mma_result = rFactor(splitk_sum, {Kf_dim, Ki_dim});
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
        TensorView* smem_epilogue = cacheAfter(mma_result);
        smem_epilogue->setMemoryType(MemoryType::Shared);
        smem_epilogues_.push_back(smem_epilogue);
        // smem_epilogue = [..., iMo, iNo, iKf, iMi, iNi]
      }
    }
  }

  //! This calls orig->rFactor(axes) and also updates the permissive graph to
  //! reflect the new IterDomain mappings
  // TODO: a utility like this might be useful at the IdModel level, where it
  // could update not just the permissive map but the exact map also. But for
  // scheduling, the permissive map is probably the most useful anyway.
  TensorView* rFactor(TensorView* orig, const std::vector<int64_t>& axes) {
    const std::vector<IterDomain*> orig_logical = orig->getLogicalDomain();
    const std::vector<IterDomain*> orig_loop = orig->getLoopDomain();

    TensorView* partial = orig->rFactor(axes);

    // rFactor does a replay of the loop domain in orig and changes the
    // IterType of the reduction domains that are not in "axes". All the
    // domains in partial_loop should map to orig_loop. All the domains in
    // full_loop map to those in noReductions(partial_loop); Partial root
    // domain maps to the original logical domain.
    const std::vector<IterDomain*> full_loop = orig->getLoopDomain();
    const std::vector<IterDomain*> partial_root = partial->getMaybeRootDomain();
    const std::vector<IterDomain*> partial_loop = partial->getLoopDomain();
    const std::vector<IterDomain*> nored_partial_loop =
        TensorDomain::noReductions(partial->getLoopDomain());

    NVF_ERROR(partial_root.size() == orig_logical.size());
    NVF_ERROR(partial_loop.size() == orig_loop.size());
    NVF_ERROR(full_loop.size() == nored_partial_loop.size());

    for (size_t i : c10::irange(orig_logical.size())) {
      ValGroup vg = graph_->toGroup(orig_logical[i]);
      graph_->initializeVal(partial_root[i], vg);
    }
    for (size_t i : c10::irange(orig_loop.size())) {
      ValGroup vg = graph_->toGroup(orig_loop[i]);
      graph_->initializeVal(partial_loop[i], vg);
    }
    for (size_t i : c10::irange(full_loop.size())) {
      ValGroup vg = graph_->toGroup(nored_partial_loop[i]);
      graph_->initializeVal(full_loop[i], vg);
    }

    return partial;
  }

  //! This calls orig->cacheAfter() and also updates the permissive graph to
  //! reflect the new IterDomain mappings
  TensorView* cacheAfter(
      TensorView* orig,
      LoadStoreOpType op_type = LoadStoreOpType::Set,
      CacheOp cache_op = CacheOp::AllLevels,
      bool propagate_allocation_domain = false) {
    const std::vector<IterDomain*> orig_alloc =
        orig->getMaybeAllocationDomain();

    TensorView* c =
        orig->cacheAfter(op_type, cache_op, propagate_allocation_domain);

    if (propagate_allocation_domain) {
      const std::vector<IterDomain*> cache_alloc =
          c->getMaybeAllocationDomain();
      NVF_ERROR(orig_alloc.size() == cache_alloc.size());
      for (size_t i : c10::irange(orig_alloc.size())) {
        ValGroup vg = graph_->toGroup(orig_alloc[i]);
        graph_->initializeVal(cache_alloc[i], vg);
      }
    }

    const std::vector<IterDomain*> orig_logical =
        TensorDomain::noReductions(orig->getLogicalDomain());
    const std::vector<IterDomain*> cache_logical = c->getLogicalDomain();
    // in split-K we do rFactor which gives us a full = sum(partial)
    // where partial has root domain that matches the logical domain of the
    // original tensor. The logical domain contains Iteration transforms of the
    // Reduction axis in the original mma output.
    NVF_ERROR(orig_logical.size() == cache_logical.size());
    for (size_t i : c10::irange(orig_logical.size())) {
      ValGroup vg = graph_->toGroup(orig_logical[i]);
      graph_->initializeVal(cache_logical[i], vg);
    }

    return c;
  }

  //! Given a base AbstractMatmulTensor, perform a shared memory swizzle on it.
  //!
  //! Since some tensors might be missing in the actual tensors we plan to
  //! apply this schedule to, we cannot just look at the two innermost
  //! dimensions. Instead, the roles of those two inner dimensions should be
  //! provided, and we will find the two inner-most dimensions with those roles.
  //!
  //! apply_swizzle indicates whether we should apply AbstractTensor::swizzle.
  //! This should only be done when the tensors we will apply this schedule to
  //! reside in shared memory.
  // NOTE: legacy=false by default here since DispatchLegacySwizzle does not
  // support swizzle of two ValGroups
  template <bool legacy = false>
  mma_utils::AbstractMatmulTensor swizzleSharedMemory(
      const mma_utils::AbstractMatmulTensor& abten,
      const std::vector<MatmulDimRole>& inner_dim_roles,
      int64_t data_type_size,
      bool apply_swizzle) {
    // Find x and y dimensions
    int64_t x_dim = -1, y_dim = -1;
    for (int64_t pos = (int64_t)abten.size() - 1; pos >= 0; --pos) {
      if (std::find_if(
              inner_dim_roles.begin(),
              inner_dim_roles.end(),
              [pos, &abten](MatmulDimRole role) {
                return abten.hasTag(pos, role);
              }) != inner_dim_roles.end()) {
        if (y_dim == -1) {
          y_dim = pos;
        } else if (x_dim == -1) {
          x_dim = pos;
          break;
        }
      }
    }
    NVF_ERROR(
        x_dim != -1 && y_dim != -1,
        "Could not find inner dims with provided roles");

    mma_utils::AbstractMatmulTensor swizzle_domain = abten;

    // Check that the innermost 2 dimensions are concrete and static
    //  sized so that the swizzle function can be defined.
    checkConcreteStaticDim(swizzle_domain[x_dim]);
    checkConcreteStaticDim(swizzle_domain[y_dim]);

    // Extract the constant sizes of the swizzled tile
    auto abstractIdConstantExtent = [](const AbstractId& abs_id) {
      if (abs_id.is<IterDomain*>()) {
        return abs_id.as<IterDomain*>()->extent()->evaluate().as<int64_t>();
      } else if (abs_id.is<ValGroupAndItsGraph>()) {
        auto vgg = abs_id.as<ValGroupAndItsGraph>();
        for (Val* v : *vgg.group) {
          // Not all the IDs in this group might have the same constant extent,
          // so check them until we find one that does.
          PolymorphicValue ext = v->as<IterDomain>()->extent()->evaluate();
          if (!ext.hasValue()) {
            continue;
          }
          return ext.as<int64_t>();
        }
        NVF_ERROR(
            false, "Could not find IterDomain in group with constant extent");
      }
      NVF_ERROR(false, "Could not convert AbstractId to concrete IterDomain");
      return 0l;
    };
    const int64_t tile_size_x = abstractIdConstantExtent(swizzle_domain[x_dim]);
    const int64_t tile_size_y = abstractIdConstantExtent(swizzle_domain[y_dim]);

    // Only tested for (1) ldmatrix access with sizeof(T) == 16bit (i.e.
    // half/bfloat16) and (2) epilogue general access with sizeof(T) == 32bit
    // (i.e. float)
    NVF_ERROR(data_type_size == 2 || data_type_size == 4);

    // For main loop, ldmatrix loads a n_rows x n_cols = 8 x 8 matrix each time.
    // For epilogue, threads in a warp is organized as 8 rows x 4 columns.
    // Each thread vectorized write 2 items, so 8 items per row.
    //--0--1--2--3
    //--4--5--6--7
    //--8--9--10-11
    //--12-13-14-15
    //--16-17-18-19
    //--20-21-22-23
    //--24-25-26-27
    //--28-29-30-31
    constexpr int64_t n_rows = 8;
    constexpr int64_t n_cols = 8;

    // Column size of the tile needs to be multiples of 8 for ldmatrix to work.
    NVF_ERROR(
        tile_size_x >= n_rows && tile_size_x % n_rows == 0 &&
            tile_size_y >= n_cols && tile_size_y % n_cols == 0,
        "Prolog swizzle for ldmatrix, illegal tile size for prolog swizzle",
        tile_size_x,
        "x",
        tile_size_y);

    /* Note [How to remove bank conflict for ldmatrix?]
     *
     * **This note is interleaved with code, I suggest reading this note like
     *   reading a jupyter notebook**
     *
     * Our task is to make sure different rows does not fall into the same
     * bank of shared memory.
     *
     * Introduction to bank conflict can be found at page 54-72 of:
     * https://on-demand.gputechconf.com/gtc/2018/presentation/s81006-volta-architecture-and-performance-optimization.pdf
     *
     * When we talk about bank conflict removal, we are talking about the
     * following task:
     *   "there are 32 banks, and each bank contains one 4-byte word, we want to
     *    make sure different lanes in a warp does not access different word
     *    addresses in the same bank"
     * For example, if thread 0 is accessing word address 1, and thread 1 is
     * accessing word address 33, then these two threads will have a bank
     * conflict because they are accessing different word addresses in the same
     * bank. However, if thread 0 is accessing byte address 4 and thread 1 is
     * accessing byte address 6 then there will be no bank conflict because 4
     * and 6 both belong to word 1.
     */

    constexpr int64_t smem_bytes_per_word = 4;
    constexpr int64_t smem_banks = 32;

    /* but here, for our convenience, because ldmatrix always use vectorized
     * access of 8 items = 16 bytes = 4 words, we further group words into
     * units: we consider each 4 words as a "unit", and each 4 banks as a
     * "megabank". So we can rephrase our task as:
     *   "there are 8 megabanks, and each megabanks contains one 4-word unit, we
     *    want to make sure different lanes in a warp does not access different
     *    unit addresses in the same megabank"
     * In this terminology, matrices are in the row major format, each matrix
     * has 8 rows, and each row has exactly one unit.
     */

    constexpr int64_t items_per_unit = n_cols;
    const int64_t bytes_per_unit = items_per_unit * data_type_size;
    const int64_t words_per_unit = bytes_per_unit / smem_bytes_per_word;
    const int64_t num_megabanks = smem_banks / words_per_unit;

    /* In the following example, each CTA tile contains 2 rows and 3 colums of
     * matrices, each 8x8 size:
     *   +----------+----------+----------+
     *   | matrix 0 | matrix 1 | matrix 2 |
     *   +----------+----------+----------+
     *   | matrix 3 | matrix 4 | matrix 5 |
     *   +----------+----------+----------+
     * The addresses of different rows in the same matrix are offset by 3 units.
     * In this perspective, loading a matrix is a strided memory access with the
     * following stride (in units):
     */

    // number of units per row
    int64_t row_stride = tile_size_y / items_per_unit;

    /* So the bank conflicting problem is now converted to the following game:
     *   I have a clock that has one pointer and `num_megabanks` ticks. I start
     *   my game by making my pointer pointing to somewhere, and turn forward
     *   the pointer `n_rows` times, each time by `row_stride` ticks.
     * This problem can be well modeled by modular arithmetic in number theory
     * using the concept "integers modulo n" a.k.a. "Z/nZ"[1].
     * Take n = 6 as an example, Z/6Z only has 6 elements: 0, 1, 2, 3, 4, 5.
     * Additions and multiplications are defined in a cyclic manner:
     *   5 + 1 = 0, 5 + 2 = 1, 5 + 3 = 2, 5 + 4 = 3, ...
     *   2 * 1 = 2, 2 * 2 = 4, 2 * 3 = 0, 2 * 4 = 2, ...
     * With this definition, Z is mapped to Z/nZ naturally by i -> i % n [2]
     *
     * It worth mention that Z/nZ is a "commutative ring", that is, we can use
     * addition and multiplication rules just like using normal integers:
     *   a + b = b + a, a * (b + c) = a * b + a * c, ...
     * In short, we can reason about Z/nZ just like we are reasoning about
     * integers, except that every number is automatically "% n".
     *
     * Reference:
     * [1] https://en.wikipedia.org/wiki/Modular_arithmetic#Integers_modulo_n
     * [2] The % is under Euclidean definition, that is -1 % 6 is 5 instead of
     *     -1, see [The Mathematics of Integer Arithmetic] for more detail. But
     *     we are only interested in non-negative numbers here, so there is no
     *     need to worry about this problem
     */

    // row_stride in Z/nZ, where n is num_megabanks:
    // assert(row_stride >= 0);
    // assert(num_megabanks >= 0);
    int64_t row_stride_znz = row_stride % num_megabanks;
    /* Consider the following function in Z/nZ:
     *   f(i; init) = init + i * stride
     * where init is the initial position of the pointer in the clock when we
     * start the game, and stride is the number of ticks we move forward each
     * time, and i is the number of times we move forward. For a fixed init, we
     * abbrivate f(i; init) as f(i).
     *
     * In our problem, f(i) is the megabank of the `i`th row of the matrix, and
     * `init` is the megabank of the 0th row of the matrix.
     *
     * One very important property of f(i) is:
     * - if f(i1) == f(i2), then for every j, f(i1 + j) = f(i2 + j)
     * This property is true because:
     *   f(i1 + j) = f(i1) + j * stride = f(i2) + j * stride = f(i2 + j)
     *
     * The above property tells us, as we turn the clock forward:
     * - initially, we will go to a never-visited tick in each turn, but,
     * - at some point, we will return back to our original position, and,
     * - after we return, we start repeat the pervious pattern again and again.
     *
     * As an example, consider f(i) where init = 0, stride = 6, under Z/8Z:
     *     i  0 1 2 3 4 5 6 7
     *   f(i) 0 6 4 2 0 6 4 2
     * We can see that f(i) is repeating a pattern of four unique numbers
     * "0 6 4 2" twice. In our bank conflict problem, this means we are using 4
     * different megabanks, and we have a 2-way conflict.
     *
     * The question of interest is, does the above observation generalize? That
     * is, does f(i) always repeat a pattern of p unique numbers q times? Note
     * that p and q must satisfy p * q = n.
     *
     * The answer to the above question is: yes! Consider the following
     * equation:
     *    f(i1 + j) == f(i1)
     * We want to know what is the smallest positive number j that makes the
     * above equation true. Because this tells us in how many steps we will see
     * repeat. This equation can be simplified as:
     *   f(i1 + j) == f(i1) + j * stride == f(i1)
     *   ==> j * stride == 0
     *
     * An important tool to study this equation is multiplicative inverse:
     * https://en.wikipedia.org/wiki/Modular_multiplicative_inverse
     * A number i has multiplicative inverse `minv(i)` in Z/nZ if and only if it
     * coprime with n. `minv(i)` is the number that `i * minv(i) == 1`. So in
     * Z/nZ, the equation `ax = b` has solution `x = minv(a)*b` if a has
     * multiplicative inverse. For example, in Z/15Z, `minv(2) = 8` because
     *   (2 * 8) % 15 = 1
     *
     * stride has an multiplicative inverse if and only if stride coprime with
     * n, that is, g := gcd(stride, n) == 1. In such case, the solution to our
     * equation j * stride == 0 is j = minv(stride) * 0 = 0, that is: f(i) does
     * not repeat, that is: there is no bank conflict.
     */

    int64_t g = std::gcd(num_megabanks, row_stride_znz);
    if (g == 1) {
      return swizzle_domain; // No need to swizzle in this case.
    }

    /* For the case where stride does not coprime with n, we note that
     * j * stride == 0 in Z/nZ is equivalent to (j * stride) % n = 0 in Z. We
     * can write stride and n as:
     *   stride = s * g, n = m * g
     * According to Theorem 4.13 in [The Mathematics of Integer Arithmetic], we
     * have:
     *   (j * stride) % n = 0
     *   ==> (j * s) % m * g = 0
     *   ==> (j * s) % m = 0
     * which is equivalent to j * s == 0 in Z/mZ. Because s coprime with m, we
     * further get:
     *   j == 0 (in Z/mZ)
     * That is, j is a multiple of m in Z. So the smallest positive j that make
     * the equation hold is n / g.
     *
     * That is: f(i) always repeat a pattern of n/g unique numbers g times.
     * In other word: we are using n/g megabanks, and we have a g-way bank
     * conflict.
     *
     * Let's use the word "pattern" to refer to the set of values of `f` at
     * different `i`, that is:
     *   pattern k = { f(i; init=k) | i in Z/nZ }
     * For the example of stride = 6 under Z/8Z, we have the following patterns
     *        f(i): 01234567
     *   pattern 0: x_x_x_x_
     *   pattern 1: _x_x_x_x
     *   (x => occupied, _ => unoccupied)
     */

    int64_t repeated_pattern_size = num_megabanks / g;

    if (repeated_pattern_size >= n_rows) {
      return swizzle_domain; // No need to swizzle in this case.
    }

    /* Now we know that we have a g-way bank conflict. How do we remove this
     * bank conflict? The answer is to mix the storage of different matrices.
     * We first split the matrices along the row axis into g pieces, each piece
     * has n/g rows. With this split, each piece occupies exactly one pattern.
     * We want to use some non-traditional storage to let different pieces of
     * the same matrix to occupy different patterns.
     *
     * Because Z/nZ has n items, each pattern has n/g different items, so we
     * have in total g different patterns. We want to find the corresponding
     * `init` values of these g different patterns.
     *
     * Consider two different init values `init1` and `init2`. When do they
     * represent the same pattern? They represent the same pattern if and only
     * if `f(0; init2)` falls on the pattern of `init1`, that is, there exist an
     * i such that
     *   f(i; init1) == f(0; init2)
     * which simplifies to
     *   init1 + i * stride == init2
     *   ==> init2 - init1 == i * stride
     * What values can `i * stride` be? It can be an arbitrary multiple of g:
     * i * stride in Z/nZ is (i * stride) % n in Z. Let m = n/g, according to
     * Theorem 4.13 in [The Mathematics of Integer Arithmetic]
     *   (i * stride) % n = (i * s) % m * g
     * Because s coprime with m, we know that for an arbitrary value `j` in
     * Z/mZ, we can take `i = minv(s) * j` to make `i * s == j`.
     *
     * That said, for init values that are off by a multiple of g they
     * correspond to the same pattern, otherwise they belongs to different
     * patterns. So, we can use
     *   init = 0, 1, ..., g - 1
     * to canonically represent g patterns. Let's call the above
     * `init` values "pattern id".
     *
     * Now we have the idea about how to remove bank conflict: We can do an
     * inner split of our row dimension by `repeated_pattern_size` to get
     * (repeat, pattern), then different indices of the "repeat" dimension will
     * be using the same megabank, and different indices of the "pattern"
     * dimension will be using different megabank. We don't need to touch the
     * "pattern" dimension, but we need to play with the "repeat" dimension to
     * interleave it with matrice ids so that each matrix is distributed across
     * different banks.
     *
     * For example, if we have repeated_pattern_size = 4, we would want to do
     * something like below:
     *    +----------+----------+
     *   0|          |          |
     *   1| matrix 0 | matrix 1 |
     *   2|          |          |
     *   3|          |          |
     *    +----------+----------+
     *   4|          |          |
     *   5| matrix 1 | matrix 0 |
     *   6|          |          |
     *   7|          |          |
     *    +----------+----------+
     *
     * We can consider each repeated_pattern_size rows as a gigarow, and each
     * repeated_pattern_size megabanks as a gigabank. Note that megabank is a
     * contiguous chunk of banks, but gigabank is not contiguous. Indeed,
     * nearby megabanks in a gigabank has a distance of `g` megabanks
     */

    NVF_ERROR(
        n_rows % repeated_pattern_size == 0,
        "Can not partition matrix into megarows");
    int64_t num_gigarows = n_rows / repeated_pattern_size;
    int64_t num_gigabanks = g; // also = num_megabanks / repeated_pattern_size

    //    x    y
    //   -2   -1
    // [row, col]
    if (repeated_pattern_size > 1) {
      swizzle_domain.split(x_dim, repeated_pattern_size);
      y_dim++;
    }
    swizzle_domain.split(y_dim, n_cols);
    //       x        x+1        y       y+1
    //      -4         -3       -2        -1
    // [gigarow id, gigarow, matrix id, matrix]
    swizzle_domain.split(y_dim, num_gigabanks);
    //       x       x+1         y       y+1        y+2
    //      -5        -4        -3        -2         -1
    // [gigarow id, gigarow, y outer, gigabank id, matrix]
    // Note that megabanks inside a gigabank are not contiguous, so the gigabank
    // id is -2 instead of -3

    /* We want to evenly distribute gigarows across gigabanks, for example, if
     * we have 7 gigarows and 3 gigabanks, then we might distribute them as:
     *  +---+
     *  |x  |
     *  | x |
     *  |  x|
     *  |x  |
     *  | x |
     *  |  x|
     *  |x  |
     *  +---+
     * considering all matrices, this is a swizzle function like:
     *  +---+
     *  |012|
     *  |201|
     *  |120|
     *  |012|
     *  |201|
     *  |120|
     *  |012|
     *  +---+
     * which is a cyclic shift.
     *
     * Note that because num_gigabanks (a.k.a. g) divide num_megabanks and
     * row_stride_znz (which is row_stride % num_megabanks), g should also
     * divide row_stride, because according to the fundamental
     * division-with-remainder property (see doc/math/integer-division.md):
     *   row_stride = q * num_megabanks + row_stride_znz
     * which means, we can just consider each num_gigabanks matrices as a group,
     * and we always have complete groups (i.e. no group has less than
     * num_gigabanks matrices). Interleaving the memory of matrices within each
     * group should be enough to fully remove bank conflict.
     */

    /* To further simplify the problem, if we assume: */
    NVF_ERROR(
        num_gigarows % num_gigabanks == 0,
        "Requires non-square swizzle, which is not supported yet");
    /* Then we can partition gigarows into full waves, each wave has
     * num_gigabanks gigarows. This partition creates square dimensions, making
     * the swizzle implementation easier */

    //       x       x+1         y       y+1        y+2
    //      -5        -4        -3        -2         -1
    // [gigarow id, gigarow, y outer, gigabank id, matrix]
    int axis_of_gigarow_id = repeated_pattern_size > 1 ? x_dim : x_dim + 1;
    swizzle_domain.split(axis_of_gigarow_id, num_gigabanks);
    y_dim++;
    //      x    x+1    x+2        y       y+1        y+2
    //     -6     -5     -4       -3        -2         -1
    // [wave id, wave, gigarow, y outer, gigabank id, matrix]

    // swizzle wave with gigabank id to make threads in a wave access different
    // gigabank. Apply swizzle only when shared_mem_tv is stored in shared
    // memory.
    // TODO: This is a temporary workaround for the following issue:
    // For the mma output, we have the following schedule:
    // rFactor: [...., X, Y] -> mma-swizzle transformations -> loop
    // For epilogue smem tensor, the schedule is
    // rFactor: [...., X, Y] -> split -> [...., X1, X2, X3, Y1, Y2, Y3]
    //   -> swizzle X2, Y2 -> [...., X1, X2', X3, Y1, Y2', Y3]
    //   -> merge back -> [...., X', Y']
    //   -> mma-swizzle transformations -> loop
    // The mma-swizzle transformations for the mma output and epilogue smem
    // tensor are the same. In indexing, we do require {X, X'} and {Y, Y'} to be
    // mapped in CA map, however, we currently can not handle that. So we have
    // to do the same split and merge to the mma output without actually
    // applying the swizzle, and this check is to detect and handle this
    // specific case. We should remove this special handling when we fix our CA
    // mapping.
    if (apply_swizzle) {
      int axis_of_gigarow_id =
          repeated_pattern_size > 1 ? x_dim + 1 : x_dim + 2;
      using SwizzleTypeMaybeLegacy =
          std::conditional_t<legacy, Swizzle2DType, SwizzleType>;
      if (isPowOf2(num_gigabanks)) {
        swizzle_domain.swizzle(
            SwizzleTypeMaybeLegacy::XOR, axis_of_gigarow_id, y_dim + 1);
      } else {
        swizzle_domain.swizzle(
            SwizzleTypeMaybeLegacy::CyclicShift, axis_of_gigarow_id, y_dim + 1);
      }
    }

    if (repeated_pattern_size > 1) {
      swizzle_domain.merge(x_dim);
      //      x      x+1        y          y+1     y+2
      //     -5       -4       -3           -2      -1
      // [waves, gigarow, y outer, gigabank id, matrix]
      y_dim--;
    }
    swizzle_domain.merge(x_dim);
    y_dim--;

    //    x        y          y+1     y+2
    //   -4       -3           -2      -1
    // [wgr, y outer, gigabank id, matrix]

    // merge back tile_size_y
    swizzle_domain.merge(y_dim);
    //    x       y     y+2
    //   -3      -2      -1
    // [wgr, yo_gid, matrix]
    swizzle_domain.merge(y_dim);
    //    x        y
    //   -3       -2
    // [wgr, yo_gid_matrix]

    return swizzle_domain;
  }

  // Creates at_acw_swmem_ and at_bcw_smem_ which are used to schedule
  // acw_smems_ and bcw_smems_
  void swizzleAllSharedMemory() {
    if (params_.use_smem_epilogue) {
      // Transform mma_result through the epilogue swizzle without actually
      // swizzling the axes. This is done to enable the domains
      // are mapped between mma_result and smem_epilogue.
      at_mma_result_ = swizzleSharedMemory(
          at_tiled_,
          {MatmulDimRole::M, MatmulDimRole::N},
          dataTypeSize(mma_results_.front()->dtype()),
          /*apply_swizzle=*/false);
      // Also apply to smem_epilogue, and now apply the swizzle
      at_smem_epilogue_ = swizzleSharedMemory(
          at_tiled_,
          {MatmulDimRole::M, MatmulDimRole::N},
          dataTypeSize(smem_epilogues_.front()->dtype()),
          /*apply_swizzle=*/true);
    } else {
      at_mma_result_ = at_tiled_;
    }

    // TODO: swizzle operand smem
  }

  void scheduleWarpTileWithReduction() {
    // Assumes
    // [M, N, K]
    auto cta_tile = params_.tile_sizes.cta_tile;
    auto warp_tile = params_.tile_sizes.warp_tile;
    // auto instruction_tile = params_.tile_sizes.instruction_tile;

    // Do not split K dimension of CTA tile into multiple warp tiles
    NVF_CHECK(
        cta_tile.k == warp_tile.k,
        "CTA tile and warp tile must have same K dimension");

    // TODO: Check that has M, N, K as the inner dims
    NVF_ERROR(at_mma_result_.hasTag(-1, MatmulDimRole::K));
    NVF_ERROR(at_mma_result_.hasTag(-2, MatmulDimRole::N));
    NVF_ERROR(at_mma_result_.hasTag(-3, MatmulDimRole::M));

    /*
    mma_utils::checkDimSize(
        tv, {-3, -2, -1}, {cta_tile.m, cta_tile.n, cta_tile.k});

    //       -3   -2  -1
    //[...    M,   N,  K]
    // Distribute warp tile:
    tv->split(-3, warp_tile.m);
    tv->split(-2, warp_tile.n);

    //  -5   -4   -3   -2   -1
    // [Mwo  Mw  Nwo   Nw   K]
    tv->split(-4, instruction_tile.m);
    tv->split(-2, instruction_tile.n);
    tv->split(-1, instruction_tile.k);

    //   -8  -7 -6 -5 -4 -3  -2 -1
    // [Mwo Mw Mi Nwo Nw Ni Kwo Ki]

    tv->reorder({{-7, -5}, {-6, -3}, {-5, -6}, {-3, -2}, {-2, -8}, {-8, -7}});
    //   -8  -7 -6  -5 -4 -3 -2 -1
    // [Kwo Mwo Nwo Mw Nw Mi Ni Ki]
    */
  }

  void applyFinalTransforms() const {
    auto forwardAndApply = [](const mma_utils::AbstractMatmulTensor& abten,
                              TensorView* tv) {
      AbstractTensor local_abten = forwardAroundMissingAxes(abten, tv);
      applyAbstractTransforms(local_abten, tv);
      // TODO: look up original ValGroup for each axis in local_abten and find
      // corresponding ParallelType if there is one set and apply it here
    };

    for (TensorView* tv : mma_results_) {
      forwardAndApply(at_mma_result_, tv);
    }

    if (params_.use_smem_epilogue) {
      for (TensorView* tv : smem_epilogues_) {
        forwardAndApply(at_smem_epilogue_, tv);
      }
    }

    // TODO: all tensors
  }

 private:
  Fusion* fusion_;
  const MatmulParams& params_;
  IdModel id_model_;
  // Permissive graph of id_model_, which we modify at times using e.g.
  // AbstractTensor.split or by mapping vals in cacheAfter and rFactor
  ValGraph* graph_;
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

  mma_utils::AbstractMatmulTensor at_tiled_, at_acw_smem_, at_bcw_smem_,
      at_mma_result_, at_smem_epilogue_;
};

} // namespace

void scheduleMultipleMatmuls(Fusion* fusion, const MatmulParams& params) {
  FusionGuard fg(fusion);

  MultipleMatmulScheduler(fusion, params).run();

  // TODO: translate starting from matmul.cpp:1027
  // if (params.use_smem_epilogue) {
}

} // namespace nvfuser
