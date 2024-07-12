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

    swizzleBlockTiles();

    doSplitKRFactor();

    // Creates at_acw_smem_, at_bcw_smem_, and at_smem_epilogue_ all of which
    // swizzle the shared memory writes
    swizzleAllSharedMemory();

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
        cacheAfter(acw_smems_[i]);
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
        cacheAfter(bcw_smems_[i]);
      }
      NVF_ERROR(bcw_smems_[i]->uses().size() == 1);
    }

    // We add two LoadStore operators to the inputs of our fusions. The first
    // one is for a read from global memory and the second one (below) is for a
    // cache read. As an optimizaton, we avoid adding an operator if there's an
    // existing LoadStoreOp present. Please note that for the second LoadStore
    // we don't propagte the allocation domain, since the scheduler sets the
    // allocation domain in the registers.
    auto addSetForCacheRead = [&](TensorView* tv_smem, TensorView** tv_r) {
      if (auto ldst = dynamic_cast<LoadStoreOp*>(tv_smem->uses().at(0))) {
        *tv_r = ldst->out()->as<TensorView>();
        ldst->setOpType(LoadStoreOpType::LdMatrix);
      } else {
        *tv_r = cacheAfter(
            tv_smem,
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
  TensorView* rFactor(TensorView* orig, const std::vector<int64_t>& axes) {
    const std::vector<IterDomain*> orig_logical = orig->getLogicalDomain();
    const std::vector<IterDomain*> orig_loop = orig->getLoopDomain();

    TensorView* partial = orig->rFactor(axes);

    // rFactor does a replay of the loop domain in orig and changes the
    // IterType of the reduction domains that are not in "axes". All the
    // domains in partial_loop should map to orig_loop. All the domains in
    // full_loop map to those in noReductions(partial_loop);
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

    // cacheAfter replays loop transforms, so we need to map thsoe replayed
    // transforms.
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
          /*apply_swizzle=*/false);
    } else {
      at_mma_result_ = at_tiled_;
    }

    // TODO: swizzle operand smem
  }

  void applyFinalTransforms() const {
    auto forwardAndApply = [](const mma_utils::AbstractMatmulTensor& abten,
                              TensorView* tv) {
      AbstractTensor local_abten = forwardAroundMissingAxes(abten, tv);
      applyAbstractTransforms(local_abten, tv);
    };

    for (TensorView* tv : mma_results_) {
      forwardAndApply(at_mma_result_, tv);
    }

    for (TensorView* tv : smem_epilogues_) {
      forwardAndApply(at_smem_epilogue_, tv);
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
