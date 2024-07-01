// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <inlining.h>
#include <instrumentation.h>
#include <ir/utils.h>
#include <multidevice/utils.h>
#include <scheduler/debug_utils.h>
#include <scheduler/matmul.h>
#include <scheduler/matmul_utils.h>
#include <scheduler/mma_utils.h>
#include <scheduler/utils.h>

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
  }

  void findPatterns() {
    std::vector<mma_utils::MatmulPattern> patterns_ =
        mma_utils::findMatmulPatterns(fusion_);
    NVF_ERROR(!patterns_.empty(), "No matmul patterns were found");
  }

  void translatePatterns() {
    mma_ops_.reserve(patterns_.size());
    for (mma_utils::MatmulPattern& pattern : patterns_) {
      mma_ops_.push_back(pattern.translateToMmaOp());
    }
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
    auto& [id_roles_, tensor_roles_] = roles_opt.value();

    mma_utils::MatmulOperandInnerDimsOpt inner_dims_opt =
        mma_utils::getOperandInnerDims(id_model_, id_roles_, tensor_roles_);
    NVF_ERROR(inner_dims_opt.isValid(), inner_dims_opt.getErrorMsg());
    inner_dims_ = inner_dims_opt.getData();

    as_ = tensor_roles_.at(MatmulRole::OPERAND_A);
    bs_ = tensor_roles_.at(MatmulRole::OPERAND_B);

    const bool has_epilogue =
        std::any_of(mma_ops_.begin(), mma_ops_.end(), [](MmaOp* mma) {
          return !mma->out()->isFusionOutput();
        });
    const bool has_fusion_c_roles =
        (0 != tensor_roles_.count(MatmulRole::EPILOGUE_INPUT));
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
  }

  // Gets canonical dim ordering then uses it to canonicalize each tensor in the
  // fusion, then create tiles and swizzle their ordering.
  void makeAllTiles() {
    canonical_dim_ordering_ = mma_utils::canonicalDimOrdering(
        tensor_roles_, id_roles_, id_model_.idGraph(IdMappingMode::PERMISSIVE));

    for (Val* v : fusion_->usedMathVals()) {
      if (auto tv = dynamic_cast<TensorView*>(v)) {
        makeTile(tv);
      }
    }
  }

  void makeTile(TensorView* tv) {
    // We record which dims the tensor has after canonicalization
    all_tv_dims_.emplace(
        tv,
        mma_utils::canonicalizeMmaTvOrdering(
            tv,
            id_model_.idGraph(IdMappingMode::PERMISSIVE),
            id_roles_,
            canonical_dim_ordering_));

    // Make a CTA tile
    // ------------------------------------------------------------------
    // Dimensions ordered as: [ (device dims), (batch dims), M, N, K ]
    auto dims_it = all_tv_dims_.find(tv);
    NVF_ERROR(dims_it != all_tv_dims_.end());
    std::vector<MatmulDomain>& tv_dims = dims_it->second;
    const auto dimPos = [&tv_dims](MatmulDomain dim) -> int64_t {
      const auto it = std::find(tv_dims.begin(), tv_dims.end(), dim);
      return it == tv_dims.end() ? -1l : std::distance(tv_dims.begin(), it);
    };
    int64_t m_pos = dimPos(MatmulDomain::M);
    int64_t n_pos = dimPos(MatmulDomain::N);
    int64_t k_pos = dimPos(MatmulDomain::K);

    const std::vector<IterDomain*> old_loop(tv->getLoopDomain());
    const std::vector<MatmulDomain> old_tv_dims(tv_dims);
    mma_utils::makeTile(
        tv, params_.tile_sizes.cta_tile.toVector(), m_pos, n_pos, k_pos);
    // Update tv_dims to reflect roles of new split axes
    tv_dims.clear();
    for (IterDomain* id : tv->getLoopDomain()) {
      auto it = std::find(old_loop.begin(), old_loop.end(), id);
      if (it == old_loop.end()) {
        // This is a new dimension that resulted from a split in makeTile
        NVF_ERROR(id->definition()->isA<Split>());
        it = std::find(
            old_loop.begin(),
            old_loop.end(),
            id->definition()->input(0)->as<TensorView>());
      }
      NVF_ERROR(it != old_loop.end());
      size_t pos = std::distance(old_loop.begin(), it);
      tv_dims.push_back(old_tv_dims[pos]);
    }

    // [..., Mo, No, Ko, Mi, Ni, Ki]
    // Swizzle block tiles:
    if (params_.grid_swizzle_factor != 1) {
      int factor = std::max(1, params_.grid_swizzle_factor); // must be >=1
      if (params_.cta_order == MatmulParams::TileRasterizationOrder::RowMajor) {
        if (n_pos != -1) {
          tv->split(n_pos, factor);
          // [I1, I2/factor, factor]
          tv->reorder({{n_pos, n_pos + 1}});
          if (m_pos > n_pos) {
            m_pos++;
          }
          if (k_pos > n_pos) {
            k_pos++;
          }
          // [I1, factor, I2/factor]
          if (m_pos != -1) {
            tv->merge(m_pos);
            // [I1*factor, I2/factor]
          }
        }
      } else if (
          params_.cta_order ==
          MatmulParams::TileRasterizationOrder::ColumnMajor) {
        if (m_pos != -1 && n_pos != -1) {
          tv->split(m_pos, factor);
          if (n_pos > m_pos) {
            n_pos++;
          }
          if (k_pos > m_pos) {
            k_pos++;
          }
          // [I1/factor, factor, I2]
          tv->reorder({{m_pos + 1, n_pos}});
          // [I1/factor, I2, factor]
          tv->merge(m_pos + 1, n_pos);
          // [I1/factor, I2*factor]
        }
      }
    }

    // [..., iMo, iNo, rKo, iMi, iNi, rKi]
    if (params_.splitk_factor != 1 && k_pos != -1) {
      // Split the outer K axis by splitk_factor
      // Split Ko -> [rKf, rKg]
      tv->split(k_pos, params_.splitk_factor, /*inner*/ false);
      // After split [..., iMo, iNo, rKf, rKg, iMi, iNi, rKi]
    }
  }

  void doSplitKRFactor() {
    // rFactor to define splitk_sum
    splitk_sums_.resize(mma_results_.size(), nullptr);
    smem_epilogues_.resize(mma_results_.size(), nullptr);
    for (size_t i : c10::irange(mma_results_.size())) {
      TensorView* mma_result = mma_results_[i];
      // We will assign these in this loop
      TensorView*& smem_epilogue = smem_epilogues_[i];
      TensorView*& splitk_sum = splitk_sums_[i];

      if (params_.splitk_factor != 1) {
        // rFactor converts
        //   mma_result = mma(A, B, {/*Kf*/-5, /*Kg*/-4, /*Ki*/-1});
        // to
        //   intermediate = mma(A, B, {-4, -1});
        //   final_sum = sum(intermediate, {/*Kf*/-3});
        // and the method returns "intermediate". We need mma_result to refer to
        // the actual MmaOp output, so here we reassign that to the
        // intermediate.
        splitk_sum = mma_result;
        mma_result = splitk_sum->rFactor({-4, -1});
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
        smem_epilogue = mma_result->cacheAfter();
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
  std::vector<ValGroup> canonical_dim_ordering_;
  // Track the role of each axis for each tensor in the Fusion
  std::unordered_map<TensorView*, std::vector<MatmulDomain>> all_tv_dims_;
};

} // namespace

void scheduleMatmul(Fusion* fusion, const MatmulParams& params) {
  FusionGuard fg(fusion);

  MultipleMatmulScheduler(fusion, params).run();

  // Unswizzle mma result in shared memory
  // Note that if we are using split-K, we will set up this buffer after
  // rfactoring the matmul, between the MmaOp and the ReductionOp, in order to
  // take advantage of unswizzling during the grid reduction
  TensorView* smem_epilogue = mma_result;

  // Swizzle block tiles:
  if (params.grid_swizzle_factor != 1) {
    int factor = std::max(1, params.grid_swizzle_factor); // must be >=1
    if (params.cta_order == MatmulParams::TileRasterizationOrder::RowMajor) {
      mma_result->split(num_device_and_batch_dims + 1, factor);
      // [I1, I2/factor, factor]
      mma_result->reorder(
          {{num_device_and_batch_dims + 1, num_device_and_batch_dims + 2}});
      // [I1, factor, I2/factor]
      mma_result->merge(num_device_and_batch_dims);
      // [I1*factor, I2/factor]
    } else if (
        params.cta_order == MatmulParams::TileRasterizationOrder::ColumnMajor) {
      mma_result->split(num_device_and_batch_dims, factor);
      // [I1/factor, factor, I2]
      mma_result->reorder(
          {{num_device_and_batch_dims + 1, num_device_and_batch_dims + 2}});
      // [I1/factor, I2, factor]
      mma_result->merge(num_device_and_batch_dims + 1);
      // [I1/factor, I2*factor]
    }
  }

  // [..., iMo, iNo, rKo, iMi, iNi, rKi]
  int num_splitk_dims = 0;
  TensorView* splitk_sum = nullptr;
  if (params.splitk_factor != 1) {
    // Split Ko -> [rKf, rKg]
    mma_result->split(-4, params.splitk_factor, /*inner*/ false);
    // After split [..., iMo, iNo, rKf, rKg, iMi, iNi, rKi]
    // rFactor converts
    //   mma_result = mma(A, B, {/*Kf*/-5, /*Kg*/-4, /*Ki*/-1});
    // to
    //   intermediate = mma(A, B, {-4, -1});
    //   final_sum = sum(intermediate, {/*Kf*/-3});
    // and the method returns "intermediate". We need mma_result to refer to
    // the actual MmaOp output, so here we reassign that to the intermediate.
    splitk_sum = mma_result;
    mma_result = splitk_sum->rFactor({-4, -1});

    num_splitk_dims = 1;
  }

  // At this point we have the following schedule:
  //   No split-K
  //     mma_result      [..., iMo, iNo, rKo, iMi, iNi, rKi]
  //   Split-K
  //     mma_result      [..., iMo, iNo, iKf, rKg, iMi, iNi, rKi]
  //     splitk_sum      [..., iMo, iNo, rKf, iMi, iNi]

  if (params.use_smem_epilogue) {
    // Note that for split-K
    //   splitk_sum = sum(mma_result)
    // becomes
    //   smem_epilogue = set(mma_result)
    //   splitk_sum = sum(smem_epilogue)
    smem_epilogue = mma_result->cacheAfter();
    // smem_epilogue = [..., iMo, iNo, iKf, iMi, iNi]
  }

  // Propagate tiling globally
  scheduler_utils::transformPropagateToAllFrom(mma_result, -1);

  if (params.use_smem_epilogue) {
    // Transform mma_result through the epilogue swizzle without actually
    // swizzling the axes. This is done to enable the domains
    // are mapped between mma_result and smem_epilogue.
    auto swizzled_dom = swizzleSharedMemory(mma_result);
    mma_result->setLoopDomain(swizzled_dom.as<IterDomain*>());
  }

  // Schedule warp tile
  // Incoming mma_result = [... iMo iNo (iKf) rKg iMi iNi rKi]
  mma_utils::scheduleWarpTileWithReduction(mma_result, gemm_tile);
  // After scheduling warp tile, the last three dimensions are split and
  // rearranged:
  //        -3 -2 -1
  //   [...  M  N  K]
  // maps to
  //         -8  -7 -6  -5 -4 -3 -2 -1
  //   [... Kwo Mwo Nwo Mw Nw Mi Ni Ki]
  // so now
  //                   -12 -11  -10   -9   -8   -7   -6  -5  -4   -3   -2   -1
  // mma_result = [... iMo iNo (iKf) rKg rKwo iMwo iNwo iMw iNw iMin iNin rKin]
  // splitk_sum = [... iMo iNo  rKf  iMi  iNi]

  // Propagate warp tile to main loop and epilog/output tvs
  scheduler_utils::BoundedDirectionalTransformPropagator::bothWays(
      mma_result, -1, {acw_smem, bcw_smem}, {smem_epilogue});

  // No (cross-CTA) split-K
  //   mma_result      [..., iMo iNo rKo rKwo iMwo iNwo iMw iNw iMin iNin rKin]
  //   smem_epilogue   (unscheduled, same as original or current mma_result)
  //   splitk_sum      (nullptr)
  //
  // With split-K
  //   mma_result   [... iMo iNo iKf  rKg rKwo iMwo iNwo iMw iNw iMin iNin rKin]
  //   splitk_sum   [... iMo iNo rKf  iMi  iNi]

  // Schedule prolog:
  //   TODO: this section needs more configurability.
  // ------------------------------------------------------------------
  scheduleProlog(acw_smem, params.supported_vec_size.a, params);
  scheduleProlog(bcw_smem, params.supported_vec_size.b, params);

  // Get the input to the mma op.
  mma = mma_result->definition()->as<MmaOp>();
  auto ab = mma->inA()->as<TensorView>();
  auto bb = mma->inB()->as<TensorView>();

  // Add mma swizzle:
  //   TODO: this section goes to a separate matmul util,
  //   and needs more configurability.
  // ------------------------------------------------------------------
  if (isTuring(params.mma_macro) || isAmpere(params.mma_macro)) {
    moveInnerBroadcastLeft(ab);
    moveInnerBroadcastLeft(bb);
  }

  ab->applyMmaSwizzle(MmaOperand::A);
  bb->applyMmaSwizzle(MmaOperand::B);

  // Propagate mma input swizzle up the DAG
  //  to all the tensors before mma op and after shared mem read.
  auto propagate_mma_input_schedule_to = [&](TensorView* a_boundary,
                                             TensorView* b_boundary) {
    if (a_boundary != nullptr) {
      scheduler_utils::BoundedDirectionalTransformPropagator::backward(
          ab,
          -1,
          {a_boundary},
          scheduler_utils::BoundedDirectionalTransformPropagator::Options()
              .propagateParallelType());
    }
    if (b_boundary != nullptr) {
      scheduler_utils::BoundedDirectionalTransformPropagator::backward(
          bb,
          -1,
          {b_boundary},
          scheduler_utils::BoundedDirectionalTransformPropagator::Options()
              .propagateParallelType());
    }
  };
  propagate_mma_input_schedule_to(acw_smem, bcw_smem);

  // This does a split-reorder-merge swizzle of the last two M and N dimensions
  // (and a possible final reduction dim).
  // eg. [M64, N24, R]  -> [WarpGroup128, N3, M2, N2, Ro, R4, R2]
  // Before
  //   mma_result  [... iMo iNo (iKf) rKg rKwo iMwo iNwo iMw iNw iMin iNin rKin]
  // After
  //   mma_result  [... iMo iNo (iKf) rKg rKwo iMwo iNwo iMw
  //                              iNw iMino iNino iMin2 iNin2 rKino rKin4 rKin2]
  mma_result->applyMmaSwizzle(MmaOperand::Accumulator);

  // Set parallelization:
  //   TODO: this section goes to a separate matmul util,
  //   and needs more configurability.
  // ------------------------------------------------------------------

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

  // Parallelization strategy:
  // Here the top two rows indicate how we can index each axis. The third row
  // is what it represents: note that a suffix i means inner and o means outer
  // here. The fourth row is the parallelization strategy:
  //   - i means iterate (produce one value per element i.e. don't reduce)
  //   - r means reduce this dimension
  //   - B: block
  //   - T: thread
  //   - S: serial. This will become a for loop in the generated kernel
  //   - iMMA: uncontracted axis in an MMA tensor core operation.
  //   - rMMA: contract in an MMA tensor core operation.
  //
  // With split-K:
  //   mma_result
  //     nbatch +   1    2    3    4    5    6   7   8
  //              -15  -14  -13  -12  -11  -10  -9  -8
  //     [... iMo iNo (iKf) rKg rKwo iMwo iNwo iMw iNw     ...
  //          iBx iBy  iBz   rS   rS  iTz  iTy  iS  iS
  //                              9    10    11    12    13    14    15
  //                             -7    -6    -5    -4    -3    -2    -1
  //                    ...   iMino iNino iMin2 iNin2 rKino rKin4 rKin2]
  //                            iTx  iMMA  iMMA  iMMA  rMMA  rMMA  rMMA
  //   smem_epilogue   (unscheduled, same as original mma_result)
  //   splitk_sum      (nullptr)
  //
  // Without split-K:
  //   mma_result
  //     nbatch +   1   2    3    4    5   6   7    8
  //              -14 -13  -12  -11  -10  -9  -8   -7
  //     [... iMo iNo rKg rKwo iMwo iNwo iMw iNw iMino
  //    (iBz) iBx iBy  rS   rS  iTz  iTy  iS  iS  iTx
  //                                   9    10    11     12    13    14
  //                                  -6    -5    -4     -3    -2    -1
  //                               iNino iMin2 iNin2  rKino rKin4 rKin2]
  //                                iMMA  iMMA  iMMA   rMMA  rMMA  rMMA
  //   smem_epilogue   (unscheduled, same as original mma_result)
  //   splitk_sum
  //     [... iMo iNo rKf  iMi  iNi]

  // When we have both batch dims and splitk, parallelize splitk only.
  // If we only have batch dim, parallelize the batch dim.
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
      // output or bias-like inputs. In those cases, we will further split this
      // dimension with an outer unrolled loop to achieve the proper
      // vectorization as specified by params.supported_vec_size.epilogue.
      NVF_ERROR(d->axis(-1)->extent()->isConst());
      int64_t d_extent = d->axis(-1)->extent()->value().as<int64_t>();
      if (d_extent > params.supported_vec_size.epilogue) {
        // Should always be a divisible split
        NVF_ERROR(d_extent % params.supported_vec_size.epilogue == 0);
        d->split(-1, params.supported_vec_size.epilogue, /*inner_split=*/true);
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

  // if auto inline, will inline to position-7, leads to performance regression
  inlineSelectedAt(
      {acr, bcr, ab, bb},
      mma_result,
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
    acr->circularBuffer(/*number_of_stages=*/2);
    bcr->circularBuffer(/*number_of_stages=*/2);
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
      /*smem_a_reuse_guaranteed=*/guaranteed_operand_reuse,
      /*smem_b_reuse_guaranteed=*/guaranteed_operand_reuse);
  fusion->setExpectedDynamicSmemBytes(estimated_smem);
}

} // namespace nvfuser
