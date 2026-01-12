// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <complex>
#include <limits>
#include <sstream>

#include <exceptions.h>
#include <fusion.h>
#include <ir/builder_passkey.h>
#include <ir/internal_base_nodes.h>
#include <ir/internal_nodes.h>
#include <mma_type.h>
#include <multidevice/device_mesh.h>
#include <type.h>
#include <visibility.h>

//! Nodes in here are intended to be "user facing" users in this sense being
//! those that want to be able to generate CUDA code.

//! IR header hierarchy
//! 1. utils.h - PolymorphicBase and NonCopyable
//! 2. ir/base_nodes.h - Statement, Expr, and Val
//! 3. ir/internal_base_nodes.h - IterDomain and TensorDomain
//! 4. ** ir/interface_nodes.h ** - TensorView and Scalar
//! 5. ir/internal_nodes.h - Any internal-only IR nodes

namespace nvfuser {

class ViewTransform;

class IrCloner;

namespace ir_utils {
std::string varName(const Val* val);
}

template <typename T>
T& Expr::attribute(size_t index) const {
  if constexpr (PolymorphicValue::is_candidate_type<T>) {
    return attributeVal(index)->value().as<T>();
  } else {
    return attributeVal(index)->value().as<Opaque>().as<T>();
  }
}

//! Mode during propagation of computeAt, standard will throw an error if
//! computeAt position provided can't be satisfied, best effort will lower the
//! computeAt position as needed during traversal, most inlined will increase
//! the compute at position to maximum possible through traversal.
enum class ComputeAtMode { Standard, BestEffort, MostInlined };

class TransformPropagator;
struct MostInlinedTransformPropagator;
class TransformIter;
class TransformReplay;
class OptOutMutator;
class TensorDomain;

class MaxPosCalculator;

namespace ir_utils {
class TVDomainGuard;
}

// [Circular buffering]
//
// A non-circle-buffered loop looks like below (assuming both the load and the
// compute are async ops):
//   for i in range(data.size):
//     load data[i] to buffer
//     wait buffer to be ready (RAW sync)
//     compute buffer
//     wait compute to be done (WAR sync)
//
// Circular buffering allows removing RAW and WAR hazards to maximize
// overlapping of memory load and compute. Both the load and compute operations
// are pipelined. In order to pipeline the load operations, the RAW hazards need
// to be removed, so that at every iteration, the data needed for computation is
// already prefetched a few iterations ago. In order to pipeline the compute,
// WAR hazards need to be removed, so that each iterations's compute is not
// required to be completed immediately in this iteration to avoid next
// iteration's load overwriting the current iteration's operand for the compute
// operation.
//
// With circular buffering, we want to prefetch a few iterations ahead of the
// compute, and defer the load to the just-used buffer a few iterations, so that
// both the load and the compute can be pipelined, minimizing the idle time.
//
// Circular buffering is controlled by two parameters: stage and prefetch. The
// stage parameter determines the size of the circular buffer, and the prefetch
// parameter determines how many iterations ahead of the compute the data is
// prefetched. Note that prefetch must be < stage. Both the removal of RAW and
// WAR hazards require additional storage space. The prefetch parameter
// determines how buffers are partitioned between RAW and WAR hazards. If we are
// not interested in pipelining the compute, then use prefetch = stage - 1, so
// that all buffers are used for RAW removal.
//
// The figure below illustrates the timeline of a circular buffered loop, where
// each row represents an iteration:

// clang-format off

//
//                 /load 0;\                 \.
//                / load 1; [prefetch = 3]    | [prefetching]
//          [stage] load 2;/                 /'
//          [ = 6 ] load 3;  wait load 0;  compute 0;                  \.
//                \ load 4;  wait load 1;  compute 1;                   |
//                 \load 5;  wait load 2;  compute 2;  wait compute 0;  |
//                  load 0;  wait load 3;  compute 3;  wait compute 1;  |
//                  load 1;  wait load 4;  compute 4;  wait compute 2;  |
//                  load 2;  wait load 5;  compute 5;  wait compute 3;  |
//                  load 3;  wait load 0;  compute 0;  wait compute 4;  |
//                  load 4;  wait load 1;  compute 1;  wait compute 5;  | [main]
//                  load 5;  wait load 2;  compute 2;  wait compute 0;  |
//                  ..................................................  |
//                  ..................................................  |
//                  ..................................................  |
//                  load  ;  wait load  ;  compute  ;  wait compute  ;  |
//                  load  ;  wait load  ;  compute  ;  wait compute  ;  |
//                  load  ;  wait load  ;  compute  ;  wait compute  ;  |
//                  load  ;  wait load  ;  compute  ;                  /'
//                          /wait load  ;  compute  ;                      \.
// [same number as prefetch] wait load  ;  compute  ;                       | [draining]
//                          \wait load  ;  compute  ;  wait all computes;  /'

// clang-format on

// In the above figure, we have:
// storage required = stage * tile_size
// load pipeline depth = prefetch + 1
// compute pipeline depth = stage - prefetch
//
// There are two ways to implement the above timeline: pipelined, and
// warp-specialization.
//
// In the pipelined way, the prefetching stage is implemented as a prologue
// loop, and main stage is implemented as a main loop, and the draining stage is
// implemented as an epilogue loop. That is, we will have the following loop
// structure:
//
// Prologue loop:
//   for i in range(prefetch):
//     load data[i] to buffer[i]
//
// Main loop (using syncthreads to avoid WAR harzard):
//   for i in range(data.size - prefetch):
//     load data[i + prefetch] to buffer[(i + prefetch) % stage]
//     wait buffer[i % stage] to be loaded
//     compute buffer[i % stage]
//     wait until the first compute in the queue is done
//       (i.e. stage - prefetch - 1 in flight computes remaining)
//     __syncthreads();
//
// Main loop (using mbarrier to avoid WAR harzard):
//   for i in range(data.size - prefetch):
//     wait buffer[(i + prefetch) % stage] to be empty
//     load data[i + prefetch] to buffer[(i + prefetch) % stage]
//     wait buffer[i % stage] to be loaded
//     compute buffer[i % stage]
//     wait until the first compute in the queue is done
//       (i.e. stage - prefetch - 1 in flight computes remaining)
//     signal that buffer (i + prefetch + 1) % stage is empty and ready to be
//       loaded again
//
// Epilogue loop:
//   for i in range(data.size - prefetch, data.size):
//     wait buffer[i % stage] to be ready
//     compute buffer[i % stage]
//   wait until all computes are done
//
// Note that in the above loop structure, the "wait compute" in the first
// stage - prefetch - 1 iterations and last iteration of the main loop is
// redundant. We can remove them to further optimize the performance, but
// we decide to keep them for simplicity.
//
// In the warp-specialized approach, we will use different warp/warp-group
// for loading and computing. We will generate code like below (assuming warp
// specialized on TIDy):
//
//   if (threadIdx.y == blockDim.y - 1) {
//     // If we use warp specialization on TIDy, then the blockDim.y of the
//     // kernel will be (whatever_value_inferred_from_schedule + 1), and the
//     // last threadIdx.y will be used as async warp
//     for i in range(data.size):
//       wait buffer[i % stage] to be empty
//       load data[i] to buffer[i % stage]
//   } else {
//     // Every threadIdx.y other than the last will be used for compute
//     for i in range(prefetch + 1):
//       signal that buffer i % stage is empty and ready to load
//     for i in range(data.size):
//       wait buffer[i % stage] to be loaded
//       compute buffer[i % stage]
//       wait until the first compute in the queue is done
//         (i.e. stage - prefetch - 1 in flight computes remaining)
//       signal that buffer (i + prefetch + 1) % stage is empty and ready to be
//         loaded again
//   }

struct Pipelined {
  bool uses_mbarrier_for_war = false;
  explicit Pipelined(bool uses_mbarrier_for_war)
      : uses_mbarrier_for_war(uses_mbarrier_for_war) {}
  Pipelined() = default;
  bool operator==(const Pipelined& other) const {
    return uses_mbarrier_for_war == other.uses_mbarrier_for_war;
  }
};

inline std::ostream& operator<<(std::ostream& os, const Pipelined& pipelined) {
  if (pipelined.uses_mbarrier_for_war) {
    return os << "PipelinedMBarrierForWAR";
  }
  return os << "Pipelined";
}

struct WarpSpecialized {
  ParallelType on = ParallelType::Serial;
  // The number of registers for load and compute warps respectively.
  std::optional<std::pair<int64_t, int64_t>> num_registers = std::nullopt;
  // The iterDomain position to define the shape of the circular buffer stage.
  std::optional<int64_t> stage_slice_position = std::nullopt;

  explicit WarpSpecialized(
      ParallelType on,
      std::pair<int64_t, int64_t> num_registers,
      int64_t stage_slice_position)
      : on(on),
        num_registers(num_registers),
        stage_slice_position(stage_slice_position) {
    validateRegisterSharing();
  }
  explicit WarpSpecialized(ParallelType on, int64_t stage_slice_position)
      : on(on), stage_slice_position(stage_slice_position) {}
  explicit WarpSpecialized(
      ParallelType on,
      std::pair<int64_t, int64_t> num_registers)
      : on(on), num_registers(num_registers) {
    validateRegisterSharing();
  }
  explicit WarpSpecialized(ParallelType on)
      : on(on), num_registers(std::nullopt) {}
  WarpSpecialized() = default;

  void validateRegisterSharing() {
    // short-circuit: register sharing is not used.
    if (!num_registers.has_value()) {
      return;
    }
    auto validate_num_registers = [](int64_t a) {
      NVF_ERROR(
          a >= 24 && a <= 256 && a % 8 == 0,
          "The number of registers for setmaxnreg must be between 24 and",
          " 256 (inclusive) and be a multiple of 8.");
    };
    validate_num_registers(num_registers.value().first);
    validate_num_registers(num_registers.value().second);
    NVF_ERROR(
        num_registers.value().first <= num_registers.value().second,
        "The number of registers for async warp group must be <= to the number",
        " of registers for the compute warp groups.");
  }

  bool operator==(const WarpSpecialized& other) const {
    return on == other.on && num_registers == other.num_registers &&
        stage_slice_position == other.stage_slice_position;
  }
};

inline std::ostream& operator<<(
    std::ostream& os,
    const WarpSpecialized& warp_specialized) {
  std::string parallel_type_str = "";
  switch (warp_specialized.on) {
    case ParallelType::TIDx:
      parallel_type_str = "TIDx";
      break;
    case ParallelType::TIDy:
      parallel_type_str = "TIDy";
      break;
    case ParallelType::TIDz:
      parallel_type_str = "TIDz";
      break;
    default:
      NVF_THROW("Invalid parallel type");
  }
  std::string num_registers = "RegisterSharing_None";
  if (warp_specialized.num_registers.has_value()) {
    auto&& [decrease_num_reg, increase_num_reg] =
        warp_specialized.num_registers.value();
    std::stringstream s;
    s << "RegisterSharing_" << decrease_num_reg << "_" << increase_num_reg;
    num_registers = s.str();
  }
  std::string slice_position = "StageSlicePosition_None";
  if (warp_specialized.stage_slice_position.has_value()) {
    std::stringstream s;
    s << "StageSlicePosition_" << warp_specialized.stage_slice_position.value();
    slice_position = s.str();
  }
  return os << "WarpSpecializedOn" << parallel_type_str << num_registers
            << slice_position;
}

using CircularBufferType = std::variant<Pipelined, WarpSpecialized>;

inline std::ostream& operator<<(
    std::ostream& os,
    const CircularBufferType& type) {
  return std::visit(
      [&os](const auto& t) -> std::ostream& { return os << t; }, type);
}

struct CircularBufferOptions {
  CircularBufferType type =
      Pipelined(false); // Type of circular buffer. Currently supports:
                        // - pipelined using syncthreads for WAR hazards
                        // - pipelined using mbarrier for WAR hazards.
  int64_t stage = 0; // Size of the circular buffer (number of buffers)
  int64_t prefetch = 0; // Number of iterations ahead of the compute to
                        // prefetch, can only be < stage.

  bool isEnable() const {
    return stage > 1;
  }

  bool usesMBarrierForWAR() const {
    return (std::holds_alternative<Pipelined>(type) &&
            std::get<Pipelined>(type).uses_mbarrier_for_war) ||
        std::holds_alternative<WarpSpecialized>(type);
    return false;
  }

  bool operator==(const CircularBufferOptions& other) const {
    return type == other.type && stage == other.stage &&
        prefetch == other.prefetch;
  }
};

inline std::ostream& operator<<(
    std::ostream& os,
    const CircularBufferOptions& options) {
  return os << "CircularBufferOptions{ stage=" << options.stage
            << ", prefetch=" << options.prefetch << ", type=" << options.type
            << " }";
}

//! TensorView is our primitive Tensor Type used in code generation. It can be
//! thought of as representing physical memory, however, its dimensionality is
//! modifed as split/merge/computeAt functions are called. The history of
//! these transformations are kept and used for generating actual code
//! referencing physical memory. Generally when users are thinking of code
//! generation in reference to a Tensor, this is the class they should be
//! interacting with.
//!
//! The reason we need both TensorView and TensorDomain is that we need to have
//! a record of both what is being computed and how it is being computed. For
//! example we may have the operation:
//!
//!   TV3[I, J, K] = TV2[I, J, K] + TV1[I, J, K]
//!
//! The mathematical operations here are on the tensor views TV1, TV2, and
//! TV3. This operation is a pointwise operation. To compute this pointwise
//! operation we iterate over the 3D TensorDomain [I, J, K], where K is the
//! fastest changing dimension.
//!
//! \todo Need to work on the const model for TensorView, making all functions
//! that should be const, const. Gave this a try but expanded really quickly.
//! getComputeAtAxis not being const because it can return a TV that some expect
//! to be non-const is the biggest headache.
//!
class NVF_API TensorView : public Val {
 public:
  TensorView(
      IrBuilderPasskey passkey,
      TensorDomain* domain,
      DataType dtype,
      MemoryType mtype = MemoryType::Local);

  TensorView(const TensorView* src, IrCloner* ir_cloner);

  NVFUSER_DECLARE_CLONE

  bool sameDefinition(const Val* other) const override;

  std::string toString(int indent_size = 0) const override;

  std::string toInlineString(int indent_size = 0) const override;

  void printTransforms() const;

  TensorDomain* domain() const {
    return domain_;
  }

  void setContiguity(const std::vector<std::optional<bool>>& contig) {
    domain()->setContiguity(contig);
  }

  void setContiguity(bool contig) {
    setContiguity(TensorDomain::getContiguityFilledWith(
        getMaybeAllocationDomain(), contig));
  }

  const std::vector<std::optional<bool>>& getContiguity() const {
    return domain()->contiguity();
  }

  bool hasReduction() const {
    return domain()->hasReduction();
  }

  bool hasBlockReduction() const {
    return domain()->hasBlockReduction();
  }

  bool hasGridReduction() const {
    return domain()->hasGridReduction();
  }

  bool hasClusterReduction() const {
    return domain()->hasClusterReduction();
  }

  bool hasBroadcast() const {
    return domain()->hasBroadcast();
  }

  bool hasRoot() const {
    return domain()->hasRoot();
  }

  bool hasAllocation() const {
    return domain()->hasAllocation();
  }

  //! Returns true if this tensor is zero dimensional,
  //!  i.e. a wrapped scalar or an empty placeholder.
  bool isZeroDim() const {
    return nDims() == 0;
  }

  //! Returns true if this tensor does not contain
  //!  any value.
  bool isEmptyTensor() const;

  std::optional<int64_t> getReductionAxis() const {
    return domain()->getReductionAxis();
  }

  const std::vector<IterDomain*>& getRootDomain() const {
    return domain()->root();
  };

  const std::vector<IterDomain*>& getMaybeRootDomain() const {
    return domain()->maybeRoot();
  };

  const std::vector<IterDomain*>& getLogicalDomain() const {
    return domain()->logical();
  };

  const std::vector<IterDomain*>& getAllocationDomain() const {
    return domain()->allocation();
  };

  const std::vector<IterDomain*>& getLoopDomain() const {
    return domain()->loop();
  };

  const std::optional<std::vector<IterDomain*>>& getAlternateLoopDomain()
      const {
    return domain()->alternateLoop();
  };

  const std::vector<IterDomain*>& getInitialLoopDomain() const {
    return domain()->initialLoop();
  };

  // If allocation domain exists in domain() return it, otherwise return
  // logical domain
  const std::vector<IterDomain*>& getMaybeAllocationDomain() const {
    return domain()->maybeAllocation();
  };

  void setLoopDomain(std::vector<IterDomain*> new_loop_domain) {
    domain()->setLoopDomain(std::move(new_loop_domain));
  }

  void setAlternateLoopDomain(std::vector<IterDomain*> new_loop_domain) {
    domain()->setAlternateLoopDomain(std::move(new_loop_domain));
  }

  void setAllocationDomain(
      std::vector<IterDomain*> new_allocation_domain,
      std::vector<std::optional<bool>> new_contiguity) {
    domain()->setAllocationDomain(
        std::move(new_allocation_domain), std::move(new_contiguity));
  }

  void setAllocationDomain(
      std::vector<IterDomain*> new_allocation_domain,
      bool new_contiguity) {
    domain()->setAllocationDomain(
        std::move(new_allocation_domain), new_contiguity);
  }

  IterDomain* axis(int64_t pos) const;

  // Does it share outer axes with other tensors?
  bool hasComputeAt() const {
    return compute_at_pos_ > 0;
  }

  bool hasMaxProducerPosition() const {
    return max_producer_pos_ > 0;
  }

  int64_t nDims() const {
    return domain()->nDims();
  }

  // sets cpu_scalar_ value, which is special handling for CPU based zero-dim
  // tensors (i.e. CPU Tensors that only have one value). This is only used if
  // on an input value, otherwise ignored. This is important as special handling
  // because these "scalars" should be type promoted as a tensor, but we want to
  // avoid explicit copying of the data, so we want to pass the data value as a
  // standard kernel argument value.
  void setCpuScalar(bool is_cpu_scalar);

  // returns cpu_scalar_ value, which is special handling for CPU based zero-dim
  // tensors (i.e. CPU Tensors that only have one value). This is only used if
  // on an input value, otherwise ignored. This is important as special handling
  // because these "scalars" should be type promoted as a tensor, but we want to
  // avoid explicit copying of the data, so we want to pass the data value as a
  // standard kernel argument value.
  bool isCpuScalar() const {
    return cpu_scalar_;
  }

  // Returns the position that this tensor is produced at relative to its axes.
  int64_t getComputeAtPosition() const {
    return compute_at_pos_;
  }

  // Returns the maximum position of producers are being computed at relative to
  // this tensor. This position dictates the clear expectations of producers.
  int64_t getMaxProducerPosition() const {
    return max_producer_pos_;
  }

  int64_t getMaybeMaxProducerPosition() const {
    return maybe_max_producer_pos_;
  }

  //! This is used when we disconnect a tensorview from a reduction
  //!  operation and connect it to a non-reduction operator. We need
  //!  to remove the reduction ids on the tv in this case.
  //! Currently only used in translate welford, and this function may
  //!  be refactored or extended if any more use cases appear.
  void clearReductionIterDomains();

  //! Compute this TensorView relative to a consumer position, -1 will
  //! compute tensors inline with each other, 0 doesn't share
  //! any loop nests between the tensors. It's an error when the given
  //! position is not legally viable. Alternatively, when the mode
  //! parameter is ComputeAtMode::BestEffort, the position is lowered
  //! one by one until a valid position is found. When
  //! ComputeAtMode::MostInlined is given, the position parameter is
  //! ignored, and the deepest possible position is searched.
  TensorView* computeAt(
      TensorView* consumer,
      int64_t position,
      ComputeAtMode mode = ComputeAtMode::Standard);

  //! Create a new broadcast IterDomain with the given extent in the loop domain
  TensorView* broadcast(int64_t axis, int64_t extent = 1);
  TensorView* broadcast(int64_t axis, Val* extent);

  // Split "axis" into 2 axes
  //! inner_split dictates if the factor section of the split should be inside
  //! the
  //! remainer or outside.
  //! e.g. split(0, 4, inner_split = true) will result in:
  //! tv[id{extent}] -> tv[id{ceilDiv(extent, factor)}, id{factor}]
  //! e.g. split(0, 4, inner_split = false) will result in:
  //! tv[id{extent}] -> tv[id{factor}, id{ceilDiv(extent, factor)}]
  TensorView* split(int64_t axis, int64_t factor, bool inner_split = true);

  // Split "axis" into 2 axes where the inner axes is size of "factor"
  // and outer axis is size axis.size() / factor. Factor can be a symbolic
  // value instead of constant. This requires setting the symbolic value as an
  // input, or using a parallel dim from NamedScalar::getParallelDim
  TensorView* split(int64_t axis, Val* factor, bool inner_split = true);

  template <typename FactorType>
  TensorView* inner_split(int64_t axis, FactorType factor) {
    return split(axis, factor, /*inner_split=*/true);
  }

  template <typename FactorType>
  TensorView* outer_split(int64_t axis, FactorType factor) {
    return split(axis, factor, /*inner_split=*/false);
  }

  // Merge axis_o and axis_i into 1 IterDomain
  TensorView* merge(int64_t axis_o, int64_t axis_i);

  // Merge axis and axis+1 into 1 IterDomain
  TensorView* merge(int64_t axis) {
    return merge(axis, axis + 1);
  }

  // Partition "axis" into component and ragged dimensions based on extents
  // The extents tensor directly specifies the size of each component:
  //   Shape: [num_components], values: [extent0, extent1, ..., extent(n-1)]
  // Returns this TensorView with the axis replaced by component and ragged dims
  // e.g. partition(0, extents) on tv[id{N}] results in:
  //   tv[id{num_components}, ragged_id{extents}]
  TensorView* partition(int64_t axis, TensorView* extents);

  // Flatten the axis from `from` to `to` into a single axis.
  // Both `from` and `to` are inclusive.
  TensorView* flatten(int64_t from = 0, int64_t to = -1);

  // Reorder axes according to old2new[old_pos] = new_pos
  TensorView* reorder(const std::unordered_map<int64_t, int64_t>& old2new);
  TensorView* reorder(
      const std::initializer_list<std::pair<const int64_t, int64_t>>& old2new);

  // Reorder axes based on the vector permutation.
  // In terms of the function above, this can be seen as old2new[index] =
  // permutation[index]
  TensorView* reorder(const std::vector<int64_t>& permutation);
  TensorView* reorder(const std::initializer_list<int64_t>& permutation);

  //! Swizzle the rectangular tile defined by the iterdomains corresponding
  //!  to the 2 given indices.
  TensorView* swizzle(SwizzleType swizzle_type, int64_t x, int64_t y);
  TensorView* swizzle(
      Swizzle2DType swizzle_type,
      int64_t x,
      int64_t y,
      SwizzleMode swizzle_mode = SwizzleMode::Data);

  //! Resize an IterDomain by expanding both the left and right sides
  //! by given widths. The resulting IterDomain has an extent of
  //! (left_expansion + axis->extent() + right_expansion).
  TensorView* resize(
      int64_t axis,
      Val* left_expansion,
      Val* right_expansion,
      std::optional<IterType> iter_type = std::nullopt);

  // WARNING: rFactor does not return this TensorView, ir returns a new
  //  tensorview consumed by this!
  //
  // Take reduction axes out of this domain, and create a new
  // domain. New domain will be used to create this domain.
  //
  // For example:
  //  TV1[I0, R1, R2, I3] = TV0[I0, I1, I2, I3]
  //
  // After:
  //  TV1->rfactor({1}), TV1 is transformed to -> TV1[I0, R2, I3]
  //
  // The TensorView returned is: TV2[I0, R1, I2, I3]
  //
  // The reduction will now beset as:
  //  TV2[I0, R1, I2, I3] = TV0[I0, I1, I2, I3]
  //  TV1[I0, R2, I3] = TV2[I0, R1, I2, I3]
  //
  TensorView* rFactor(const std::vector<int64_t>& axes);

  //! Multi-output version of rFactor, semantically similar with
  //! the reduction version except that the rfactor is done
  //! for all outputs in a consistent way
  std::vector<TensorView*> rFactor(
      const std::vector<int64_t>& axes,
      const std::vector<TensorView*>& tvs);

  //! Create a TensorView before the original tensor. A common use case is to
  //! write results into shared memory or registers before moving to global
  //! memory. Analogous to TVM Cache_Write
  //!
  //! @param op_type: memory operator to use for the inserted op between
  //!   the the data tensor and the cache tensor
  TensorView* cacheBefore(LoadStoreOpType op_type = LoadStoreOpType::Set);

  //! Create a TensorView after the original tensor. A common use case is to
  //! read tensor into shared memory or registers. Analogous to TVM Cache_Read
  //!
  //! @param op_type: memory operator to use for the inserted op between
  //!   the the data tensor and the cache tensor
  //! @param cache_op: cache operator, see enum class CacheOp
  //! @param propagate_allocation_domain: replay allocation domain on cached
  //! load
  //! @param cached_uses: if empty, cache all uses; otherwise, only try to cache
  //! uses in cached_uses.
  TensorView* cacheAfter(
      LoadStoreOpType op_type = LoadStoreOpType::Set,
      CacheOp cache_op = CacheOp::Unspecified,
      bool propagate_allocation_domain = true,
      std::vector<Expr*> cached_uses = {});

  // For a fusion output with other uses, we want to avoid writing to global
  // memory and then reading the output again. We write to global memory
  // separately after an operation. We replace this fusion output with the
  // direct write TensorView.
  TensorView* cacheFork();

  MemoryType getMemoryType() const {
    return memory_type_;
  }

  void setMemoryType(MemoryType mt);

  // Apply circular buffering transformation. Negative prefetch_distance
  // means "all but", for example, -1 means number_of_stages - 1.
  void circularBuffer(
      int64_t number_of_stages,
      int64_t prefetch_distance = -1,
      CircularBufferType type = Pipelined(false));

  // Returns true if this tensor is circular buffered.
  bool isCircularBuffered() const {
    return circular_buffer_options_.isEnable();
  }

  const CircularBufferOptions& circularBufferOptions() const {
    return circular_buffer_options_;
  }

  //! Transforms the innermost iterdomains according to the given mma swizzle,
  //!  this should be used on the tvs that are either inputs/outputs of an
  //!  MmaOp, or any tv's that are involved in prolog/epilog fusions and need to
  //!  have a matching thread swizzle with the mma operand/result.
  //! More detail on usage see [MmaSwizzler] in scheduler/mma_utils.h .
  void applyMmaSwizzle(MmaOperand operand);
  void applyMmaSwizzle(MmaInputSmemSwizzle swizzle);

  //! Function to schedule the swizzled TMA box.
  //! This functions works on the assumption that the TMA box is 2D
  //! and the inner-dimension is less or equal to the swizzle size.
  //! This doesn't work for the swizzle none mode. For more details
  //! refer to the figure doc/dev/tma/swizzle.svg
  void swizzleTMABox(MmaInputSmemSwizzle swizzle);

  //! Transforms the innermost iterdomains according to the given mma swizzle,
  //!  this should be used on the tvs that are inputs of a MmaOp or are loaded
  //!  using TMA.
  void applyMmaSwizzleForTMALoad(MmaInputSmemSwizzle swizzle);

  //! Returns if this tensor view has swizzle operator on its tensor domain.
  //!  This is the temporary flag for indicating that the new swizzle
  //!  implementation is used and will be removed in follow ups.
  bool hasSwizzleOp() const {
    return has_swizzle_op_;
  }

  //! A temporary helper function for the transition from Swizzle2D to Swizzle
  void setHasSwizzleOp() {
    has_swizzle_op_ = true;
  }

  friend TransformPropagator;
  friend MostInlinedTransformPropagator;
  friend TransformReplay;
  friend OptOutMutator;
  friend class ir_utils::TVDomainGuard;

  // Inline the computation of this tensor into its consumer at the given
  // position. If this tensor is already inlined in a higher position, then this
  // call is a no-op. If the right most dimensions before `pos` are
  // broadcasting, then will not inline into these broadcastings. If
  // best_effort, then will inline into the highest allowed position that is <=
  // `pos`.
  void inlineAt(
      int64_t pos,
      bool best_effort = false,
      MaxPosCalculator* calc = nullptr);

  //! Inline the computation of this tensor into a consumer at the given
  //! position. The consumer to compute with is determined when the
  //! fusion is lowered. Specifically, it is the first consumer tensor
  //! in the topologically ordered dependency graph. Before the
  //! lowering, its compute-with consumer is considered unresolved,
  //! which is then resolved by resolveComputeWith below.
  //!
  //! The position is relative to its own domain. It is an
  //! error if the position is smaller than the compute-at position. If this
  //! tensor is already inlined in a higher position with the same
  //! consumer, then this call is a no-op. The actual position is
  //! computed in the same way as inlineAt, except that computeWith
  //! does not have the constraint of the persistent data-dependency pattern.
  void computeWith(int64_t pos, bool best_effort = false);

  //! Set the actual consumer tensors that this tensor is
  //! computed with. Requires a topologically sorted list expressions,
  //! which can be obtained reorderExprsForComputeAt. Return true if
  //! resolution is actually done. This should only be done in the
  //! Kernel container.
  bool resolveComputeWith(const std::vector<Expr*>& sorted_exprs);

  bool hasComputeWith() const {
    return getComputeWithPosition() > getComputeAtPosition();
  }

  bool hasResolvedComputeWith() const {
    return !compute_with_consumers_.empty();
  }

  //! Query if this tensor is computed with a given consumer.
  bool isComputedWith(const TensorView* consumer) const;

  //! Return the tensors with which this tensor is computed. It is an
  //! error to use this function without first resolving computeWith.
  const std::vector<TensorView*>& getComputeWithConsumers() const;

  int64_t getComputeWithPosition() const {
    return compute_with_pos_;
  }

  int64_t getMaxComputePosition() const {
    return std::max(getComputeWithPosition(), getComputeAtPosition());
  }

  //! Returns the position that this tensor is produced at for a given
  //! consumer. If this tensor is computed with the given consumer,
  //! which also means its computeWith needs to have been resolved, the
  //! computeWith position is returned. Otherwise, the default computeAt
  //! position is retured.
  int64_t getComputePosition(const TensorView* consumer) const;

  // Update the max producer position of the current tensor. This is required
  // when we modify producer-consumer relationship of a scheduled tensor, for
  // example, grouping multiple reductions.
  void updateMaxProducerPosition(MaxPosCalculator* calc = nullptr);

  // Initialize compute and prodocuer positions. Fusion can result in
  // an inconsistent state. Use with extreme care.
  void clearComputePosition();

  // Commit the current changes in loop domain into rFactor domain. This
  // function can be used to do implicit transpose and view, but today, only
  // implicit transpose is being tested. This function can be dangerous: it
  // changes the the semantics of the current tensor without updating its
  // consumers consistently, and there is no reliable way to detect this
  // inconsistency. It is the responsibility of the caller of this function to
  // ensure consistency.
  void commitLeafToLogical();

  //! Request that we reclaim the memory of this tv before any subsequent
  //! tensors are allocated.
  //!
  //! This method influences the shared memory allocator that assigns shared
  //! memory addresses at lowering. It ensures that the proper synchronization
  //! is present in the kernel to reuse memory and inserts new block
  //! synchronizations if necessary.
  void promoteReuse(bool b = true) {
    NVF_CHECK(
        memory_type_ == MemoryType::Shared,
        "promoteReuse should only be called on shared memory tensors");
    promote_reuse_ = b;
  }

  //! Returns whether we should insert syncs if needed in order to reuse the
  //! memory of this tensor.
  bool shouldPromoteReuse() const {
    return promote_reuse_;
  }

  void setDeviceMesh(const DeviceMesh& mesh) {
    mesh_ = mesh;
  }

  const DeviceMesh& getDeviceMesh() const {
    return mesh_;
  }

  bool hasDeviceMesh() const {
    return !mesh_.vector().empty();
  }

  // Get/set the "Tensor Memory Dimension Separator Position"
  // This is an allocation domain position for tensors with MemoryType::Tensor
  // that separates the row and column of tensor memory.
  // See doc/dev/tmem.md for more details.
  int64_t getTMemDimSepPos() const {
    return tmem_dim_sep_pos_;
  }
  void setTMemDimSepPos(int64_t pos);

 protected:
  void setDomain(TensorDomain* td) {
    domain_ = td;
  }

 private:
  int64_t wrapDim(int64_t dim) const {
    return nvfuser::wrapDim(dim, nDims());
  }

  //! A helper function to maintain the consistency of schedules of
  //! multiple outputs when doing rfactor on multi-output reduction ops.
  TensorView* multiOutputRFactorHelper(
      TensorView* tv,
      const std::vector<int64_t>& axes);

  void clearComputeWith();

 private:
  TensorDomain* domain_ = nullptr;
  int64_t compute_at_pos_ = 0;
  int64_t max_producer_pos_ = 0;
  MemoryType memory_type_ = MemoryType::Local;

  //! Indicates the circular buffering options if applicable.
  CircularBufferOptions circular_buffer_options_;

  // special handling for CPU based zero-dim tensors (i.e. CPU Tensors that
  // only have one value). This is only used if on an input value, otherwise
  // ignored. This is important as special handling because these "scalars"
  // should be type promoted as a tensor, but we want to avoid explicit
  // copying of the data, so we want to pass the data value as a standard
  // kernel argument value.
  bool cpu_scalar_ = false;

  //! Indicates if this tensor view has swizzle operator on its tensor domain.
  //!  This is the temporary flag for indicating that the new swizzle
  //!  implementation is used and will be removed in follow ups.
  bool has_swizzle_op_ = false;

  //! Direct consumer tensors that this tensor is computed with
  std::vector<TensorView*> compute_with_consumers_;

  //! Position where this tensor is computed with the compute-with
  //! consumer tensors. It should be always be equal or greater than
  //! the computeAt position
  int64_t compute_with_pos_ = 0;

  //! Maximum position where producers may be computed at, including
  //! unresolved computeWith. This is equal to max_producer_pos_ when
  //! no producer has unresolved computeWith. It is only used before
  //! resolving computeWith so that no IterDomain should never be
  //! transformed when there may actually be a producer tensor that
  //! may be computed at.
  int64_t maybe_max_producer_pos_ = 0;

  //! When this is true, it indicates, if this is a shared memory tensor and
  //! there other shared memory tensors whose lifetimes do not overlap and come
  //! later than this tensor's lifetime, that we should ensure that thread
  //! blocks are synchronized such that all threads have performed their last
  //! read of this tensor (or any tensors aliasing in) before writing to the
  //! current tensor. This will then allow us to safely reuse the memory
  //! allocated to this tensor.
  bool promote_reuse_ = false;

  // Device Mesh on which the Tensor is sharded
  DeviceMesh mesh_;

  // The "Tensor Memory Dimension Separator Position"
  // This is an allocation domain position for tensors with MemoryType::Tensor
  // that separates the row and column of tensor memory.
  // See doc/dev/tmem.md for more details.
  int64_t tmem_dim_sep_pos_ = 0;
};

//! A simple TensorView builder
//!
//! Example usage:
//!
//!   auto tv = TensorViewBuilder()
//!       .ndims(ndims)
//!       .dtype(dtype)
//!       .contiguity(contiguity)
//!       .build();
//!
class NVF_API TensorViewBuilder {
 public:
  //! Set the number of dimensions of the tensor (default 0, meaning scalar)
  TensorViewBuilder& ndims(int64_t ndims);

  //! Set the data type of the tensor (default DataType::Float)
  TensorViewBuilder& dtype(DataType dtype);

  //! Set the contiguity information (default non-contiguous)
  TensorViewBuilder& contiguity(std::vector<std::optional<bool>> contiguity);
  TensorViewBuilder& contiguity(bool contiguity);

  //! Set the shape (default 0 dimensional, ie. scalar)
  TensorViewBuilder& shape(std::vector<Val*> shape);
  TensorViewBuilder& shape(const std::vector<int64_t>& shape);

  //! Set if a dimension is expanded
  TensorViewBuilder& expanded(std::vector<bool> expanded);

  //! Set the permutation from allocation domain on root domain
  TensorViewBuilder& strideOrder(std::vector<int64_t> stride_order);

  //! Creates a new TensorView with the specified options
  TensorView* build() const;

 private:
  int64_t ndims_ = 0;
  DataType dtype_ = DataType::Float;

  // contiguity_ is the vector that you will pass to the constructor of
  // TensorDomain. However, constructing this vector can be non-trivial, because
  // it is required to be nullopt for broadcast dimensions. We often want to
  // create contiguity vector that represents all contiguous or all
  // discontiguous. uniform_contiguity_ is there to make this use case more
  // convenient. If set, then TensorViewBuilder will automatically fill the
  // contiguity with the value of uniform_contiguity_ where it is not required
  // to be nullopt. Note that you can only set one of contiguity_ or
  // uniform_contiguity_.
  std::vector<std::optional<bool>> contiguity_;
  std::optional<bool> uniform_contiguity_ = std::nullopt;

  std::vector<Val*> shape_;

  std::vector<int64_t> stride_order_;
  std::vector<bool> expanded_;
};

} // namespace nvfuser
