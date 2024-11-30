// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/native/cuda/jit_utils.h>

#include <c10/util/irange.h>

#include <contiguity.h>
#include <debug.h>
#include <driver_api.h>
#include <instrumentation.h>
#include <ir/all_nodes.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <options.h>
#include <runtime/executor_utils.h>
#include <tensor_metadata.h>
#include <utils.h>

#include <cuda_occupancy.h>

#include <cstdlib>
#include <fstream>
#include <variant>

namespace nvfuser {
namespace executor_utils {

namespace {

// Return true if all the tensors have the same stride, assumes all tensors are
// contiguous
bool checkSameStride(const std::vector<c10::IValue>& tensors) {
  if (tensors.size() < 2) {
    return true;
  }
  for (const auto idx : c10::irange(tensors.size() - 1)) {
    auto current = tensors[idx];
    auto next = tensors[idx + 1];
    if (!current.isTensor() || !next.isTensor()) {
      return false;
    }

    const auto& current_tensor = current.toTensor();
    const auto& next_tensor = next.toTensor();
    if (current_tensor.ndimension() != next_tensor.ndimension()) {
      return false;
    }

    for (const auto i : c10::irange(current_tensor.ndimension())) {
      if (current_tensor.stride(i) != next_tensor.stride(i)) {
        return false;
      }
    }
  }
  return true;
}

// Return true if all the tensors are contiguous and have the same striding
bool checkSameContiguity(const std::vector<c10::IValue>& tensors) {
  if (tensors.size() < 2) {
    return true;
  }

  auto reference = tensors.front();
  if (!reference.isTensor()) {
    return false;
  }

  // Determine if the reference tensor is contiguous
  const auto& reference_tensor = reference.toTensor();
  int64_t expected_stride = 1;
  for (const auto i : c10::irange(1, reference_tensor.ndimension() + 1)) {
    int64_t ind = reference_tensor.ndimension() - i;
    if (reference_tensor.size(ind) == 1) {
      continue;
    }
    if (reference_tensor.stride(ind) != expected_stride) {
      return false;
    }
    expected_stride *= reference_tensor.size(ind);
  }

  // Check if all the tensors have the same contiguity
  return checkSameStride(tensors);
}

bool checkValidMisalignedTensors(
    const std::unordered_set<TensorView*>& inp_tv,
    const std::unordered_set<TensorView*>& out_tv,
    const std::vector<c10::IValue>& inp_tensors,
    const std::vector<c10::IValue>& out_tensors) {
  if (out_tv.empty()) {
    // Only check input tensors
    return checkSameStride(inp_tensors);
  } else if (!out_tv.empty() && out_tensors.empty()) {
    // out_tensors is empty unless outputs are given to runFusion.
    // Assume out tensors are contiguous
    return checkSameContiguity(inp_tensors);
  } else {
    // Only check input and output tensors
    std::vector<c10::IValue> tensors;
    tensors.insert(tensors.end(), inp_tensors.begin(), inp_tensors.end());
    tensors.insert(tensors.end(), out_tensors.begin(), out_tensors.end());
    return checkSameStride(tensors);
  }
}

// Finds a fusion input or output tensor, this function is used to grab tensors
// to validate the strides of the tensors for vectorization.
//
// Returns a pair consisting of a flag indicating if it's a fusion input (else
// is output) and an integer position within in the input or output tensor list.
std::vector<std::pair<bool, int64_t>> getVectorizedFusionInputOutput(
    TensorView* producer_tv,
    TensorView* consumer_tv,
    Fusion* fusion) {
  std::vector<std::pair<bool, int64_t>> input_output;

  // When the producer is a fusion input, only return the producer
  // (vectorization validation assumes consumer of input is vectorizable).
  // Similarly, when the consumer is a fusion output, only return the consumer
  // (vectorization validation assumes producer of output is vectorizable). If
  // producer is input and consumer is output, return both.

  if (producer_tv->isFusionInput()) {
    auto producer_it = std::find(
        fusion->inputs().begin(), fusion->inputs().end(), producer_tv);
    NVF_ERROR(
        producer_it != fusion->inputs().end(),
        "Could not find ",
        producer_tv,
        " in fusion inputs.");
    auto pos = std::distance(fusion->inputs().begin(), producer_it);
    input_output.push_back(
        std::make_pair<bool, int64_t>(true, static_cast<int64_t>(pos)));
  }

  if (consumer_tv->isFusionOutput()) {
    auto consumer_it = std::find(
        fusion->outputs().begin(), fusion->outputs().end(), consumer_tv);
    NVF_ERROR(
        consumer_it != fusion->outputs().end(),
        "Could not find ",
        consumer_tv,
        " in fusion outputs.");
    auto pos = std::distance(fusion->outputs().begin(), consumer_it);
    input_output.push_back(
        std::make_pair<bool, int64_t>(false, static_cast<int64_t>(pos)));
  }

  return input_output;
}

//! Returns the information of vectorized input/output tensors
//! in the given fusion.
std::unique_ptr<caching::VectorizedTensorInfo> getVectorizedTensorValidationInfo(
    kir::Kernel* kernel) {
  auto vectorized_tensor_info_ptr =
      std::make_unique<caching::VectorizedTensorInfo>();

  for (const auto& vector_info : kernel->summary().vectorized_set_info) {
    auto consumer_tv = vector_info.consumer_tv;
    auto producer_tv = vector_info.producer_tv;

    auto vector_dim = vector_info.vectorized_loop_id;
    const auto is_aligned =
        vector_dim->getParallelType() == ParallelType::Vectorize;

    // Find fusion inputs and outputs that are used with misaligned
    // vectorization.
    if (!is_aligned) {
      NVF_ERROR(
          producer_tv->isFusionInput() || consumer_tv->isFusionOutput(),
          "MisalignedVectorize is assumed to be used with either input or output tensor");
      if (consumer_tv->getMemoryType() == MemoryType::Global &&
          producer_tv->getMemoryType() == MemoryType::Local) {
        vectorized_tensor_info_ptr->global_out_misaligned_tv.insert(
            consumer_tv);
      } else if (
          producer_tv->getMemoryType() == MemoryType::Global &&
          consumer_tv->getMemoryType() == MemoryType::Local) {
        vectorized_tensor_info_ptr->global_inp_misaligned_tv.insert(
            producer_tv);
      } else {
        NVF_THROW(
            "Unsupported memory configuration for misaligned vectorization.");
      }
    }

    // Collect information on corresponding fusion input and output
    // tensors to verify strides.
    auto inp_or_out_info =
        getVectorizedFusionInputOutput(producer_tv, consumer_tv, kernel);

    // If both producer and consumer are contig and intermediate,
    // nothing to validate with respect to strides.
    if (inp_or_out_info.empty()) {
      continue;
    }

    // Misaligned vectorize only allows from input to local or local
    // to output
    if (!is_aligned) {
      NVF_ERROR(inp_or_out_info.size() == 1);
    }

    for (const auto& inp_or_out : inp_or_out_info) {
      const bool is_input = inp_or_out.first;
      const int64_t pos = inp_or_out.second;

      if (is_aligned) {
        auto& pos_list = is_input
            ? vectorized_tensor_info_ptr->aligned_vectorized_inp_tensor_pos
            : vectorized_tensor_info_ptr->aligned_vectorized_out_tensor_pos;
        pos_list.push_back(pos);
      } else {
        auto& map = is_input
            ? vectorized_tensor_info_ptr->inp_misaligned_tensors_pos
            : vectorized_tensor_info_ptr->out_misaligned_tensors_pos;
        map.emplace_back(pos);
      }
    }
  }

  return vectorized_tensor_info_ptr;
}

// Make sure the root domain(s) comprising the vectorized loop domain
// have the (merged) extent that is divisible by the vectorization
// word size.
void validateAlignedVectorizeExtents(
    const VectorizedSetInfo& info,
    ExpressionEvaluator& expr_eval) {
  if (info.contig_alloc_ids.empty()) {
    // This happens when device lowering removes the `Expr` that computes
    // `info.consumer_tv` because it's merely an alias.
    // `getTensorIndexFromIdGraph` captures only remaining `Expr`s.
    return;
  }

  // TODO: Rewrite validation of the vectorized dimension
  // int64_t vectorized_merged_domain_extent = 1;
  for (auto id : info.contig_alloc_ids) {
    auto extent_val = expr_eval.evaluate(id->extent());
    NVF_ERROR(
        extent_val.hasValue(),
        "Error vectorizing, ",
        info.consumer_tv->toString(),
        " as the extent of a vectorized root domain, ",
        id->toString(),
        ", is unknown.");
    // TODO: Rewrite validation of the vectorized dimension
    // vectorized_merged_domain_extent *= extent_val->as<int64_t>();
  }

  // TODO: Rewrite validation of the vectorized dimension, we can't just used a
  // single merged extent because we could be splitting a dimension then merging
  // it in order to the right of it. Contig merged index simply isn't exactly
  // what we need to validate for vectorization, and we're relying on better
  // vectorization support than that would offer. This validation needs to be
  // rewritten based on updated indexing logic that traverses loop->logical
  // domains and tracks partial mappings like scheduler/vectorize_helper.cpp
  //
  // NVF_ERROR(
  //     vectorized_merged_domain_extent % info.word_size == 0,
  //     "Error vectorizing, ",
  //     info.consumer_tv->toString(),
  //     " as the extent of the indexed domain, ",
  //     vectorized_merged_domain_extent,
  //     ", is not divisible by vector word size ",
  //     info.word_size);
}

// Return offsets of the first points accessed as well as sliced root
// domains. Currently only non-zero when tensor is sliced.
std::pair<std::unordered_set<size_t>, std::unordered_set<IterDomain*>>
getTensorOffsets(
    TensorView* tv,
    c10::IntArrayRef logical_strides,
    ExpressionEvaluator& eval) {
  if (!tv->isFusionInput()) {
    return {{0}, {}};
  }

  std::unordered_set<size_t> offsets;
  std::unordered_set<IterDomain*> sliced_domains;

  const auto logical_ids = TensorDomain::noReductions(tv->getLogicalDomain());

  for (auto use : tv->uses()) {
    auto slice = dynamic_cast<SliceOp*>(use);

    if (slice == nullptr) {
      offsets.insert(0);
      continue;
    }

    NVF_ERROR(logical_strides.size() == logical_ids.size());
    const auto slice_info = slice->getRanges();

    size_t offset = 0;
    for (const auto i : c10::irange(logical_ids.size())) {
      auto slice_start_eval = eval.evaluate(slice_info.at(i).start);
      NVF_ERROR(slice_start_eval.hasValue());
      auto slice_stop_eval = eval.evaluate(slice_info.at(i).stop);
      NVF_ERROR(slice_stop_eval.hasValue());
      auto extent_eval =
          eval.evaluate(logical_ids.at(i)->getMaybeExpandedExtent());
      NVF_ERROR(extent_eval.hasValue());

      offset += static_cast<size_t>(
          slice_start_eval.as<int64_t>() * logical_strides.at(i));

      // Keep track of the root domain unless this slice is
      // effectively no-op
      if (slice_start_eval.as<int64_t>() != 0 ||
          slice_stop_eval.as<int64_t>() != extent_eval.as<int64_t>()) {
        sliced_domains.insert(logical_ids.at(i));
      }
    }

    offsets.insert(offset);
  }

  return std::make_pair(offsets, sliced_domains);
}

void validateAlignedVectorizedFusionInputOutput(
    const at::Tensor& aten_tensor,
    int64_t word_size,
    TensorView* tv,
    ExpressionEvaluator& eval) {
  eval.bind(tv, aten_tensor);
  const auto& metadata = eval.evaluate(IrBuilder::metadataExpr(tv));

  const auto [offsets, sliced_domains] =
      getTensorOffsets(tv, metadata->*&TensorMetaData::logical_stride, eval);
  const bool is_sliced = !sliced_domains.empty();

  const auto& domain_to_validate =
      is_sliced ? tv->getLogicalDomain() : tv->getMaybeAllocationDomain();

  std::vector<int64_t> no_reduction_to_full;
  for (int64_t i : c10::irange((int64_t)domain_to_validate.size())) {
    auto alloc_id = domain_to_validate.at(i);
    if (!alloc_id->isReduction()) {
      no_reduction_to_full.emplace_back(i);
    }
  }

  const auto& sizes = is_sliced ? metadata->*&TensorMetaData::logical_size
                                : metadata->*&TensorMetaData::alloc_size;
  const auto& strides = is_sliced ? metadata->*&TensorMetaData::logical_stride
                                  : metadata->*&TensorMetaData::alloc_stride;
  NVF_ERROR(sizes.size() == no_reduction_to_full.size());
  NVF_ERROR(strides.size() == no_reduction_to_full.size());

  for (auto offset : offsets) {
    NVF_ERROR(
        (reinterpret_cast<size_t>(aten_tensor.data_ptr()) +
         offset * aten_tensor.dtype().itemsize()) %
                (word_size * aten_tensor.dtype().itemsize()) ==
            0,
        "Vectorization of ",
        tv->toString(),
        " not possible as the memory address is not aligned. ",
        "Address: ",
        aten_tensor.data_ptr(),
        ", offset: ",
        offset,
        ", vector word size: ",
        word_size,
        ", data type: ",
        aten_tensor.dtype());
  }

  // Traverse strides from the right-most domains. The rightmost
  // domain must have stride 1.
  int64_t cur_contig_stride = 1;
  bool still_rightmost = true;
  bool non_contig_due_to_slice = false;
  for (int64_t i = (int64_t)sizes.size() - 1; i >= 0; --i) {
    const auto size = sizes.at(i);
    const auto stride = strides.at(i);
    auto id = domain_to_validate.at(no_reduction_to_full.at(i));
    const auto is_expanded_broadcasting =
        id->isBroadcast() && id->hasExpandedExtent();

    if (is_expanded_broadcasting) {
      NVF_ERROR(
          stride == 0,
          "Dimension ",
          i,
          " should be an expanded broadcasting, but it does not have stride zero.");
    }

    // If this domain is contiguous or size == 1, then not necessary to check
    // the stride. Otherwise, stride must be 1 if it's rightmost or
    // divisible by word_size

    bool is_contiguous =
        stride == cur_contig_stride && !non_contig_due_to_slice;

    NVF_ERROR(
        is_contiguous || size == 1 || is_expanded_broadcasting ||
            (still_rightmost && stride == 1) ||
            (!still_rightmost && stride % word_size == 0),
        "Vectorization of ",
        tv->toString(),
        " with word size ",
        word_size,
        " not possible due to invalid stride.",
        " Domain: ",
        tv->axis(i)->toString(),
        ", stride: ",
        stride,
        ", cur_contig_stride: ",
        cur_contig_stride,
        ", non contig due to slice: ",
        non_contig_due_to_slice);
    // If the domain is size-1, the next domain is still considered
    // rightmost.
    still_rightmost =
        still_rightmost && (size == 1 || is_expanded_broadcasting);
    // We do not update cur_contig_stride for size==1 dimensions,
    // since we have specialized vectorization stride check for
    // them. Same for non_contig_due_to_slice.
    if (size != 1 && !is_expanded_broadcasting) {
      cur_contig_stride = stride * size;
      // Note that when a domain is sliced, the next outer domain is
      // no longer contiguous.
      non_contig_due_to_slice = sliced_domains.count(
          domain_to_validate.at(no_reduction_to_full.at(i)));
    }
  }
}

void validateAlignedVectorizedTensors(
    kir::Kernel* kernel,
    const KernelArgumentHolder& args,
    const std::vector<at::Tensor>& outputs,
    caching::ExecutorCompileTimeInfoCache* data_cache,
    ExpressionEvaluator& expr_eval) {
  // Verify extents of aligned vectorized tensors
  for (const auto& vec_info : kernel->summary().vectorized_set_info) {
    if (vec_info.vectorized_loop_id->getParallelType() ==
        ParallelType::Vectorize) {
      validateAlignedVectorizeExtents(vec_info, expr_eval);
    }
  }

  // Validate input and output tensors with aligend
  // vectorization.
  auto tensor_vectorization_validation_entry =
      executor_utils::caching::ExecutorCompileTimeEntry<
          executor_utils::caching::VectorizedTensorValidation>(
          data_cache, [kernel]() {
            return executor_utils::getVectorizedTensorValidationInfo(kernel);
          });
  for (auto pos : tensor_vectorization_validation_entry.get()
                      .aligned_vectorized_inp_tensor_pos) {
    auto tv = kernel->inputs().at(pos)->as<TensorView>();
    auto word_size = kernel->summary().vectorized_accesses.at(tv);
    NVF_ERROR(args[pos]->is<at::Tensor>(), "alias io only supports tensor");
    validateAlignedVectorizedFusionInputOutput(
        args[pos]->as<at::Tensor>(), word_size, tv, expr_eval);
  }
  if (!outputs.empty()) {
    for (auto pos : tensor_vectorization_validation_entry.get()
                        .aligned_vectorized_out_tensor_pos) {
      auto tv = kernel->outputs().at(pos)->as<TensorView>();
      auto word_size = kernel->summary().vectorized_accesses.at(tv);
      validateAlignedVectorizedFusionInputOutput(
          outputs[pos], word_size, tv, expr_eval);
    }
  }
}

// Misaligned vectorization check. Currently misaligned vectorization is limited
// to global-register and register-global load/store patterns. However, this
// could be improved to include shared memory.
void validateMisalignedVectorizedTensors(
    kir::Kernel* kernel,
    const KernelArgumentHolder& args,
    const std::vector<at::Tensor>& outputs,
    caching::ExecutorCompileTimeInfoCache* data_cache,
    ExpressionEvaluator& expr_eval) {
  auto tensor_vectorization_validation_entry =
      executor_utils::caching::ExecutorCompileTimeEntry<
          executor_utils::caching::VectorizedTensorValidation>(
          data_cache, [kernel]() {
            return executor_utils::getVectorizedTensorValidationInfo(kernel);
          });

  std::vector<c10::IValue> inp_misaligned_tensors;
  std::vector<c10::IValue> out_misaligned_tensors;

  const auto& inp_misaligned_tensors_pos =
      tensor_vectorization_validation_entry.get().inp_misaligned_tensors_pos;
  inp_misaligned_tensors.reserve(inp_misaligned_tensors_pos.size());
  std::transform(
      inp_misaligned_tensors_pos.begin(),
      inp_misaligned_tensors_pos.end(),
      std::back_inserter(inp_misaligned_tensors),
      [&args](int idx) {
        NVF_ERROR(args[idx]->is<at::Tensor>(), "alias io only supports tensor");
        return args[idx]->as<at::Tensor>();
      });

  const auto& out_misaligned_tensors_pos =
      tensor_vectorization_validation_entry.get().out_misaligned_tensors_pos;
  if (!outputs.empty()) {
    out_misaligned_tensors.reserve(out_misaligned_tensors_pos.size());
    std::transform(
        out_misaligned_tensors_pos.begin(),
        out_misaligned_tensors_pos.end(),
        std::back_inserter(out_misaligned_tensors),
        [&outputs](int idx) { return outputs[idx]; });
  }
  // If input stride is non-contiguous + no outputs, return false
  NVF_ERROR(
      checkValidMisalignedTensors(
          tensor_vectorization_validation_entry.get().global_inp_misaligned_tv,
          tensor_vectorization_validation_entry.get().global_out_misaligned_tv,
          inp_misaligned_tensors,
          out_misaligned_tensors),
      "All global tensors must have the same stride for misaligned vectorization.");
}

} // namespace

void validateCircularBuffering(
    kir::Kernel* kernel,
    ExpressionEvaluator& expr_eval) {
  const CircularBufferInfo& cb_info = kernel->summary().circular_buffer_info;
  for (const TensorView* cb_tv : cb_info.getCircularBufferTvs()) {
    IterDomain* axis = cb_info.getCircularBufferAxis(cb_tv);
    NVF_ERROR(axis != nullptr);
    PolymorphicValue runtime_axis_size = expr_eval.evaluate(axis->extent());
    NVF_ERROR(
        runtime_axis_size >= cb_tv->circularBufferOptions().stage,
        "This kernel fails to fill the circular buffer pipeline at runtime. ",
        "The extent of the circular buffer axis is ",
        runtime_axis_size,
        " while ",
        cb_tv->circularBufferOptions().stage,
        " is the number of stages in the circular buffer.");
  }
}

void validateVectorizedTensors(
    kir::Kernel* kernel,
    const KernelArgumentHolder& args,
    const std::vector<at::Tensor>& outputs,
    caching::ExecutorCompileTimeInfoCache* data_cache,
    ExpressionEvaluator& expr_eval) {
  FUSER_PERF_SCOPE("KernelExecutor::validateVectorizedTensors");

  validateAlignedVectorizedTensors(
      kernel, args, outputs, data_cache, expr_eval);

  validateMisalignedVectorizedTensors(
      kernel, args, outputs, data_cache, expr_eval);
}

ExpressionEvaluator bindInputs(
    const KernelArgumentHolder& args,
    Fusion* kernel) {
  FUSER_PERF_SCOPE("executor_utils::bindInputs");

  // args may contains more than just inputs, but inputs are always at the
  // beginning.
  NVF_ERROR(
      kernel->inputs().size() <= args.size(),
      "KernelArgumentHolder contains less argument than kernel's input.");

  ExpressionEvaluator expr_eval;
  const auto& inputs = kernel->inputs();
  for (const auto i : c10::irange(inputs.size())) {
    // NOTE: we bind all inputs here, including at::Tensors. This means that
    // expr_eval will create a PolymorphicValue containing *args[i], which means
    // that at::Tensor's lifetime will be at least as long as that of expr_eval.
    try {
      expr_eval.bind(inputs[i], *args[i], true);
    } catch (const nvfError& e) {
      std::stringstream ss;
      ss << "When trying to run the provided host program,"
         << " there was an error with the provided input " << i
         << ". Provided input was:" << std::endl;
      indent(ss, 1) << PolymorphicValue_functions::toString(*args[i])
                    << std::endl;
      ss << "Fusion input was:" << std::endl;
      indent(ss, 1) << inputs[i]->toString() << std::endl;
      ss << "Expr eval provided the error:" << std::endl;
      ss << R"(""")" << e.msg() << R"(""")" << std::endl;
      NVF_THROW(ss.str());
    }
  }

  return expr_eval;
}

CudaExecutable::~CudaExecutable() {
  if (module != nullptr) {
    NVFUSER_CUDA_SAFE_CALL(cuModuleUnload(module));
    module = (CUmodule)0x2a2a2a2a2a2a2a2a;
  }
}

namespace caching {

//! CompileTimeInfo is the actual subclass of CompileTimeInfoBase that will
//!  be stored in the data cache. It owns a data_ state internally of the
//!  dataType defined within the entry class, which are listed in header file.
template <typename EntryClass>
class CompileTimeInfo : public CompileTimeInfoBase {
 public:
  CompileTimeInfo(std::unique_ptr<typename EntryClass::DataType> data)
      : CompileTimeInfoBase(EntryClass::EntryType), data_(std::move(data)) {}

  typename EntryClass::DataType* get() {
    return data_.get();
  }

 private:
  std::unique_ptr<typename EntryClass::DataType> data_;
};

void ExecutorCompileTimeInfoCache::insert(EntryOwningPtr new_entry) {
  // Just overwrite when insertion duplicates, equality not checked.
  entry_type_map_[new_entry->type()] = new_entry.get();
  entries_.emplace_back(std::move(new_entry));
}

template <typename EntryClass>
ExecutorCompileTimeEntry<EntryClass>::ExecutorCompileTimeEntry(
    ExecutorCompileTimeInfoCache* data_cache,
    MakerFnType fn) {
  using InfoType = CompileTimeInfo<EntryClass>;

  if (!data_cache || !data_cache->has(EntryClass::EntryType)) {
    owned_data_ = fn();
    data_ptr_ = owned_data_.get();

    if (data_cache) {
      std::unique_ptr<CompileTimeInfoBase> new_entry =
          std::make_unique<InfoType>(std::move(owned_data_));
      data_cache->insert(std::move(new_entry));
    }
  } else {
    data_ptr_ =
        data_cache->at(EntryClass::EntryType)->template as<InfoType>()->get();
  }
}

// Template instantiation
template class ExecutorCompileTimeEntry<ParallelBindingIterDomains>;
template class ExecutorCompileTimeEntry<ParallelIterExtentMap>;
template class ExecutorCompileTimeEntry<VectorizedTensorValidation>;

} // namespace caching

std::vector<IterDomain*> getParallelBindingsIterDomains(
    GpuLower* lower,
    const std::vector<TensorView*>& used_tvs) {
  std::vector<IterDomain*> parallel_ids;
  for (auto tv : used_tvs) {
    for (auto id : tv->getLoopDomain()) {
      if (id->isThread()) {
        if (id->isBroadcast()) {
          // Want to keep the broadcast dimensions if they are not resolved
          // TODO: piping down the parallel dimension map here would
          //  be helpful
          if (lower->caMap()->getConcreteMappedID(id, IdMappingMode::LOOP) ==
              id) {
            parallel_ids.push_back(id);
          }
        } else {
          // Non broadcast ids are directly added to the binding
          //  ids.
          parallel_ids.push_back(id);
        }
      }
    }
  }
  return parallel_ids;
}

namespace {

void insertParallelExtent(
    IterDomain* binding_id,
    const std::unique_ptr<ParallelExtentMap>& parallel_iter_extents_ptr) {
  auto extent = binding_id->extent();
  const auto it =
      parallel_iter_extents_ptr->find(binding_id->getParallelType());
  if (it != parallel_iter_extents_ptr->end()) {
    it->second.push_back(extent);
  } else {
    parallel_iter_extents_ptr->operator[](binding_id->getParallelType()) = {
        extent};
  }
}

} // namespace

std::unique_ptr<ParallelExtentMap> getParallelIterExtents(
    std::vector<IterDomain*>& parallel_binding_ids) {
  auto parallel_iter_extents_ptr = std::make_unique<ParallelExtentMap>();
  for (auto id : parallel_binding_ids) {
    insertParallelExtent(id, parallel_iter_extents_ptr);
  }

  return parallel_iter_extents_ptr;
}

} // namespace executor_utils
} // namespace nvfuser
