// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <c10/util/irange.h>

// Extract size and strides
#include <kernel_cache.h>

#include <executor_kernel_arg.h>

namespace nvfuser {

std::string TensorArgAbstract::toString() const {
  std::stringstream ss;
  auto rank = getRank();
  ss << "tensor dtype: " << getDataType() << " sizes: (";
  for (auto i = 0; i < rank; i++) {
    ss << getSize(i) << ", ";
  }
  ss << ") stride: (";
  for (auto i = 0; i < rank; i++) {
    ss << getStride(i) << ", ";
  }
  ss << ") pointer: " << getPointer();
  return ss.str();
}

namespace {

template <typename T, int NAllocDims, typename nvfuser_index_t>
std::unique_ptr<TensorArgAbstract> getTensorArg(
    const at::Tensor& tensor,
    TensorView* tv,
    ExpressionEvaluator& eval,
    bool index_type_resolved) {
  switch (tensor.ndimension()) {
    case (0):
      return std::make_unique<
          TensorArg<TensorArgCodegen<T, 0, NAllocDims, nvfuser_index_t>>>(
          tensor, tv, eval, index_type_resolved);
    case (1):
      return std::make_unique<
          TensorArg<TensorArgCodegen<T, 1, NAllocDims, nvfuser_index_t>>>(
          tensor, tv, eval, index_type_resolved);
    case (2):
      return std::make_unique<
          TensorArg<TensorArgCodegen<T, 2, NAllocDims, nvfuser_index_t>>>(
          tensor, tv, eval, index_type_resolved);
    case (3):
      return std::make_unique<
          TensorArg<TensorArgCodegen<T, 3, NAllocDims, nvfuser_index_t>>>(
          tensor, tv, eval, index_type_resolved);
    case (4):
      return std::make_unique<
          TensorArg<TensorArgCodegen<T, 4, NAllocDims, nvfuser_index_t>>>(
          tensor, tv, eval, index_type_resolved);
    case (5):
      return std::make_unique<
          TensorArg<TensorArgCodegen<T, 5, NAllocDims, nvfuser_index_t>>>(
          tensor, tv, eval, index_type_resolved);
    case (6):
      return std::make_unique<
          TensorArg<TensorArgCodegen<T, 6, NAllocDims, nvfuser_index_t>>>(
          tensor, tv, eval, index_type_resolved);
    case (7):
      return std::make_unique<
          TensorArg<TensorArgCodegen<T, 7, NAllocDims, nvfuser_index_t>>>(
          tensor, tv, eval, index_type_resolved);
    case (8):
      return std::make_unique<
          TensorArg<TensorArgCodegen<T, 8, NAllocDims, nvfuser_index_t>>>(
          tensor, tv, eval, index_type_resolved);
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Tried to generate a tensor to run a generated kernel with ",
          tensor.ndimension(),
          " dimensions, however only 0 to 8 dimensional tensor are supported.");
  }
  return nullptr;
}

template <typename T, typename nvfuser_index_t>
std::unique_ptr<TensorArgAbstract> getTensorArg(
    const at::Tensor& tensor,
    TensorView* tv,
    ExpressionEvaluator& eval,
    bool index_type_resolved) {
  switch (TensorDomain::noReductions(tv->getMaybeAllocationDomain()).size()) {
    case (0):
      return getTensorArg<T, 0, nvfuser_index_t>(
          tensor, tv, eval, index_type_resolved);
    case (1):
      return getTensorArg<T, 1, nvfuser_index_t>(
          tensor, tv, eval, index_type_resolved);
    case (2):
      return getTensorArg<T, 2, nvfuser_index_t>(
          tensor, tv, eval, index_type_resolved);
    case (3):
      return getTensorArg<T, 3, nvfuser_index_t>(
          tensor, tv, eval, index_type_resolved);
    case (4):
      return getTensorArg<T, 4, nvfuser_index_t>(
          tensor, tv, eval, index_type_resolved);
    case (5):
      return getTensorArg<T, 5, nvfuser_index_t>(
          tensor, tv, eval, index_type_resolved);
    case (6):
      return getTensorArg<T, 6, nvfuser_index_t>(
          tensor, tv, eval, index_type_resolved);
    case (7):
      return getTensorArg<T, 7, nvfuser_index_t>(
          tensor, tv, eval, index_type_resolved);
    case (8):
      return getTensorArg<T, 8, nvfuser_index_t>(
          tensor, tv, eval, index_type_resolved);
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Tried to generate a tensor to run a generated kernel with ",
          tensor.ndimension(),
          " dimensions, however only 0 to 8 dimensional tensor are supported.");
  }
  return nullptr;
}

template <typename nvfuser_index_t>
struct GetTensorArgWithNativeType {
  template <typename T>
  std::unique_ptr<TensorArgAbstract> operator()(
      const at::Tensor& tensor,
      TensorView* tv,
      ExpressionEvaluator& eval,
      bool index_type_resolved) {
    return getTensorArg<T, nvfuser_index_t>(
        tensor, tv, eval, index_type_resolved);
  };
};

template <typename INDEX_TYPE>
std::unique_ptr<TensorArgAbstract> getTensorArg(
    const at::Tensor& tensor,
    TensorView* tv,
    ExpressionEvaluator& eval,
    bool index_type_resolved) {
  return atenTypeDispatchWithC10Complex(
      tensor.scalar_type(),
      GetTensorArgWithNativeType<INDEX_TYPE>(),
      tensor,
      tv,
      eval,
      index_type_resolved);
}

std::unique_ptr<TensorArgAbstract> getTensorArg(
    const at::Tensor& tensor,
    TensorView* tv,
    ExpressionEvaluator& eval,
    std::optional<PrimDataType> index_type) {
  if (index_type.has_value()) {
    switch (index_type.value()) {
      case PrimDataType::Int32:
        return getTensorArg<int>(tensor, tv, eval, true);
      case PrimDataType::Int:
        return getTensorArg<int64_t>(tensor, tv, eval, true);
      default:
        TORCH_INTERNAL_ASSERT(false, "unknown index mode");
        break;
    }
  } else {
    // Tentatively create TensorArgAbstract with int64_t
    return getTensorArg<int64_t>(tensor, tv, eval, false);
  }
}

} // namespace

KernelArgumentHolder KernelArgumentHolder::createKernelArgumentHolder(
    const c10::ArrayRef<c10::IValue>& inputs,
    const std::vector<Val*>& vals,
    ExpressionEvaluator eval) {
  if (inputs.empty()) {
    // default to device 0
    KernelArgumentHolder args;
    args.setDeviceIndex(0);
    return args;
  }
  auto device_index = getCommonDeviceCUDA(inputs);

  KernelArgumentHolder args;
  args.setDeviceIndex(device_index);
  args.push(inputs, vals, eval);

  return args;
}

namespace {

struct MakeCpuScalarTensor {
  template <typename T>
  std::unique_ptr<ArgAbstract> operator()(const at::Tensor& tensor) const {
    return std::make_unique<CpuScalarTensorArg<CpuScalarTensorCodegen<T>>>(
        tensor.data_ptr<T>()[0]);
  }
};

PrimDataType getIndexTypeOfAtenTensor(const at::Tensor& tensor) {
  KernelIndexTypeCompute index_type_helper;
  for (const auto i : c10::irange(tensor.ndimension())) {
    index_type_helper.addDim(tensor.sizes()[i], tensor.strides()[i]);
  }
  return index_type_helper.getType();
}

} // namespace

std::vector<std::pair<int64_t, int64_t>> getAllocationSizesAndStrides(
    const at::Tensor& tensor,
    TensorView* tv,
    ExpressionEvaluator& eval) {
  const auto& alloc_dom =
      TensorDomain::noReductions(tv->getMaybeAllocationDomain());
  const auto& rfactor_dom =
      TensorDomain::noReductions(tv->getMaybeRFactorDomain());
  // active IDs and their shape and stride
  std::unordered_map<IterDomain*, std::pair<int64_t, int64_t>> active_ids;
  int64_t no_reduction_i = 0;
  for (auto rf_id : rfactor_dom) {
    if (!rf_id->isReduction()) {
      active_ids[rf_id] = {
          tensor.size(no_reduction_i), tensor.stride(no_reduction_i)};
      no_reduction_i++;
    }
  }
  // traverse forward from rfactor to alloc
  auto forward_exprs = StmtSort::getExprsBetween(
      tv->fusion(),
      {rfactor_dom.begin(), rfactor_dom.end()},
      {alloc_dom.begin(), alloc_dom.end()});
  for (auto expr : forward_exprs) {
    if (auto split = dynamic_cast<Split*>(expr)) {
      auto in = split->in();
      auto inner = split->inner();
      auto outer = split->outer();
      auto [in_size, in_stride] = active_ids.at(in);
      auto factor = eval.evaluate(split->factor())->as<int64_t>();
      TORCH_INTERNAL_ASSERT(
          in_size % factor == 0,
          "non-divisible split is not allowed in allocation domain");
      TORCH_INTERNAL_ASSERT(active_ids.erase(in) == 1);
      TORCH_INTERNAL_ASSERT(
          active_ids
              .emplace(inner, std::pair<int64_t, int64_t>{factor, in_stride})
              .second);
      TORCH_INTERNAL_ASSERT(active_ids
                                .emplace(
                                    outer,
                                    std::pair<int64_t, int64_t>{
                                        in_size / factor, in_stride * factor})
                                .second);
    } else if (auto merge = dynamic_cast<Merge*>(expr)) {
      auto inner = merge->inner();
      auto outer = merge->outer();
      auto out = merge->out();
      auto [inner_size, inner_stride] = active_ids.at(inner);
      auto [outer_size, outer_stride] = active_ids.at(outer);
      TORCH_INTERNAL_ASSERT(
          inner_stride * inner_size == outer_stride,
          "Merging of discontiguous dimensions is not allowed in allocation domain");
      TORCH_INTERNAL_ASSERT(active_ids.erase(inner) == 1);
      TORCH_INTERNAL_ASSERT(active_ids.erase(outer) == 1);
      TORCH_INTERNAL_ASSERT(active_ids
                                .emplace(
                                    out,
                                    std::pair<int64_t, int64_t>{
                                        inner_size * outer_size, inner_stride})
                                .second);
    } else {
      TORCH_INTERNAL_ASSERT(
          false, "Unsupported transormation in allocation domain");
    }
  }
  // traverse backward from rfactor to allocation
  auto backward_exprs = StmtSort::getExprsBetween(
      tv->fusion(),
      {alloc_dom.begin(), alloc_dom.end()},
      {rfactor_dom.begin(), rfactor_dom.end()});
  std::reverse(backward_exprs.begin(), backward_exprs.end());
  for (auto expr : backward_exprs) {
    if (auto split = dynamic_cast<Split*>(expr)) {
      auto in = split->in();
      auto inner = split->inner();
      auto outer = split->outer();
      auto [inner_size, inner_stride] = active_ids.at(inner);
      auto [outer_size, outer_stride] = active_ids.at(outer);
      TORCH_INTERNAL_ASSERT(
          inner_stride * inner_size == outer_stride,
          "Splitting one dimension into discontiguous dimensions is not allowed in allocation domain");
      TORCH_INTERNAL_ASSERT(active_ids.erase(inner) == 1);
      TORCH_INTERNAL_ASSERT(active_ids.erase(outer) == 1);
      TORCH_INTERNAL_ASSERT(active_ids
                                .emplace(
                                    in,
                                    std::pair<int64_t, int64_t>{
                                        inner_size * outer_size, inner_stride})
                                .second);
    } else if (auto merge = dynamic_cast<Merge*>(expr)) {
      auto inner = merge->inner();
      auto outer = merge->outer();
      auto out = merge->out();
      auto factor = eval.evaluate(inner->extent())->as<int64_t>();
      auto [out_size, out_stride] = active_ids.at(out);
      TORCH_INTERNAL_ASSERT(
          out_size % factor == 0,
          "The size of the output must divisible by the size of inner dimension");
      TORCH_INTERNAL_ASSERT(active_ids.erase(out) == 1);
      TORCH_INTERNAL_ASSERT(
          active_ids
              .emplace(inner, std::pair<int64_t, int64_t>{factor, out_stride})
              .second);
      TORCH_INTERNAL_ASSERT(active_ids
                                .emplace(
                                    outer,
                                    std::pair<int64_t, int64_t>{
                                        out_size / factor, out_stride * factor})
                                .second);
    } else {
      TORCH_INTERNAL_ASSERT(
          false, "Unsupported transormation in allocation domain");
    }
  }
  // compute final result
  std::vector<std::pair<int64_t, int64_t>> sizes_strides;
  sizes_strides.reserve(alloc_dom.size());
  for (auto i : c10::irange(alloc_dom.size())) {
    auto id = alloc_dom.at(i);
    sizes_strides.emplace_back(active_ids.at(id));
  }
  // validate final strides with contiguity
  int64_t contiguous_stride = 1;
  std::vector<std::optional<bool>> contiguity = tv->getContiguity();
  for (int64_t i = sizes_strides.size() - 1; i >= 0; i--) {
    if (alloc_dom.at(i)->isBroadcast()) {
      continue;
    }
    while (!contiguity.back().has_value()) {
      contiguity.pop_back();
      continue;
    }
    auto [size, stride] = sizes_strides.at(i);
    if (*contiguity.back()) {
      TORCH_CHECK(
          stride == contiguous_stride,
          "Stride mismatch with contiguity info. ",
          "tv: ",
          tv->toString(),
          " allocation domain: ",
          ir_utils::toString(tv->getMaybeAllocationDomain()),
          " dim: ",
          i,
          " expected stride: ",
          contiguous_stride,
          " actual stride: ",
          stride);
    }
    contiguous_stride = stride * size;
    contiguity.pop_back();
  }
  return sizes_strides;
}

// Push a tensor to the arguments
void KernelArgumentHolder::push(
    const at::Tensor& tensor,
    TensorView* tv,
    ExpressionEvaluator& eval) {
  if (is_cpu_scalar(tensor)) {
    arguments_.push_back(atenTypeDispatchWithC10Complex(
        tensor.scalar_type(), MakeCpuScalarTensor(), tensor));
  } else {
    arguments_.push_back(getTensorArg(tensor, tv, eval, std::nullopt));
  }
}

// Push a scalar or integer to the arguments
void KernelArgumentHolder::push(const c10::IValue& val) {
  TORCH_INTERNAL_ASSERT(
      val.isScalar(),
      "Tried to push an arg to run in a fused kernel, expected a scalar but got, ",
      val.tagKind());
  auto scalar_val = val.toScalar();
  switch (scalar_val.type()) {
    case c10::ScalarType::ComplexDouble:
      arguments_.push_back(
          std::make_unique<ComplexDoubleArg>(scalar_val.toComplexDouble()));
      return;
    case c10::ScalarType::Double:
      arguments_.push_back(std::make_unique<DoubleArg>(scalar_val.toDouble()));
      return;
    case c10::ScalarType::Long:
      arguments_.push_back(std::make_unique<LongArg>(scalar_val.toLong()));
      return;
    case c10::ScalarType::Bool:
      arguments_.push_back(std::make_unique<BoolArg>(scalar_val.toBool()));
      return;
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          " Tried to create argument to send to a fused kernel, but got an unexpected type.");
  }
  TORCH_INTERNAL_ASSERT(
      false,
      " Tried to create argument to send to a fused kernel, but got a non-scalar type.");
}

void KernelArgumentHolder::push(int64_t val) {
  arguments_.push_back(std::make_unique<LongArg>(val));
}

void KernelArgumentHolder::push(const at::PhiloxCudaState& val) {
  arguments_.push_back(std::make_unique<PhiloxCudaStateArg>(val));
}

// Create buffer, flatten arguments into it, align by 8 Bytes, return pointers
// in the buffer
void** KernelArgumentHolder::getBuffer(
    PrimDataType index_type,
    ExpressionEvaluator& eval) {
  if (void_ptrs_.size() < arguments_.size()) {
    void_ptrs_.resize(arguments_.size());
  }
  for (const auto i : c10::irange(arguments_.size())) {
    auto arg = arguments_.at(i).get();
    if (auto tensor_arg = dynamic_cast<TensorArgAbstract*>(arg)) {
      if (!tensor_arg->isIndexTypeResolved() ||
          tensor_arg->getIndexType() != index_type) {
        auto resolved_arg = getTensorArg(
            tensor_arg->getTensor(),
            tensor_arg->getTensorView(),
            eval,
            index_type);
        arguments_.at(i) = std::move(resolved_arg);
      }
    }
    void_ptrs_.at(i) = static_cast<void*>(arguments_.at(i)->arg());
  }
  return void_ptrs_.data();
}

void KernelArgumentHolder::push(
    const c10::ArrayRef<c10::IValue>& args,
    const std::vector<Val*>& vals,
    ExpressionEvaluator& eval) {
  // Naive I/O setup, I'm ignoring all the potential transformation (i.e. I/O
  // allocated here from the subgraph could be, and very likely are, different
  // from I/O expected by the generated CUDA kernel.
  TORCH_INTERNAL_ASSERT(args.size() == vals.size());
  for (auto i : c10::irange(args.size())) {
    auto arg = args.at(i);
    if (arg.isTensor()) {
      push(arg.toTensor(), vals.at(i)->as<TensorView>(), eval);
    } else {
      push(arg);
    }
  }
}

void KernelArgumentHolder::push(
    const std::vector<at::Tensor>& tensors,
    const std::vector<TensorView*>& tvs,
    ExpressionEvaluator& eval) {
  TORCH_INTERNAL_ASSERT(tensors.size() == tvs.size());
  for (auto i : c10::irange(tensors.size())) {
    push(tensors.at(i), tvs.at(i), eval);
  }
}

void KernelArgumentHolder::push(const ArgAbstract* arg) {
  arguments_.emplace_back(arg->copy_unique_ptr());
}

void KernelArgumentHolder::erase(const ArgAbstract* arg_to_delete) {
  auto iter = std::remove_if(
      arguments_.begin(),
      arguments_.end(),
      [&](const std::unique_ptr<ArgAbstract>& ref) {
        return arg_to_delete == ref.get();
      });
  arguments_.erase(iter, arguments_.end());
}

void KernelArgumentHolder::swap(int i, const ArgAbstract* arg) {
  auto holder = arg->copy_unique_ptr();
  arguments_[i].swap(holder);
}

void KernelArgumentHolder::appendPhiloxRNGSeed(uint64_t rand_offset) {
  at::PhiloxCudaState philox_engine_inputs;
  auto gen = at::cuda::detail::getDefaultCUDAGenerator();
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen.mutex());
    philox_engine_inputs =
        at::check_generator<at::CUDAGeneratorImpl>(gen)->philox_cuda_state(
            rand_offset);
  }
  push(philox_engine_inputs);
}

std::string KernelArgumentHolder::toString() const {
  std::stringstream ss;
  for (const auto& arg : arguments_) {
    ss << arg->toString() << "\n";
  }
  return ss.str();
}

PrimDataType KernelArgumentHolder::getSmallestIndexTypeOfArguments() const {
  for (const auto& arg : arguments_) {
    auto tensor_arg = dynamic_cast<const TensorArgAbstract*>(arg.get());
    if (tensor_arg == nullptr) {
      continue;
    }
    KernelIndexTypeCompute index_type_helper;
    for (const auto dim_i : c10::irange(tensor_arg->getRank())) {
      auto size = tensor_arg->getSize(dim_i);
      auto stride = tensor_arg->getStride(dim_i);
      if (index_type_helper.addDim(size, stride) == PrimDataType::Int) {
        return PrimDataType::Int;
      }
    }
  }
  return PrimDataType::Int32;
}

} // namespace nvfuser
