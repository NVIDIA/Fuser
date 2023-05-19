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

namespace {

// Forward traverse from rFactor domain to allocation domain, compute frontier
// sizes and strides, validate that splits are divisible and merges are
// contiguous, and update active_ids_ correspondingly.
class ForwardTraverseFromRFactorToAlloc {
  ExpressionEvaluator& ee_;
  std::unordered_map<IterDomain*, std::pair<int64_t, int64_t>>& active_ids_;

  void handle(Split* split) {
    auto in = split->in();
    auto inner = split->inner();
    auto outer = split->outer();
    auto in_it = active_ids_.find(in);
    // TORCH_INTERNAL_ASSERT(in_it != active_ids_.end())
    if (in_it == active_ids_.end()) {
      // TODO: see [Allocation domain on both side of rFactor]
      return;
    }
    auto [in_size, in_stride] = in_it->second;
    auto factor = ee_.evaluate(split->factor())->as<int64_t>();
    TORCH_INTERNAL_ASSERT(
        in_size % factor == 0,
        "The rFactor domain and allocation domain of fusion input/output ",
        "tensors must be a one-to-one map, therefore, ",
        "non-divisible split is not allowed in allocation domain");
    TORCH_INTERNAL_ASSERT(active_ids_.erase(in) == 1);
    TORCH_INTERNAL_ASSERT(
        active_ids_
            .emplace(inner, std::pair<int64_t, int64_t>{factor, in_stride})
            .second);
    TORCH_INTERNAL_ASSERT(active_ids_
                              .emplace(
                                  outer,
                                  std::pair<int64_t, int64_t>{
                                      in_size / factor, in_stride * factor})
                              .second);
  }

  void handle(Merge* merge) {
    auto inner = merge->inner();
    auto outer = merge->outer();
    auto out = merge->out();
    auto inner_it = active_ids_.find(inner);
    auto outer_it = active_ids_.find(outer);
    // TORCH_INTERNAL_ASSERT(inner_it != active_ids_.end())
    // TORCH_INTERNAL_ASSERT(outer_it != active_ids_.end())
    if (inner_it == active_ids_.end() || outer_it == active_ids_.end()) {
      // TODO: see [Allocation domain on both side of rFactor]
      return;
    }
    auto [inner_size, inner_stride] = inner_it->second;
    auto [outer_size, outer_stride] = outer_it->second;
    TORCH_INTERNAL_ASSERT(
        inner_stride * inner_size == outer_stride,
        "The rFactor domain and allocation domain of fusion input/output ",
        "tensors must be a one-to-one map, therefore, ",
        "merging of discontiguous dimensions is not allowed in allocation domain");
    TORCH_INTERNAL_ASSERT(active_ids_.erase(inner) == 1);
    TORCH_INTERNAL_ASSERT(active_ids_.erase(outer) == 1);
    TORCH_INTERNAL_ASSERT(active_ids_
                              .emplace(
                                  out,
                                  std::pair<int64_t, int64_t>{
                                      inner_size * outer_size, inner_stride})
                              .second);
  }

  void handle(Expr* expr) {
    if (auto split = dynamic_cast<Split*>(expr)) {
      handle(split);
    } else if (auto merge = dynamic_cast<Merge*>(expr)) {
      handle(merge);
    } else {
      TORCH_INTERNAL_ASSERT(
          false, "Unsupported transormation in allocation domain");
    }
  }

 public:
  ForwardTraverseFromRFactorToAlloc(
      ExpressionEvaluator& ee,
      std::unordered_map<IterDomain*, std::pair<int64_t, int64_t>>& active_ids)
      : ee_(ee), active_ids_(active_ids) {}

  void run(
      TensorView* tv,
      const std::vector<IterDomain*>& rfactor,
      const std::vector<IterDomain*>& alloc) {
    auto forward_exprs = StmtSort::getExprsBetween(
        tv->fusion(),
        {rfactor.begin(), rfactor.end()},
        {alloc.begin(), alloc.end()});
    for (auto expr : forward_exprs) {
      handle(expr);
    }
  }
};

// Similar to ForwardTraverseFromRFactorToAlloc, but in the opposite direction.
class BackwardTraverseFromRFactorToAlloc {
  at::Tensor tensor_;
  ExpressionEvaluator& ee_;
  std::unordered_map<IterDomain*, std::pair<int64_t, int64_t>>& active_ids_;

  void handle(Split* split) {
    auto in = split->in();
    auto inner = split->inner();
    auto outer = split->outer();
    auto inner_it = active_ids_.find(inner);
    auto outer_it = active_ids_.find(outer);
    // TORCH_INTERNAL_ASSERT(inner_it != active_ids_.end())
    // TORCH_INTERNAL_ASSERT(outer_it != active_ids_.end())
    if (inner_it == active_ids_.end() || outer_it == active_ids_.end()) {
      // TODO: see [Allocation domain on both side of rFactor]
      return;
    }
    auto [inner_size, inner_stride] = inner_it->second;
    auto [outer_size, outer_stride] = outer_it->second;
    TORCH_INTERNAL_ASSERT(
        inner_stride * inner_size == outer_stride,
        "The rFactor domain and allocation domain of fusion input/output ",
        "tensors must be a one-to-one map, therefore, ",
        "splitting one dimension into discontiguous dimensions is not allowed in allocation domain");
    TORCH_INTERNAL_ASSERT(active_ids_.erase(inner) == 1);
    TORCH_INTERNAL_ASSERT(active_ids_.erase(outer) == 1);
    TORCH_INTERNAL_ASSERT(active_ids_
                              .emplace(
                                  in,
                                  std::pair<int64_t, int64_t>{
                                      inner_size * outer_size, inner_stride})
                              .second);
  }

  void handle(Merge* merge) {
    auto inner = merge->inner();
    auto outer = merge->outer();
    auto out = merge->out();
    auto factor = ee_.evaluate(inner->extent())->as<int64_t>();
    auto out_it = active_ids_.find(out);
    // TORCH_INTERNAL_ASSERT(out_it != active_ids_.end())
    if (out_it == active_ids_.end()) {
      // TODO: see [Allocation domain on both side of rFactor]
      return;
    }
    auto [out_size, out_stride] = out_it->second;
    TORCH_INTERNAL_ASSERT(
        out_size % factor == 0,
        "The rFactor domain and allocation domain of fusion input/output ",
        "tensors must be a one-to-one map, therefore, ",
        "the size of the output must divisible by the size of inner dimension");
    TORCH_INTERNAL_ASSERT(active_ids_.erase(out) == 1);
    TORCH_INTERNAL_ASSERT(
        active_ids_
            .emplace(inner, std::pair<int64_t, int64_t>{factor, out_stride})
            .second);
    TORCH_INTERNAL_ASSERT(active_ids_
                              .emplace(
                                  outer,
                                  std::pair<int64_t, int64_t>{
                                      out_size / factor, out_stride * factor})
                              .second);
  }

  void handle(Expr* expr) {
    if (auto split = dynamic_cast<Split*>(expr)) {
      handle(split);
    } else if (auto merge = dynamic_cast<Merge*>(expr)) {
      handle(merge);
    } else {
      TORCH_INTERNAL_ASSERT(
          false, "Unsupported transormation in allocation domain");
    }
  }

 public:
  BackwardTraverseFromRFactorToAlloc(
      ExpressionEvaluator& ee,
      std::unordered_map<IterDomain*, std::pair<int64_t, int64_t>>& active_ids)
      : ee_(ee), active_ids_(active_ids) {}

  void run(
      TensorView* tv,
      const std::vector<IterDomain*>& rfactor,
      const std::vector<IterDomain*>& alloc) {
    auto backward_exprs = StmtSort::getExprsBetween(
        tv->fusion(),
        {alloc.begin(), alloc.end()},
        {rfactor.begin(), rfactor.end()});
    std::reverse(backward_exprs.begin(), backward_exprs.end());
    for (auto expr : backward_exprs) {
      handle(expr);
    }
  }
};

} // namespace

// Given an ATen tensor, whose sizes and strides are w.r.t to the rFactor domain
// of its corresponding TensorView, compute the sizes and strides of the tensor
// with respect to its allocation domain.
// For example, if the rFactor domain is [I1, I2], and the allocation domain is
// [I2*I1], and the tensor's size is [5, 3] and stride is [2, 10], then the
// resulting size will be [15] and stride will be [2]
// Another example, if the rFactor domain is [I1*I2] and the allocation domain
// is [I1, I2], and the tensor's size is [15] and stride is [7], and the extent
// of I2 is 5, then the resulting size will be [3, 5] and stride will be [35, 7]
std::vector<std::pair<int64_t, int64_t>>
inferAndValidateAllocationSizesAndStrides(
    const at::Tensor& tensor,
    TensorView* tv,
    ExpressionEvaluator& ee) {
  if (tv == nullptr || !tv->hasAllocation()) {
    // When tv is nullptr, or tv does not have allocation, the given sizes and
    // strides should already be in the target format. So nothing to do here.
    std::vector<std::pair<int64_t, int64_t>> result;
    for (auto i : c10::irange(tensor.dim())) {
      result.emplace_back(tensor.size(i), tensor.stride(i));
    }
    return result;
  }
  const auto& alloc =
      TensorDomain::noReductions(tv->getMaybeAllocationDomain());
  const auto& rfactor = TensorDomain::noReductions(tv->getMaybeRFactorDomain());

  // active IDs and their shape and stride
  std::unordered_map<IterDomain*, std::pair<int64_t, int64_t>> active_ids;
  TORCH_INTERNAL_ASSERT((int64_t)rfactor.size() == tensor.dim());
  for (int64_t i : c10::irange((int64_t)rfactor.size())) {
    auto rf_id = rfactor.at(i);
    active_ids[rf_id] = {tensor.size(i), tensor.stride(i)};
  }

  ForwardTraverseFromRFactorToAlloc(ee, active_ids).run(tv, rfactor, alloc);
  BackwardTraverseFromRFactorToAlloc(ee, active_ids).run(tv, rfactor, alloc);

  // Now active_ids should contain the final sizes and strides, unordered. We
  // need to put them to the correct order.
  std::vector<std::pair<int64_t, int64_t>> sizes_strides;
  sizes_strides.reserve(alloc.size());
  for (auto i : c10::irange(alloc.size())) {
    auto id = alloc.at(i);
    sizes_strides.emplace_back(active_ids.at(id));
  }
  // Validate final sizes and strides with contiguity
  int64_t contiguous_stride = 1;
  std::vector<std::optional<bool>> contiguity = tv->getContiguity();
  for (int64_t i = (int64_t)sizes_strides.size() - 1; i >= 0; i--) {
    if (alloc.at(i)->isBroadcast()) {
      continue;
    }
    while (!contiguity.back().has_value()) {
      contiguity.pop_back();
    }
    auto [size, stride] = sizes_strides.at(i);
    TORCH_INTERNAL_ASSERT(!contiguity.empty());
    auto last_contiguity = contiguity.back();
    TORCH_INTERNAL_ASSERT(
        last_contiguity.has_value(),
        "I don't think this check makes sense, but unfortunately ",
        "clang-tidy is not smart enough to infer from the context that this is always true.");
    if (*last_contiguity) {
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
  TORCH_INTERNAL_ASSERT(
      contiguity.empty(),
      "The size of contiguity mismatch with the dimensionality of allocation domain");
  // Validate that for expanded broadcast, the stride must be zero.
  for (int64_t i : c10::irange((int64_t)sizes_strides.size())) {
    if (auto alloc_id = alloc.at(i); alloc_id->hasExpandedExtent()) {
      auto [_, stride] = sizes_strides.at(i);
      TORCH_CHECK(
          stride == 0,
          "Expecting an expanded dimension on dimension ",
          i,
          " but found stride ",
          stride);
    }
  }
  return sizes_strides;
}

PrimDataType TensorArgAbstract::getSmallestIndexType() const {
  KernelIndexTypeCompute index_type_helper;
  for (const auto dim_i : c10::irange(tensor_.ndimension())) {
    auto size = tensor_.size(dim_i);
    auto stride = tensor_.stride(dim_i);
    if (index_type_helper.addDim(size, stride) == PrimDataType::Int) {
      return PrimDataType::Int;
    }
  }
  return PrimDataType::Int32;
}

namespace {

template <int nalloc, typename nvfuser_index_t>
std::unique_ptr<TensorArgAbstract> getTensorArg(
    at::Tensor tensor,
    TensorView* tv,
    ExpressionEvaluator& eval) {
  switch (tensor.ndimension()) {
    case (0):
      return std::make_unique<
          TensorArg<TensorArgCodegen<0, nalloc, nvfuser_index_t>>>(
          std::move(tensor), tv, eval);
    case (1):
      return std::make_unique<
          TensorArg<TensorArgCodegen<1, nalloc, nvfuser_index_t>>>(
          std::move(tensor), tv, eval);
    case (2):
      return std::make_unique<
          TensorArg<TensorArgCodegen<2, nalloc, nvfuser_index_t>>>(
          std::move(tensor), tv, eval);
    case (3):
      return std::make_unique<
          TensorArg<TensorArgCodegen<3, nalloc, nvfuser_index_t>>>(
          std::move(tensor), tv, eval);
    case (4):
      return std::make_unique<
          TensorArg<TensorArgCodegen<4, nalloc, nvfuser_index_t>>>(
          std::move(tensor), tv, eval);
    case (5):
      return std::make_unique<
          TensorArg<TensorArgCodegen<5, nalloc, nvfuser_index_t>>>(
          std::move(tensor), tv, eval);
    case (6):
      return std::make_unique<
          TensorArg<TensorArgCodegen<6, nalloc, nvfuser_index_t>>>(
          std::move(tensor), tv, eval);
    case (7):
      return std::make_unique<
          TensorArg<TensorArgCodegen<7, nalloc, nvfuser_index_t>>>(
          std::move(tensor), tv, eval);
    case (8):
      return std::make_unique<
          TensorArg<TensorArgCodegen<8, nalloc, nvfuser_index_t>>>(
          std::move(tensor), tv, eval);
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
std::unique_ptr<TensorArgAbstract> getTensorArg(
    at::Tensor tensor,
    TensorView* tv,
    ExpressionEvaluator& eval) {
  // When tv is nullptr, the given sizes and strides should already be in the
  // target format.
  int64_t alloc_size =
      (tv != nullptr
           ? (int64_t)TensorDomain::noReductions(tv->getMaybeAllocationDomain())
                 .size()
           : tensor.dim());
  switch (alloc_size) {
    case (0):
      return getTensorArg<0, nvfuser_index_t>(tensor, tv, eval);
    case (1):
      return getTensorArg<1, nvfuser_index_t>(tensor, tv, eval);
    case (2):
      return getTensorArg<2, nvfuser_index_t>(tensor, tv, eval);
    case (3):
      return getTensorArg<3, nvfuser_index_t>(tensor, tv, eval);
    case (4):
      return getTensorArg<4, nvfuser_index_t>(tensor, tv, eval);
    case (5):
      return getTensorArg<5, nvfuser_index_t>(tensor, tv, eval);
    case (6):
      return getTensorArg<6, nvfuser_index_t>(tensor, tv, eval);
    case (7):
      return getTensorArg<7, nvfuser_index_t>(tensor, tv, eval);
    case (8):
      return getTensorArg<8, nvfuser_index_t>(tensor, tv, eval);
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Tried to generate a tensor to run a generated kernel with ",
          tensor.ndimension(),
          " dimensions, however only 0 to 8 dimensional tensor are supported.");
  }
  return nullptr;
}

std::unique_ptr<TensorArgAbstract> getAbstractTensorArg(at::Tensor tensor) {
  return std::make_unique<TensorArgAbstract>(std::move(tensor));
}

std::unique_ptr<TensorArgAbstract> getTensorArg(
    at::Tensor tensor,
    TensorView* tv,
    ExpressionEvaluator& eval,
    PrimDataType index_type) {
  switch (index_type) {
    case PrimDataType::Int32:
      return getTensorArg<int>(std::move(tensor), tv, eval);
    case PrimDataType::Int:
      return getTensorArg<int64_t>(std::move(tensor), tv, eval);
    default:
      TORCH_INTERNAL_ASSERT(false, "unknown index mode");
      break;
  }
}

} // namespace

KernelArgumentHolder KernelArgumentHolder::createKernelArgumentHolder(
    const c10::ArrayRef<c10::IValue>& inputs) {
  if (inputs.empty()) {
    // default to device 0
    KernelArgumentHolder args;
    args.setDeviceIndex(0);
    return args;
  }
  auto device_index = getCommonDeviceCUDA(inputs);

  KernelArgumentHolder args;
  args.setDeviceIndex(device_index);
  args.push(inputs);

  return args;
}

namespace {

template <size_t size>
std::unique_ptr<ArgAbstract> makeCpuScalarTensorArg(const at::Tensor& tensor) {
  auto ptr = std::make_unique<CpuScalarTensorArg<size>>();
  static_assert(sizeof(ptr->instance_) == size);
  std::memcpy(&(ptr->instance_), tensor.data_ptr(), size);
  return ptr;
}

} // namespace

// Push a tensor to the arguments
void KernelArgumentHolder::push(const at::Tensor& tensor) {
  if (is_cpu_scalar(tensor)) {
    switch (tensor.element_size()) {
      case 1:
        arguments_.push_back(makeCpuScalarTensorArg<1>(tensor));
        break;
      case 2:
        arguments_.push_back(makeCpuScalarTensorArg<2>(tensor));
        break;
      case 4:
        arguments_.push_back(makeCpuScalarTensorArg<4>(tensor));
        break;
      case 8:
        arguments_.push_back(makeCpuScalarTensorArg<8>(tensor));
        break;
      case 16:
        arguments_.push_back(makeCpuScalarTensorArg<16>(tensor));
        break;
    }
  } else {
    arguments_.push_back(getAbstractTensorArg(tensor));
  }
}

// Push a scalar or integer to the arguments
void KernelArgumentHolder::push(const c10::IValue& val) {
  TORCH_INTERNAL_ASSERT(
      val.isScalar(),
      "Tried to push an arg to run in a fused kernel, expected a scalar but got, ",
      val);
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
    std::vector<TensorView*> tvs,
    ExpressionEvaluator& eval) {
  TORCH_INTERNAL_ASSERT(
      arguments_.size() == tvs.size(),
      "The size of arguments and the size of tvs does not match.");
  if (void_ptrs_.size() < arguments_.size()) {
    void_ptrs_.resize(arguments_.size());
  }
  for (const auto i : c10::irange(arguments_.size())) {
    if (auto tensor_arg =
            dynamic_cast<TensorArgAbstract*>(arguments_.at(i).get())) {
      if (tensor_arg->isAbstract() ||
          tensor_arg->getIndexType() != index_type) {
        auto resolved_arg =
            getTensorArg(tensor_arg->getTensor(), tvs.at(i), eval, index_type);
        arguments_.at(i) = std::move(resolved_arg);
      }
    }
    void_ptrs_.at(i) = static_cast<void*>(arguments_.at(i)->arg());
  }
  return void_ptrs_.data();
}

void KernelArgumentHolder::push(const c10::ArrayRef<c10::IValue>& args) {
  // Naive I/O setup, I'm ignoring all the potential transformation (i.e. I/O
  // allocated here from the subgraph could be, and very likely are, different
  // from I/O expected by the generated CUDA kernel.
  for (const auto& arg : args) {
    if (arg.isTensor()) {
      push(arg.toTensor());
    } else {
      push(arg);
    }
  }
}

void KernelArgumentHolder::push(const std::vector<at::Tensor>& tensors) {
  for (const auto& tensor : tensors) {
    push(tensor);
  }
}

void KernelArgumentHolder::push(const ArgAbstract* arg) {
  arguments_.emplace_back(arg->clone());
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
  auto holder = arg->clone();
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
    if (auto tensor_arg = dynamic_cast<const TensorArgAbstract*>(arg.get())) {
      if (tensor_arg->getSmallestIndexType() == PrimDataType::Int) {
        return PrimDataType::Int;
      }
    }
  }
  return PrimDataType::Int32;
}

void KernelArgumentHolder::pushTensorProxy(
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides,
    at::ScalarType dtype) {
  TORCH_INTERNAL_ASSERT(strides.size() == sizes.size());
  auto meta_tensor = at::detail::empty_strided_meta(
      sizes,
      strides,
      dtype,
      c10::nullopt,
      c10::Device(c10::DeviceType::Meta, 0),
      c10::nullopt);
  arguments_.push_back(getAbstractTensorArg(at::Tensor(meta_tensor)));
}

} // namespace nvfuser
