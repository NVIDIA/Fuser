// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <runtime/allocations.h>

#include <expr_evaluator.h>
#include <instrumentation.h>
#include <ir/iostream.h>
#include <multidevice/utils.h>
#include <polymorphic_value.h>
#include <runtime/executor.h>
#include <runtime/executor_kernel_arg.h>
#include <runtime/executor_utils.h>
#include <tensor_metadata.h>

namespace nvfuser {

KernelArgumentHolder inferOutputSizes(
    Fusion* fusion,
    const KernelArgumentHolder& args,
    PrecomputedValues* evaluator_precomputed_values) {
  FUSER_PERF_SCOPE("fusion_executor::allocations::inferOutputSizes");
  ExpressionEvaluator expr_eval;

  std::unique_ptr<PrecomputedValues> evaluator_precomputed_values_up = nullptr;
  if (evaluator_precomputed_values == nullptr) {
    evaluator_precomputed_values_up =
        std::make_unique<PrecomputedValues>(fusion);
    evaluator_precomputed_values_up->bindInputs(args);
    evaluator_precomputed_values_up->evaluate();
    evaluator_precomputed_values = evaluator_precomputed_values_up.get();
  }
  NVF_ERROR(evaluator_precomputed_values != nullptr);
  expr_eval.precomputedValues() = evaluator_precomputed_values;

  auto arg_index_type = args.getSmallestIndexTypeOfArguments();

  KernelArgumentHolder output_tensor_proxies;
  output_tensor_proxies.setDeviceIndex(args.getDeviceIndex());

  for (Val* output : fusion->outputs()) {
    NVF_ERROR(
        output->isA<TensorView>(),
        "Cannot allocate outputs that are not tensors.");
    auto output_tv = output->as<TensorView>();
    const auto& [sizes, strides] = inferShapeOfOutput(output_tv, expr_eval);
    const auto dtype = (output_tv->dtype() == DataType::Index)
        ? data_type_to_aten(arg_index_type)
        : data_type_to_aten(output_tv->dtype());
    output_tensor_proxies.pushTensorProxy(sizes, strides, dtype);
  }
  return output_tensor_proxies;
}

int64_t computeSharedMemory(
    ExpressionEvaluator& expr_eval,
    const std::vector<const kir::Allocate*>& buffers,
    DataType index_type,
    int64_t smem_offset) {
  FUSER_PERF_SCOPE("fusion_executor::allocations::computeSharedMemory");
  int64_t total = smem_offset;
  // align smem_offset at 16 bytes
  smem_offset = (smem_offset + 15) & (~15);
  for (auto smem_alloc : buffers) {
    // If this buffer aliases another buffer,
    // then do not allocate memory for this buffer.
    if (smem_alloc->alias() == nullptr) {
      NVF_ERROR(
          smem_alloc->address(),
          "Smem address is not set for buffer T",
          smem_alloc->buffer()->name());
      const auto address_val = expr_eval.evaluate(smem_alloc->address());
      NVF_ERROR(
          address_val.hasValue(),
          "Failed to evaluate the address ",
          smem_alloc->address()->toInlineString(),
          " of shared memory buffer T",
          smem_alloc->buffer()->name());
      NVF_ERROR(
          address_val.is<int64_t>(),
          "Address val ",
          smem_alloc->address()->toInlineString(),
          " of shared memory buffer T",
          smem_alloc->buffer()->name(),
          " should be int64 but found ",
          address_val);
      const auto size_val = expr_eval.evaluate(smem_alloc->size());
      NVF_ERROR(
          size_val.hasValue(),
          "Failed to evaluate the size ",
          smem_alloc->size(),
          " of shared memory buffer - T",
          smem_alloc->buffer()->name());

      const auto first_byte = smem_offset + address_val.as<int64_t>();
      const auto data_size =
          dataTypeSizeByte(smem_alloc->buffer()->dtype(), index_type);
      const int64_t size_bytes = size_val.as<int64_t>() * data_size;
      const auto last_byte = first_byte + size_bytes;

      total = std::max(total, last_byte);
      // First byte may not equal to last byte of the previous buffer since
      // shared memory is forced to align at 128 Bytes. See PR-3023.
      // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#table-alignment-multi-dim-tma
      if (isDebugDumpEnabled(DebugDumpOption::DynamicSharedMemory)) {
        debug() << "buffer: " << smem_alloc->buffer()->toString()
                << ", first_byte: " << first_byte
                << ", last_byte: " << last_byte << ", size: " << size_bytes
                << std::endl;
      }
    }
  }
  return total;
}

namespace {
std::vector<int64_t> getContiguousStrides(
    const std::vector<int64_t>& sizes,
    const std::vector<bool>& expand_flags) {
  NVF_ERROR(sizes.size() == expand_flags.size());

  std::vector<int64_t> strides(sizes.size());
  int64_t cur_stride = 1;
  for (auto i = sizes.size(); i > 0; --i) {
    auto size = sizes.at(i - 1);
    NVF_ERROR(
        size >= 0,
        "Positive size is assumed non-negative but received: ",
        size);

    int64_t stride = cur_stride;

    // If expanded, stride is 0
    if (expand_flags.at(i - 1)) {
      stride = 0;
    } else if (size == 0) {
      // If the size is 0, the stride is 1.
      stride = 1;
    } else {
      cur_stride *= size;
    }

    strides.at(i - 1) = stride;
  }

  return strides;
}

// Infer the size and stride of each dimension
std::pair<std::vector<int64_t>, std::vector<int64_t>> inferShape(
    const TensorView* tv,
    const std::vector<Val*>& symbolic_sizes,
    const std::vector<bool>& expand_flags,
    const ExpressionEvaluator& expr_eval) {
  FUSER_PERF_SCOPE("fusion_executor::allocations::inferShape");

  std::vector<int64_t> concrete_sizes(symbolic_sizes.size(), 0);

  for (const auto i : arange(symbolic_sizes.size())) {
    auto symbolic_size = symbolic_sizes.at(i);
    const auto inferred_val = expr_eval.evaluate(symbolic_size);
    NVF_ERROR(
        inferred_val.hasValue(),
        "Could not launch kernel as program could not infer ",
        symbolic_size->toInlineString(),
        " (",
        symbolic_size->toString(),
        ") for the buffer ",
        tv->toString());

    auto concrete_size = inferred_val.as<int64_t>();
    concrete_sizes.at(i) = concrete_size;
  }

  // Adjust the last dimension of the logical domain to support DataType
  // that is not supported by PyTorch. See the comment of getLastDimAdjustment
  // in type.h for more details.
  const auto adjust_last_dim = getLastDimAdjustment(tv->dtype());
  if (!concrete_sizes.empty()) {
    auto& last_dim = concrete_sizes.back();
    last_dim = adjust_last_dim.fromNVFToATen(last_dim);
  } else {
    NVF_ERROR(
        adjust_last_dim.denominator == 1 && adjust_last_dim.numerator == 1,
        "DataType not supported");
  }

  auto strides = getContiguousStrides(concrete_sizes, expand_flags);

  return {concrete_sizes, strides};
}
} // namespace

static bool fill_allocation_with_nan_ = false;

bool shouldFillAllocationWithNan() {
  return fill_allocation_with_nan_;
}

void setFillAllocationWithNan(bool value) {
  fill_allocation_with_nan_ = value;
}

void fillTensorWithNan(at::Tensor& t) {
  switch (t.scalar_type()) {
    case at::ScalarType::Char:
      t.fill_(0x7F);
      break;
    case at::ScalarType::Short:
      t.fill_(0x7FFF);
      break;
    case at::ScalarType::Int:
      t.fill_(0x7FFFFFFF);
      break;
    case at::ScalarType::Long:
      t.fill_(0x7FFFFFFFFFFFFFFFL);
      break;
    case at::ScalarType::Byte:
      t.fill_(0xFF);
      break;
    case at::ScalarType::UInt16:
      t.fill_(0xFFFF);
      break;
    case at::ScalarType::UInt32:
      t.fill_(0xFFFFFFFF);
      break;
    case at::ScalarType::UInt64:
      t.fill_(0xFFFFFFFFFFFFFFFFL);
      break;
    case at::ScalarType::Bool:
      t.fill_(true);
      break;
    case at::ScalarType::Half:
    case at::ScalarType::Float:
    case at::ScalarType::Double:
    case at::ScalarType::BFloat16:
    case at::ScalarType::Float8_e4m3fn:
    case at::ScalarType::Float8_e5m2:
    case at::ScalarType::Float8_e8m0fnu:
      t.fill_(std::nan(""));
      break;
#if NVF_TORCH_VERSION_NO_LESS(2, 8, 0)
    case at::ScalarType::Float4_e2m1fn_x2:
      t.view(torch::kByte).fill_(0xFF);
      break;
#endif
    case at::ScalarType::ComplexHalf:
    case at::ScalarType::ComplexFloat:
    case at::ScalarType::ComplexDouble:
      t.fill_(c10::complex<double>(std::nan(""), std::nan("")));
      break;
    default:
      NVF_THROW("Unknown dtype");
  }
}

KernelArgumentHolder allocateOutputs(
    const Fusion* fusion,
    const std::vector<GlobalBufferInfo>& output_infos,
    const std::vector<int>& output_alias_to_input_map,
    const c10::Device& device,
    const KernelArgumentHolder& args,
    bool dynamic_evaluate) {
  FUSER_PERF_SCOPE("fusion_executor::allocations::allocateOutputs");

  KernelArgumentHolder out_tensors;
  out_tensors.resize(output_infos.size());
  for (auto out_idx : arange(output_infos.size())) {
    auto out_info = output_infos.at(out_idx);
    if (output_alias_to_input_map.at(out_idx) == -1) {
      auto alloc_tensor = at::native::empty_strided_cuda(
          out_info.shape_info.logical_sizes,
          out_info.shape_info.logical_strides,
          out_info.type,
          c10::nullopt,
          device,
          c10::nullopt);
      if (shouldFillAllocationWithNan()) {
        fillTensorWithNan(alloc_tensor);
      }
      out_tensors[out_idx] = alloc_tensor;
    } else if (
        fusion->getOutputAlias(out_info.tv).type ==
        AllocationType::ReuseBuffer) {
      const auto& inp = args[output_alias_to_input_map.at(out_idx)];
      NVF_ERROR(inp.is<at::Tensor>(), "Input is not a Tensor");
      out_tensors[out_idx] = inp;
    } else if (
        fusion->getOutputAlias(out_info.tv).type == AllocationType::Evaluate) {
      if (dynamic_evaluate) {
        out_tensors[out_idx] = std::monostate();
        continue;
      }

      ExpressionEvaluator ee;
      ee.bind(
          fusion->getOutputAlias(out_info.tv).aliased_io,
          args[output_alias_to_input_map.at(out_idx)]);
      out_tensors[out_idx] = ee.evaluate(out_info.tv);
    } else {
      NVF_THROW(
          "Unexpected allocation path, internal logic around allocations must "
          "be incorrect.");
    }
  }
  return out_tensors;
}

std::vector<GlobalBufferInfo> getBufferInfos(
    ExpressionEvaluator& expr_eval,
    DataType index_dtype,
    const std::vector<Val*>& tvs) {
  FUSER_PERF_SCOPE("fusion_executor::allocations::getBufferInfos");
  std::vector<GlobalBufferInfo> buffer_infos;
  buffer_infos.reserve(tvs.size());
  for (Val* v : tvs) {
    auto* tv = dynamic_cast<TensorView*>(v);
    NVF_ERROR(
        tv != nullptr, "Cannot allocate outputs that are not tensors: ", v);

    GlobalBufferInfo info;
    info.tv = tv;
    info.shape_info = inferTensorShapes(tv, expr_eval);
    auto dtype = (tv->dtype() == DataType::Index ? index_dtype : tv->dtype());
    info.type = data_type_to_aten(dtype);

    buffer_infos.emplace_back(info);
  }
  return buffer_infos;
}

namespace {

class ForwardTraverseFromAllocToLogical {
  at::Tensor tensor_;
  const ExpressionEvaluator& ee_;
  std::list<IterDomain*>& frontier_;

  // Forward traverse split from allocation to logical. Needs to, for example,
  // view tensor with shape [..., 15, ...] as [..., 3, 5, ...]
  void handle(Split* split) {
    auto in = split->in();
    auto inner = split->inner();
    auto outer = split->outer();
    auto factor = ee_.evaluate(split->factor()).as<int64_t>();
    auto in_it = std::find(frontier_.begin(), frontier_.end(), in);
    // NVF_ERROR(in_it != frontier_.end());
    if (in_it == frontier_.end()) {
      // TODO: We should get rid of this return and enable the above assert.
      // Note [Allocation domain on both side of logical]
      // For cases where the allocation domain is on both side of logical, for
      // example, in Tensor3d_To_NHWC4d_FwdBwd_CUDA:
      // [alloc,root]   [alloc,root]           [root]
      //          \     /                      /    |
      //         [logical]                  split   [logical]
      //                                    /  \         |
      //                      [alloc,logical] [logical]  |
      //                                             \   |
      //                                             [alloc]
      // I have no idea why StmtSort::getExprsBetween is not returning the
      // expected set of exprs, but for now, I will just skip these illegal
      // exprs.
      return;
    }
    // view tensor
    int64_t dim = std::distance(frontier_.begin(), in_it);
    std::vector<int64_t> new_shape;
    for (auto i : arange(tensor_.dim())) {
      if (i == dim) {
        new_shape.emplace_back(-1);
        new_shape.emplace_back(factor);
      } else {
        new_shape.emplace_back(tensor_.size(i));
      }
    }
    tensor_ = tensor_.view(new_shape);
    // update frontier
    frontier_.insert(in_it, outer);
    frontier_.insert(in_it, inner);
    frontier_.erase(in_it);
  }

  // Forward traverse split from allocation to logical. Needs to, for example,
  // view tensor with shape [..., 3, 5, ...] as [..., 15, ...]
  void handle(Merge* merge) {
    auto inner = merge->inner();
    auto outer = merge->outer();
    auto out = merge->out();
    auto inner_it = std::find(frontier_.begin(), frontier_.end(), inner);
    auto outer_it = std::find(frontier_.begin(), frontier_.end(), outer);
    // NVF_ERROR(inner_it != frontier_.end());
    // NVF_ERROR(outer_it != frontier_.end());
    if (inner_it == frontier_.end() || outer_it == frontier_.end()) {
      // TODO: see [Allocation domain on both side of logical]
      return;
    }
    int64_t inner_dim = std::distance(frontier_.begin(), inner_it);
    int64_t outer_dim = std::distance(frontier_.begin(), outer_it);
    int64_t left = std::min(inner_dim, outer_dim);
    // view the tensor
    if (outer_dim + 1 != inner_dim) {
      // need to permute the tensor in order to do a merging view
      // before: [..., outer, ..., inner, ...]
      // after: [..., outer, inner, ...]
      std::vector<int64_t> dims;
      int64_t i = 0;
      while (i < tensor_.dim() && i != left) {
        dims.emplace_back(i);
        i++;
      }
      dims.emplace_back(outer_dim);
      dims.emplace_back(inner_dim);
      while (i < tensor_.dim()) {
        if (i != outer_dim && i != inner_dim) {
          dims.emplace_back(i);
        }
        i++;
      }
      tensor_ = tensor_.permute(dims);
    }
    std::vector<int64_t> new_shape;
    for (auto i : arange(tensor_.dim())) {
      if (i == left) {
        new_shape.emplace_back(-1);
      } else if (i != left + 1) {
        new_shape.emplace_back(tensor_.size(i));
      }
    }
    tensor_ = tensor_.view(new_shape);
    // update frontier
    if (inner_dim < outer_dim) {
      *inner_it = out;
      frontier_.erase(outer_it);
    } else {
      *outer_it = out;
      frontier_.erase(inner_it);
    }
  }

  void handle(Expr* expr) {
    if (auto split = dynamic_cast<Split*>(expr)) {
      handle(split);
    } else if (auto merge = dynamic_cast<Merge*>(expr)) {
      handle(merge);
    } else {
      NVF_THROW("Unsupported transormation in allocation domain");
    }
  }

 public:
  ForwardTraverseFromAllocToLogical(
      at::Tensor tensor,
      const ExpressionEvaluator& ee,
      std::list<IterDomain*>& frontier)
      : tensor_(std::move(tensor)), ee_(ee), frontier_(frontier) {}

  at::Tensor run(
      const std::vector<IterDomain*>& logical,
      const std::vector<IterDomain*>& alloc) {
    auto forward_exprs = StmtSort::getExprsBetween(
        {alloc.begin(), alloc.end()}, {logical.begin(), logical.end()});
    for (auto expr : forward_exprs) {
      handle(expr);
    }
    return tensor_;
  }
};

// Backward traverse is similar to forward traverse, but we need to do opposite
// transformations.
class BackwardTraverseFromAllocToLogical {
  at::Tensor tensor_;
  const ExpressionEvaluator& ee_;
  std::list<IterDomain*>& frontier_;

  // Backward traverse split from allocation to logical. Needs to, for example,
  // view tensor with shape [..., 3, 5, ...] as [..., 15, ...]
  void handle(Split* split) {
    auto inner = split->inner();
    auto outer = split->outer();
    auto in = split->in();
    auto inner_it = std::find(frontier_.begin(), frontier_.end(), inner);
    auto outer_it = std::find(frontier_.begin(), frontier_.end(), outer);
    // NVF_ERROR(inner_it != frontier_.end());
    // NVF_ERROR(outer_it != frontier_.end());
    if (inner_it == frontier_.end() || outer_it == frontier_.end()) {
      // TODO: see [Allocation domain on both side of logical]
      return;
    }
    int64_t inner_dim = std::distance(frontier_.begin(), inner_it);
    int64_t outer_dim = std::distance(frontier_.begin(), outer_it);
    int64_t left = std::min(inner_dim, outer_dim);
    // view the tensor
    if (outer_dim + 1 != inner_dim) {
      // need to permute the tensor in order to do a merging view
      // before: [..., outer, ..., inner, ...]
      // after: [..., outer, inner, ...]
      std::vector<int64_t> dims;
      int64_t i = 0;
      while (i < tensor_.dim() && i != left) {
        dims.emplace_back(i);
        i++;
      }
      dims.emplace_back(outer_dim);
      dims.emplace_back(inner_dim);
      while (i < tensor_.dim()) {
        if (i != outer_dim && i != inner_dim) {
          dims.emplace_back(i);
        }
        i++;
      }
      tensor_ = tensor_.permute(dims);
    }

    std::vector<int64_t> new_shape;
    for (auto i : arange(tensor_.dim())) {
      if (i == left) {
        new_shape.emplace_back(-1);
      } else if (i != left + 1) {
        new_shape.emplace_back(tensor_.size(i));
      }
    }

    // Copy tensor_ shape into std::vector<int64_t>
    std::vector<int64_t> tensor_shape_vec(
        tensor_.sizes().begin(), tensor_.sizes().end());
    std::vector<int64_t> tensor_new_shape;
    size_t i = 0;
    while (i < new_shape.size()) {
      if (new_shape[i] != -1) {
        tensor_new_shape.push_back(new_shape[i]);
        ++i;
      } else {
        // Multiply the corresponding entry and the next entry in
        // tensor_shape_vec
        NVF_ERROR(
            i + 1 < tensor_shape_vec.size(),
            "Index out of bounds for -1 handling in new_shape");
        tensor_new_shape.push_back(
            tensor_shape_vec[i] * tensor_shape_vec[i + 1]);
        ++i;
      }
    }

    // Compute cumulative product from highest index to 0-th index
    std::vector<int64_t> tensor_new_strides(tensor_new_shape.size(), 1);
    int64_t prod = 1;
    for (int i = static_cast<int>(tensor_new_shape.size()) - 1; i >= 0; --i) {
      prod *= tensor_new_shape[i];
      tensor_new_strides[i] = prod;
    }

    tensor_ = tensor_.as_strided(tensor_new_shape, tensor_new_strides);

    // update frontier
    if (inner_dim < outer_dim) {
      *inner_it = in;
      frontier_.erase(outer_it);
    } else {
      *outer_it = in;
      frontier_.erase(inner_it);
    }
  }

  // Backward traverse split from allocation to logical. Needs to, for example,
  // view tensor with shape [..., 15, ...] as [..., 3, 5, ...]
  void handle(Merge* merge) {
    auto out = merge->out();
    auto inner = merge->inner();
    auto outer = merge->outer();
    auto factor = ee_.evaluate(inner->extent()).as<int64_t>();
    auto out_it = std::find(frontier_.begin(), frontier_.end(), out);
    // NVF_ERROR(out_it != frontier_.end());
    if (out_it == frontier_.end()) {
      // TODO: see [Allocation domain on both side of logical]
      return;
    }
    // view tensor
    int64_t dim = std::distance(frontier_.begin(), out_it);
    std::vector<int64_t> new_shape;
    for (auto i : arange(tensor_.dim())) {
      if (i == dim) {
        new_shape.emplace_back(-1);
        new_shape.emplace_back(factor);
      } else {
        new_shape.emplace_back(tensor_.size(i));
      }
    }
    tensor_ = tensor_.view(new_shape);
    // update frontier
    frontier_.insert(out_it, outer);
    frontier_.insert(out_it, inner);
    frontier_.erase(out_it);
  }

  void handle(Expr* expr) {
    if (auto split = dynamic_cast<Split*>(expr)) {
      handle(split);
    } else if (auto merge = dynamic_cast<Merge*>(expr)) {
      handle(merge);
    } else {
      NVF_THROW("Unsupported transormation in allocation domain");
    }
  }

 public:
  BackwardTraverseFromAllocToLogical(
      at::Tensor tensor,
      const ExpressionEvaluator& ee,
      std::list<IterDomain*>& frontier)
      : tensor_(std::move(tensor)), ee_(ee), frontier_(frontier) {}

  at::Tensor run(
      const std::vector<IterDomain*>& logical,
      const std::vector<IterDomain*>& alloc) {
    auto backward_exprs = StmtSort::getExprsBetween(
        {logical.begin(), logical.end()}, {alloc.begin(), alloc.end()});
    std::reverse(backward_exprs.begin(), backward_exprs.end());
    for (auto expr : backward_exprs) {
      handle(expr);
    }
    return tensor_;
  }
};

// Start from a tensor whose dimensions are consistent with the allocation
// domain of tv, apply a sequence of view/permute to the tensor to transform it
// into a format whose dimensions are consistent with the logical domain of tv.
// For example, if the logical domain is [I1, I2], and the allocation domain is
// [I2*I1], then we will allocate as [I2*I1], then do a tensor.view(I2, I1).t()
// to get a tensor whose semantics is [I1, I2] but its memory is [I2*I1].
// Another example, if the logical domain is [I1*I2] and the allocation domain
// is [I1, I2], then we will allocate as [I1, I2] and do a tensor.view(I1*I2) to
// get a tensor whose semantics is [I1*I2] but memory is [I1,I2]
at::Tensor transformFromAllocationToLogical(
    at::Tensor tensor,
    TensorView* tv,
    const ExpressionEvaluator& ee) {
  FUSER_PERF_SCOPE("allocations::transformFromAllocationToLogical");
  // Ignore reductions because reductions does not exist in tensor's definition
  auto logical = TensorDomain::noReductions(tv->getLogicalDomain());
  auto alloc = TensorDomain::noReductions(tv->getMaybeAllocationDomain());
  // Traverse all affine transformations from allocation domain. Because
  // allocation domain can be before or after the logical domain, we need both a
  // forward and a backward traverse.
  std::list<IterDomain*> frontier(alloc.begin(), alloc.end());
  NVF_ERROR(tensor.dim() == (int64_t)frontier.size());
  tensor = ForwardTraverseFromAllocToLogical(tensor, ee, frontier)
               .run(logical, alloc);
  tensor = BackwardTraverseFromAllocToLogical(tensor, ee, frontier)
               .run(logical, alloc);
  NVF_ERROR(frontier.size() == logical.size());
  // Now that all affine transformations are handled, and frontiers should
  // contain the same set of IDs as logical. We still need to do a final
  // permutation so that their orders are also consistent.
  std::unordered_map<IterDomain*, int64_t> current_dims;
  int64_t counter = 0;
  for (auto id : frontier) {
    current_dims[id] = counter++;
  }
  std::vector<int64_t> dims;
  dims.reserve(frontier.size());
  for (auto id : logical) {
    dims.emplace_back(current_dims.at(id));
  }
  return tensor.permute(dims);
}

std::pair<std::vector<int64_t>, std::vector<int64_t>> inferAllocationShape(
    TensorView* tv,
    const ExpressionEvaluator& expr_eval) {
  std::vector<Val*> symbolic_sizes;
  std::vector<bool> expand_flags;

  // Allocate the allocation domain
  for (const auto id : tv->getMaybeAllocationDomain()) {
    if (id->isReduction() || id->isStride()) {
      continue;
    }

    if (id->isDeviceDim()) {
      symbolic_sizes.push_back(id->container()->oneVal());
    } else {
      symbolic_sizes.push_back(id->getMaybeExpandedExtent());
    }
    if (id->hasExpandedExtent()) {
      NVF_ERROR(
          id->isBroadcast(),
          "Non-broadcast domain should not have an expanded extent: ",
          id->toString());
      expand_flags.push_back(true);
    } else {
      expand_flags.push_back(false);
    }
  }
  return inferShape(tv, symbolic_sizes, expand_flags, expr_eval);
}

} // namespace

std::pair<std::vector<int64_t>, std::vector<int64_t>> inferShapeOfOutput(
    TensorView* tv,
    const ExpressionEvaluator& expr_eval) {
  FUSER_PERF_SCOPE("fusion_executor::allocations::inferShapeOfOutput");
  // Fusion outputs do not come with Allocate and
  // need to be allocated while taking expanded broadcasts into
  // account.

  auto size_stride = inferAllocationShape(tv, expr_eval);
  if (!tv->hasAllocation()) {
    return size_stride;
  }
  auto options =
      c10::TensorOptions().device(c10::Device(c10::DeviceType::Meta));
  auto meta_tensor =
      at::empty_strided(size_stride.first, size_stride.second, options);
  // TODO(jiej): we should refactor it here, there's no need to use
  // meta_tensor at all, size + stride should be used directly in the
  // `transformFromAllocationToLogical`
  meta_tensor = transformFromAllocationToLogical(meta_tensor, tv, expr_eval);
  return {meta_tensor.sizes().vec(), meta_tensor.strides().vec()};
}

TensorShapeInfo inferTensorShapes(
    TensorView* tv,
    const ExpressionEvaluator& expr_eval) {
  // Alias handling:
  auto alias_info = tv->fusion()->getOutputAlias(tv);
  if (alias_info.type != AllocationType::New) {
    // For reuse buffer alias, we need to get the aliased_io's size/stride
    if (alias_info.type == AllocationType::ReuseBuffer) {
      tv = alias_info.aliased_io->as<TensorView>();
    }

    auto val = expr_eval.evaluate(tv);
    NVF_ERROR(val.is<at::Tensor>(), "Output is not a Tensor");
    auto tensor = val.as<at::Tensor>();

    if (!tv->hasAllocation()) {
      return TensorShapeInfo{
          tensor.sizes().vec(),
          tensor.strides().vec(),
          isSharded(tv) ? unshardedSizes(tv, tensor.sizes().vec())
                        : std::vector<int64_t>(),
      };
    }
    auto allocation_size_stride =
        inferAndValidateAllocationSizesAndStrides(tensor, tv, expr_eval);
    return TensorShapeInfo{
        tensor.sizes().vec(),
        tensor.strides().vec(),
        isSharded(tv) ? unshardedSizes(tv, tensor.sizes().vec())
                      : std::vector<int64_t>(),
        allocation_size_stride.first,
        allocation_size_stride.second};
  }

  // Non-alias handling:
  auto allocation_size_stride = inferAllocationShape(tv, expr_eval);
  if (!tv->hasAllocation()) {
    return TensorShapeInfo{
        allocation_size_stride.first,
        allocation_size_stride.second,
        isSharded(tv) ? unshardedSizes(tv, allocation_size_stride.first)
                      : std::vector<int64_t>(),
    };
  }

  auto options =
      c10::TensorOptions().device(c10::Device(c10::DeviceType::Meta));
  auto logical_meta_tensor = at::empty_strided(
      allocation_size_stride.first, allocation_size_stride.second, options);
  // TODO(jiej): we should refactor it here, there's no need to use
  // logical_meta_tensor at all, size + stride should be used directly in the
  // `transformFromAllocationToLogical`
  logical_meta_tensor =
      transformFromAllocationToLogical(logical_meta_tensor, tv, expr_eval);
  return TensorShapeInfo{
      logical_meta_tensor.sizes().vec(),
      logical_meta_tensor.strides().vec(),
      isSharded(tv) ? unshardedSizes(tv, logical_meta_tensor.sizes().vec())
                    : std::vector<int64_t>(),
      allocation_size_stride.first,
      allocation_size_stride.second};
}

} // namespace nvfuser
