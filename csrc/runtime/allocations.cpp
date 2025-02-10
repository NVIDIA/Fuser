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
#include <multidevice/utils.h>
#include <polymorphic_value.h>
#include <runtime/executor_kernel_arg.h>
#include <runtime/executor_utils.h>
#include <tensor_metadata.h>

#include <ir/iostream.h>

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
          dataTypeSize(smem_alloc->buffer()->dtype(), index_type);
      const int64_t size_bytes = size_val.as<int64_t>() * data_size;
      const auto last_byte = first_byte + size_bytes;

      total = std::max(total, last_byte);
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
    std::vector<Val*> symbolic_sizes,
    std::vector<bool> expand_flags,
    const ExpressionEvaluator& expr_eval) {
  FUSER_PERF_SCOPE("fusion_executor::allocations::inferShape");

  // Allocate should be provided for intermediates. We just need to
  // grab a chunk of memory of the size dicatated by
  // Allocate::shape(). Fusion outputs do not come with Allocate and
  // need to be allocated while taking expanded broadcasts into
  // account.

  std::vector<int64_t> concrete_sizes(symbolic_sizes.size(), 0);

  for (const auto i : c10::irange(symbolic_sizes.size())) {
    auto symbolic_size = symbolic_sizes.at(i);
    const auto inferred_val = expr_eval.evaluate(symbolic_size);
    NVF_ERROR(
        inferred_val.hasValue(),
        "Could not launch kernel as program could not infer ",
        symbolic_size->toInlineString(),
        "(",
        symbolic_size->toString(),
        ") for the buffer ",
        tv->toString());

    auto concrete_size = inferred_val.as<int64_t>();
    concrete_sizes.at(i) = concrete_size;
  }

  auto strides = getContiguousStrides(concrete_sizes, expand_flags);

  return {concrete_sizes, strides};
}
} // namespace

std::pair<std::vector<int64_t>, std::vector<int64_t>> inferShapeOfIntermediate(
    const TensorView* tv,
    const kir::Allocate* alloc,
    ExpressionEvaluator& expr_eval) {
  FUSER_PERF_SCOPE("fusion_executor::allocations::inferShapeOfIntermediate");
  // The allocation domain represents the logical allocation domain,
  // bu its actual allocation size may be different, e.g., for
  // supporting halo accesses. The actual size is currently computed
  // when creating the Allocate expr.
  NVF_ERROR(alloc != nullptr);
  const auto& symbolic_sizes = alloc->shape();
  // For intermediate tensors, we just need to allocate a memory chunk
  // of the specified size. Broadcast expansion does not need to be considered.
  const auto expand_flags = std::vector<bool>(symbolic_sizes.size(), false);

  return inferShape(tv, symbolic_sizes, expand_flags, expr_eval);
}

bool fill_allocation_with_nan_ = false;

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
      t.fill_(std::nan(""));
      break;
    case at::ScalarType::ComplexHalf:
    case at::ScalarType::ComplexFloat:
    case at::ScalarType::ComplexDouble:
      t.fill_(c10::complex<double>(std::nan(""), std::nan("")));
      break;
    default:
      NVF_THROW("Unknown dtype");
  }
}

at::Tensor allocateTensor(
    const GlobalBufferInfo& out_info,
    const AliasInfo& alias_info,
    const c10::Device& device,
    ExpressionEvaluator& ee) {
  FUSER_PERF_SCOPE("fusion_executor::allocations::allocateTensor");
  // Handle a fusion with duplicated outputs.
  TensorView* out_tv = out_info.tv;
  if (ee.isKnown(out_tv)) {
    return ee.evaluate(out_tv).as<at::Tensor>();
  }

  std::optional<at::Tensor> aliased_io_tensor = std::nullopt;
  Val* aliased_io = alias_info.aliased_io;
  if (aliased_io != nullptr) {
    NVF_ERROR(
        aliased_io->isFusionInput() || aliased_io->isFusionOutput(),
        aliased_io->toInlineString(),
        " is expected to be a fusion input/output. `ee.evaluate` ",
        "an intermediate tensor may involve GPU computation to materialize it ",
        "to global memory.");
    const PolymorphicValue& aliased_io_val = ee.evaluate(aliased_io);
    NVF_ERROR(
        aliased_io_val.is<at::Tensor>(),
        "Alias io only supports tensor. Found ",
        PolymorphicValue_functions::toString(aliased_io_val));
    aliased_io_tensor = aliased_io_val.as<at::Tensor>();
  }

  switch (alias_info.type) {
    case AllocationType::New: {
      auto alloc_tensor = at::native::empty_strided_cuda(
          out_info.sizes,
          out_info.strides,
          out_info.type,
          c10::nullopt,
          device,
          c10::nullopt);
      if (shouldFillAllocationWithNan()) {
        fillTensorWithNan(alloc_tensor);
      }
      return alloc_tensor;
    }
    case AllocationType::ReuseBuffer:
      // Unlike for `AllocationType::Evaluate`, don't use
      // ExpressionEvaluator to compute the output tensor. This is because
      // the output tensor may hold different data from the input, e.g., an
      // updated running mean.  `ExpressionEvaluator::evaluate(out_tv)`
      // would trigger non-trivial host computation.
      return aliased_io_tensor.value();
    case AllocationType::Evaluate: {
      auto out_tensor = ee.evaluate(out_tv).as<at::Tensor>();
      if (aliased_io_tensor.has_value()) {
        NVF_ERROR(
            out_tensor.is_alias_of(aliased_io_tensor.value()),
            "ExpressionEvaluator failed to evaluate ",
            out_tv->toString(),
            " as an alias of ",
            aliased_io->toString());
        // TODO: Validate output sizes and strides
      }
      return out_tensor;
    }
    default:
      NVF_THROW("Unrecognized AllocationType.");
  }
}

std::vector<at::Tensor> allocateOutputs(
    const Fusion* fusion,
    const std::vector<GlobalBufferInfo>& output_info,
    const c10::Device& device,
    ExpressionEvaluator& ee) {
  FUSER_PERF_SCOPE("fusion_executor::allocations::allocateOutputs");

  const auto num_outs = output_info.size();

  // Sort the outputs so we compute aliases after allocating non-aliases. The
  // order between aliases can be arbitrary. E.g.,
  //
  // ```
  // non_alias_out = ...
  // alias_out_0 = reshape(non_alias_out, ...)
  // alias_out_1 = reshape(alias_out_0, ...)
  // ```
  //
  // It's fine to compute `alias_out_1` before computing `alias_out_0`: when we
  // compute `alias_out_1`, `alias_out_0` will be recursively
  // `ExpressionEvaluator::evaluate`ed. However, `non_alias_out` must be
  // allocated first so `alias_out_*` can refer them.
  std::vector<std::pair<int64_t, Val*>> sorted_outs;
  sorted_outs.reserve(num_outs);
  for (const auto out_index : c10::irange(num_outs)) {
    sorted_outs.emplace_back(out_index, fusion->outputs()[out_index]);
  }
  std::sort(
      sorted_outs.begin(),
      sorted_outs.end(),
      [fusion](
          const std::pair<int64_t, Val*>& lhs,
          const std::pair<int64_t, Val*>& rhs) {
        return (
            fusion->getOutputAlias(lhs.second).type == AllocationType::New &&
            fusion->getOutputAlias(rhs.second).type != AllocationType::New);
      });

  std::vector<at::Tensor> out_tensors(num_outs);
  for (const auto& [out_index, out] : sorted_outs) {
    at::Tensor out_tensor = allocateTensor(
        output_info[out_index], fusion->getOutputAlias(out), device, ee);
    // Bind `out_tensor` so
    // 1. duplicated outputs map to the same tensor,
    // 2. an output that aliases another output can be evaluated via
    // ExpressionEvaluator cheaply.
    // std::cout << "Bind: " << output_info[out_index].tv->toString()
    //           << "\n  With logical: "
    //           << TensorDomain::noReductions(
    //                  output_info[out_index].tv->getLogicalDomain())
    //           << "\n To sizes: " << out_tensor.sizes()
    //           << "\n GBI: " << output_info[out_index].sizes << std::endl;

    ee.bind(out, out_tensor);
    // std::cout << "After" << std::endl;
    out_tensors[out_index] = out_tensor;
  }
  return out_tensors;
}

namespace {
GlobalBufferInfo getBufferInfo(
    ExpressionEvaluator& expr_eval,
    DataType index_dtype,
    TensorView* tv) {
  FUSER_PERF_SCOPE("fusion_executor::allocations::getBufferInfo");
  GlobalBufferInfo info;
  info.tv = tv;
  std::tie(info.sizes, info.strides) = inferShapeOfOutput(info.tv, expr_eval);
  auto dtype =
      (info.tv->dtype() == DataType::Index ? index_dtype : info.tv->dtype());
  info.type = data_type_to_aten(dtype);
  return info;
}

} // namespace
std::vector<GlobalBufferInfo> getBufferInfos(
    ExpressionEvaluator& expr_eval,
    DataType index_dtype,
    const std::vector<Val*>& fusion_outputs) {
  FUSER_PERF_SCOPE("fusion_executor::allocations::getOutbufferInfo");
  std::vector<GlobalBufferInfo> output_buffer_infos;
  output_buffer_infos.reserve(fusion_outputs.size());
  for (const auto out : fusion_outputs) {
    NVF_ERROR(
        out->isA<TensorView>(),
        "Cannot allocate outputs that are not tensors.");

    output_buffer_infos.emplace_back(
        getBufferInfo(expr_eval, index_dtype, out->as<TensorView>()));
  }
  return output_buffer_infos;
}

std::pair<std::vector<int64_t>, std::vector<int64_t>> inferShapeOfOutput(
    TensorView* tv,
    const ExpressionEvaluator& expr_eval) {
  // std::cout << "Processing: TV" << tv->name() << "\n  logical: "
  //           << TensorDomain::noReductions(tv->getLogicalDomain())
  //           << "\n  alloc: "
  //           << TensorDomain::noReductions(tv->getMaybeAllocationDomain())
  //           << std::endl;
  FUSER_PERF_SCOPE("fusion_executor::allocations::inferShapeOfOutput");
  // nvFuser treats allocation domain as the underlying memory format, but it
  // operates on the logical domain. Therefore we need to be able to project
  // through pure stride manipulations from the two domains. We may extend this
  // capability in the future to have non-stride manipulations between the two.
  // This could be used to have nvFuser kernels internally have more complex
  // layouts. However, when returning a tensor back to the framework or when
  // using expression evaluator on tensors, the logical domain must
  // representable exclusively through stride transforms from the allocation
  // domain.
  //
  // For Multi Device scheduling the process can be more complicated than when
  // only considering single GPU allocations. For early development logical
  // domains are sharded, and allocation domains are consistently sharded.
  //
  // This worked well for experimentation, but it means we're changing the
  // logical domain which should be semantic. Therefore to use typical parallel
  // transforms like split, merge, parallelize(DID.), the logical domain will be
  // unsharded but the allocation domain will be sharded.
  //
  // In this scenario logical domain will map to unsharded sizes, we need to
  // (1) project the unsharded dimensions to the logical domain to infer the
  // unsharded sizes of the allocation domain. (2) Modify the allocation domain
  // sizes to change any axis parallelized with DID to size 1 (stride can remain
  // or be changed as well). (3) Project the unsharded allocation domain back to
  // the logical domain.
  //
  // Since this is a more intensive process, we will detect if it needs to be
  // done, if it doesn't need to be done all we need to do is step (3).

  if (tv->hasAllocation()) {
    auto logical_domain = TensorDomain::noReductions(tv->getLogicalDomain());
    auto alloc_domain =
        TensorDomain::noReductions(tv->getMaybeAllocationDomain());

    // Step (1)
    std::vector<Val*> symbolic_sizes(logical_domain.size(), nullptr);
    std::vector<bool> logical_expand_flags(logical_domain.size(), false);

    for (auto id_i : c10::irange(logical_domain.size())) {
      if (logical_domain[id_i]->isDeviceDim() && consistentDomainSharding(tv)) {
        symbolic_sizes[id_i] = logical_domain[id_i]->container()->oneVal();
      } else {
        symbolic_sizes[id_i] = logical_domain[id_i]->getMaybeExpandedExtent();
      }
      if (logical_domain[id_i]->hasExpandedExtent()) {
        logical_expand_flags[id_i] = true;
      }
    }

    auto logical_size_stride =
        inferShape(tv, symbolic_sizes, logical_expand_flags, expr_eval);
    // std::cout << "Logical: " << logical_size_stride << std::endl;
    // Project to allocation

    auto alloc_proj = inferAndValidateProjection(
        logical_size_stride.first,
        logical_size_stride.second,
        logical_domain,
        alloc_domain,
        false,
        expr_eval);

    std::vector<bool> alloc_expand_flags(alloc_domain.size(), false);
    for (auto id_i : c10::irange(alloc_domain.size())) {
      if (alloc_domain[id_i]->hasExpandedExtent()) {
        alloc_expand_flags[id_i] = true;
      }
    }

    // Allocation should be contiguous, not logical. Get contiguous allocation
    // strides.
    alloc_proj.second =
        getContiguousStrides(alloc_proj.first, alloc_expand_flags);
    // std::cout << "Alloc: " << alloc_proj << std::endl;
    alloc_proj =
        removeSharding(alloc_domain, alloc_proj.first, alloc_proj.second);

    if (std::any_of(
            alloc_proj.first.begin(), alloc_proj.first.end(), [](int64_t size) {
              return size > 1;
            })) {
      // std::cout << 0 << std::endl;
      validateContiguity(
          tv->getMaybeAllocationDomain(),
          tv->getContiguity(),
          alloc_proj.first,
          alloc_proj.second);
      // std::cout << 1 << std::endl;
    }

    // std::cout << "Alloc no sharding: " << alloc_proj << std::endl;
    auto logical_proj = inferAndValidateProjection(
        alloc_proj.first,
        alloc_proj.second,
        alloc_domain,
        logical_domain,
        true,
        expr_eval);
    // std::cout << "Final logical proj: " << logical_proj << std::endl;
    return logical_proj;
  }

  auto logical_domain =
      TensorDomain::noReductions(tv->getMaybeAllocationDomain());
  std::vector<Val*> symbolic_sizes(logical_domain.size(), nullptr);
  std::vector<bool> expand_flags(logical_domain.size(), false);

  for (auto id_i : c10::irange(logical_domain.size())) {
    symbolic_sizes[id_i] = logical_domain[id_i]->getMaybeExpandedExtent();
    if (logical_domain[id_i]->hasExpandedExtent()) {
      expand_flags[id_i] = true;
    }
  }
  auto logical_sizes = inferShape(tv, symbolic_sizes, expand_flags, expr_eval);
  // std::cout << "Logical: " << logical_sizes << std::endl;
  if (std::any_of(
          logical_domain.begin(), logical_domain.end(), [](IterDomain* id) {
            return id->isDeviceDim();
          })) {
    logical_sizes = removeSharding(
        logical_domain, logical_sizes.first, logical_sizes.second);
    // std::cout << "Logical no sharding: " << logical_sizes << std::endl;
  }

  if (std::any_of(
          logical_sizes.first.begin(),
          logical_sizes.first.end(),
          [](int64_t size) { return size > 1; })) {
    // std::cout << 2 << std::endl;
    // std::cout << tv->getLogicalDomain() << std::endl;
    // for (auto cont : tv->getContiguity()) {
    //   std::cout << cont << ", ";
    // }
    // std::cout << std::endl;
    // std::cout << logical_sizes.first << std::endl;
    // std::cout << logical_sizes.second << std::endl;
    validateContiguity(
        tv->getLogicalDomain(),
        tv->getContiguity(),
        logical_sizes.first,
        logical_sizes.second);
    // std::cout << 3 << std::endl;
  }

  return logical_sizes;
}

} // namespace nvfuser
