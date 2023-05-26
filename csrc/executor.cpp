// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <executor.h>

#include <codegen.h>
#include <device_lower/analysis/bank_conflict.h>
#include <executor_kernel_arg.h>
#include <executor_utils.h>
#include <instrumentation.h>
#include <ir/all_nodes.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <kernel_ir.h>
#include <utils.h>

#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/llvm_jit_strings.h>
#include <ATen/native/cuda/jit_utils.h>
#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/irange.h>

#include <cmath>
#include <fstream>

namespace nvfuser {

int64_t FusionExecutor::fusion_id_counter_ = 0;

bool fill_allocation_with_nan_ = false;

bool shouldFillAllocationWithNan() {
  return fill_allocation_with_nan_;
}

void setFillAllocationWithNan(bool value) {
  fill_allocation_with_nan_ = value;
}

namespace {

static const char* defineIndexType(PrimDataType index_type) {
  if (index_type == DataType::Int32) {
    return "typedef int nvfuser_index_t;\n";
  } else if (index_type == DataType::Int) {
    return "typedef int64_t nvfuser_index_t;\n";
  } else {
    TORCH_INTERNAL_ASSERT(false, "invalid indexing type: ", index_type);
  }
}

static const char* defineIntegerTypes() {
  return R"(
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef short int int16_t;
typedef unsigned short int uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long long int int64_t;
typedef unsigned long long int uint64_t;
)";
}

static const std::string& includeStdComplex() {
  static std::string result = std::string(R"ESCAPE(
#ifdef __NVCC__
#include <complex>
#endif // __NVCC__
)ESCAPE");
  return result;
}

} // namespace

std::unique_ptr<PrecomputedValues>& FusionExecutor::
    evaluatorPrecomputedValues() {
  if (!evaluator_precomputed_values_) {
    evaluator_precomputed_values_ =
        std::make_unique<PrecomputedValues>(lowered_->kernel());
  }
  return evaluator_precomputed_values_;
}

std::string FusionExecutor::getStructuredCode(
    const std::string& kernel_str,
    PrimDataType index_type) const {
  // generating cuda code;
  std::string code = "";
  code += includeStdComplex();
  code += std::string("namespace ") + FusionExecutor::kernelNamespace() +
      " {\n" + defineIntegerTypes() + defineIndexType(index_type) +
      executor_utils::kernelPreamble() + kernel_str + "}\n";

  if (isDebugDumpEnabled(DebugDumpOption::CudaKernel)) {
    std::cout << "\n======= Codegen output for kernel: " << kernelName()
              << " =======\n\n"
              << kernel_str << "\n======================================\n\n";
  } else if (isDebugDumpEnabled(DebugDumpOption::CudaFull)) {
    std::cout << "\n======= Codegen output for kernel: " << kernelName()
              << " =======\n\n"
              << code << "\n======================================\n\n";
  }
  if (isDebugDumpEnabled(DebugDumpOption::CudaToFile) ||
      isDebugDumpEnabled(DebugDumpOption::DebugInfo)) {
    std::stringstream file_name;
    file_name << "__tmp_kernel" << fusion_id_ << ".cu";
    std::cout << "PRINTING: " << file_name.str() << std::endl;
    std::ofstream out(file_name.str());
    out << code << std::endl;
    out.close();
  }

  return code;
}

std::string FusionExecutor::getStructuredCode() const {
  return getStructuredCode(kernelString(), kernel()->indexType());
}

// TODO: come up with a more user friendly interface
void FusionExecutor::debugCompileFusionFromStr(
    Fusion* fusion,
    const std::string& code,
    const std::string& name,
    int id,
    CompileOptions options) {
  options_ = options;

  if (isDebugDumpEnabled(DebugDumpOption::FusionIr)) {
    fusion->print();
  } else if (isDebugDumpEnabled(DebugDumpOption::FusionIrMath)) {
    fusion->printMath();
  }

  if (isDebugDumpEnabled(DebugDumpOption::CudaFull)) {
    std::cout << "\n==== codegen output for kernel: " << kernelName()
              << " ====" << std::endl
              << code << std::endl
              << "======================================\n"
              << std::endl;
  }

  lowered_ = std::make_unique<GpuLower>(fusion);
  const auto kernel = lowered_->kernel();
  fusion_ = lowered_->kernel();

  fusion_id_ = id;
  setUsedTVs();

  if (isDebugDumpEnabled(DebugDumpOption::KernelIr)) {
    kernel->print();
  }

  const auto& kernel_summary = kernel->summary();

  if (!kernel_summary.static_smem_allocations.empty()) {
    ExpressionEvaluator static_evaluator;
    const auto static_smem_size = computeSharedMemory(
        static_evaluator, kernel_summary.static_smem_allocations);
    TORCH_INTERNAL_ASSERT(
        static_smem_size < max_static_smem_,
        "The static shared memory allocation is larger than available memory.");
  }

  std::tie(compiled_kernel_, last_compiler_log_, last_compiled_binary_) =
      executor_utils::getCompiledKernel(c10::nullopt, code, name, fusion_id_);
  TORCH_INTERNAL_ASSERT(
      fusion_id_ > 0, "assign a fusion_id_ <= 0 is not accepted.");
}

void FusionExecutor::compileFusion(
    Fusion* fusion,
    const KernelArgumentHolder& args,
    const LaunchParams& launch_constraints,
    CompileParams compile_params) {
  FUSER_PERF_SCOPE("compileFusion");

  TORCH_INTERNAL_ASSERT(
      !fusion->outputs().empty(), "No output found for this kernel, aborting.");

  for (auto out : fusion->outputs()) {
    TORCH_INTERNAL_ASSERT(
        out->getValType() == ValType::TensorView,
        "Output types from fusions that are not tensors are not supported at this point.");

    const auto maybe_rfactor_domain =
        out->as<TensorView>()->getMaybeRFactorDomain();
    // walking through outputs to see if output shapes are dependent on
    // non-tensor inputs. For which case, we should have disabled output
    // allocation, since the caching id only looks at tensor shapes.
    // See issue https://github.com/csarofeen/pytorch/issues/2002
    std::vector<Val*> output_extents;
    for (const auto id : maybe_rfactor_domain) {
      Val* extent = nullptr;
      if (id->isReduction() || id->isStride()) {
        continue;
      } else if (id->isBroadcast() && id->hasExpandedExtent()) {
        extent = id->expandedExtent();
      } else {
        extent = id->extent();
      }
      output_extents.emplace_back(extent);
    }
    auto dependencies = InputsOf::outputs(fusion, output_extents);
    if (std::any_of(dependencies.begin(), dependencies.end(), [](Val* val) {
          return val->isFusionInput();
        })) {
      // TODO: parameter cache is too big a hammer here. We should consider
      // separate the caching logic of output sizes & launch params. Since
      // output size dependency should only invalidate the output sizes
      disable_parameter_cache_ = true;
      break;
    }
  }

  if (isDebugDumpEnabled(DebugDumpOption::FusionIr)) {
    fusion->print();
  } else if (isDebugDumpEnabled(DebugDumpOption::FusionIrMath)) {
    fusion->printMath();
  }

  // TODO: refactor the options_ passed through
  options_.device =
      c10::Device(c10::DeviceType::CUDA, (int8_t)args.getDeviceIndex());

  // Set the index type of compile params if not already set. If set,
  // make sure the compile param type is valid with the given kernel
  // arguments.
  auto arg_index_type = args.getSmallestIndexTypeOfArguments();
  if (compile_params.index_type.has_value()) {
    // If the int32 compilation is requested, but the arguments demand
    // int64, that's an error
    TORCH_INTERNAL_ASSERT(
        !(compile_params.index_type.value() == PrimDataType::Int32 &&
          arg_index_type == PrimDataType::Int),
        "Compilation with int32 is requested but int64 is required for the arguments");
  } else if (arg_index_type == PrimDataType::Int) {
    // If the given compile option doesn't specify the index type, and
    // the arguments require 64-bit indexing, we need to use 64-bit
    // indexing. Note that if the arg type is 32-bit, it doesn't mean
    // it's safe to use 32-bit for the whole kernel, so unless it's
    // specified through CompileParams, we do not use 32-bit indexing.
    compile_params.index_type = arg_index_type;
  }

  c10::DeviceGuard dg(options_.device);

  TORCH_INTERNAL_ASSERT(
      options_.device.is_cuda(), "Provided device to CUDA fuser is the CPU.");
  auto properties = at::cuda::getDeviceProperties(options_.device.index());
  // TODO: These properties should be set as part of the constructor so that it
  // can be const
  device_smem_limit_ = static_cast<int64_t>(properties->sharedMemPerBlockOptin);
  warp_size_ = properties->warpSize;

  lowered_ = std::make_unique<GpuLower>(fusion, compile_params);

  const auto kernel = lowered_->kernel();
  fusion_ = lowered_->kernel()->as<Fusion>();

  fusion_id_ = ++fusion_id_counter_;
  setUsedTVs();

  if (isDebugDumpEnabled(DebugDumpOption::KernelIr)) {
    kernel->print();
  }

  if (isDebugDumpEnabled(DebugDumpOption::BankConflictInfo)) {
    auto bank_conflict_info = getBankConflictInfo(kernel);
    if (bank_conflict_info.empty()) {
      std::cout << "===== No bank confliction =====" << std::endl;
    } else {
      std::cout << "======= Bank confliction =======" << std::endl;
      for (auto info : bank_conflict_info) {
        std::cout << "Expr: " << info.first->toString() << std::endl;
        auto conflict = info.second;
        if (conflict.first > 1) {
          std::cout << "input conflict: " << conflict.first << " way, ";
        }
        if (conflict.second > 1) {
          std::cout << "output conflict: " << conflict.second << " way";
        }
        std::cout << std::endl;
      }
      std::cout << "================================" << std::endl;
    }
  }

  kernel_code_ = codegen::generateCudaKernel(kernel, kernelName());

  auto load_external_code = [](const char* external_code_path) {
    std::cout << "--------> Compiling external cuda code: "
              << external_code_path << std::endl;
    std::ifstream cuda_src(external_code_path);
    std::stringstream buffer;
    buffer << cuda_src.rdbuf();
    return buffer.str();
  };
  auto external_code_path = std::getenv("PYTORCH_NVFUSER_EXTERNAL_SRC");
  const auto structured_code = external_code_path
      ? load_external_code(external_code_path)
      : getStructuredCode();

  const auto& kernel_summary = kernel->summary();

  // We currently shouldn't allocate any more shared mem
  //  tensors statically but could keep this path if
  //  needed in later development.
  if (!kernel_summary.static_smem_allocations.empty()) {
    ExpressionEvaluator static_evaluator;
    const auto static_smem_size = computeSharedMemory(
        static_evaluator, kernel_summary.static_smem_allocations);
    TORCH_INTERNAL_ASSERT(
        static_smem_size < max_static_smem_,
        "The static shared memory allocation is larger than available memory.");
  }

  if (kernel_summary.has_dynamic_local_memory_allocations) {
    std::stringstream ss;
    ss << "Allocations must be based on constant integers for local memory. However, found: ";
    for (auto alloc : kernel_summary.dynamic_lmem_allocations) {
      ss << alloc->buffer()->toString() << ", ";
    }
    ss << " have dynamic allocations but are placed in local memory.";
    TORCH_INTERNAL_ASSERT(false, ss.str());
  }

  // TODO: pass block_size here;
  std::optional<int64_t> dynamic_smem = std::nullopt;
  std::optional<int64_t> block_size = std::nullopt;
  if (!args.empty()) {
    auto expr_eval = executor_utils::bindInputs(args, kernel);
    auto launch_params =
        computeLaunchParams(launch_constraints, expr_eval, warp_size_);
    block_size = launch_params.nThreads();
    dynamic_smem = launch_params.smem();
    TORCH_INTERNAL_ASSERT(
        block_size > 0, "launch param inferred block size < 0");
  }

  // TODO: high water mark should be computed via occupancy API after
  // compilation.

  // Basically setting high water martk as 1 when we don't provide args for
  // compilation, it will just generate a kernel that gets ditched at the first
  // run - not great. We should have better heuristics.
  block_size_high_water_mark_ = std::max<int64_t>(
      (block_size.has_value() ? block_size.value() : 1),
      block_size_high_water_mark_);
  maxrregcount_high_water_mark_ = compile_params.maxrregcount;
  std::tie(compiled_kernel_, last_compiler_log_, last_compiled_binary_) =
      executor_utils::getCompiledKernel(
          kernel_code_,
          structured_code,
          getCanonicalKernelName(),
          fusion_id_,
          block_size,
          maxrregcount_high_water_mark_,
          save_compiled_binary_ || isDebugDumpEnabled(DebugDumpOption::Sass));
  TORCH_INTERNAL_ASSERT(
      fusion_id_ > 0, "failed to assign a fusion_id_ after compilation.");

  // These should be nullopt at this point, but reset just in case
  resetCompiledKernelProperties();

  // If the dynamic shmem size is known, make sure the compiled kernel
  // has at least that size of dynamic shmem
  if (dynamic_smem.has_value()) {
    ensureAvailableDynamicSmemSize(dynamic_smem.value());
  }

  if (isDebugDumpEnabled(DebugDumpOption::Sass)) {
    std::cout << disassembledKernelSASS() << std::endl;
  }
}

namespace {

void fillTensorWithNan(at::Tensor& t) {
  switch (t.scalar_type()) {
    case at::ScalarType::Byte:
      t.fill_(0xFF);
      break;
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
    case at::ScalarType::Bool:
      t.fill_(true);
      break;
    case at::ScalarType::Half:
    case at::ScalarType::Float:
    case at::ScalarType::Double:
    case at::ScalarType::BFloat16:
      t.fill_(std::nan(""));
      break;
    case at::ScalarType::ComplexHalf:
    case at::ScalarType::ComplexFloat:
    case at::ScalarType::ComplexDouble:
      t.fill_(c10::complex<double>(std::nan(""), std::nan("")));
      break;
    default:
      TORCH_INTERNAL_ASSERT(false, "Unknown dtype");
  }
}

std::vector<int64_t> getContiguousStrides(
    const std::vector<int64_t>& sizes,
    const std::vector<bool>& expand_flags) {
  TORCH_INTERNAL_ASSERT(sizes.size() == expand_flags.size());

  std::vector<int64_t> strides(sizes.size());
  int64_t cur_stride = 1;
  for (auto i = sizes.size(); i > 0; --i) {
    auto size = sizes.at(i - 1);
    TORCH_INTERNAL_ASSERT(
        size >= 0,
        "Positive size is assumed non-negative but received: ",
        size);

    int64_t stride = cur_stride;

    // If expanded, stride is 0
    if (expand_flags.at(i - 1)) {
      stride = 0;
    } else if (size == 0) {
      // If the size is 0, the stride is 1
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
    ExpressionEvaluator& expr_eval) {
  FUSER_PERF_SCOPE("inferShape");

  // Allocate should be provided for intermediates. We just need to
  // grab a chunk of memory of the size dicatated by
  // Allocate::shape(). Fusion outputs do not come with Allocate and
  // need to be allocated while taking expanded broadcasts into
  // account.

  std::vector<int64_t> concrete_sizes(symbolic_sizes.size(), 0);

  for (const auto i : c10::irange(symbolic_sizes.size())) {
    auto symbolic_size = symbolic_sizes.at(i);
    const auto inferred_val = expr_eval.evaluate(symbolic_size);
    TORCH_INTERNAL_ASSERT(
        inferred_val.has_value(),
        "Could not launch kernel as program could not infer ",
        symbolic_size->toInlineString(),
        "(",
        symbolic_size->toString(),
        ") for the buffer ",
        tv->toString());

    auto concrete_size = inferred_val->as<int64_t>();
    concrete_sizes.at(i) = concrete_size;
  }

  auto strides = getContiguousStrides(concrete_sizes, expand_flags);

  return {concrete_sizes, strides};
}

// Infer the shape of an intemediate tensor using kir::Allocate
std::pair<std::vector<int64_t>, std::vector<int64_t>> inferShapeOfIntermediate(
    const TensorView* tv,
    ExpressionEvaluator& expr_eval) {
  auto alloc_dom = TensorDomain::noReductions(tv->getMaybeAllocationDomain());
  std::vector<nvfuser::Val*> symbolic_sizes;
  symbolic_sizes.reserve(alloc_dom.size());
  for (auto id : alloc_dom) {
    if (id->isBroadcast()) {
      symbolic_sizes.emplace_back(id->container()->oneVal());
    } else {
      symbolic_sizes.emplace_back(id->extent());
    }
  }

  // For intermediate tensors, we just need to allocate a memory chunk
  // of the specified size. Broadcast expansion does not need to be considered.
  const auto expand_flags = std::vector<bool>(symbolic_sizes.size(), false);

  return inferShape(tv, symbolic_sizes, expand_flags, expr_eval);
}

// Infer the sizes and strides of an output tensor
std::pair<std::vector<int64_t>, std::vector<int64_t>> inferShapeOfOutput(
    const TensorView* tv,
    ExpressionEvaluator& expr_eval) {
  // Fusion outputs do not come with Allocate and
  // need to be allocated while taking expanded broadcasts into
  // account.

  std::vector<Val*> symbolic_sizes;
  std::vector<bool> expand_flags;

  // Allocate the allocation domain
  for (const auto id : tv->getMaybeAllocationDomain()) {
    if (id->isReduction() || id->isStride()) {
      continue;
    }
    symbolic_sizes.push_back(id->getMaybeExpandedExtent());
    if (id->hasExpandedExtent()) {
      TORCH_INTERNAL_ASSERT(
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

namespace {

class ForwardTraverseFromAllocToRFactor {
  at::Tensor tensor_;
  TensorView* tv_;
  ExpressionEvaluator& ee_;
  std::list<IterDomain*>& frontier_;

  // Forward traverse split from allocation to rFactor. Needs to, for example,
  // view tensor with shape [..., 15, ...] as [..., 3, 5, ...]
  void handle(Split* split) {
    auto in = split->in();
    auto inner = split->inner();
    auto outer = split->outer();
    auto factor = ee_.evaluate(split->factor())->as<int64_t>();
    auto in_it = std::find(frontier_.begin(), frontier_.end(), in);
    // TORCH_INTERNAL_ASSERT(in_it != frontier_.end());
    if (in_it == frontier_.end()) {
      // TODO: We should get rid of this return and enable the above assert.
      // Note [Allocation domain on both side of rFactor]
      // For cases where the allocation domain is on both side of rFactor, for
      // example, in Tensor3d_To_NHWC4d_FwdBwd_CUDA:
      // [alloc,root]   [alloc,root]           [root]
      //          \     /                      /    |
      //         [rFactor]                  split   [rFactor]
      //                                    /  \         |
      //                      [alloc,rFactor] [rFactor]  |
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
    for (auto i : c10::irange(tensor_.dim())) {
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

  // Forward traverse split from allocation to rFactor. Needs to, for example,
  // view tensor with shape [..., 3, 5, ...] as [..., 15, ...]
  void handle(Merge* merge) {
    auto inner = merge->inner();
    auto outer = merge->outer();
    auto out = merge->out();
    auto inner_it = std::find(frontier_.begin(), frontier_.end(), inner);
    auto outer_it = std::find(frontier_.begin(), frontier_.end(), outer);
    // TORCH_INTERNAL_ASSERT(inner_it != frontier_.end());
    // TORCH_INTERNAL_ASSERT(outer_it != frontier_.end());
    if (inner_it == frontier_.end() || outer_it == frontier_.end()) {
      // TODO: see [Allocation domain on both side of rFactor]
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
    for (auto i : c10::irange(tensor_.dim())) {
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
      TORCH_INTERNAL_ASSERT(
          false, "Unsupported transormation in allocation domain");
    }
  }

 public:
  ForwardTraverseFromAllocToRFactor(
      at::Tensor tensor,
      TensorView* tv,
      ExpressionEvaluator& ee,
      std::list<IterDomain*>& frontier)
      : tensor_(std::move(tensor)), tv_(tv), ee_(ee), frontier_(frontier) {}

  at::Tensor run(
      const std::vector<IterDomain*>& rfactor,
      const std::vector<IterDomain*>& alloc) {
    auto forward_exprs = StmtSort::getExprsBetween(
        tv_->fusion(),
        {alloc.begin(), alloc.end()},
        {rfactor.begin(), rfactor.end()});
    for (auto expr : forward_exprs) {
      handle(expr);
    }
    return tensor_;
  }
};

// Backward traverse is similar to forward traverse, but we need to do opposite
// transformations.
class BackwardTraverseFromAllocToRFactor {
  at::Tensor tensor_;
  TensorView* tv_;
  ExpressionEvaluator& ee_;
  std::list<IterDomain*>& frontier_;

  // Backward traverse split from allocation to rFactor. Needs to, for example,
  // view tensor with shape [..., 3, 5, ...] as [..., 15, ...]
  void handle(Split* split) {
    auto inner = split->inner();
    auto outer = split->outer();
    auto in = split->in();
    auto inner_it = std::find(frontier_.begin(), frontier_.end(), inner);
    auto outer_it = std::find(frontier_.begin(), frontier_.end(), outer);
    // TORCH_INTERNAL_ASSERT(inner_it != frontier_.end());
    // TORCH_INTERNAL_ASSERT(outer_it != frontier_.end());
    if (inner_it == frontier_.end() || outer_it == frontier_.end()) {
      // TODO: see [Allocation domain on both side of rFactor]
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
    for (auto i : c10::irange(tensor_.dim())) {
      if (i == left) {
        new_shape.emplace_back(-1);
      } else if (i != left + 1) {
        new_shape.emplace_back(tensor_.size(i));
      }
    }
    tensor_ = tensor_.view(new_shape);
    // update frontier
    if (inner_dim < outer_dim) {
      *inner_it = in;
      frontier_.erase(outer_it);
    } else {
      *outer_it = in;
      frontier_.erase(inner_it);
    }
  }

  // Backward traverse split from allocation to rFactor. Needs to, for example,
  // view tensor with shape [..., 15, ...] as [..., 3, 5, ...]
  void handle(Merge* merge) {
    auto out = merge->out();
    auto inner = merge->inner();
    auto outer = merge->outer();
    auto factor = ee_.evaluate(inner->extent())->as<int64_t>();
    auto out_it = std::find(frontier_.begin(), frontier_.end(), out);
    // TORCH_INTERNAL_ASSERT(out_it != frontier_.end());
    if (out_it == frontier_.end()) {
      // TODO: see [Allocation domain on both side of rFactor]
      return;
    }
    // view tensor
    int64_t dim = std::distance(frontier_.begin(), out_it);
    std::vector<int64_t> new_shape;
    for (auto i : c10::irange(tensor_.dim())) {
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
      TORCH_INTERNAL_ASSERT(
          false, "Unsupported transormation in allocation domain");
    }
  }

 public:
  BackwardTraverseFromAllocToRFactor(
      at::Tensor tensor,
      TensorView* tv,
      ExpressionEvaluator& ee,
      std::list<IterDomain*>& frontier)
      : tensor_(std::move(tensor)), tv_(tv), ee_(ee), frontier_(frontier) {}

  at::Tensor run(
      const std::vector<IterDomain*>& rfactor,
      const std::vector<IterDomain*>& alloc) {
    auto backward_exprs = StmtSort::getExprsBetween(
        tv_->fusion(),
        {rfactor.begin(), rfactor.end()},
        {alloc.begin(), alloc.end()});
    std::reverse(backward_exprs.begin(), backward_exprs.end());
    for (auto expr : backward_exprs) {
      handle(expr);
    }
    return tensor_;
  }
};

// Start from a tensor whose dimensions are consistent with the allocation
// domain of tv, apply a sequence of view/permute to the tensor to transform it
// into a format whose dimensions are consistent with the rFactor domain of tv.
// For example, if the rFactor domain is [I1, I2], and the allocation domain is
// [I2*I1], then we will allocate as [I2*I1], then do a tensor.view(I2, I1).t()
// to get a tensor whose semantics is [I1, I2] but its memory is [I2*I1].
// Another example, if the rFactor domain is [I1*I2] and the allocation domain
// is [I1, I2], then we will allocate as [I1, I2] and do a tensor.view(I1*I2) to
// get a tensor whose semantics is [I1*I2] but memory is [I1,I2]
at::Tensor transformOutputFromAllocationToRFactor(
    at::Tensor tensor,
    TensorView* tv,
    ExpressionEvaluator& ee) {
  // Ignore reductions because reductions does not exist in tensor's definition
  auto rfactor = TensorDomain::noReductions(tv->getMaybeRFactorDomain());
  auto alloc = TensorDomain::noReductions(tv->getMaybeAllocationDomain());
  // Traverse all affine transformations from allocation domain. Because
  // allocation domain can be before or after the rFactor domain, we need both a
  // forward and a backward traverse.
  std::list<IterDomain*> frontier(alloc.begin(), alloc.end());
  TORCH_INTERNAL_ASSERT(tensor.dim() == (int64_t)frontier.size());
  tensor = ForwardTraverseFromAllocToRFactor(tensor, tv, ee, frontier)
               .run(rfactor, alloc);
  tensor = BackwardTraverseFromAllocToRFactor(tensor, tv, ee, frontier)
               .run(rfactor, alloc);
  TORCH_INTERNAL_ASSERT(frontier.size() == rfactor.size());
  // Now that all affine transformations are handled, and frontiers should
  // contain the same set of IDs as rfactor. We still need to do a final
  // permutation so that their orders are also consistent.
  std::unordered_map<IterDomain*, int64_t> current_dims;
  int64_t counter = 0;
  for (auto id : frontier) {
    current_dims[id] = counter++;
  }
  std::vector<int64_t> dims;
  dims.reserve(frontier.size());
  for (auto id : rfactor) {
    dims.emplace_back(current_dims.at(id));
  }
  return tensor.permute(dims);
}

} // namespace

// Allocate output tensors for a given kernel. Outputs may alias inputs, in
// that case output tensors are shallow copies of the aliased inputs
std::vector<at::Tensor> allocOutputs(
    const kir::Kernel* kernel,
    const std::vector<FusionExecutor::GlobalBufferInfo>& output_info,
    const std::vector<std::pair<int, int>>& output_to_input_aliases,
    const KernelArgumentHolder& inputs,
    const c10::Device& device,
    ExpressionEvaluator& ee) {
  FUSER_PERF_SCOPE("ExecutorRunFusion::OutputAlloc");

  std::vector<at::Tensor> outputs;

  for (const auto output_idx : c10::irange(output_info.size())) {
    const auto& buf_info = output_info.at(output_idx);

    auto alias_it = std::find_if(
        output_to_input_aliases.begin(),
        output_to_input_aliases.end(),
        [&](const auto output_to_input) {
          return output_to_input.first == (int)output_idx;
        });

    // Note: aliased output is not returned as output. But we still need it
    // for kernel execution, so would need to push them to args
    if (alias_it != output_to_input_aliases.end()) {
      auto aliased_input_index = alias_it->second;
      auto tensor_arg_abstract = dynamic_cast<const TensorArgAbstract*>(
          inputs.at(aliased_input_index));
      TORCH_INTERNAL_ASSERT(
          tensor_arg_abstract, "alias io only supports tensor");
      outputs.emplace_back(tensor_arg_abstract->getTensor());
    } else if (kernel->outputs().at(output_idx)->isFusionInput()) {
      // pushing empty tensor for trivial forwarding. Since we handle this in
      // integration, see step 1 - note [trivial forwarding]
      auto alloc_dom =
          TensorDomain::noReductions(kernel->outputs()
                                         .at(output_idx)
                                         ->as<TensorView>()
                                         ->getMaybeAllocationDomain());
      const auto tensor_options =
          at::TensorOptions().dtype(at::kFloat).device(device);
      outputs.emplace_back(
          at::empty(std::vector<int64_t>(alloc_dom.size(), 0), tensor_options));
    } else {
      auto alloc_tensor = at::native::empty_strided_cuda(
          buf_info.sizes,
          buf_info.strides,
          buf_info.type,
          c10::nullopt,
          device,
          c10::nullopt);
      if (shouldFillAllocationWithNan()) {
        fillTensorWithNan(alloc_tensor);
      }
      outputs.emplace_back(transformOutputFromAllocationToRFactor(
          alloc_tensor, buf_info.tv, ee));
    }
  }

  return outputs;
}

} // namespace

int64_t FusionExecutor::computeSharedMemory(
    ExpressionEvaluator& expr_eval,
    const std::vector<const kir::Allocate*>& buffers,
    bool align_padding,
    int64_t total) {
  FUSER_PERF_SCOPE("computeSharedMemory");
  for (auto smem_alloc : buffers) {
    // If this buffer aliases another buffer,
    // then do not allocate memory for this buffer.
    if (smem_alloc->alias() == nullptr) {
      const auto inferred_val = expr_eval.evaluate(smem_alloc->size());
      if (inferred_val.has_value()) {
        const auto data_size =
            static_cast<int64_t>(dataTypeSize(smem_alloc->buffer()->dtype()));
        // Add padding to align dynamic shared memory
        if (align_padding) {
          const int align_size = 16; // always align to 16B/128b.
          total = ceilDiv(total, align_size) * align_size;
        }
        total += inferred_val->as<int64_t>() * data_size;
      } else {
        TORCH_INTERNAL_ASSERT(
            false,
            "Failed to evaluate the size ",
            smem_alloc->size(),
            " of shared memory buffer - T",
            smem_alloc->buffer()->name());
      }
    }
  }
  return total;
}

LaunchParams FusionExecutor::computeLaunchParams(
    const LaunchParams& launch_constraints,
    ExpressionEvaluator& expr_eval,
    const int64_t warp_size) {
  FUSER_PERF_SCOPE("FusionExecutor::ComputeLaunchParams");
  TORCH_INTERNAL_ASSERT(warp_size > 0, "WARP_SIZE should be larger than 0");

  LaunchParams launch_params;

  auto data_cache = compileTimeDataCache();

  auto lower = lowered_.get();
  auto& used_tvs = getUsedTVs();
  auto parallel_binding_ids_entry =
      executor_utils::caching::ExecutorCompileTimeEntry<
          executor_utils::caching::ParallelBindingIterDomains>(
          data_cache, [&used_tvs, &lower]() {
            return std::make_unique<std::vector<IterDomain*>>(
                executor_utils::getParallelBindingsIterDomains(
                    lower, used_tvs));
          });
  auto& parallel_binding_ids = parallel_binding_ids_entry.get();

  auto parallel_iter_extent_entry =
      executor_utils::caching::ExecutorCompileTimeEntry<
          executor_utils::caching::ParallelIterExtentMap>(
          data_cache, [&parallel_binding_ids]() {
            return executor_utils::getParallelIterExtents(parallel_binding_ids);
          });
  auto& parallel_iter_extents = parallel_iter_extent_entry.get();

  const auto& simplified_parallel_iter_extents =
      lower->parallelDimensionMap().getMap();

  // TODO: Need to redesign this part a bit to
  //   find the right place to trigger evaluate
  if (expr_eval.precomputedValues()) {
    expr_eval.precomputedValues()->bindParallelExtents(
        parallel_iter_extents, launch_constraints);
    expr_eval.precomputedValues()->evaluate();
  }

  // If any dimension was set in launch constraints we need to run through
  // IterDomains that have been parallelized, and bind those values. Or make
  // sure if they could be inferred the inference matches what was set.
  for (auto& entry : parallel_iter_extents) {
    auto p_type = entry.first;
    if (launch_constraints.hasDim(p_type)) {
      auto parallel_extents = entry.second;
      for (auto extent : parallel_extents) {
        auto inferred_val = expr_eval.evaluate(extent);
        if (inferred_val.has_value()) {
          // This value could have been inferred, make sure it was set right.
          bool valid = inferred_val->as<int64_t>() ==
                  launch_constraints.getDim(p_type) ||
              launch_constraints.getRawVal(p_type) == -1;
          if (!useFallback() && !valid) {
            TORCH_WARN_ONCE(
                "Cannot validate parallelization scheme, "
                "this may be due to mixed broadcast axes that are parallelized.");
          }
        } else if (!expr_eval.precomputedValues()) {
          expr_eval.bind(extent, launch_constraints.getDim(p_type));
        }
        if (!launch_params.hasDim(p_type)) {
          // Bind the launch constraint into our evaluation context
          launch_params.bind(launch_constraints.getDim(p_type), p_type);
          // Makes sure the p-types bound to evaluators are the
          //  final values that will become the actual launch
          //  param size to ensure accurate smem buffer size
          //  computation.
          expr_eval.bind(p_type, launch_constraints.getDim(p_type));
        }
      }
    }
  }

  // Run through the rest of the parallel IterDomains and infer their size
  for (auto [p_type, extent] : simplified_parallel_iter_extents) {
    FUSER_PERF_SCOPE("FusionExecutor::ParallelBindingResolution");
    auto val = expr_eval.evaluate(extent);
    TORCH_INTERNAL_ASSERT(
        val.has_value(),
        "Tried to evaluate the extent, ",
        extent->toInlineString(),
        " for the ptype: ",
        p_type,
        " to set launch bounds but could not.");

    if (val->as<int64_t>() > 0) {
      expr_eval.bind(p_type, val->as<int64_t>());
      launch_params.bind(val->as<int64_t>(), p_type);
    }
  }

  // Re-run the integer machine with all
  //  the thread sizes now determined.
  if (expr_eval.precomputedValues()) {
    expr_eval.precomputedValues()->evaluate();
  }

  const auto kernel = lowered_->kernel();
  const auto& kernel_summary = kernel->summary();

  // Calculate Dynamic Shared Memory Size
  // Add workspace for reduction and broadcast
  int64_t reduction_broadcast_workspace = 0;
  const bool has_workspace = kernel_summary.has_block_reductions ||
      kernel_summary.has_grid_reductions ||
      kernel_summary.has_block_broadcasts || kernel_summary.has_grid_broadcasts;
  if (has_workspace &&
      kernel_summary.largest_smem_data_type != DataType::Null) {
    // Not using nThreads here since it does not handle uninitialized value

    // TODO: here is an optimization opportunity since welford uses int64_t for
    // N while the data type is not neccessarily double. But it may need more
    // work on the alignment
    const int welford_factor =
        kernel_summary.has_block_welford || kernel_summary.has_grid_welford ? 3
                                                                            : 1;
    reduction_broadcast_workspace =
        (int64_t)dataTypeSize(kernel_summary.largest_smem_data_type) *
        welford_factor * launch_params.bdimx() * launch_params.bdimy() *
        launch_params.bdimz();

    if (kernel_summary.has_outer_grouped_grid_welford) {
      reduction_broadcast_workspace = std::max(
          reduction_broadcast_workspace,
          (int64_t)kernel_summary.outer_grouped_grid_welford_largest_smem_size);
    }
  }

  const auto dynamic_smem_size = computeSharedMemory(
      expr_eval,
      kernel_summary.dynamic_smem_allocations,
      true,
      reduction_broadcast_workspace);

  // Check that requested smem size can be dynamically allocated.
  //  This check is only done once a kernel has been compiled, since
  //  maybe_available_dynamic_smem_ needs to be evaluated on
  //  a compiled kernel.
  if (compiled()) {
    validateDynamicSmemSize(dynamic_smem_size);
  }

  launch_params.setSmem(dynamic_smem_size);

  return launch_params;
}

std::vector<FusionExecutor::GlobalBufferInfo> FusionExecutor::
    getIntermediateBufferInfo(ExpressionEvaluator& expr_eval) {
  FUSER_PERF_SCOPE("FusionExecutor::GetIntermediateBufferInfo");

  std::vector<GlobalBufferInfo> global_buffers;

  const auto kernel = lowered_->kernel();
  const auto& kernel_summary = kernel->summary();

  for (auto alloc : kernel_summary.global_allocations) {
    TORCH_INTERNAL_ASSERT(
        alloc->buffer()->isA<TensorView>(),
        "Cannot allocate global buffers that are not tensors.");
    auto tv = alloc->buffer()->as<TensorView>();
    if (tv->isFusionOutput()) {
      continue;
    }
    GlobalBufferInfo info;
    info.zero_init = alloc->zeroInit();
    std::tie(info.sizes, info.strides) =
        inferShapeOfIntermediate(tv, expr_eval);
    info.type = data_type_to_aten(tv->dtype());

    // Remember the tensor buffer used for storing kernel profile
    if (isOptionEnabled(EnableOption::KernelProfile) &&
        tv == kernel->profile().getBuffer()) {
      info.is_profile_buffer = true;
    }

    global_buffers.emplace_back(info);
  }

  return global_buffers;
}

std::vector<at::Tensor> FusionExecutor::allocOutputSpace(
    const at::ArrayRef<c10::IValue>& inputs) {
  auto kernel_inputs = KernelArgumentHolder::createKernelArgumentHolder(inputs);
  auto expr_eval =
      executor_utils::bindInputs(kernel_inputs, lowered_->kernel());

  auto input_alias_indices_entry =
      executor_utils::caching::ExecutorCompileTimeEntry<
          executor_utils::caching::InputAliasIndices>(
          compileTimeDataCache(), [&]() {
            return std::make_unique<std::vector<std::pair<int, int>>>(
                fusion_->getOutputToInputAliasIndices());
          });

  const auto& output_to_input_aliases = input_alias_indices_entry.get();

  auto output_info =
      getOutputBufferInfo(kernel_inputs, expr_eval, output_to_input_aliases);

  return allocOutputs(
      kernel(),
      output_info,
      output_to_input_aliases,
      kernel_inputs,
      options_.device,
      expr_eval);
}

std::vector<FusionExecutor::GlobalBufferInfo> FusionExecutor::
    getOutputBufferInfo(
        const KernelArgumentHolder& args,
        ExpressionEvaluator& expr_eval,
        const std::vector<std::pair<int, int>>& output_to_input_aliases) {
  FUSER_PERF_SCOPE("FusionExecutor::GetOutbufferInfo");
  const auto kernel = lowered_->kernel();
  std::vector<GlobalBufferInfo> outputs;
  TORCH_INTERNAL_ASSERT(
      args.size() == kernel->inputs().size(),
      "kernel arguments length does not match runtime arguments.");
  for (const auto out_i : c10::irange(kernel->outputs().size())) {
    GlobalBufferInfo info;
    auto out_val = kernel->outputs()[out_i];
    info.tv = dynamic_cast<TensorView*>(out_val);
    if (out_val->isFusionInput()) {
      // pushing empty tensor for trivial forwarding. Since we handle this in
      // integration, see step 1 - note [trivial forwarding]
      info.type = at::kFloat;
      info.sizes = {0};
    } else {
      TORCH_INTERNAL_ASSERT(
          info.tv != nullptr, "Cannot allocate outputs that are not tensors.");
      auto output = out_val->as<TensorView>();
      auto alias_it = std::find_if(
          output_to_input_aliases.begin(),
          output_to_input_aliases.end(),
          [&](const auto output_to_input) {
            return output_to_input.first == (int)out_i;
          });
      if (alias_it != output_to_input_aliases.end()) {
        // Aliased to an input, no need to gather allocation
        // info. Leave it as is
      } else {
        std::tie(info.sizes, info.strides) =
            inferShapeOfOutput(output, expr_eval);
        info.type = data_type_to_aten(output->dtype());
        info.zero_init = false;
      }
    }
    outputs.emplace_back(info);
  }
  return outputs;
}

void FusionExecutor::setUsedTVs() {
  auto used_vals = fusion_->usedMathVals();
  auto used_tvs = ir_utils::filterByType<TensorView>(used_vals);
  used_tvs_.clear();
  used_tvs_.insert(used_tvs_.begin(), used_tvs.begin(), used_tvs.end());
}

KernelArgumentHolder FusionExecutor::inferOutputSizes(
    Fusion* fusion,
    const KernelArgumentHolder& args) {
  FUSER_PERF_SCOPE("FusionExecutor::inferOutputSizes");
  std::unique_ptr<PrecomputedValues> evaluator_precomputed_values =
      std::make_unique<PrecomputedValues>(fusion);
  evaluator_precomputed_values->bindInputs(args);
  evaluator_precomputed_values->evaluate();

  ExpressionEvaluator expr_eval;
  expr_eval.precomputedValues() = evaluator_precomputed_values.get();

  auto arg_index_type = args.getSmallestIndexTypeOfArguments();
  const auto& output_to_input_aliases = fusion->getOutputToInputAliasIndices();

  KernelArgumentHolder ret;
  ret.setDeviceIndex(args.getDeviceIndex());

  for (const auto out_i : c10::irange(fusion->outputs().size())) {
    // If the output is just trivially the input, just "copy" it over.
    // See note [trivial forwarding]
    if (fusion->outputs()[out_i]->isFusionInput()) {
      auto input_it = std::find(
          fusion->inputs().begin(),
          fusion->inputs().end(),
          fusion->outputs()[out_i]);
      TORCH_INTERNAL_ASSERT(
          input_it != fusion->inputs().end(),
          "Issue with an input showing up as output but could not find input.");
      auto inp_i = std::distance(fusion->inputs().begin(), input_it);
      auto tensor_arg_abstract =
          dynamic_cast<const TensorArgAbstract*>(args[inp_i]);
      TORCH_INTERNAL_ASSERT(
          tensor_arg_abstract,
          "Cannot register a scalar as an output in a fusion.");
      ret.push(tensor_arg_abstract);
    } else {
      TORCH_INTERNAL_ASSERT(
          fusion->outputs()[out_i]->isA<TensorView>(),
          "Cannot allocate outputs that are not tensors.");
      auto output_tv = fusion->outputs()[out_i]->as<TensorView>();

      auto alias_it = std::find_if(
          output_to_input_aliases.begin(),
          output_to_input_aliases.end(),
          [&](const auto& output_to_input) {
            return output_to_input.first == (int)out_i;
          });

      if (alias_it != output_to_input_aliases.end()) {
        // When aliasing output to an input, we do not need to allocate a new
        // output but still need to push an entry.
        ret.push(int64_t(0));
      } else {
        const auto& [sizes, strides] = inferShapeOfOutput(output_tv, expr_eval);
        const auto dtype = (output_tv->dtype() == DataType::Index)
            ? data_type_to_aten(arg_index_type)
            : data_type_to_aten(output_tv->dtype());
        ret.pushTensorProxy(sizes, strides, dtype);
      }
    }
  }

  for (const auto& [aliased_output_index, aliased_input_index] :
       output_to_input_aliases) {
    TORCH_INTERNAL_ASSERT(
        args[aliased_input_index]->isType(ArgType::Tensor),
        "alias io only supports tensor");
    ret.swap(aliased_output_index, args[aliased_input_index]);
  }
  return ret;
}

namespace {

// Make sure the index type of Kernel is valid
void validateIndexType(
    kir::Kernel* kernel,
    const CompileParams& compile_params) {
  TORCH_INTERNAL_ASSERT(
      !compile_params.index_type.has_value() ||
          kernel->indexType() == compile_params.index_type.value(),
      "Kernel index type and compilation index type don't match. Kernel type: ",
      kernel->indexType(),
      ". Compilation index type: ",
      compile_params.index_type.value());
}

void validateCooperativeLaunch(
    CUfunction kernel,
    const LaunchParams& launch_params,
    int64_t device_index) {
  int num_blocks_per_SM = -1;
  auto block_size =
      launch_params.bdimx() * launch_params.bdimy() * launch_params.bdimz();
  CUDA_SAFE_CALL(cuOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks_per_SM,
      kernel,
      (int)block_size,
      (size_t)launch_params.smem()));

  auto grid_size =
      launch_params.gdimx() * launch_params.gdimy() * launch_params.gdimz();
  auto max_active_blocks = num_blocks_per_SM *
      at::cuda::getDeviceProperties(device_index)->multiProcessorCount;
  TORCH_INTERNAL_ASSERT(
      (int64_t)(max_active_blocks) >= grid_size,
      "Wanted to launch a cooperative kernel, however the number of blocks is greater than ",
      "what can be resident on the GPU at once. Need: ",
      grid_size,
      " (",
      launch_params.gdimx(),
      " * ",
      launch_params.gdimy(),
      " * ",
      launch_params.gdimz(),
      ") but limited to ",
      num_blocks_per_SM,
      " * ",
      at::cuda::getDeviceProperties(device_index)->multiProcessorCount);
}

// Dump fusion inputs and outputs as well as some useful fusion
// information. Note that inputs and outputs are those that are passed
// to FusionExecutor::runFusion, so outputs may not be given.
void dumpFusionArgs(
    int64_t fusion_id,
    const KernelArgumentHolder& args,
    const LaunchParams& launch_constraints,
    const CompileParams& compile_params,
    const std::vector<at::Tensor>& outputs) {
  std::cout << "Arguments for fusion" << fusion_id << ":" << std::endl
            << "Inputs:" << std::endl;
  for (auto i : c10::irange(args.size())) {
    std::cout << "  " << args[i]->toString() << std::endl;
  }
  std::cout << "Outputs:" << std::endl;
  for (const auto& output : outputs) {
    std::cout << "  " << output.scalar_type() << " " << output.sizes()
              << " (strides = " << output.strides() << ")" << std::endl;
  }
  std::cout << launch_constraints.toString();
  std::cout << "maxrregcount= " << compile_params.maxrregcount << std::endl;
}

// Dump arguments that are passed to a CUDA kernel call, which include
// the inputs and outputs of the fusion as well as temporary
// global-memory buffers. Unlike dumpFusionArgs, which dumps inputs
// and outputs passed to FusionExecutor::runFusion, this function
// dumps those that are passed to a CUDA kernel.
void dumpKernelArgs(
    int64_t fusion_id,
    const KernelArgumentHolder& args,
    size_t num_inputs,
    const std::vector<at::Tensor>& allocated_outputs,
    const std::vector<at::Tensor>& intermediates,
    const std::vector<FusionExecutor::GlobalBufferInfo>& intermediates_info) {
  std::cout << "Arguments for kernel" << fusion_id << ":" << std::endl
            << "Inputs:" << std::endl;
  for (auto i : c10::irange(num_inputs)) {
    std::cout << "  " << args[i]->toString() << std::endl;
  }
  std::cout << "Outputs:" << std::endl;
  // note: add aliased outputs here.
  for (const auto& output : allocated_outputs) {
    std::cout << "  " << output.scalar_type() << " " << output.sizes()
              << " (strides = " << output.strides()
              << ", address = " << output.data_ptr() << ")" << std::endl;
  }
  std::cout << "Intermediate global buffers:" << std::endl;
  for (const auto i : c10::irange(intermediates.size())) {
    const auto& buffer = intermediates.at(i);
    const auto& zero_init = intermediates_info.at(i).zero_init;
    std::cout << "  " << buffer.scalar_type() << " " << buffer.sizes()
              << " is_zero_initialized: " << zero_init << std::endl;
  }
}

FusionExecutor::GlobalBufferInfo getGlobalBufferAllocationInfo(
    const at::Tensor& at_tensor) {
  FusionExecutor::GlobalBufferInfo info{
      .sizes = at_tensor.sizes().vec(),
      .strides = at_tensor.strides().vec(),
      .type = at_tensor.scalar_type()};
  return info;
}

} // namespace

void FusionExecutor::initializeExecutorEntry(
    ExecutorEntry& executor_entry,
    const KernelArgumentHolder& args,
    const LaunchParams& launch_constraints,
    const CompileParams& compile_params,
    const std::vector<at::Tensor>& outputs) {
  FUSER_PERF_SCOPE("ExecutorRunFusion::InitializeExecutorEntry");

  // code path to take when either:
  //   1. no opt_code is provided or
  //   2. `executor_entry` is not initialized
  executor_utils::validateKernelInputs(fusion_, args, options_.device);

  ExpressionEvaluator expr_eval;
  evaluatorPrecomputedValues()->bindInputs(args);
  expr_eval.precomputedValues() = evaluatorPrecomputedValues().get();

  auto launch_params =
      computeLaunchParams(launch_constraints, expr_eval, warp_size_);

  executor_utils::validateVectorizedTensors(
      kernel(), args, outputs, compileTimeDataCache(), expr_eval);

  auto input_alias_indices_entry =
      executor_utils::caching::ExecutorCompileTimeEntry<
          executor_utils::caching::InputAliasIndices>(
          compileTimeDataCache(), [&]() {
            return std::make_unique<std::vector<std::pair<int, int>>>(
                fusion_->getOutputToInputAliasIndices());
          });

  const auto& output_to_input_aliases = input_alias_indices_entry.get();

  std::vector<GlobalBufferInfo> output_info;

  if (outputs.empty()) {
    output_info = getOutputBufferInfo(args, expr_eval, output_to_input_aliases);
  } else {
    // Need to save the information necessary for allocations as
    // future uses of this ExecutorEntry may not be provided with
    // allocated outputs
    for (const auto& output : outputs) {
      output_info.emplace_back(getGlobalBufferAllocationInfo(output));
    }
  }

  auto intermediates = getIntermediateBufferInfo(expr_eval);

  uint64_t rand_offset = 0;
  if (kernel()->summary().max_rng_offsets >= 0) {
    // NOTE: this is how we map offset to PW kernels in order to have
    // identical random number generator to match native PyTorch results.
    // But it doesn't really work as it takes assumption how threads are
    // binded but is not generally how we handle that in scheduler.
    // Refer to `Philox` in generated kernel to understand how the mapping
    // works.
    rand_offset = (uint64_t)(kernel()->summary().max_rng_offsets + 1) * 4;
  }

  // All information is gathered. Save it to ExecutorEntry
  executor_entry.launch_params = launch_params;
  executor_entry.output_to_input_aliases = output_to_input_aliases;
  executor_entry.outputs = output_info;
  executor_entry.intermediates = intermediates;
  executor_entry.rand_offset = rand_offset;
  executor_entry.init = true;
}

void FusionExecutor::recompileKernel(
    const LaunchParams& new_launch_params,
    const CompileParams& new_compile_params) {
  if (new_launch_params.nThreads() <= block_size_high_water_mark_ &&
      new_compile_params.maxrregcount == maxrregcount_high_water_mark_) {
    return;
  }

  const auto structured_code = getStructuredCode();
  block_size_high_water_mark_ = new_launch_params.nThreads();
  maxrregcount_high_water_mark_ = new_compile_params.maxrregcount;

  std::tie(compiled_kernel_, last_compiler_log_, last_compiled_binary_) =
      executor_utils::getCompiledKernel(
          kernel_code_,
          structured_code,
          getCanonicalKernelName(),
          fusion_id_,
          block_size_high_water_mark_,
          maxrregcount_high_water_mark_,
          save_compiled_binary_);

  resetCompiledKernelProperties();

  if (kernel()->summary().has_cooperative_grid_reduction) {
    // We need to increase shared memory before kernel launch, but also before
    // calling into `validateCooperativeLaunch`!
    // So we need to do it there before calling into the validation, to avoid
    // false positives
    ensureAvailableDynamicSmemSize(new_launch_params.smem());
    validateCooperativeLaunch(
        compiled_kernel_.function, new_launch_params, options_.device.index());
  }
}

int64_t FusionExecutor::getAvailableDynamicSmemSize() {
  TORCH_INTERNAL_ASSERT(
      compiled(), "Cannot get dynamic smem size unless kernel is compiled");
  if (!available_dynamic_smem_size_.has_value()) {
    int size = 0;
    CUDA_SAFE_CALL(cuFuncGetAttribute(
        &size,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        compiled_kernel_.function));
    available_dynamic_smem_size_ = size;
  }
  return available_dynamic_smem_size_.value();
}

int64_t FusionExecutor::getStaticSmemSize() {
  TORCH_INTERNAL_ASSERT(
      compiled(), "Cannot get static smem size unless kernel is compiled");
  if (!static_smem_size_.has_value()) {
    int size = 0;
    // Is this really a costly operation worth caching?
    CUDA_SAFE_CALL(cuFuncGetAttribute(
        &size, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, compiled_kernel_.function));
    static_smem_size_ = size;
  }
  return static_smem_size_.value();
}

void FusionExecutor::validateDynamicSmemSize(int64_t dynamic_smem_size) {
  TORCH_INTERNAL_ASSERT(
      getStaticSmemSize() + dynamic_smem_size < device_smem_limit_,
      "The total shared memory allocation is larger than available memory.",
      " Dynamic size: ",
      dynamic_smem_size,
      ". Static size: ",
      getStaticSmemSize(),
      ". Required total size: ",
      getStaticSmemSize() + dynamic_smem_size,
      ". Device limit size: ",
      device_smem_limit_);
}

int64_t FusionExecutor::ensureAvailableDynamicSmemSize(
    int64_t dynamic_smem_size) {
  TORCH_INTERNAL_ASSERT(
      compiled(), "Cannot set dynamic smem size unless kernel is compiled");
  if (dynamic_smem_size > getAvailableDynamicSmemSize()) {
    validateDynamicSmemSize(dynamic_smem_size);
    CUDA_SAFE_CALL(cuFuncSetAttribute(
        compiled_kernel_.function,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        dynamic_smem_size));
    available_dynamic_smem_size_ = dynamic_smem_size;
  }
  return getAvailableDynamicSmemSize();
}

void FusionExecutor::resetCompiledKernelProperties() {
  available_dynamic_smem_size_.reset();
  static_smem_size_.reset();
}

std::vector<TensorView*> FusionExecutor::getTvsForKernelArguments() const {
  std::vector<TensorView*> tvs;
  for (auto val : kernel()->inputs()) {
    tvs.emplace_back(dynamic_cast<TensorView*>(val));
  }
  for (auto val : kernel()->outputs()) {
    tvs.emplace_back(dynamic_cast<TensorView*>(val));
  }
  for (auto alloc : kernel()->summary().global_allocations) {
    auto tv = alloc->buffer()->as<TensorView>();
    if (tv->isFusionOutput()) {
      continue;
    }
    tvs.emplace_back(tv);
  }
  if (lowered_->kernel()->summary().max_rng_offsets >= 0) {
    tvs.emplace_back(nullptr);
  }
  return tvs;
}

std::vector<at::Tensor> FusionExecutor::runFusion(
    KernelArgumentHolder& args,
    const LaunchParams& launch_constraints,
    CompileParams compile_params,
    std::vector<at::Tensor> outputs) {
  FUSER_PERF_SCOPE("FusionExecutor::RunFusion");
  TORCH_INTERNAL_ASSERT(compiled());
  TORCH_INTERNAL_ASSERT(
      fusion_id_ > 0, "Cannot run fusion, it was not compiled.");
  TORCH_INTERNAL_ASSERT(
      !args.getCacheId().has_value() || outputs.empty(),
      "short cut input cache is not compatible with pre-allocated output");

  validateIndexType(kernel(), compile_params);

  const auto num_inputs = args.size();

  if (isDebugDumpEnabled(DebugDumpOption::FusionArgs)) {
    dumpFusionArgs(
        fusion_id_, args, launch_constraints, compile_params, outputs);
  }

  c10::DeviceGuard dg(options_.device);
  auto stream = at::cuda::getCurrentCUDAStream();
  at::cuda::jit::initializeCudaContext();
  TORCH_INTERNAL_ASSERT(lowered_);

  // Placeholder for the case where parameter cache is not used
  ExecutorEntry temporary_executor_entry;

  ExecutorEntry* executor_entry =
      args.getCacheId().has_value() && !disable_parameter_cache_
      ? &executor_entry_lookup_[*args.getCacheId()]
      : &temporary_executor_entry;

  // Initialize the executor entry if not initlized
  if (!executor_entry->init) {
    initializeExecutorEntry(
        *executor_entry, args, launch_constraints, compile_params, outputs);
  }

  recompileKernel(executor_entry->launch_params, compile_params);

  // TODO: Why does this need to be stored in the class?
  launch_params_ = executor_entry->launch_params;

  // context manager to disable auto grad for `empty_cuda` calls later
  at::AutoDispatchBelowADInplaceOrView non_variable_type_mode;

  // only allocate outputs when not given
  if (outputs.empty()) {
    auto expr_eval = executor_utils::bindInputs(args, lowered_->kernel());
    outputs = allocOutputs(
        kernel(),
        executor_entry->outputs,
        executor_entry->output_to_input_aliases,
        args,
        options_.device,
        expr_eval);
  } else {
    // TODO: Use validateKernelOutputs
    TORCH_INTERNAL_ASSERT(
        outputs.size() == fusion_->outputs().size(),
        __func__,
        " provided number of outputs does not match fusion output");
  }
  args.push(outputs);

  std::vector<at::Tensor> intermediates;
  at::Tensor profile_buffer;
  {
    FUSER_PERF_SCOPE("ExecutorRunFusion::IntermediateBufferAlloc");
    for (const auto i : c10::irange(executor_entry->intermediates.size())) {
      const auto& buf_info = executor_entry->intermediates.at(i);
      at::Tensor intermediate_buffer;
      if (buf_info.zero_init) {
        intermediate_buffer = at::zeros(
            buf_info.sizes,
            at::TensorOptions().dtype(buf_info.type).device(options_.device));
      } else {
        intermediate_buffer = at::native::empty_cuda(
            buf_info.sizes,
            buf_info.type,
            c10::nullopt,
            options_.device,
            c10::nullopt);
        if (shouldFillAllocationWithNan()) {
          fillTensorWithNan(intermediate_buffer);
        }
      }
      args.push(intermediate_buffer);
      intermediates.push_back(intermediate_buffer);
      if (buf_info.is_profile_buffer) {
        profile_buffer = intermediate_buffer;
      }
    }
  }

  // push back RNG state if needed
  if (lowered_->kernel()->summary().max_rng_offsets >= 0) {
    args.appendPhiloxRNGSeed(executor_entry->rand_offset);
  }

  if (isDebugDumpEnabled(DebugDumpOption::LaunchParam)) {
    launch_params_.print();
  }

  if (isDebugDumpEnabled(DebugDumpOption::KernelArgs)) {
    dumpKernelArgs(
        fusion_id_,
        args,
        num_inputs,
        outputs,
        intermediates,
        executor_entry->intermediates);
  }

  if (isDebugDumpEnabled(DebugDumpOption::IndexType)) {
    std::cout << "Index type: " << kernel()->indexType() << std::endl;
  }

  cudaEvent_t start_event = {};
  cudaEvent_t finish_event = {};

  if (measure_kernel_time_ ||
      isDebugDumpEnabled(DebugDumpOption::EffectiveBandwidth) ||
      isDebugDumpEnabled(DebugDumpOption::PerfDebugVerbose)) {
    CUDA_RT_SAFE_CALL(cudaEventCreate(&start_event));
    CUDA_RT_SAFE_CALL(cudaEventCreate(&finish_event));
    CUDA_RT_SAFE_CALL(cudaEventRecord(start_event, stream));
  }

  if (execute_kernel_) {
    ensureAvailableDynamicSmemSize(executor_entry->launch_params.smem());
    auto ee = executor_utils::bindInputs(args, kernel());
    auto arg_buffer =
        args.getBuffer(kernel()->indexType(), getTvsForKernelArguments(), ee);
    if (!kernel()->summary().has_cooperative_grid_reduction) {
      FUSER_PERF_SCOPE("ExecutorRunFusion::cuLaunchKernel");
      CUDA_SAFE_CALL(cuLaunchKernel(
          compiled_kernel_.function,
          launch_params_.gdimx(),
          launch_params_.gdimy(),
          launch_params_.gdimz(),
          launch_params_.bdimx(),
          launch_params_.bdimy(),
          launch_params_.bdimz(),
          launch_params_.smem(),
          stream,
          arg_buffer,
          nullptr));
    } else {
      FUSER_PERF_SCOPE("ExecutorRunFusion::cuLaunchCooperativeKernel");
      CUDA_SAFE_CALL(cuLaunchCooperativeKernel(
          compiled_kernel_.function,
          launch_params_.gdimx(),
          launch_params_.gdimy(),
          launch_params_.gdimz(),
          launch_params_.bdimx(),
          launch_params_.bdimy(),
          launch_params_.bdimz(),
          launch_params_.smem(),
          stream,
          arg_buffer));
    }
  }

  if (measure_kernel_time_ ||
      isDebugDumpEnabled(DebugDumpOption::EffectiveBandwidth) ||
      isDebugDumpEnabled(DebugDumpOption::PerfDebugVerbose)) {
    CUDA_RT_SAFE_CALL(cudaEventRecord(finish_event, stream));
    CUDA_RT_SAFE_CALL(cudaEventSynchronize(start_event));
    CUDA_RT_SAFE_CALL(cudaEventSynchronize(finish_event));
    CUDA_RT_SAFE_CALL(
        cudaEventElapsedTime(&kernel_time_ms_, start_event, finish_event));
    CUDA_RT_SAFE_CALL(cudaEventDestroy(start_event));
    CUDA_RT_SAFE_CALL(cudaEventDestroy(finish_event));

    bytes_processed_ = 0;
    // Figure how many bytes are inputs, outputs, and temporary buffers
    for (auto i : c10::irange(num_inputs)) {
      if (auto tensor_arg_abstract =
              dynamic_cast<const TensorArgAbstract*>(args[i])) {
        bytes_processed_ += tensor_arg_abstract->numel() *
            (int64_t)dataTypeSize(tensor_arg_abstract->getDataType());
      }
    }
    for (const auto& output : outputs) {
      bytes_processed_ += output.numel() *
          (int64_t)dataTypeSize(aten_to_data_type(output.scalar_type()));
    }

    if (isDebugDumpEnabled(DebugDumpOption::EffectiveBandwidth)) {
      double gb_per_s =
          ((double)bytes_processed_ / ((double)kernel_time_ms_ / 1000)) /
          (double)1.0e9;
      std::cout << "kernel" << fusion_id_ << " run in " << kernel_time_ms_
                << " ms, achieved: " << gb_per_s << " GB/s" << std::endl;
    }
  }

  if (isOptionEnabled(EnableOption::KernelProfile)) {
    std::cout << kernel()->profile().toString(profile_buffer);
  }

  return outputs;
}

void FusionExecutor::compileRtc(
    const std::string& code,
    const std::string& name,
    bool structured,
    PrimDataType index_type) {
  FUSER_PERF_SCOPE("ExecutorRunFusion::compileRtc");
  TORCH_INTERNAL_ASSERT(
      index_type == PrimDataType::Int || index_type == PrimDataType::Int32 ||
          "Invalid index type: ",
      index_type);
  std::string scode;
  if (!structured) {
    scode = getStructuredCode(code, index_type);
  } else {
    scode = code;
  }
  fusion_id_ = 1;

  std::tie(compiled_kernel_, last_compiler_log_, last_compiled_binary_) =
      executor_utils::getCompiledKernel(c10::nullopt, scode, name, fusion_id_);
}

float FusionExecutor::runRtc(
    const LaunchParams& launch_params,
    const std::vector<at::Tensor>& args,
    PrimDataType index_type) {
  FUSER_PERF_SCOPE("runFusion");

  c10::DeviceGuard dg(options_.device);
  auto stream = at::cuda::getCurrentCUDAStream();

  cudaEvent_t start_event = {};
  cudaEvent_t finish_event = {};

  CUDA_RT_SAFE_CALL(cudaEventCreate(&start_event));
  CUDA_RT_SAFE_CALL(cudaEventCreate(&finish_event));

  KernelArgumentHolder kernel_arguments;
  kernel_arguments.push(args);

  CUDA_RT_SAFE_CALL(cudaEventRecord(start_event, stream));

  ExpressionEvaluator ee;
  std::vector<TensorView*> tvs(args.size(), nullptr);
  CUDA_SAFE_CALL(cuLaunchKernel(
      compiled_kernel_.function,
      launch_params.gdimx(),
      launch_params.gdimy(),
      launch_params.gdimz(),
      launch_params.bdimx(),
      launch_params.bdimy(),
      launch_params.bdimz(),
      launch_params.smem(),
      stream,
      kernel_arguments.getBuffer(index_type, tvs, ee),
      nullptr));

  CUDA_RT_SAFE_CALL(cudaEventRecord(finish_event, stream));
  CUDA_RT_SAFE_CALL(cudaEventSynchronize(start_event));
  CUDA_RT_SAFE_CALL(cudaEventSynchronize(finish_event));

  float kernel_time_ms = 0;
  CUDA_RT_SAFE_CALL(
      cudaEventElapsedTime(&kernel_time_ms, start_event, finish_event));
  CUDA_RT_SAFE_CALL(cudaEventDestroy(start_event));
  CUDA_RT_SAFE_CALL(cudaEventDestroy(finish_event));

  return kernel_time_ms;
}

} // namespace nvfuser
