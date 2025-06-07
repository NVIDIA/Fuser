#include <host_ir/host_ir_llvm_jit.h>
#include <host_ir/lower_to_llvm.h>

#include "llvm/Support/TargetSelect.h"
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include "llvm/Support/Error.h"

#include <chrono>

namespace nvfuser {

// PIMPL implementation for HostIrLlvmJit
struct HostIrLlvmJit::LlvmJitImpl {
  std::unique_ptr<llvm::orc::LLJIT> jit;
  FuncType shape_infer_fn = nullptr;
  FuncType stride_infer_fn = nullptr;
};

// Constructor implementation
HostIrLlvmJit::HostIrLlvmJit(int num_threads) : pimpl_(new LlvmJitImpl) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  pimpl_->jit = ExitOnErr(
      llvm::orc::LLJITBuilder().setNumCompileThreads(num_threads).create());
}

// The destructor must be defined here where LlvmJitImpl is a complete type.
HostIrLlvmJit::~HostIrLlvmJit() = default;

// Move constructor and assignment operator
HostIrLlvmJit::HostIrLlvmJit(HostIrLlvmJit&&) noexcept = default;
HostIrLlvmJit& HostIrLlvmJit::operator=(HostIrLlvmJit&&) noexcept = default;

void HostIrLlvmJit::compile(TensorView* output_tv) {
  Fusion* fusion = output_tv->fusion();
  NVF_ERROR(fusion != nullptr, "Output TensorView must belong to a fusion.");

  // This simplified API assumes a single input TensorView.
  // This can be extended to handle multiple inputs.
  TensorView* input_tv = nullptr;
  for (auto inp : fusion->inputs()) {
    if (auto tv = dynamic_cast<TensorView*>(inp)) {
      NVF_ERROR(
          input_tv == nullptr,
          "Multiple input TensorViews not yet supported in this simplified API");
      input_tv = tv;
    }
  }
  NVF_ERROR(input_tv != nullptr, "No input TensorView found in fusion");

  auto input_domain = input_tv->getLogicalDomain();
  auto output_logical_domain = output_tv->getLogicalDomain();
  auto allocation_domain = output_tv->getMaybeAllocationDomain();

  // Store the output dimension for the run method
  output_tensor_dim_ = output_logical_domain.size();

  // JIT compile shape inference module
  auto TSM_shape =
      generate_infer_shape_module(input_domain, output_logical_domain, *fusion);
  if (auto Err = pimpl_->jit->addIRModule(std::move(TSM_shape))) {
    llvm::errs() << "Error adding shape infer module to JIT: "
                 << llvm::toString(std::move(Err)) << "\n";
  }

  // JIT compile stride inference module
  auto TSM_stride =
      generate_infer_stride_module(allocation_domain, output_logical_domain, *fusion);
  if (auto Err = pimpl_->jit->addIRModule(std::move(TSM_stride))) {
    llvm::errs() << "Error adding stride infer module to JIT: "
                 << llvm::toString(std::move(Err)) << "\n";
  }

  // Look up the function pointers and store them
  pimpl_->shape_infer_fn =
      ExitOnErr(pimpl_->jit->lookup("infer_shape")).toPtr<FuncType>();
  pimpl_->stride_infer_fn =
      ExitOnErr(pimpl_->jit->lookup("infer_stride")).toPtr<FuncType>();
}

at::Tensor HostIrLlvmJit::run(const at::Tensor& input) {
  NVF_ERROR(
      pimpl_->shape_infer_fn != nullptr && pimpl_->stride_infer_fn != nullptr,
      "JIT must be compiled before running.");

  // Allocate memory for shape result
  std::vector<int64_t> shape_result(output_tensor_dim_);

  // Run shape inference
  pimpl_->shape_infer_fn(
      input.sizes().data(),
      input.sizes().size(),
      shape_result.data(),
      shape_result.size());

  // Allocate memory for stride result
  std::vector<int64_t> stride_result(output_tensor_dim_);

  // Run stride inference
  pimpl_->stride_infer_fn(
      shape_result.data(),
      shape_result.size(),
      stride_result.data(),
      stride_result.size());

  // Create the output tensor with the computed shape and strides
  at::Tensor output_tensor =
      at::empty_strided(shape_result, stride_result, input.options());
  return output_tensor;
}

} // namespace nvfuser