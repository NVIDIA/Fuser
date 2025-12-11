// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include "host_ir/jit_constants.h"

#include <cstdint>

#include <ATen/ATen.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/jit_utils.h>
#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"

#include "driver_api.h"
#include "fusion_profiler.h"
#include "instrumentation.h"

namespace nvfuser {

namespace {

// Helper function to register external functions in JIT
void registerExternalFunction(
    void* func_ptr,
    llvm::orc::SymbolMap& symbolMap,
    llvm::orc::MangleAndInterner& mangler,
    std::string_view func_name) {
  auto addr = llvm::orc::ExecutorAddr::fromPtr(func_ptr);
  symbolMap[mangler(func_name)] =
      llvm::orc::ExecutorSymbolDef(addr, llvm::JITSymbolFlags::Exported);
}

} // anonymous namespace

void registerExternalFunctionsImpl(
    llvm::orc::LLJIT* jit,
    llvm::orc::JITDylib& dest_dynamic_lib) {
  auto mangler = llvm::orc::MangleAndInterner(
      dest_dynamic_lib.getExecutionSession(), jit->getDataLayout());

  // tensor size extraction function
  void* extract_tensor_size_func_ptr =
      reinterpret_cast<void*>(+[](at::Tensor* tensor, int64_t dim) -> int64_t {
        NVF_ERROR(tensor != nullptr, kTensorSizeFuncName, " tensor is nullptr");
        NVF_ERROR(dim >= 0 && dim < tensor->dim(), "dim is out of range");
        return tensor->size(dim);
      });

  // tensor stride extraction function
  void* extract_tensor_stride_func_ptr =
      reinterpret_cast<void*>(+[](at::Tensor* tensor, int64_t dim) -> int64_t {
        NVF_ERROR(
            tensor != nullptr, kTensorStrideFuncName, " tensor is nullptr");
        NVF_ERROR(dim >= 0 && dim < tensor->dim(), "dim is out of range");
        return tensor->stride(dim);
      });

  // tensor data pointer extraction function
  void* extract_tensor_data_ptr_func_ptr =
      reinterpret_cast<void*>(+[](at::Tensor* tensor) -> void* {
        NVF_ERROR(tensor != nullptr, "tensor_data_ptr: tensor is nullptr");
        return tensor->data_ptr();
      });

  // new at::Tensor() wrapper instead of real tensor allocation
  void* new_tensor_func_ptr = reinterpret_cast<void*>(
      +[]() -> at::Tensor* { return new at::Tensor(); });

  // copy and return tensor
  void* set_tensor_func_ptr =
      reinterpret_cast<void*>(+[](at::Tensor* in) -> at::Tensor* {
        NVF_ERROR(in != nullptr, kSetTensorFuncName, " in is nullptr");
        auto* out = new at::Tensor();
        *out = *in;
        return out;
      });

  // delete a newed tensor
  void* delete_tensor_func_ptr = reinterpret_cast<void*>(
      +[](at::Tensor* tensor) -> void { delete tensor; });

  // at::native::empty_strided_cuda
  void* empty_strided_cuda_func_ptr =
      reinterpret_cast<void*>(+[](const int64_t* sizes,
                                  int64_t ndim,
                                  const int64_t* strides,
                                  int64_t strides_ndim,
                                  int32_t dtype,
                                  int64_t device_index,
                                  at::Tensor* out_tensor) {
        at::IntArrayRef aten_sizes(sizes, ndim);
        at::IntArrayRef aten_strides(strides, strides_ndim);
        auto scalar_type = static_cast<at::ScalarType>(dtype);
        at::Device device =
            at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(device_index));
        *out_tensor = at::native::empty_strided_cuda(
            aten_sizes,
            aten_strides,
            scalar_type,
            c10::nullopt,
            device,
            c10::nullopt);
      });

  // launch_kernel_direct function: simpler wrapper that just calls
  // cuLaunchKernelEx All argument packing is done in LLVM IR before calling
  // this
  void* launch_kernel_direct_func_ptr =
      reinterpret_cast<void*>(+[](void** kernel_args,
                                  void* cuda_function_ptr,
                                  int64_t gdimx,
                                  int64_t gdimy,
                                  int64_t gdimz,
                                  int64_t bdimx,
                                  int64_t bdimy,
                                  int64_t bdimz,
                                  int64_t smem) {
        FUSER_PERF_SCOPE("launch_kernel_direct");

        CUlaunchConfig config = {};
        auto stream = at::cuda::getCurrentCUDAStream();

        config.gridDimX = gdimx;
        config.gridDimY = gdimy;
        config.gridDimZ = gdimz;
        config.blockDimX = bdimx;
        config.blockDimY = bdimy;
        config.blockDimZ = bdimz;
        config.sharedMemBytes = smem;
        config.hStream = stream;
        config.attrs = nullptr;
        config.numAttrs = 0;

        // Launch the kernel
        {
          FUSER_PERF_SCOPE("cuLaunchKernelEx");
          NVFUSER_CUDA_SAFE_CALL(cuLaunchKernelEx(
              &config,
              reinterpret_cast<CUfunction>(cuda_function_ptr),
              kernel_args,
              nullptr));
        }
      });

  // matmul_out function
  void* matmul_out_func_ptr = reinterpret_cast<void*>(
      +[](at::Tensor* t_out, at::Tensor* t_a, at::Tensor* t_b) {
        at::matmul_out(*t_out, *t_a, *t_b);
      });

  // linear function in place
  void* linear_out_func_ptr = reinterpret_cast<void*>(+[](at::Tensor* out,
                                                          at::Tensor* in,
                                                          at::Tensor* weight,
                                                          at::Tensor* bias) {
    std::optional<at::Tensor> bias_opt = std::nullopt;
    if (bias != nullptr) {
      bias_opt = *bias;
    }
    at::linear_out(*out, *in, *weight, bias_opt);
  });

  // permute a tensor and return a new tensor
  void* permute_func_ptr = reinterpret_cast<void*>(
      +[](at::Tensor* in,
          const int64_t* permutation,
          int64_t perm_size) -> at::Tensor* {
        // Convert pointer to vector for permute function
        std::vector<int64_t> perm_vec(permutation, permutation + perm_size);
        return new at::Tensor(in->permute(perm_vec));
      });

  // reshape a tensor and return a new tensor
  void* reshape_func_ptr = reinterpret_cast<void*>(
      +[](at::Tensor* in,
          const int64_t* shape,
          int64_t shape_size) -> at::Tensor* {
        // Convert pointer to IntArrayRef for reshape function
        at::IntArrayRef shape_ref(shape, shape_size);
        return new at::Tensor(in->reshape(shape_ref));
      });

  // insert fuser perf scope
  void* nvtx_range_push_func_ptr = reinterpret_cast<void*>(
      +[](const char* name) -> void { nvtxRangePush(name); });

  void* nvtx_range_pop_func_ptr =
      reinterpret_cast<void*>(+[]() -> void { nvtxRangePop(); });

  // Register wrapper functions in JIT
  llvm::orc::SymbolMap name_to_symbol;
  registerExternalFunction(
      extract_tensor_size_func_ptr,
      name_to_symbol,
      mangler,
      kTensorSizeFuncName);
  registerExternalFunction(
      extract_tensor_stride_func_ptr,
      name_to_symbol,
      mangler,
      kTensorStrideFuncName);
  registerExternalFunction(
      extract_tensor_data_ptr_func_ptr,
      name_to_symbol,
      mangler,
      kTensorDataPtrFuncName);
  registerExternalFunction(
      new_tensor_func_ptr, name_to_symbol, mangler, kNewTensorFuncName);
  registerExternalFunction(
      delete_tensor_func_ptr, name_to_symbol, mangler, kDeleteTensorFuncName);
  registerExternalFunction(
      set_tensor_func_ptr, name_to_symbol, mangler, kSetTensorFuncName);
  registerExternalFunction(
      empty_strided_cuda_func_ptr,
      name_to_symbol,
      mangler,
      kAtEmptyStridedCudaWrapper);
  registerExternalFunction(
      nvtx_range_push_func_ptr,
      name_to_symbol,
      mangler,
      kNvtxRangePushFuncName);
  registerExternalFunction(
      nvtx_range_pop_func_ptr, name_to_symbol, mangler, kNvtxRangePopFuncName);
  registerExternalFunction(
      launch_kernel_direct_func_ptr,
      name_to_symbol,
      mangler,
      kLaunchKernelDirectFuncName);
  registerExternalFunction(
      matmul_out_func_ptr, name_to_symbol, mangler, kMatmulOutFuncName);
  registerExternalFunction(
      linear_out_func_ptr, name_to_symbol, mangler, kLinearOutFuncName);
  registerExternalFunction(
      permute_func_ptr, name_to_symbol, mangler, kPermuteFuncName);
  registerExternalFunction(
      reshape_func_ptr, name_to_symbol, mangler, kReshapeFuncName);

  auto err = dest_dynamic_lib.define(llvm::orc::absoluteSymbols(name_to_symbol));
  if (err) {
    NVF_THROW(llvm::toString(std::move(err)));
  }
}

} // namespace nvfuser
