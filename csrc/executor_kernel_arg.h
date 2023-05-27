// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ATen/core/ivalue.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/util/Exception.h>
#include <expr_evaluator.h>
#include <ir/all_nodes.h>
#include <torch/csrc/jit/ir/ir.h>
#include <type.h>
#include <array>
#include <cstddef>
#include <optional>

namespace nvfuser {

// TODO: macro this and the printer below
enum class ArgType {
  PhiloxCudaState,
  Long,
  Double,
  ComplexDouble,
  Bool,
  Tensor,
  CpuScalarTensor
};

inline std::string argTypeToString(ArgType type) {
  std::string ret;
  switch (type) {
    case ArgType::PhiloxCudaState:
      ret = "PhiloxCudaState";
      break;
    case ArgType::Long:
      ret = "Long";
      break;
    case ArgType::Double:
      ret = "Double";
      break;
    case ArgType::ComplexDouble:
      ret = "ComplexDouble";
      break;
    case ArgType::Bool:
      ret = "Bool";
      break;
    case ArgType::Tensor:
      ret = "Tensor";
      break;
    case ArgType::CpuScalarTensor:
      ret = "CpuScalarTensor";
      break;
  }
  return ret;
}

// This should match the tensor used in the code generation (almost exactly)
template <int ndims, int nalloc, typename nvfuser_index_t>
struct TensorArgCodegen {
  using index_type = nvfuser_index_t;

  void* data;
  std::array<nvfuser_index_t, ndims> size;
  std::array<nvfuser_index_t, nalloc> stride;

  static constexpr int nDims() {
    return ndims;
  }
  static constexpr int nAllocationDims() {
    return nalloc;
  }
  void setSize(int64_t i, nvfuser_index_t s) {
    size[i] = s;
  }
  void setStride(int64_t i, nvfuser_index_t s) {
    stride[i] = s;
  }
  nvfuser_index_t getSize(int64_t i) const {
    return size[i];
  }
  nvfuser_index_t getStride(int64_t i) const {
    return stride[i];
  }
};

// 0-Dim GPU based tensor
template <typename nvfuser_index_t>
struct TensorArgCodegen<0, 0, nvfuser_index_t> {
  using index_type = nvfuser_index_t;

  void* data;

  static constexpr int nDims() {
    return 0;
  }
  static constexpr int nAllocationDims() {
    return 0;
  }
  void setSize(int64_t, nvfuser_index_t) {
    TORCH_INTERNAL_ASSERT(false, "Tried to set size of a 0-dim tensor");
  }
  void setStride(int64_t, nvfuser_index_t) {
    TORCH_INTERNAL_ASSERT(false, "Tried to set stride of a 0-dim tensor");
  }
  nvfuser_index_t getSize(int64_t i) const {
    TORCH_INTERNAL_ASSERT(false, "Tried to get size of a 0-dim tensor");
  }
  nvfuser_index_t getStride(int64_t i) const {
    TORCH_INTERNAL_ASSERT(false, "Tried to get stride of a 0-dim tensor");
  }
};

struct ArgAbstract {
  virtual ~ArgAbstract() = default;
  virtual const void* arg() const = 0;
  virtual void* arg() = 0;
  virtual bool isType(ArgType type) const = 0;
  virtual ArgType type() const = 0;
  virtual std::unique_ptr<ArgAbstract> clone() const = 0;
  virtual std::string toString() const {
    return "input type: " + argTypeToString(type());
  };
};

#define DEF_HELPEE_FUNC(TARGET_TYPE, ARG_NAME)          \
  bool isType(ArgType type) const override {            \
    return ArgType::TARGET_TYPE == type;                \
  }                                                     \
  ArgType type() const override {                       \
    return ArgType::TARGET_TYPE;                        \
  }                                                     \
  const void* arg() const override {                    \
    return &ARG_NAME;                                   \
  }                                                     \
  void* arg() override {                                \
    return &ARG_NAME;                                   \
  }                                                     \
  std::unique_ptr<ArgAbstract> clone() const override { \
    return std::make_unique<TARGET_TYPE##Arg>(*this);   \
  }

#define DEF_TOSTRING_FUNC                 \
  std::string toString() const override { \
    std::stringstream ss;                 \
    ss << val_;                           \
    return ss.str();                      \
  }

struct PhiloxCudaStateArg : public ArgAbstract {
  at::PhiloxCudaState val_;
  PhiloxCudaStateArg(at::PhiloxCudaState _val) : val_(_val){};
  DEF_HELPEE_FUNC(PhiloxCudaState, val_)
};

struct LongArg : public ArgAbstract {
  int64_t val_;
  explicit LongArg(int64_t _val) : val_(_val) {}
  DEF_HELPEE_FUNC(Long, val_)
  DEF_TOSTRING_FUNC
};

struct DoubleArg : public ArgAbstract {
  double val_;
  explicit DoubleArg(double _val) : val_(_val) {}
  DEF_HELPEE_FUNC(Double, val_)
  DEF_TOSTRING_FUNC
};

struct ComplexDoubleArg : public ArgAbstract {
  c10::complex<double> val_;
  explicit ComplexDoubleArg(c10::complex<double> _val) : val_(_val) {}
  DEF_HELPEE_FUNC(ComplexDouble, val_)
  DEF_TOSTRING_FUNC
};

struct BoolArg : public ArgAbstract {
  bool val_;
  explicit BoolArg(bool _val) : val_(_val) {}
  DEF_HELPEE_FUNC(Bool, val_)
  DEF_TOSTRING_FUNC
};

struct TensorArgAbstract : ArgAbstract {
  at::Tensor tensor_;

  TensorArgAbstract(at::Tensor tensor) : tensor_(std::move(tensor)) {}
  TensorArgAbstract(const TensorArgAbstract&) = default;

  int64_t getRank() const {
    return tensor_.ndimension();
  }

  int64_t getSize(int64_t i) const {
    return tensor_.size(i);
  }

  virtual int64_t getAllocRank() const {
    TORCH_INTERNAL_ASSERT(
        false, "The allocation rank of an abstract tensor arg is not known.");
  }

  virtual int64_t getAllocSize(int64_t i) const {
    TORCH_INTERNAL_ASSERT(
        false, "The allocation shape of an abstract tensor arg is not known.");
  }

  virtual int64_t getAllocStride(int64_t i) const {
    TORCH_INTERNAL_ASSERT(
        false, "The allocation stride of an abstract tensor arg is not known.");
  }

  size_t getPointerAddress() const {
    return (size_t)tensor_.data_ptr();
  }

  DataType getDataType() const {
    return aten_to_data_type(tensor_.scalar_type());
  }

  int64_t numel() const {
    return tensor_.numel();
  }

  at::Tensor getTensor() const {
    return tensor_;
  }

  virtual bool isAbstract() const {
    return true;
  }

  virtual PrimDataType getIndexType() const {
    TORCH_INTERNAL_ASSERT(
        false, "The index type of an abstract tensor arg is not known.");
  }

  PrimDataType getSmallestIndexType() const;

  bool isType(ArgType t) const override {
    return type() == t;
  }

  ArgType type() const override {
    return ArgType::Tensor;
  }

  //! Returns the address of an tensor argument struct.
  const void* arg() const override {
    TORCH_INTERNAL_ASSERT(false, "Abstract tensor arg does not have arg");
  }

  //! Returns the address of an tensor argument struct.
  void* arg() override {
    TORCH_INTERNAL_ASSERT(false, "Abstract tensor arg does not have arg");
  }

  std::string toString() const override {
    std::stringstream ss;
    auto rank = getRank();
    ss << "tensor dtype: " << getDataType() << " sizes: (";
    for (auto i = 0; i < rank; i++) {
      ss << getSize(i) << ", ";
    }
    ss << ") pointer: " << getPointerAddress();
    return ss.str();
  }

  std::unique_ptr<ArgAbstract> clone() const override {
    return std::make_unique<TensorArgAbstract>(*this);
  }
};

std::vector<std::pair<int64_t, int64_t>>
inferAndValidateAllocationSizesAndStrides(
    const at::Tensor& tensor,
    TensorView* tv,
    ExpressionEvaluator& ee);

template <typename TENSOR_TYPE>
struct TensorArg : public TensorArgAbstract {
  TENSOR_TYPE instance_;
  std::array<int64_t, TENSOR_TYPE::nAllocationDims()> alloc_sizes;

  TensorArg(const at::Tensor& tensor, TensorView* tv, ExpressionEvaluator& eval)
      : TensorArgAbstract(tensor) {
    instance_.data = tensor.data_ptr();
    for (const auto i : c10::irange(tensor.ndimension())) {
      instance_.setSize(i, (typename TENSOR_TYPE::index_type)tensor.size(i));
    }
    inferSetAndValidateStrides(tensor, tv, eval);
  }

  void inferSetAndValidateStrides(
      const at::Tensor& tensor,
      TensorView* tv,
      ExpressionEvaluator& eval) {
    auto sizes_strides =
        inferAndValidateAllocationSizesAndStrides(tensor, tv, eval);
    TORCH_INTERNAL_ASSERT(
        (size_t)instance_.nAllocationDims() == sizes_strides.size());
    for (auto i : c10::irange((int64_t)sizes_strides.size())) {
      alloc_sizes.at(i) = sizes_strides.at(i).first;
      using stride_t = typename TENSOR_TYPE::index_type;
      instance_.setStride(i, (stride_t)sizes_strides.at(i).second);
    }
  }

  int64_t getAllocRank() const override {
    return instance_.nAllocationDims();
  }

  int64_t getAllocSize(int64_t i) const override {
    return alloc_sizes.at(i);
  }

  int64_t getAllocStride(int64_t i) const override {
    return instance_.getStride(i);
  }

  //! Returns the address of an tensor argument struct.
  const void* arg() const override {
    return &instance_;
  }

  //! Returns the address of an tensor argument struct.
  void* arg() override {
    return &instance_;
  }

  bool isAbstract() const override {
    return false;
  }

  PrimDataType getIndexType() const override {
    return NativeTypeToDataType<typename TENSOR_TYPE::index_type>::type;
  }

  std::string toString() const override {
    std::stringstream ss;
    ss << TensorArgAbstract::toString();
    ss << " allocation size: (";
    for (auto i = 0; i < getAllocRank(); i++) {
      ss << getAllocSize(i) << ", ";
    }
    ss << ") allocation stride: (";
    for (auto i = 0; i < getAllocRank(); i++) {
      ss << getAllocStride(i) << ", ";
    }
    ss << ")";
    return ss.str();
  }

  std::unique_ptr<ArgAbstract> clone() const override {
    return std::make_unique<TensorArg>(*this);
  }
};

template <size_t size>
struct CpuScalarTensorArg : public ArgAbstract {
  std::array<std::byte, size> instance_;
  DEF_HELPEE_FUNC(CpuScalarTensor, instance_)
};

// TODO: This class needs some further clean up and refactor
//! KernelArgumentHolder copies meta information from kernel inputs, including
//! tensor sizes/shapes/dtype/memory_ptr and copies scalar inputs. It is used
//! for both compilation as well as kernel execution. The important thing is to
//! strip ownership of tensor from KernelArgumentHolder, so that during async
//! compilation, we are not unnecessarily holding memory that is not needed.
class TORCH_CUDA_CU_API KernelArgumentHolder {
 public:
  //! create KernelArgumentHolder from c10 inputs. Note that we we not taking
  //! the ownership of the memory from the original inputs, but just recording
  //! its meta data for kernel execution/compilation.
  static KernelArgumentHolder createKernelArgumentHolder(
      const c10::ArrayRef<c10::IValue>& inputs,
      std::optional<int8_t> device = std::nullopt);

  KernelArgumentHolder() = default;

  KernelArgumentHolder(const KernelArgumentHolder& self)
      : device_index_(self.getDeviceIndex()), cache_id_(self.getCacheId()) {
    for (const auto& arg : self.arguments_) {
      push(arg.get());
    }
  }

  KernelArgumentHolder& operator=(const KernelArgumentHolder& self) {
    device_index_ = self.getDeviceIndex();
    for (const auto& arg : self.arguments_) {
      push(arg.get());
    }
    return *this;
  }

  //! Computes the smallest index type for the currently held
  //! arguments. It does not consider any other tensors used in a kernel.
  PrimDataType getSmallestIndexTypeOfArguments() const;

  // Push a tensor proxy to the arguments
  void pushTensorProxy(
      const std::vector<int64_t>& sizes,
      const std::vector<int64_t>& strides,
      at::ScalarType dtype);

  // Push a tensor to the arguments
  void push(const at::Tensor& tensor);

  // Push a scalar or integer to the arguments
  void push(const c10::IValue& val);

  void push(const at::PhiloxCudaState& val);

  // Create a buffer, flatten arguments into it, align by 8 Bytes, return
  // pointers in the buffer. Tensor arguments are passed with the given index
  // type.
  void** getBuffer(
      PrimDataType index_type,
      std::vector<TensorView*> tvs,
      ExpressionEvaluator& eval);

  void push(const c10::ArrayRef<c10::IValue>& args);

  void push(const std::vector<at::Tensor>& tensors);

  void push(const ArgAbstract* arg);

  void erase(const ArgAbstract* arg);

  void swap(int i, const ArgAbstract* arg);

  // push int64
  void push(int64_t val);

  const ArgAbstract* back() const {
    return arguments_.back().get();
  }

  void appendPhiloxRNGSeed(uint64_t rand_offset);

  const ArgAbstract* at(size_t ind) const {
    return arguments_.at(ind).get();
  };

  const ArgAbstract* operator[](size_t ind) const {
    return at(ind);
  };

  size_t size() const {
    return arguments_.size();
  }

  bool empty() const {
    return arguments_.empty();
  }

  void setDeviceIndex(int8_t index) {
    device_index_ = index;
  }

  int8_t getDeviceIndex() const {
    return device_index_;
  }

  void setCacheId(size_t id) {
    cache_id_ = id;
  }

  std::optional<size_t> getCacheId() const {
    return cache_id_;
  }

  std::string toString() const;

 private:
  std::vector<std::unique_ptr<ArgAbstract>> arguments_;
  std::vector<void*> void_ptrs_;

  int8_t device_index_ = 0;
  std::optional<size_t> cache_id_ = std::nullopt;
};

} // namespace nvfuser
