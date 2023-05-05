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
#include <torch/csrc/jit/ir/ir.h>
#include <type.h>
#include <array>
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
template <typename T, int N, typename nvfuser_index_t>
struct TensorArgCodegen {
  using data_type = T;
  using index_type = nvfuser_index_t;
  static constexpr int ndims = N;

  T& operator[](nvfuser_index_t ind) {
    return data[ind];
  };

  T* data;
  std::array<nvfuser_index_t, N> size;
  std::array<nvfuser_index_t, N> stride;
  constexpr int nDims() const {
    return N;
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
template <typename T, typename nvfuser_index_t>
struct TensorArgCodegen<T, 0, nvfuser_index_t> {
  using data_type = T;
  using index_type = nvfuser_index_t;
  static constexpr int ndims = 0;

  T& operator[](nvfuser_index_t ind) {
    return data[ind];
  };

  T* data;
  constexpr int nDims() const {
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

// Specialization for 0-dim case that's easy to pass in a CPU based tensor
// without memcpy
template <typename T>
struct CpuScalarTensorCodegen {
  T& operator[](int) {
    return data;
  };

  T data;
};

struct ArgAbstract {
  virtual ~ArgAbstract() = default;
  virtual const void* arg() const = 0;
  virtual void* arg() = 0;
  virtual bool isType(ArgType type) const = 0;
  virtual ArgType type() const = 0;
  virtual std::unique_ptr<ArgAbstract> copy_unique_ptr() const = 0;
  virtual std::string toString() const {
    return "input type: " + argTypeToString(type());
  };
};

#define DEF_HELPEE_FUNC(TARGET_TYPE, ARG_NAME)                    \
  bool isType(ArgType type) const override {                      \
    return ArgType::TARGET_TYPE == type;                          \
  }                                                               \
  ArgType type() const override {                                 \
    return ArgType::TARGET_TYPE;                                  \
  }                                                               \
  const void* arg() const override {                              \
    return &ARG_NAME;                                             \
  }                                                               \
  void* arg() override {                                          \
    return &ARG_NAME;                                             \
  }                                                               \
  std::unique_ptr<ArgAbstract> copy_unique_ptr() const override { \
    return std::make_unique<TARGET_TYPE##Arg>(*this);             \
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
  virtual int64_t getRank() const = 0;
  virtual int64_t getSize(int64_t i) const = 0;
  virtual int64_t getStride(int64_t i) const = 0;
  virtual void* getPointer() const = 0;
  virtual DataType getDataType() const = 0;
  virtual int64_t numel() const = 0;
  virtual at::Tensor getTensor() const = 0;
  virtual bool isIndexTypeResolved() const = 0;
  //! Returns the index type of the tensor. It's an error if the
  //! tensor does not have a resolved index type.
  virtual PrimDataType getIndexType() const = 0;

  std::string toString() const override;
};

template <typename TENSOR_TYPE>
struct TensorArg : public TensorArgAbstract {
  TENSOR_TYPE instance_;
  at::Tensor tensor_;
  bool index_type_resolved_ = false;

  TensorArg(const at::Tensor& tensor, bool index_type_resolved)
      : tensor_(tensor), index_type_resolved_(index_type_resolved) {
    setPointer(tensor.data_ptr());
    for (const auto i : c10::irange(tensor.ndimension())) {
      setSize(i, tensor.sizes()[i]);
      setStride(i, tensor.strides()[i]);
    }
  }

  void setSize(int64_t i, int64_t size) {
    instance_.setSize(i, (typename TENSOR_TYPE::index_type)size);
  }
  void setStride(int64_t i, int64_t stride) {
    instance_.setStride(i, (typename TENSOR_TYPE::index_type)stride);
  }
  void setPointer(void* ptr) {
    instance_.data = static_cast<decltype(TENSOR_TYPE::data)>(ptr);
  }
  void setTensor(at::Tensor tensor) {
    tensor_ = tensor;
  }

  int64_t getSize(int64_t i) const override {
    return instance_.getSize(i);
  }
  int64_t getStride(int64_t i) const override {
    return instance_.getStride(i);
  }
  int64_t getRank() const override {
    return instance_.nDims();
  }
  void* getPointer() const override {
    return instance_.data;
  }
  DataType getDataType() const override {
    return NativeTypeWithC10ComplexToDataType<
        typename TENSOR_TYPE::data_type>::type;
  }
  at::Tensor getTensor() const override {
    return tensor_;
  }
  int64_t numel() const override {
    int64_t ret = 1;
    for (auto i : c10::irange(instance_.nDims())) {
      ret *= instance_.getSize(i);
    }
    return ret;
  }

  bool isIndexTypeResolved() const override {
    return index_type_resolved_;
  }

  PrimDataType getIndexType() const override {
    TORCH_INTERNAL_ASSERT(isIndexTypeResolved());
    return NativeTypeToDataType<typename TENSOR_TYPE::index_type>::type;
  }

  bool isType(ArgType t) const override {
    return type() == t;
  }

  ArgType type() const override {
    return ArgType::Tensor;
  }

  //! Returns the address of an tensor argument struct. It's an error
  //! if called with a tensor with no resolved index type
  const void* arg() const override {
    TORCH_INTERNAL_ASSERT(isIndexTypeResolved());
    return &instance_;
  }

  //! Returns the address of an tensor argument struct. It's an error
  //! if called with a tensor with no resolved index type
  void* arg() override {
    TORCH_INTERNAL_ASSERT(isIndexTypeResolved());
    return &instance_;
  }

  std::unique_ptr<ArgAbstract> copy_unique_ptr() const override {
    return std::make_unique<TensorArg>(*this);
  }
};

template <typename CPU_TENSOR_TYPE>
struct CpuScalarTensorArg : public ArgAbstract {
  CPU_TENSOR_TYPE instance_;

  CpuScalarTensorArg() = delete;

  explicit CpuScalarTensorArg(decltype(CPU_TENSOR_TYPE::data) _data) {
    instance_.data = _data;
  }

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
      const c10::ArrayRef<c10::IValue>& inputs);

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

  // Push a tensor to the arguments
  void push(const at::Tensor& tensor);

  // Push a scalar or integer to the arguments
  void push(const c10::IValue& val);

  void push(const at::PhiloxCudaState& val);

  // Create a buffer, flatten arguments into it, align by 8 Bytes, return
  // pointers in the buffer. Tensor arguments are passed with the given index
  // type.
  void** getBuffer(PrimDataType index_type);

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
