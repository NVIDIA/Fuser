// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#pragma once
#include <nanobind/nanobind.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/autograd/python_variable.h>

namespace nanobind {
namespace detail {

// Register a caster for CPP at::Tensor and Python tensor.Tensor
template <>
struct type_caster<at::Tensor> {
  // Macro to define the type name exposed to Python error messages
  NB_TYPE_CASTER(at::Tensor, const_name("torch.Tensor"));

  bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) {
    if (!src || !THPVariable_Check(src.ptr())) {
      return false;
    }

    try {
      // Unpack the C++ tensor from the Python object
      // THPVariable_Unpack returns 'at::Tensor'
      value = THPVariable_Unpack(src.ptr());
      return true;
    } catch (const std::exception& e) {
      // Let nanobind handle the exception
      return false;
    }
  }

  static handle from_cpp(
      const at::Tensor& src,
      rv_policy policy,
      cleanup_list* cleanup) {
    // Wrap the C++ tensor into a new Python object
    // THPVariable_Wrap creates a new reference (PyObject*)
    PyObject* obj = THPVariable_Wrap(src);
    if (!obj) {
      throw std::runtime_error("Failed to wrap at::Tensor to Python object");
    }
    return handle(obj);
  }
};

// Register a caster for CPP c10::ScalarType and Python torch.dtype
template <>
struct type_caster<c10::ScalarType> {
  NB_TYPE_CASTER(c10::ScalarType, const_name("torch.dtype"));

  bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) {
    if (!THPDtype_Check(src.ptr())) {
      return false;
    }

    // Unpack the scalar type from the Python object
    value = ((THPDtype*)src.ptr())->scalar_type;
    return true;
  }

  static handle from_cpp(c10::ScalarType src, rv_policy, cleanup_list*) {
    return nb::none();
  }
};

} // namespace detail
} // namespace nanobind
