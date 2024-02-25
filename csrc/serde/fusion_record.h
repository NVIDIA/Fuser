// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <exceptions.h>
#include <python_frontend/fusion_record.h>
#include <serde/factory.h>

namespace nvfuser::serde {

// Forward definition for RecordFunctor
struct RecordFunctor;

// OpRecord Function Signatures
// ========================================================================
// Unary Functions
typedef std::function<TensorView*(TensorView*)> unary_tv_fn;
typedef std::function<Val*(Val*)> unary_val_fn;

// ========================================================================
// Binary Functions
typedef std::function<TensorView*(TensorView*, TensorView*)> binary_tv_fn;
typedef std::function<Val*(Val*, Val*)> binary_val_fn;
typedef std::function<TensorView*(TensorView*, Val*)> binary_tv_val_fn;
typedef std::function<TensorView*(Val*, TensorView*)> binary_val_tv_fn;

// ========================================================================
// Ternary Functions
// Binary with Alpha Functions
typedef std::function<TensorView*(TensorView*, TensorView*, TensorView*)>
    ternary_tv_fn;
typedef std::function<Val*(Val*, Val*, Val*)> ternary_val_fn;
typedef std::function<TensorView*(TensorView*, TensorView*, Val*)>
    ternary_tv_tv_val_fn;
typedef std::function<TensorView*(TensorView*, Val*, TensorView*)>
    ternary_tv_val_tv_fn;
typedef std::function<TensorView*(Val*, TensorView*, TensorView*)>
    ternary_val_tv_tv_fn;
typedef std::function<TensorView*(Val*, Val*, TensorView*)>
    ternary_val_val_tv_fn;
typedef std::function<TensorView*(TensorView*, Val*, Val*)>
    ternary_tv_val_val_fn;
typedef std::function<TensorView*(Val*, TensorView*, Val*)>
    ternary_val_tv_val_fn;

// ========================================================================
// Ternary with Alpha Functions
typedef std::function<TensorView*(TensorView*, TensorView*, TensorView*, Val*)>
    ternary_alpha_tv_fn;
typedef std::function<Val*(Val*, Val*, Val*, Val*)> ternary_alpha_val_fn;
typedef std::function<TensorView*(TensorView*, TensorView*, Val*, Val*)>
    ternary_alpha_tv_tv_val_fn;
typedef std::function<TensorView*(TensorView*, Val*, TensorView*, Val*)>
    ternary_alpha_tv_val_tv_fn;
typedef std::function<TensorView*(Val*, TensorView*, TensorView*, Val*)>
    ternary_alpha_val_tv_tv_fn;
typedef std::function<TensorView*(Val*, Val*, TensorView*, Val*)>
    ternary_alpha_val_val_tv_fn;
typedef std::function<TensorView*(TensorView*, Val*, Val*, Val*)>
    ternary_alpha_tv_val_val_fn;
typedef std::function<TensorView*(Val*, TensorView*, Val*, Val*)>
    ternary_alpha_val_tv_val_fn;
// ========================================================================

//! The RecordFunctorFactory class is used to deserialize the flatbuffer
//! RecordFunctor table. We create an enum type for each RecordFunctor class.
//! Each template specialization has a unique RecordType and parser function.
class RecordFunctorFactory
    : public Factory<RecordFunctor, python_frontend::RecordFunctor*> {
 public:
  RecordFunctorFactory()
      : Factory((nvfuser::toUnderlying(RecordType::MAX) + 1)) {
    setupFunctionMaps();
    registerAllParsers();
  }

 private:
  void registerAllParsers();
  void setupFunctionMaps();

  // String to Operation maps
  // Unary Functions
  std::unordered_map<std::string, unary_tv_fn> unary_tv;
  std::unordered_map<std::string, unary_val_fn> unary_val;

  // Binary Functions
  std::unordered_map<std::string, binary_tv_fn> binary_tv;
  std::unordered_map<std::string, binary_val_fn> binary_val;
  std::unordered_map<std::string, binary_tv_val_fn> binary_tv_val;
  std::unordered_map<std::string, binary_val_tv_fn> binary_val_tv;

  // Ternary Functions
  // Binary with Alpha Functions
  std::unordered_map<std::string, ternary_tv_fn> ternary_tv;
  std::unordered_map<std::string, ternary_val_fn> ternary_val;
  std::unordered_map<std::string, ternary_tv_tv_val_fn> ternary_tv_tv_val;
  std::unordered_map<std::string, ternary_tv_val_tv_fn> ternary_tv_val_tv;
  std::unordered_map<std::string, ternary_val_tv_tv_fn> ternary_val_tv_tv;
  std::unordered_map<std::string, ternary_val_val_tv_fn> ternary_val_val_tv;
  std::unordered_map<std::string, ternary_tv_val_val_fn> ternary_tv_val_val;
  std::unordered_map<std::string, ternary_val_tv_val_fn> ternary_val_tv_val;

  // Ternary with Alpha Functions
  std::unordered_map<std::string, ternary_alpha_tv_fn> ternary_alpha_tv;
  std::unordered_map<std::string, ternary_alpha_val_fn> ternary_alpha_val;
  std::unordered_map<std::string, ternary_alpha_tv_tv_val_fn>
      ternary_alpha_tv_tv_val;
  std::unordered_map<std::string, ternary_alpha_tv_val_tv_fn>
      ternary_alpha_tv_val_tv;
  std::unordered_map<std::string, ternary_alpha_val_tv_tv_fn>
      ternary_alpha_val_tv_tv;
  std::unordered_map<std::string, ternary_alpha_val_val_tv_fn>
      ternary_alpha_val_val_tv;
  std::unordered_map<std::string, ternary_alpha_tv_val_val_fn>
      ternary_alpha_tv_val_val;
  std::unordered_map<std::string, ternary_alpha_val_tv_val_fn>
      ternary_alpha_val_tv_val;
};

} // namespace nvfuser::serde
