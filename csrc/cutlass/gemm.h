// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exception>
#include <string>

namespace nvfuser {

class Fusion;
class Val;

class CutlassParams;
class ScaledMmaOp;

namespace cutlass_codegen {

//! Instead of using something like DataWrapperOpt, we will throw these
//! exceptions whenever we fail to translate a Fusion
class UnsupportedFusion : public std::exception {
 public:
  UnsupportedFusion(const std::string& message) : message_(message) {}

  const char* what() const noexcept override {
    return message_.c_str();
  }

 private:
  const std::string message_;
};

#define NVF_CUTLASS_REJECT(msg, ...)                   \
  throw ::nvfuser::cutlass_codegen::UnsupportedFusion( \
      ::nvfuser::to_str(msg, ##__VA_ARGS__));

//! Unlike NVF_ERROR, this throws UnsupportedFusion whenever the condition is
//! TRUE
#define NVF_CUTLASS_REJECT_IF(cond, msg, ...) \
  if (cond) {                                 \
    NVF_CUTLASS_REJECT(msg, ##__VA_ARGS__);   \
  }

ScaledMmaOp* findScaledMmaOp(Fusion* fusion);

//! Simply finds the position of a Val in fusion->inputs().
int64_t fusionInputPosition(Fusion* fusion, Val* v);

//! Simply finds the position of a Val in fusion->outputs().
int64_t fusionOutputPosition(Fusion* fusion, Val* v);

std::string generateNvfp4ScaledMmKernel(
    Fusion* fusion,
    const CutlassParams& params);

std::string getGemmRejectReason(Fusion* fusion);

} // namespace cutlass_codegen

} // namespace nvfuser
