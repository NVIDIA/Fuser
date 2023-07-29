// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <disjoint_set.h>
#include <ir/base_nodes.h>
#include <ir/printer.h>

/*
  This file implements the IRs composing a Pipeline:
  - PipelineVals
  - PipelineStage
  - PipelineCommunication
*/

namespace nvfuser {

class PipelineStageDescriptor;

/*
  A PipelineStage represents a Stage of a Pipeline.
  It is derived from Expr, and represents the composition
  of all the exprs of the originalFusion to go from the stage's inputs to the
  stage's output.

  It is instantiated from a list of I/O and a PipelineStageDescriptor (pointing
  to the originalFusion's Vals contained in the Stage)
*/
class TORCH_CUDA_CU_API PipelineStage : public Expr {
  using ValSet = VectorOfUniqueEntries<Val*>;

 public:
  using Expr::Expr;

  PipelineStage(
      IrBuilderPasskey passkey,
      const PipelineStageDescriptor* descriptor,
      ValSet input_vals,
      ValSet output_vals);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  const char* getOpString() const override {
    return "PipelineStage";
  }

  bool sameAs(const Statement* other) const override;

  const PipelineStageDescriptor* descriptor() const {
    return attribute<const PipelineStageDescriptor*>(0);
  }
};

/*
  PipelineVals are the only Vals of a Pipeline.
  Each instance contains a pointer to a Val from the originalFusion,
  as well as a pointer to the PipelineStage to which it belongs.
  Currently, PipelineVal is necessarily an input or output of a (unique)
  PipelineStage
*/
class TORCH_CUDA_CU_API PipelineVal : public Val {
 public:
  PipelineVal(IrBuilderPasskey passkey, Val* val);
  PipelineVal(const PipelineVal* src, IrCloner* ir_cloner);

  NVFUSER_DECLARE_CLONE

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  // returns the Val from which this PipelineVal has been created
  Val* getOriginalVal() const {
    return original_val_;
  }

  void setStage(PipelineStage* stage) {
    stage_ = stage;
  }

  // returns the PipelineStage to which the instance belongs
  PipelineStage* getStage() const {
    return stage_;
  }

  bool sameAs(const Statement* other) const override;

 private:
  // stores the Val used in the instance's constructor
  Val* original_val_;
  // stores the PipelineStage to which it belongs
  PipelineStage* stage_ = nullptr;
};

/*
  PipelineCommunication is a unary operation between two PipelineVals
  It represents the data communication in-between two PipelineStages
*/
class TORCH_CUDA_CU_API PipelineCommunication : public Expr {
 public:
  using Expr::Expr;
  PipelineCommunication(IrBuilderPasskey, Val* in, Val* out);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  const char* getOpString() const override {
    return "PipelineCommunication";
  }

  const auto& in() const {
    return inputs().at(0);
  }

  const auto& out() const {
    return outputs().at(0);
  }
};

} // namespace nvfuser
