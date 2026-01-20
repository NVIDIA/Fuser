// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <fusion.h>
#include <ir/base_nodes.h>
#include <ir/interface_nodes.h>
#include <mma_type.h>
#include <parallel_type_bitmap.h>
#include <visibility.h>

namespace nvfuser {

//! Matmul Operator to be expression evaluated without decomposition.
class MatmulOp : public Expr {
 public:
  using Expr::Expr;

  MatmulOp(IrBuilderPasskey, Val* out, Val* in_a, Val* in_b);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "MatmulOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  TensorView* out() const {
    return output(0)->as<TensorView>();
  }

  TensorView* inA() const {
    return input(0)->as<TensorView>();
  }

  TensorView* inB() const {
    return input(1)->as<TensorView>();
  }

  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;
};

// Linear node with same functionality as F.linear
// (https://pytorch.org/docs/stable/generated/torch.nn.functional.linear.html#torch.nn.functional.linear)
class LinearOp : public Expr {
 public:
  using Expr::Expr;

  LinearOp(IrBuilderPasskey, Val* out, Val* in_a, Val* in_b, Val* bias);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "LinearOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  TensorView* out() const {
    return output(0)->as<TensorView>();
  }

  TensorView* inA() const {
    return input(0)->as<TensorView>();
  }

  TensorView* inB() const {
    return input(1)->as<TensorView>();
  }

  TensorView* bias() const {
    if (hasBias()) {
      return input(2)->as<TensorView>();
    } else {
      return nullptr;
    }
  }

  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;

  bool hasBias() const {
    return inputs().size() == 3;
  }
};

// SDPA node with same functionality at::_scaled_dot_product_flash_attention
// output = [N, H, L, Ev]
// logsumexp = [N, H, L]
// query_seq_len = scalar(int)
// key_seq_len = scalar(int)
// philox_seed = CPU scalar tensor or uint64_t[2] tensor (for > 2.7.0)
// philox_offset = CPU scalar tensor or empty uint64_t tensor (for > 2.7.0)
// debug_attn_mask = scalar tensor (Thunder does not return a debug attn mask by
// setting `return_debug_mask=False` when invoking flash attention)

// Note: For older versions, torch returns CPU scalar tensors for philox_seed
// and philox_offset. For torch 2.7.0 and above, torch returns philox_seed ->
// rng_state (uint64_t[2]) and philox_offset -> _unused (empty tensor). The rng
// state contains both seed and offset.

// query = [N, H, L, E]
// key = [N, H, S, E]
// value = [N, H, S, Ev]
// dropout_p = scalar(double)
// is_causal = scalar(bool)
// scale = scalar(double)

// N = number of sequences / batch size
// H = num of heads
// L = query sequence length / target sequence length
// S = key/value sequence length / src sequence length
// E = query/key embd dimension
// Ev = value embd dimension

// For flash attention, E = Ev
class SdpaFwdOp : public Expr {
 public:
  using Expr::Expr;

  SdpaFwdOp(
      IrBuilderPasskey,
      TensorView* output,
      TensorView* log_sumexp,
      TensorView* philox_seed,
      TensorView* philox_offset,
      TensorView* query,
      TensorView* key,
      TensorView* value,
      TensorView* bias,
      TensorView* mask,
      Val* dropout_p,
      Val* is_causal,
      Val* scale);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "SdpaFwdOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  TensorView* attn_out() const {
    return output(0)->as<TensorView>();
  }

  TensorView* logsumexp() const {
    return output(1)->as<TensorView>();
  }

  TensorView* philox_seed() const {
    return output(2)->as<TensorView>();
  }

  TensorView* philox_offset() const {
    return output(3)->as<TensorView>();
  }

  TensorView* query() const {
    return input(0)->as<TensorView>();
  }

  TensorView* key() const {
    return input(1)->as<TensorView>();
  }

  TensorView* value() const {
    return input(2)->as<TensorView>();
  }

  int64_t bias_input_index() const {
    return attribute<int64_t>(1);
  }

  TensorView* bias() const {
    return bias_input_index() >= 0 ? input(bias_input_index())->as<TensorView>()
                                   : nullptr;
  }

  int64_t mask_input_index() const {
    return attribute<int64_t>(2);
  }

  TensorView* mask() const {
    return mask_input_index() >= 0 ? input(mask_input_index())->as<TensorView>()
                                   : nullptr;
  }

  Val* dropout_p() const {
    return input(3);
  }

  Val* is_causal() const {
    return input(4);
  }

  int64_t scale_input_index() const {
    return attribute<int64_t>(0);
  }

  Val* scale() const {
    return scale_input_index() >= 0 ? input(scale_input_index()) : nullptr;
  }

  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;
};

// SDPA bwd node with same functionality
// at::_scaled_dot_product_flash_attention_backward
// grad_query = [N, H, L, E]
// grad_key = [N, H, S, E]
// grad_value = [N, H, S, Ev]

// grad_output = [N, H, L, Ev]
// query = [N, H, L, E]
// key = [N, H, S, E]
// value = [N, H, S, Ev]
// output = [N, H, L, Ev]
// logsumexp = [N, H, L]
// dropout_p = scalar(double)
// is_causal = scalar(bool)
// philox_seed = CPU scalar tensor or uint64_t[2] tensor (for > 2.7.0)
// philox_offset = CPU scalar tensor or empty uint64_t tensor (for > 2.7.0)
// scale = scalar(double)

// Note: For older versions, torch accepts CPU scalar tensors for philox_seed
// and philox_offset. For torch 2.7.0 and above, torch accepts philox_seed ->
// rng_state (uint64_t[2]) and philox_offset -> _unused (empty tensor). The rng
// state contains both seed and offset.

// N = number of sequences / batch size
// H = num of heads
// L = query sequence length / target sequence length
// S = key/value sequence length / src sequence length
// E = query/key embd dimension
// Ev = value embd dimension

// For flash attention, E = Ev
class SdpaBwdOp : public Expr {
 public:
  using Expr::Expr;

  SdpaBwdOp(
      IrBuilderPasskey,
      TensorView* grad_query,
      TensorView* grad_key,
      TensorView* grad_value,
      TensorView* grad_output,
      TensorView* query,
      TensorView* key,
      TensorView* value,
      TensorView* output,
      TensorView* log_sumexp,
      Val* dropout_p,
      Val* is_causal,
      TensorView* philox_seed,
      TensorView* philox_offset,
      Val* scale);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "SdpaBwdOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  TensorView* grad_query() const {
    return output(0)->as<TensorView>();
  }

  TensorView* grad_key() const {
    return output(1)->as<TensorView>();
  }

  TensorView* grad_value() const {
    return output(2)->as<TensorView>();
  }

  TensorView* grad_attn() const {
    return input(0)->as<TensorView>();
  }

  TensorView* query() const {
    return input(1)->as<TensorView>();
  }

  TensorView* key() const {
    return input(2)->as<TensorView>();
  }

  TensorView* value() const {
    return input(3)->as<TensorView>();
  }

  TensorView* attn_out() const {
    return input(4)->as<TensorView>();
  }

  TensorView* logsumexp() const {
    return input(5)->as<TensorView>();
  }

  Val* dropout_p() const {
    return input(6);
  }

  Val* is_causal() const {
    return input(7);
  }

  Val* philox_seed() const {
    return input(8);
  }

  Val* philox_offset() const {
    return input(9);
  }

  Val* scale() const {
    if (inputs().size() > 10) {
      return input(10);
    }
    return nullptr;
  }

  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;
};

class EmbeddingFwdOp : public Expr {
 public:
  using Expr::Expr;

  EmbeddingFwdOp(
      IrBuilderPasskey,
      TensorView* output,
      TensorView* input,
      TensorView* weight,
      Val* padding_idx,
      Val* max_norm,
      Val* norm_type,
      Val* scale_grad_by_freq,
      Val* sparse);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "EmbeddingFwdOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  TensorView* out() const {
    return output(0)->as<TensorView>();
  }

  TensorView* in() const {
    return input(0)->as<TensorView>();
  }

  TensorView* weight() const {
    return input(1)->as<TensorView>();
  }

  Val* norm_type() const {
    return input(2);
  }

  Val* scale_grad_by_freq() const {
    return input(3);
  }

  Val* sparse() const {
    return input(4);
  }

  Val* padding_idx() const {
    if (has_padding_idx()) {
      return input(5);
    }
    return nullptr;
  }

  Val* max_norm() const {
    if (has_max_norm()) {
      return input(5 + has_padding_idx());
    }
    return nullptr;
  }

  bool has_padding_idx() const {
    return attribute<bool>(0);
  }

  bool has_max_norm() const {
    return attribute<bool>(1);
  }

  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;
};

class ArgsortOp : public Expr {
 public:
  using Expr::Expr;

  ArgsortOp(
      IrBuilderPasskey,
      Val* out,
      Val* in,
      int64_t dim,
      bool descending = false,
      bool stable = false);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "ArgsortOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;

  Val* out() const {
    return output(0);
  }
  Val* in() const {
    return input(0);
  }
  int64_t dim() const {
    return attribute<int64_t>(0);
  }
  bool isDescending() const {
    return attribute<bool>(1);
  }
  bool isStable() const {
    return attribute<bool>(2);
  }
};

class NVF_API TopKOp : public Expr {
 public:
  using Expr::Expr;

  TopKOp(
      IrBuilderPasskey,
      Val* out_values,
      Val* out_indices,
      Val* in,
      Val* k,
      int64_t dim,
      bool largest,
      bool sorted);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "TopKOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;

  Val* outValues() const {
    return output(0);
  }
  Val* outIndices() const {
    return output(1);
  }
  Val* in() const {
    return input(0);
  }
  Val* k() const {
    return input(1);
  }
  int64_t dim() const {
    return attribute<int64_t>(0);
  }
  bool isLargest() const {
    return attribute<bool>(1);
  }
  bool isSorted() const {
    return attribute<bool>(2);
  }
};

class GroupedMmaOp : public Expr {
 public:
  using Expr::Expr;

  GroupedMmaOp(
      IrBuilderPasskey,
      Val* out_mat,
      Val* out_scale,
      Val* out_gamma,
      Val* mat1,
      Val* mat2,
      Val* offsets,
      Val* scale1 = nullptr,
      Val* scale2 = nullptr,
      Val* alpha = nullptr,
      Val* bias = nullptr,
      Val* beta = nullptr);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "GroupedMmaOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;

  // Get output matrix
  TensorView* out() const {
    return output(0)->as<TensorView>();
  }

  // Get output block scaling factor
  TensorView* outScale() const {
    if (outputs().size() > 1) {
      return output(1)->as<TensorView>();
    }
    return nullptr;
  }

  // Get output global scaling factor
  TensorView* outGamma() const {
    if (outputs().size() > 2) {
      return output(2)->as<TensorView>();
    }
    return nullptr;
  }

  // Get first input matrix
  TensorView* matrix1() const {
    return input(0)->as<TensorView>();
  }

  // Get second input matrix
  TensorView* matrix2() const {
    return input(1)->as<TensorView>();
  }

  // Get group offset input vector
  TensorView* offsets() const {
    return input(2)->as<TensorView>();
  }

  // Get scale factor for first input matrix, returns nullptr if not present
  TensorView* scale1() const {
    if (hasScale()) {
      return input(attribute<int64_t>(0))->as<TensorView>();
    }
    return nullptr;
  }

  // Get scale factor for second input matrix, returns nullptr if not present
  TensorView* scale2() const {
    if (hasScale()) {
      return input(attribute<int64_t>(0) + 1)->as<TensorView>();
    }
    return nullptr;
  }

  TensorView* alpha() const {
    if (hasAlpha()) {
      return input(attribute<int64_t>(1))->as<TensorView>();
    }
    return nullptr;
  }

  TensorView* bias() const {
    if (hasBias()) {
      return input(attribute<int64_t>(2))->as<TensorView>();
    }
    return nullptr;
  }

  TensorView* beta() const {
    if (hasBeta()) {
      return input(attribute<int64_t>(3))->as<TensorView>();
    }
    return nullptr;
  }

  // True if scale factors are present
  bool hasScale() const {
    return attribute<int64_t>(0) != -1;
  }

  int64_t scale1Offset() const {
    return attribute<int64_t>(0);
  }

  // True if scale factors are present
  bool hasAlpha() const {
    return attribute<int64_t>(1) != -1;
  }

  int64_t alphaOffset() const {
    return attribute<int64_t>(1);
  }

  // True if bias is present
  bool hasBias() const {
    return attribute<int64_t>(2) != -1;
  }

  int64_t biasOffset() const {
    return attribute<int64_t>(2);
  }

  // True if beta is present
  bool hasBeta() const {
    return attribute<int64_t>(3) != -1;
  }

  int64_t betaOffset() const {
    return attribute<int64_t>(0);
  }

  int64_t scale2Offset() const {
    return attribute<int64_t>(0) + 1;
  }

  // Get the IterDomain for the k-dimension of the first input matrix
  IterDomain* getKDimOfMatrix1() const;

  // Get the IterDomain for the k-dimension of the second input matrix
  IterDomain* getKDimOfMatrix2() const;

  // Get the IterDomain for the group dimension of the first input matrix,
  // returns nullptr if not present
  IterDomain* getGroupDimOfMatrix1() const;

  // Get the IterDomain for the group dimension of the second input matrix,
  // returns nullptr if not present
  IterDomain* getGroupDimOfMatrix2() const;

  // Get the IterDomain for the group dimension of the output matrix, returns
  // nullptr if not present
  IterDomain* getGroupDimOfOutput() const;
};

class ScaledMmaOp : public Expr {
 public:
  using Expr::Expr;

  ScaledMmaOp(
      IrBuilderPasskey,
      Val* out_mat,
      Val* out_scale,
      Val* out_gamma,
      Val* mat1,
      Val* mat2,
      Val* scale1,
      Val* scale2,
      Val* alpha = nullptr,
      Val* bias = nullptr,
      Val* beta = nullptr);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "ScaledMmaOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;

  // Get output matrix
  TensorView* out() const {
    return output(0)->as<TensorView>();
  }

  // Get output block scaling factor
  TensorView* outScale() const {
    if (outputs().size() > 1) {
      return output(1)->as<TensorView>();
    }
    return nullptr;
  }

  // Get output global scaling factor
  TensorView* outGamma() const {
    if (outputs().size() > 2) {
      return output(2)->as<TensorView>();
    }
    return nullptr;
  }

  // Get first input matrix
  TensorView* matrix1() const {
    return input(0)->as<TensorView>();
  }

  // Get second input matrix
  TensorView* matrix2() const {
    return input(1)->as<TensorView>();
  }

  // Get scale factor for first input matrix, returns nullptr if not present
  TensorView* scale1() const {
    return input(2)->as<TensorView>();
  }

  // Get scale factor for second input matrix, returns nullptr if not present
  TensorView* scale2() const {
    return input(3)->as<TensorView>();
  }

  TensorView* alpha() const {
    if (hasAlpha()) {
      return input(attribute<int64_t>(0))->as<TensorView>();
    }
    return nullptr;
  }

  TensorView* bias() const {
    if (hasBias()) {
      return input(attribute<int64_t>(1))->as<TensorView>();
    }
    return nullptr;
  }

  TensorView* beta() const {
    if (hasBeta()) {
      return input(attribute<int64_t>(2))->as<TensorView>();
    }
    return nullptr;
  }

  // True if scale factors are present
  bool hasAlpha() const {
    return attribute<int64_t>(0) != -1;
  }

  int64_t alphaOffset() const {
    return attribute<int64_t>(0);
  }

  // True if bias is present
  bool hasBias() const {
    return attribute<int64_t>(1) != -1;
  }

  int64_t biasOffset() const {
    return attribute<int64_t>(1);
  }

  // True if beta is present
  bool hasBeta() const {
    return attribute<int64_t>(2) != -1;
  }

  int64_t betaOffset() const {
    return attribute<int64_t>(2);
  }
};

class ScanOp : public Expr {
 public:
  using Expr::Expr;

  ScanOp(
      IrBuilderPasskey,
      BinaryOpType op_type,
      Val* init,
      Val* out,
      Val* in,
      int64_t dim);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "ScanOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  //! Returns the inclusive scan output
  Val* out() const {
    return output(0);
  }

  Val* in() const {
    return input(0);
  }

  Val* init() const {
    return attributeVal(0);
  }

  BinaryOpType opType() const {
    return attribute<BinaryOpType>(1);
  }

  int64_t dim() const {
    return attribute<int64_t>(2);
  }

  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;
};

class CutlassNvfp4GroupedMmaOp : public Expr {
 public:
  using Expr::Expr;

  CutlassNvfp4GroupedMmaOp(
      IrBuilderPasskey,
      Val* out_mat,
      Val* mat1,
      Val* mat2,
      Val* scale1,
      Val* scale2,
      Val* alpha,
      Val* problem_sizes,
      Val* expert_offsets,
      Val* sf_offsets);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "CutlassNvfp4GroupedMmaOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;

  // Get output matrix
  TensorView* out() const {
    return output(0)->as<TensorView>();
  }

  // Get first input matrix
  TensorView* matrix1() const {
    return input(0)->as<TensorView>();
  }

  // Get second input matrix
  TensorView* matrix2() const {
    return input(1)->as<TensorView>();
  }

  // Get scale factor for first input matrix, returns nullptr if not present
  TensorView* scale1() const {
    return input(2)->as<TensorView>();
  }

  // Get scale factor for second input matrix, returns nullptr if not present
  TensorView* scale2() const {
    return input(3)->as<TensorView>();
  }

  TensorView* alpha() const {
    return input(4)->as<TensorView>();
  }

  TensorView* problemSizes() const {
    return input(5)->as<TensorView>();
  }

  TensorView* expertOffsets() const {
    return input(6)->as<TensorView>();
  }

  TensorView* scalingFactorOffsets() const {
    return input(7)->as<TensorView>();
  }
};

class PreprocessGroupedMatmulInputSf : public Expr {
 public:
  using Expr::Expr;

  // NOTE: row_idx and col_idx are used only for index lowering.
  PreprocessGroupedMatmulInputSf(
      IrBuilderPasskey,
      Val* output,
      Val* input,
      Val* input_offsets,
      Val* output_offsets,
      BlockScalingFactorLayout layout,
      Val* k,
      Val* g,
      Val* row_idx = nullptr,
      Val* col_idx = nullptr);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "PreprocessGroupedMatmulInputSf";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;

  Val* out() const {
    return output(0);
  }

  Val* in() const {
    return input(0);
  }

  TensorView* inputOffsets() const {
    return input(1)->as<TensorView>();
  }

  TensorView* outputOffsets() const {
    return input(2)->as<TensorView>();
  }

  // get scalar - column size
  Val* k() const {
    return input(3);
  }

  // get scalar - number of groups
  Val* g() const {
    return input(4);
  }

  // get enum - block scaling factor layout
  BlockScalingFactorLayout layout() const {
    return attribute<BlockScalingFactorLayout>(0);
  }
};

class BlockQuantizationOp : public Expr {
 public:
  using Expr::Expr;

  // This op takes in a high precision input(input)
  // and returns the quantized output(output) along with the block scaling
  // factors (output_scales). It can also take as an optional input the global
  // scaling factor and block size (though we currently only support 16).
  // logical_index is used for internal implemtation. This op is currently
  // implemented via a runtime function. During index computation, we compute
  // the index of the output_scales and pass it to the runtime function.
  BlockQuantizationOp(
      IrBuilderPasskey,
      Val* output_scales,
      Val* output,
      Val* input,
      Val* logical_index = nullptr,
      Val* global_scale = nullptr,
      int64_t block_size = 16,
      bool swizzled_scales = false);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  Val* blockScales() const {
    return output(1);
  }

  Val* quantizedOutput() const {
    return output(0);
  }

  Val* in() const {
    return input(0);
  }

  int64_t blockSize() const {
    return attribute<int64_t>(1);
  }

  bool hasGlobalScale() const {
    if (inputs().size() > 1) {
      return true;
    }
    return false;
  }

  Val* globalScale() const {
    if (hasGlobalScale()) {
      return input(1);
    }
    return nullptr;
  }

  const char* getOpString() const override {
    return "BlockQuantizationOp";
  }

  bool isSwizzledScales() const {
    return attribute<bool>(2);
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;
};

class GroupedBlockQuantizationOp : public Expr {
 public:
  using Expr::Expr;

  // This op takes in a high precision input(input)
  // and returns the quantized output(output) along with the block scaling
  // factors (output_scales). It can also take as an optional input the global
  // scaling factor and block size (though we currently only support 16).
  // logical_index is used for internal implemtation. This op is currently
  // implemented via a runtime function. During index computation, we compute
  // the index of the output_scales and pass it to the runtime function.
  GroupedBlockQuantizationOp(
      IrBuilderPasskey,
      Val* output_scales,
      Val* output,
      Val* input,
      Val* input_offsets,
      Val* output_offsets,
      BlockScalingFactorLayout layout,
      Val* k,
      Val* g,
      Val* global_scale = nullptr,
      int64_t block_size = 16,
      Val* row_idx = nullptr,
      Val* col_idx = nullptr);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  Val* blockScales() const {
    return output(1);
  }

  Val* quantizedOutput() const {
    return output(0);
  }

  Val* in() const {
    return input(0);
  }

  int64_t blockSize() const {
    return attribute<int64_t>(0);
  }

  bool hasGlobalScale() const {
    if (inputs().size() > 5) {
      return true;
    }
    return false;
  }

  Val* globalScale() const {
    if (hasGlobalScale()) {
      return input(5);
    }
    return nullptr;
  }

  const char* getOpString() const override {
    return "GroupedBlockQuantizationOp";
  }

  TensorView* inputOffsets() const {
    return input(1)->as<TensorView>();
  }

  TensorView* outputOffsets() const {
    return input(2)->as<TensorView>();
  }

  // get scalar - column size
  Val* k() const {
    return input(3);
  }

  // get scalar - number of groups
  Val* g() const {
    return input(4);
  }

  // get enum - block scaling factor layout
  BlockScalingFactorLayout layout() const {
    return attribute<BlockScalingFactorLayout>(1);
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;
};

} // namespace nvfuser
