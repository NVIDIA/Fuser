// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <list>

#include <exceptions.h>
#include <fusion.h>
#include <ir/base_nodes.h>
#include <ir/interface_nodes.h>
#include <mma_type.h>
#include <parallel_type_bitmap.h>
#include <visibility.h>

//! Nodes in here should generally not be used by users. They should be behind
//! the scenes and users shouldn't have to be aware of what they do to use the
//! code generator
//!
//! \todo improve implementation bool IterDomain::sameAs(const IterDomain*)
//! \todo Add testing of sameAs functions for these nodes
//!

//! IR header hierarchy
//! 1. utils.h - PolymorphicBase and NonCopyable
//! 2. ir/base_nodes.h - Statement, Expr, and Val
//! 3. ir/internal_base_nodes.h - IterDomain and TensorDomain
//! 4. ir/interface_nodes.h - TensorView and Scalar
//! 5. ** ir/internal_nodes.h ** - Any internal-only IR nodes

namespace nvfuser {

class ViewTransform;
class Scope;
class IrCloner;
struct AnalyzeViewResult;

class NVF_API FullOp : public Expr {
 public:
  using Expr::Expr;

  FullOp(IrBuilderPasskey, Val* out, Val* fill_value);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "FullOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;

  Val* getFillValue() const {
    return inputs().back();
  }
};

class SelectOp : public Expr {
 public:
  using Expr::Expr;

  SelectOp(IrBuilderPasskey, Val* out, Val* in, int64_t dim, Val* index);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "SelectOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;

  TensorView* lookupTv() const {
    return input(0)->as<TensorView>();
  }

  int64_t dim() const {
    return attribute<int64_t>(0);
  }

  IterDomain* getIndexedID() const;

  std::unordered_map<IterDomain*, Val*> getIndexOverridingMap() const {
    return {{getIndexedID(), input(1)}};
  }
};

class IndexSelectOp : public Expr {
 public:
  using Expr::Expr;

  IndexSelectOp(IrBuilderPasskey, Val* out, Val* in1, int64_t dim, Val* in3);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "IndexSelectOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;

  TensorView* lookupTv() const {
    return input(0)->as<TensorView>();
  }

  TensorView* indexTv() const {
    return input(1)->as<TensorView>();
  }

  IterDomain* getIndexedID() const;

  IterDomain* getConsumerOfIndexedID() const;

  int64_t dim() const {
    return attribute<int64_t>(0);
  }
};

class IndexPutAccumulateOp : public Expr {
 public:
  using Expr::Expr;

  // [ Note -- IndexPutAccumulateOp semantics ]
  //
  // logical ID groups of IndexPutAccumulateOp
  // args:
  //     acc   [ ID_indexed_g0, ID_g0 ]
  //     index [ ID_indexing_g1 ]
  //     value [ ID_indexing_g1, ID_g0 ]
  // output:
  //     out   [ ID_indexed_g0, ID_g0 ]
  //
  // Note that:
  //     1. indexed ID for `out` and `acc` share the same extent.
  //     2. indexed ID for `index` and `value` share the same extent.
  IndexPutAccumulateOp(
      IrBuilderPasskey,
      Val* out,
      Val* acc,
      Val* index,
      Val* value);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "IndexPutAccumulateOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;

  TensorView* accumulateTv() const {
    return input(0)->as<TensorView>();
  }

  TensorView* indexTv() const {
    return input(1)->as<TensorView>();
  }

  TensorView* valueTv() const {
    return input(2)->as<TensorView>();
  }

  // return ID_indexing_g1 from value
  IterDomain* getIndexingIDOfValue() const;

  // return ID_indexing_g1 from index, for IndexPutAccumulate, there's only one
  // indexing ID at this moment
  IterDomain* getIndexingID() const;
};

class GatherOp : public Expr {
 public:
  using Expr::Expr;

  //! Parameter exact_sizes indicates whether the non-indexed domains
  //! of the index tensor have the same extents of those of the input
  //! tensor. It's true in the case of take_along_axis.
  GatherOp(
      IrBuilderPasskey,
      Val* out,
      Val* in,
      int64_t dim,
      Val* index,
      bool exact_sizes);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "GatherOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;

  TensorView* lookupTv() const {
    return input(0)->as<TensorView>();
  }

  TensorView* indexTv() const {
    return input(1)->as<TensorView>();
  }

  int64_t dim() const {
    return attribute<int64_t>(0);
  }

  IterDomain* getIndexedID() const;

  IterDomain* getConsumerOfIndexedID() const;

  bool exactSizes() const {
    return attribute<bool>(1);
  }
};

// ScatterOp represents an out-of-place scatter operation as part of a
// compute definition. However, its scheduling definition is always
// based on the logical domain of the index input tensor. More
// specifically, from the viewpoint of PyTorch/Thunder, the output
// input and output tensors correspond to different tensors, however,
// from the scheduling point of view, the output tensor always has a
// loop domain that is derived from the logical domain of the index
// logical domain.
//
// IMPLEMENTATION NOTE: This is currently implemented using the
// initial loop domain of TensorDomain. To build a valid ScatterOp, the
// TensorDomain of the output tensor must have a loop domain that is differnt
// from the logical domain. The initial loop domain, kept tracked as
// TensorDomain::initial_loop_, is used to augment the Exact graph by
// adding mappings with the logical domain of the index and src input
// tensors.
class ScatterOp : public Expr {
 public:
  using Expr::Expr;

  // exact_sizes: true when non-scatter axes of all inputs are
  // guaranteed to have the same extents
  ScatterOp(
      IrBuilderPasskey,
      Val* out,
      Val* self,
      int64_t dim,
      Val* index,
      Val* src,
      bool exact_sizes,
      std::optional<BinaryOpType> accumulate_op = std::nullopt);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "ScatterOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;

  Val* in() const {
    return input(0);
  }

  Val* out() const {
    return output(0);
  }

  Val* index() const {
    return input(1);
  }

  Val* src() const {
    return input(2);
  }

  int64_t dim() const {
    return attribute<int64_t>(0);
  }

  IterDomain* getIndexedID() const;

  bool exactSizes() const {
    return attribute<bool>(1);
  }

  bool accumulate() const {
    return attribute<bool>(2);
  }

  BinaryOpType accumulateOp() const {
    NVF_ERROR(accumulate());
    return attribute<BinaryOpType>(3);
  }
};

class IotaOp : public Expr {
 public:
  using Expr::Expr;

  IotaOp(IrBuilderPasskey, Val* out, Val* length, Val* start, Val* step);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "IotaOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;

  DataType dtype() const {
    return *start()->getDataType();
  }

  Val* length() const {
    return input(0);
  }

  Val* start() const {
    return input(1);
  }

  Val* step() const {
    return input(2);
  }
};

// Tensor factory for generating identity matrices like
//
// [[1, 0, 0],
//  [0, 1, 0],
//  [0, 0, 1]]
//
// or
//
// [[1, 0, 0],
//  [0, 1, 0],
//  [0, 0, 1],
//  [0, 0, 0]]
//
// or
//
// [[1, 0, 0, 0],
//  [0, 1, 0, 0],
//  [0, 0, 1, 0]]
class EyeOp : public Expr {
 public:
  using Expr::Expr;

  EyeOp(IrBuilderPasskey, Val* out, DataType dtype);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "EyeOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;

  DataType dtype() const {
    return attribute<DataType>(0);
  }
};

//! A specialization for Unary operations. Unary operations take in a single
//! input and produce a single output. Examples include:
//!   1) Casting operation i.e. float(a_val)
//!   2) Negation i.e. val * -1
//!   3) Reduction across a dimension i.e. val.sum(axis=2)
//!   4) split/merge
class UnaryOp : public Expr {
 public:
  using Expr::Expr;

  UnaryOp(IrBuilderPasskey, UnaryOpType type, Val* out, Val* in);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "UnaryOp";
  }

  std::string getGraphvizLabel() const override;

  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Val* out() const {
    return output(0);
  }
  Val* in() const {
    return input(0);
  }

  UnaryOpType getUnaryOpType() const {
    return attribute<UnaryOpType>(0);
  }

 private:
  void printHelper(std::stringstream& ss, std::string input) const;
};

//! A specialization for Binary operations. Binary operations take in two inputs
//! and produce a single output. Examples include:
//!  1) Add/mul/div/mod/sub (A * B)
//!  2) LT (A < B)
class BinaryOp : public Expr {
 public:
  using Expr::Expr;

  BinaryOp(IrBuilderPasskey, BinaryOpType type, Val* out, Val* lhs, Val* rhs);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "BinaryOp";
  }

  std::string getGraphvizLabel() const override;

  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Val* out() const {
    return output(0);
  }
  Val* lhs() const {
    return input(0);
  }
  Val* rhs() const {
    return input(1);
  }

  BinaryOpType getBinaryOpType() const {
    return attribute<BinaryOpType>(0);
  }

 private:
  void printHelper(
      std::stringstream& ss,
      int indent_size,
      std::string lhs,
      std::string rhs) const;
};

class TernaryOp : public Expr {
 public:
  using Expr::Expr;

  TernaryOp(
      IrBuilderPasskey,
      TernaryOpType type,
      Val* out,
      Val* in1,
      Val* in2,
      Val* in3);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "TernaryOp";
  }

  std::string getGraphvizLabel() const override;

  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Val* out() const {
    return output(0);
  }

  Val* in1() const {
    return input(0);
  }
  Val* in2() const {
    return input(1);
  }
  Val* in3() const {
    return input(2);
  }

  TernaryOpType getTernaryOpType() const {
    return attribute<TernaryOpType>(0);
  }

 private:
  void printHelper(
      std::stringstream& ss,
      int indent_size,
      std::string in1,
      std::string in2,
      std::string in3) const;
};

// construct an array from a list of values
class ArrayConstruct : public Expr {
 public:
  using Expr::Expr;

  ArrayConstruct(IrBuilderPasskey, Val* output, std::vector<Val*> inputs);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "ArrayConstruct";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;

  Val* out() const {
    return output(0);
  }
};

class ReverseArray : public Expr {
 public:
  using Expr::Expr;

  ReverseArray(IrBuilderPasskey, Val* output, Val* input);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "ReverseArray";
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
};

// Get an item from an array, array[index]
class GetItem : public Expr {
 public:
  using Expr::Expr;

  GetItem(IrBuilderPasskey, Val* output, Val* array, Val* index);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "GetItem";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;

  Val* out() const {
    return output(0);
  }

  Val* array() const {
    return input(0);
  }

  Val* index() const {
    return input(1);
  }
};

// construct a struct from a list of values
class StructConstruct : public Expr {
 public:
  using Expr::Expr;

  StructConstruct(
      IrBuilderPasskey,
      Val* output,
      const std::vector<std::pair<std::string, Val*>>& fields);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "StructConstruct";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;

  std::string fieldName(size_t i) const {
    return attribute<std::string>(i);
  }

  Val* out() const {
    return output(0);
  }
};

// Get an attribute from a struct, struct.attr
class GetAttr : public Expr {
 public:
  using Expr::Expr;

  GetAttr(IrBuilderPasskey, Val* output, Val* struct_, std::string attr);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "GetAttr";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;

  Val* out() const {
    return output(0);
  }

  Val* struct_() const {
    return input(0);
  }

  std::string attr() const {
    return attribute<std::string>(0);
  }
};

// Get an attribute from a struct, struct.attr
class GetMetaData : public Expr {
 public:
  using Expr::Expr;

  GetMetaData(IrBuilderPasskey, Val* output, Val* input);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "GetMetaData";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  bool sameAs(const Statement* other) const override {
    auto other_meta = dynamic_cast<const GetMetaData*>(other);
    if (other_meta == nullptr) {
      return false;
    }
    // Do not recursively check input, because if we have
    // T1 = set(T0)
    // T2 = set(T0)
    // Then even if T1->sameAs(T2), they should not have the same metadata.
    // For example, T1 and T2 may be different fusion outputs, so their data
    // pointers are different.
    return other_meta->in() == in();
  }

  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;

  Val* out() const {
    return output(0);
  }

  Val* in() const {
    return input(0);
  }
};

// Construct a tensor from an array
class TensorConstruct : public Expr {
 public:
  using Expr::Expr;

  TensorConstruct(IrBuilderPasskey, TensorView* output, Val* input);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "TensorConstruct";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;

  TensorView* out() const {
    return output(0)->as<TensorView>();
  }

  Val* in() const {
    return input(0);
  }
};

//! A specialization for random number generator (RNG) operations. RNG
//! operations take in no tensor input and produce a single output.
class RNGOp : public Expr {
  NVF_API int64_t getOutputDims() const;

 public:
  struct Attributes {
    // default initialization for clang-tidy
    // cppcoreguidelines-pro-type-member-init
    RNGOpType rtype = RNGOpType::Undefined;
    DataType dtype;
    size_t num_parameters = 0;

    // TODO: Enable the following in C++20:
    // bool operator==(const Attributes &other) const = default;
    bool operator==(const Attributes& other) const {
      // Note: we do not need to explicitly compare num_parameters since it is
      // tied to rtype
      return rtype == other.rtype && dtype == other.dtype;
    }
  };

  using Expr::Expr;

  //! Note that if philox_offset is provided, then rng_offset will be ignored.
  RNGOp(
      IrBuilderPasskey,
      RNGOpType type,
      Val* out,
      DataType dtype,
      std::vector<Val*> parameters = {},
      Val* philox_seed = nullptr,
      Val* philox_offset = nullptr,
      Val* philox_index = nullptr);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "RNGOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  RNGOpType getRNGOpType() const {
    return attribute<Attributes>(0).rtype;
  }

  DataType dtype() const {
    return attribute<Attributes>(0).dtype;
  }

  size_t getNumParameters() const {
    return attribute<Attributes>(0).num_parameters;
  }

  std::vector<Val*> getParameters() const {
    return {
        inputs().begin() + getOutputDims(),
        inputs().begin() + (int64_t)(getOutputDims() + getNumParameters())};
  }

  std::vector<Val*> getShape() const {
    return {inputs().begin(), inputs().begin() + getOutputDims()};
  }

  Val* getRNGSeedVal() const {
    // Note that inputs() consists of:
    // output dims | parameters | philox seed | philox_offset
    auto seed_index = getOutputDims() + getNumParameters();
    return (inputs().size() > seed_index) ? inputs().at(seed_index) : nullptr;
  }

  Val* getRNGOffsetVal() const {
    // Note that inputs() consists of:
    // output dims | parameters | philox seed | philox_offset
    auto offset_index = getOutputDims() + getNumParameters() + 1;
    return (inputs().size() > offset_index) ? inputs().at(offset_index)
                                            : nullptr;
  }

  bool isDeterministic() const override {
    return inputs().size() == getOutputDims() + getNumParameters() + 2;
  }

  void setSeedAndOffset(Val* seed, Val* offset) {
    NVF_ERROR(!isDeterministic());
    addInput(seed);
    addInput(offset);
  }

  Val* getPhiloxIndex() const {
    return attributeVal(1);
  }

  int getPhiloxMultiple() const {
    return dtype() == DataType::Double ? 2 : 4;
  }
};

//! Broadcast in to match out. The semantics are identical to torch.unsqueeze.
//! is_broadcast_dims are relative to out. Where
//! is_broadcast_dims.size() == out->nDims().
class NVF_API BroadcastOp : public Expr {
 public:
  using Expr::Expr;

  //! \param out The output tensor
  //! \param in The input tensor
  //! \param is_broadcast_dims True when output dim is a new broadcast domain
  BroadcastOp(
      IrBuilderPasskey,
      Val* out,
      Val* in,
      std::vector<bool> is_broadcast_dims);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "BroadcastOp";
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

  bool isBroadcastDim(size_t dim) const {
    return getBroadcastDimFlags().at(dim);
  }

  //! The same list passed to the broadcast arithmetic op. Each
  //! element corresponds to an IterDomain of the output tensor and is
  //! true when the IterDomain is a new broadcast domain. Note
  //! that the output tensor may have other broadcast domains whose
  //! flags are false because the input tensor may already have
  //! broadcast domains.
  const std::vector<bool>& getBroadcastDimFlags() const {
    return attribute<std::vector<bool>>(0);
  }
};

//! Squeeze in to match out. is_squeeze_dims are relative to in. Where
//! is_squeeze_dims.size() == in->nDims(). Squeeze is the opposite of
//! broadcast.
class NVF_API SqueezeOp : public Expr {
 public:
  using Expr::Expr;

  //! \param out The output tensor
  //! \param in The input tensor
  //! \param is_squeeze_dims True when input dim is a removed broadcast domain
  SqueezeOp(
      IrBuilderPasskey,
      Val* out,
      Val* in,
      std::vector<bool> is_broadcast_dims);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "SqueezeOp";
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

  bool isSqueezeDim(size_t dim) const {
    return getSqueezeDimFlags().at(dim);
  }

  //! The same list passed to the squeeze arithmetic op. Each
  //! element corresponds to an IterDomain of the input tensor and is
  //! true when the IterDomain is a broadcast domain that is removed in the
  //! output. Note that the output tensor may still contain broadcast domains
  //! because the input tensor may have broadcast domains that we don't want to
  //! remove (false flag).
  const std::vector<bool>& getSqueezeDimFlags() const {
    return attribute<std::vector<bool>>(0);
  }

  //! Check that squeezed IDs in old_tv concretize to Broadcast IterType
  void checkConcretization(Val* old_tv, Val* new_tv) const override;
};

//! Reduction operation. Out is first initialized to _init. Then
//! reduction_op_type is used to update out as out = reductionOp(out, in).
//! Output's axes marked as reduction will be reduced to produce an output
//! tensor. The output tensors size will be the size of all
//! non-reduction/non-broadcast dimensions.
class ReductionOp : public Expr {
 public:
  using Expr::Expr;

  ReductionOp(
      IrBuilderPasskey,
      BinaryOpType reduction_op_type,
      Val* init,
      Val* out,
      Val* in,
      bool is_allreduce = false);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "ReductionOp";
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
  Val* init() const {
    return attributeVal(0);
  }

  BinaryOpType getReductionOpType() const {
    return attribute<BinaryOpType>(1);
  }

  bool isAllreduce() const {
    return attribute<bool>(2);
  }

  //! Scheduling method to request that this reduction be performed as a
  //! serial grid reduction. Note that it is an error to use this method on a
  //! reduction whose output has any of its reduction axes parallelized with a
  //! threadIdx, even if that parallelization occurs after this method call.
  //!
  //! Also note that this operation should not be inlined with other reductions
  //! unless they use the same parallelization pattern and they are also serial
  //! gridreductions.
  void requestSerialGridReduction(bool value = true) {
    attribute<bool>(3) = value;
  }

  bool serialGridReductionRequested() const {
    return attribute<bool>(3);
  }
};

//! Grouped reduction operation for horizontal fusions. It works like
//! batched GEMMs in the sense that multiple independent reductions are
//! performed together. The main benefit is when reducing tensors across thread
//! blocks, a single grid sync can be done for all individual
//! reductions. As grid sync is very expensive, this can be a
//! significant performance impact.
class GroupedReductionOp : public Expr {
 public:
  using Expr::Expr;

  GroupedReductionOp(
      IrBuilderPasskey,
      std::vector<BinaryOpType> reduction_op_types,
      std::vector<Val*> init,
      std::vector<Val*> out,
      std::vector<Val*> in,
      bool is_allreduce = false);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "GroupedReductionOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;

  //! Number of expressions grouped horizontally. It does not reflect
  //! iteration grouping.
  size_t numHorizontallyGroupedExprs() const {
    return getReductionOpTypes().size();
  }

  std::vector<Val*> initVals() const {
    auto size = numHorizontallyGroupedExprs();
    std::vector<Val*> result;
    result.reserve(size);
    for (auto i : arange(2, 2 + size)) {
      result.emplace_back(attribute(i)->as<Val>());
    }
    return result;
  }

  Val* initVal(size_t index) const {
    return attributeVal(2 + index);
  }

  const std::vector<BinaryOpType>& getReductionOpTypes() const {
    return attribute<std::vector<BinaryOpType>>(0);
  }

  BinaryOpType getReductionOpType(size_t index) const {
    return getReductionOpTypes().at(index);
  }

  bool isAllreduce() const {
    return attribute<bool>(1);
  }

  //! Return the index of the corresponding reduction expression for
  //! a given output val.
  int getExprIndexOfOutput(Val* output_val) const;
};

//! Average, variance and N (count) vals for Welford
class WelfordTriplet {
 public:
  //! Names of the Welford triplet vals
  enum class ValName { Avg, Var, N };

  WelfordTriplet() = default;

  WelfordTriplet(Val* avg, Val* var, Val* N) : vals_({avg, var, N}) {}

  Val* const& avg() const {
    return get(ValName::Avg);
  }

  Val*& avg() {
    return get(ValName::Avg);
  }

  TensorView* avgTv() const {
    NVF_ERROR(avg()->isA<TensorView>());
    return avg()->as<TensorView>();
  }

  Val* const& var() const {
    return get(ValName::Var);
  }

  Val*& var() {
    return get(ValName::Var);
  }

  TensorView* varTv() const {
    NVF_ERROR(var()->isA<TensorView>());
    return var()->as<TensorView>();
  }

  Val* const& N() const {
    return get(ValName::N);
  }

  Val*& N() {
    return get(ValName::N);
  }

  TensorView* NTv() const {
    NVF_ERROR(N()->isA<TensorView>());
    return N()->as<TensorView>();
  }

  //! Get the i-th val. Ordering is defined by ValName.
  Val* const& get(int i) const {
    return vals_.at(i);
  }

  //! Get the i-th val. Ordering is defined by ValName.
  Val*& get(int i) {
    return vals_.at(i);
  }

  Val* const& get(ValName name) const {
    return get(valNameToIndex(name));
  }

  Val*& get(ValName name) {
    return get(valNameToIndex(name));
  }

  //! Get the name of a given val in this triplet. None is returned if
  //! not found.
  std::optional<ValName> getNameOf(Val* val) const;

  //! Return a new triplet with outputs produced by a function applied
  //! to each of this triplet
  template <typename Func>
  WelfordTriplet transform(Func func) const {
    return WelfordTriplet(func(avg()), func(var()), func(N()));
  }

  bool sameAs(const WelfordTriplet& other) const;

  WelfordTriplet clone(IrCloner* ir_cloner) const;

  //! Clone a vector of triplets
  static std::vector<WelfordTriplet> clone(
      const std::vector<WelfordTriplet>& src,
      IrCloner* ir_cloner);

  auto begin() {
    return vals_.begin();
  }

  auto begin() const {
    return vals_.begin();
  }

  auto end() {
    return vals_.end();
  }

  auto end() const {
    return vals_.end();
  }

 private:
  //! Convert a given val name to an index
  static int valNameToIndex(ValName name) {
    return static_cast<int>(name);
  }

  //! Convert a given index to a name
  static ValName indexToValName(int index) {
    NVF_ERROR(index >= 0 && index < 3, "Invalid index: ", index);
    return static_cast<ValName>(index);
  }

 private:
  //! Holds avg, var and N in this order
  std::array<Val*, 3> vals_ = {{nullptr, nullptr, nullptr}};
};

//! Welford Scan operation.
class WelfordOp : public Expr {
 public:
  using Expr::Expr;
  static constexpr int kNumAttrs = 4;

  WelfordOp(
      IrBuilderPasskey,
      const WelfordTriplet& output,
      const WelfordTriplet& input,
      const WelfordTriplet& init,
      bool is_fused = false);

  WelfordOp(
      IrBuilderPasskey,
      Val* out_avg,
      Val* out_var,
      Val* out_N,
      Val* in_avg,
      Val* in_var,
      Val* in_N,
      Val* init_avg,
      Val* init_var,
      Val* init_N,
      bool is_fused = false);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "WelfordOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;

  Val* out() const {
    return outputTriplet().avg();
  }

  Val* in() const {
    return inputTriplet().avg();
  }

  WelfordTriplet outputTriplet() const {
    return WelfordTriplet(outAvg(), outVar(), outN());
  }

  Val* outAvg() const {
    return output(0);
  }

  Val* outVar() const {
    return output(1);
  }

  Val* outN() const {
    return output(2);
  }

  WelfordTriplet inputTriplet() const {
    return WelfordTriplet(inAvg(), inVar(), inN());
  }

  Val* inAvg() const {
    return input(0);
  }

  Val* inVar() const {
    return input(1);
  }

  Val* inN() const {
    return input(2);
  }

  WelfordTriplet initTriplet() const {
    return WelfordTriplet(initAvg(), initVar(), initN());
  }

  Val* initAvg() const {
    return attributeVal(0);
  }

  Val* initVar() const {
    return attributeVal(1);
  }

  Val* initN() const {
    return attributeVal(2);
  }

  bool singleValue() const {
    return inN()->isOneInt();
  }

  bool hasInit() const {
    return !initN()->isZeroInt();
  }

  //! True if using the fused reduction kernel (not implemented yet)
  bool isAllreduce() const {
    return attribute<bool>(3);
  }

  std::vector<Val*> getInitVals() const;

  //! Return the init val for an output val
  Val* getInitValOfOutput(Val* output_val) const;
};

class GroupedWelfordOp : public Expr {
 public:
  using Expr::Expr;

  GroupedWelfordOp(
      IrBuilderPasskey,
      std::vector<WelfordTriplet> output_vals,
      std::vector<WelfordTriplet> input_vals,
      std::vector<WelfordTriplet> init_vals,
      bool is_allreduce = false);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "GroupedWelfordOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  //! Number of expressions grouped horizontally. It does not reflect
  //! iteration grouping. As horizontal grouping is not supported,
  //! this always returns 1.
  size_t numHorizontallyGroupedExprs() const {
    return 1;
  }

  Val* out(size_t index) const {
    return outAvg(index);
  }

  Val* in(size_t index) const {
    return inAvg(index);
  }

  std::vector<WelfordTriplet> outputVals() const {
    std::vector<WelfordTriplet> result;
    auto size = outputs().size() / 3;
    result.reserve(size);
    for (auto i : arange(size)) {
      result.emplace_back(outAvg(i), outVar(i), outN(i));
    }
    return result;
  }

  std::vector<WelfordTriplet> inputVals() const {
    std::vector<WelfordTriplet> result;
    auto size = inputs().size() / 3;
    result.reserve(size);
    for (auto i : arange(size)) {
      result.emplace_back(inAvg(i), inVar(i), inN(i));
    }
    return result;
  }

  std::vector<WelfordTriplet> initVals() const {
    std::vector<WelfordTriplet> result;
    auto size = inputs().size() / 3;
    result.reserve(size);
    for (auto i : arange(size)) {
      result.emplace_back(initAvg(i), initVar(i), initN(i));
    }
    return result;
  }

  Val* outAvg(size_t index) const {
    return output(index * 3);
  }

  Val* outVar(size_t index) const {
    return output(index * 3 + 1);
  }

  Val* outN(size_t index) const {
    return output(index * 3 + 2);
  }

  Val* inAvg(size_t index) const {
    return input(index * 3);
  }

  Val* inVar(size_t index) const {
    return input(index * 3 + 1);
  }

  Val* inN(size_t index) const {
    return input(index * 3 + 2);
  }

  Val* initAvg(size_t index) const {
    return attributeVal(1 + index * 3);
  }

  Val* initVar(size_t index) const {
    return attributeVal(2 + index * 3);
  }

  Val* initN(size_t index) const {
    return attributeVal(3 + index * 3);
  }

  //! Return the index of the corresponding welford expression for
  //! a given output val
  int getExprIndexOfOutput(Val* output_val) const;

  //! Return the init val for an output val
  Val* getInitValOfOutput(Val* output_val) const;

  bool singleValue(size_t index) const {
    return inN(index)->isOneInt();
  }

  bool hasInit(size_t index) const {
    return !initN(index)->isZeroInt();
  }

  bool isAllreduce() const {
    return attribute<bool>(0);
  }
};

//! Fused Matmul operation
class MmaOp : public Expr {
 public:
  using AxesData = std::vector<int64_t>;
  using Expr::Expr;

  MmaOp(IrBuilderPasskey, Val* out, Val* in_a, Val* in_b, Val* init);

  MmaOp(
      IrBuilderPasskey,
      Val* out,
      Val* in_a,
      Val* in_b,
      Val* init,
      const MmaMacro& options);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "MmaOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Val* out() const {
    return output(0);
  }

  Val* inA() const {
    return input(0);
  }

  Val* inB() const {
    return input(1);
  }

  Val* init() const {
    return attributeVal(0);
  }

  const auto& macro() const {
    return attribute<MmaMacro>(ATTR_POS_MACRO);
  }

  int64_t m() const {
    return getM(macro());
  }

  int64_t n() const {
    return getN(macro());
  }

  int64_t k() const {
    return getK(macro());
  }

  bool isTuring() const {
    return nvfuser::isTuring(macro());
  }

  bool isAmpere() const {
    return nvfuser::isAmpere(macro());
  }

  bool isHopper() const {
    return nvfuser::isHopper(macro());
  }

  bool isBlackwell1CTA() const {
    return nvfuser::isBlackwell1CTA(macro());
  }

  bool isBlackwell2CTA() const {
    return nvfuser::isBlackwell2CTA(macro());
  }

  bool isBlackwell() const {
    return nvfuser::isBlackwell(macro());
  }

  void setMacro(MmaMacro options);

 private:
  // Predefined indices of attributes stored for this IR node, to avoid
  //  magic numbers, based on order in which attributes are initialized
  //  in constructor
  static constexpr size_t ATTR_POS_INIT = 0;
  static constexpr size_t ATTR_POS_MACRO = 1;
};

//! The semantics are identical to torch.broadcast_to.
class NVF_API ExpandOp : public Expr {
 public:
  using Expr::Expr;

  ExpandOp(
      IrBuilderPasskey,
      TensorView* out,
      TensorView* in,
      const std::vector<Val*>& expanded_extents);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "ExpandOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  TensorView* out() const {
    return output(0)->as<TensorView>();
  }

  TensorView* in() const {
    return input(0)->as<TensorView>();
  }

  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;
};

// Represents a repetition of broadcast IDs. Repetitions of
// non-broadcast IDs are represented using the broadcast, expand and
// reshape pattern. See the repeat op implementation in ops/alias.cpp
// as well as the TranslateRepeatToExpand preseg pass.
class RepeatOp : public Expr {
 public:
  using Expr::Expr;

  // in: Input tensor that have broadcast logical IDs.
  // out: Output tensor where some of the input broadcast logical IDs
  // are converted to concrete IDs. Their extents represent the
  // repetition factor of each ID.
  RepeatOp(IrBuilderPasskey, TensorView* out, TensorView* in);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "RepeatOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  TensorView* out() const {
    return output(0)->as<TensorView>();
  }

  TensorView* in() const {
    return input(0)->as<TensorView>();
  }

  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;
};

class ViewAsScalar : public Expr {
 public:
  using Expr::Expr;

  ViewAsScalar(IrBuilderPasskey, Val* out, Val* in, IterDomain* vector_id);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "ViewAsScalar";
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

  // The IterDomain of type VectorComponent newly appended to the output
  IterDomain* vector_id() const {
    return attribute(0)->as<IterDomain>();
  }
};

class NVF_API ReshapeOp : public Expr {
 public:
  using Expr::Expr;

  ReshapeOp(IrBuilderPasskey, Val* out, Val* in);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "ReshapeOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  TensorView* out() const {
    return output(0)->as<TensorView>();
  }

  TensorView* in() const {
    return input(0)->as<TensorView>();
  }

  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;
};

//! This operator explicitly models data movement between
//!   state spaces on GPU. Currently the modeled state spaces include
//!   global memory, shared memory and register.
//!
//! The main usage of this op is to facilitate generation of hardware
//!   accelerated memory ops, i.e. ldmatrix, cp.async and more to come.
class LoadStoreOp : public Expr {
 public:
  using Expr::Expr;

  LoadStoreOp(
      IrBuilderPasskey,
      LoadStoreOpType op_type,
      Val* out,
      Val* in,
      CacheOp cache_op = CacheOp::Unspecified);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "LoadStoreOp";
  }

  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Val* out() const {
    return output(0);
  }

  Val* in() const {
    return input(0);
  }

  LoadStoreOpType opType() const {
    return attribute<LoadStoreOpType>(0);
  }

  CacheOp cacheOp() const {
    return attribute<CacheOp>(1);
  }

  void setOpType(LoadStoreOpType op) {
    attribute<LoadStoreOpType>(0) = op;
    if (op != LoadStoreOpType::Set && op != LoadStoreOpType::CpAsync) {
      attribute<CacheOp>(1) = CacheOp::Unspecified;
    }
  }

  void setCacheOp(CacheOp cache_op) {
    attribute<CacheOp>(1) = cache_op;
  }
};

//! Representation a split on an IterDomain by "factor"
//! inner_split dictates if the factor section of the split should be inside the
//! remainer or outside.
class NVF_API Split : public Expr {
 public:
  using Expr::Expr;

  Split(
      IrBuilderPasskey,
      IterDomain* outer,
      IterDomain* inner,
      IterDomain* in,
      Val* factor,
      bool inner_split = true);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "Split";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  IterDomain* outer() const {
    return output(0)->as<IterDomain>();
  }
  IterDomain* inner() const {
    return output(1)->as<IterDomain>();
  }
  IterDomain* in() const {
    return input(0)->as<IterDomain>();
  }
  Val* factor() const {
    return attributeVal(0);
  }
  Val* isDivisible() const;

  bool innerSplit() const {
    return attribute<bool>(1);
  }
};

//! Merge the IterDomains outer and inner into one domain, outer and inner
//! dictate which will be traversed first (inner). Both IterDomains must be of
//! the same iter or reduction type, as well as the same parallelization
//! strategy if there is one
class NVF_API Merge : public Expr {
 public:
  using Expr::Expr;

  Merge(
      IrBuilderPasskey,
      IterDomain* out,
      IterDomain* outer,
      IterDomain* inner);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "Merge";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  IterDomain* out() const {
    return output(0)->as<IterDomain>();
  }
  IterDomain* outer() const {
    return input(0)->as<IterDomain>();
  }
  IterDomain* inner() const {
    return input(1)->as<IterDomain>();
  }
};

//! Partition an IterDomain into component and ragged dimensions
//! Creates a component IterDomain and a RaggedIterDomain based on extents
//! tensor The extents tensor contains the extent for each component
class NVF_API Partition : public Expr {
 public:
  using Expr::Expr;

  Partition(
      IrBuilderPasskey,
      IterDomain* component,
      RaggedIterDomain* ragged,
      IterDomain* in,
      TensorView* extents);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "Partition";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  //! Component dimension output (extent = num_components)
  IterDomain* component() const {
    return output(0)->as<IterDomain>();
  }

  //! Ragged dimension output (variable extents per component)
  RaggedIterDomain* ragged() const {
    return output(1)->as<RaggedIterDomain>();
  }

  //! Input IterDomain being partitioned
  IterDomain* in() const {
    return input(0)->as<IterDomain>();
  }

  //! Extents tensor containing extent for each component
  TensorView* extents() const {
    return attributeVal(0)->as<TensorView>();
  }
};

class Swizzle : public Expr {
 public:
  using Expr::Expr;

  Swizzle(
      IrBuilderPasskey,
      IterDomain* out_x,
      IterDomain* out_y,
      IterDomain* in_x,
      IterDomain* in_y,
      SwizzleType swizzle_type = SwizzleType::NoSwizzle);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "Swizzle";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  // Output iterdomain pair corresponding
  //  to the original input iterdomain pair.
  IterDomain* outX() const {
    return output(0)->as<IterDomain>();
  }

  IterDomain* outY() const {
    return output(1)->as<IterDomain>();
  }

  // Input iterdomain pair.
  IterDomain* inX() const {
    return input(0)->as<IterDomain>();
  }

  IterDomain* inY() const {
    return input(1)->as<IterDomain>();
  }

  // The type of predefined 1-to-1 functions
  //  used for swizzling math.
  auto swizzleType() const {
    return attribute<SwizzleType>(0);
  }
};

//! Applies 2D swizzles on a rectangular tile defined by 2 iterdomains.
class Swizzle2D : public Expr {
 public:
  using Expr::Expr;

  Swizzle2D(
      IrBuilderPasskey,
      IterDomain* out_x,
      IterDomain* out_y,
      IterDomain* in_x,
      IterDomain* in_y,
      Swizzle2DType swizzle_type = Swizzle2DType::NoSwizzle,
      SwizzleMode swizzle_mode = SwizzleMode::Data);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "Swizzle2D";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  // Output iterdomain pair corresponding
  //  to the original input iterdomain pair.
  IterDomain* outX() const {
    return output(0)->as<IterDomain>();
  }

  IterDomain* outY() const {
    return output(1)->as<IterDomain>();
  }

  // Input iterdomain pair.
  IterDomain* inX() const {
    return input(0)->as<IterDomain>();
  }

  IterDomain* inY() const {
    return input(1)->as<IterDomain>();
  }

  // The type of predefined 1-to-1 functions
  //  used for swizzling math.
  auto swizzleType() const {
    return attribute<Swizzle2DType>(0);
  }

  // Swizzle mode of this swizzle instance.
  // [Note on swizzle mode]
  // On the current implementations we support two modes of
  //  swizzle math, namely, data mode and loop mode.
  // `Data` mode swizzling is a swizzle that will change the
  //  data layout in shared memory, likely in global memory buffers
  //  as well in the future. see also IndexSwizzle in index_compute.cpp.
  //
  //  Most important use cases are transpose bank conflict removal, and mma
  //  swizzled shared memory layout. Example illustrated in 1D case:
  //
  // for (int i = 0; i<I; i++){
  //   # This is a `Data` mode swizzle.
  //  Tshared [swizzled(i)] = Tin[i];
  // }
  // # Now Tshared holds swizzled data, i.e. the data layout of
  //    Tshared does not map to Tin with affine relationships.
  //
  // for(int i=0;i<I;i++){
  //   Tout = Tshared[swizzled(i)];
  // }
  //
  // `Loop` mode swizzling does not affect the data layout of any buffer
  //   but only permutes the iteration order of serial or parallel loop.
  // This is useful when we want to designate non-affine mapping of thread
  //   to data or we want to generate non-affine loops.
  // Exampe illustrated in 1D case:
  //   for (int i = 0; i<I; i++){
  //     # This is a `Loop` mode swizzle
  //    Tshared [swizzled(i)] = Tin[swizzled(i)];
  //   }
  // # Now Tshared holds normal data, i.e. it still has
  //   the same data layout as if the swizzle wasn't there.
  //
  // # Consumers of Tshared does not need to know about the
  //   loop swizzle at previous op if not inlined.
  // for(int i=0;i<I;i++){
  //   Tout = Tshared[i];
  // }
  //  TODO: Loop swizzles eventually will be piped through in all mappings
  //  and replay of the fusion IR infrastructure.
  auto swizzleMode() const {
    return attribute<SwizzleMode>(1);
  }
};

//! IterDomain expression to resize
class Resize : public Expr {
 public:
  using Expr::Expr;

  // Expand the input domain by left_expand and right_expand for each
  // of the start and end sides, respectively
  Resize(
      IrBuilderPasskey,
      IterDomain* out,
      IterDomain* in,
      Val* left_expand,
      Val* right_expand);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "Resize";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  IterDomain* out() const {
    return output(0)->as<IterDomain>();
  }

  IterDomain* in() const {
    return input(0)->as<IterDomain>();
  }

  Val* leftExpand() const {
    return attributeVal(0);
  }

  Val* rightExpand() const {
    return attributeVal(1);
  }
};

//! Integer value which has a special name
//!
//! These could be:
//! - threadIdx.x
//! - blockIdx.y
//! - blockDim.z
//! - T3.stride[2]
//!
class NamedScalar : public Val {
 public:
  NamedScalar(IrBuilderPasskey passkey, std::string name, DataType dtype);

  NamedScalar(const NamedScalar* src, IrCloner* ir_cloner);

  NVFUSER_DECLARE_CLONE

  const std::string& name() const {
    return name_;
  }

  bool sameAs(const Statement* other) const override;

  std::string toString(int indent_size = 0) const override {
    return name_;
  }

  std::string toInlineString(int indent_size = 0) const override {
    return name_;
  }

  //! Check if this is threadIdx.{x,y,z}
  bool isThreadIdx() const {
    auto p = getParallelIndex();
    return (
        p == ParallelType::TIDx || p == ParallelType::TIDy ||
        p == ParallelType::TIDz);
  }

  //! Check if this is blockIdx.{x,y,z}
  bool isBlockIdx() const {
    auto p = getParallelIndex();
    return (
        p == ParallelType::BIDx || p == ParallelType::BIDy ||
        p == ParallelType::BIDz);
  }

  //! Check if this is blockDim.{x,y,z}
  bool isBlockDim() const {
    auto p = getParallelDim();
    return (
        p == ParallelType::TIDx || p == ParallelType::TIDy ||
        p == ParallelType::TIDz);
  }

  //! Check if this is gridDim.{x,y,z}
  bool isGridDim() const {
    auto p = getParallelDim();
    return (
        p == ParallelType::BIDx || p == ParallelType::BIDy ||
        p == ParallelType::BIDz);
  }

  //! Return the named scalar extent of a parallel dimension (e.g. blockDim.x)
  //! WARNING: Only works with Fusion container at the moment
  static NamedScalar* getParallelDim(ParallelType p_type);

  //! Return the named scalar index of a parallel dimension (e.g. threadIdx.x)
  //! WARNING: Only works with Fusion container at the moment
  static NamedScalar* getParallelIndex(ParallelType p_type);

  //! Return the parallel type of this NamedScalar if it is an extent of a
  //! parallel dimension
  std::optional<ParallelType> getParallelDim() const;

  //! Return the parallel type of this NamedScalar if it is an index of a
  //! parallel dimension
  std::optional<ParallelType> getParallelIndex() const;

 private:
  std::string name_;
};

class PadOp : public Expr {
 public:
  using Expr::Expr;

  //! Pad a tensor as specified by a vector of integer scalars. For
  //! the actual semantics, see the torch.pad documentation. Note that
  //! unlike torch.pad, the pad_widths vector parameter must contain
  //! width vals for all dimensions. For non-padded dimensions, width
  //! vals should be integer zero.
  PadOp(
      IrBuilderPasskey passkey,
      TensorView* out,
      TensorView* inp,
      const std::vector<Val*>& pad_widths,
      Val* value);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "PadOp";
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

  Val* value() const {
    return input(1);
  }

  //! Return axes that are actually paded, i.e., those that have
  //! non-zero pad widths
  std::vector<int64_t> getPaddedAxes() const;

  //! Return pad widths of the given axis, which are just zero for non padded
  //! dimensions
  std::pair<Val*, Val*> getPadWidths(int64_t axis) const;

  //! Return the pad widths of all dimensions, including non-padded ones
  NVF_API std::vector<Val*> getPadWidths() const;

 private:
  //! Offset of pad_width inputs in the input vector
  int64_t getPadWidthInputOffset() const {
    return 2;
  }

  //! Iterator to the first pad_width input
  auto getPadWidthInputBegin() const {
    return inputs().cbegin() + getPadWidthInputOffset();
  }

  //! Iterator to the end of the pad_width inputs
  auto getPadWidthInputEnd() const {
    return inputs().cend();
  }
};

// Similar to at::indexing::Slice
struct Slice {
  Val* start = nullptr;
  Val* stop = nullptr;
  Val* step = nullptr;

  std::string toString() const;
};

class SliceOp : public Expr {
 public:
  using Expr::Expr;

  SliceOp(
      IrBuilderPasskey passkey,
      TensorView* out,
      TensorView* inp,
      const std::vector<Slice>& ranges);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "SliceOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;

  TensorView* out() const {
    return output(0)->as<TensorView>();
  }

  TensorView* in() const {
    return input(0)->as<TensorView>();
  }

  //! Get normalized ranges for SliceOp.
  NVF_API std::vector<Slice> getRanges() const;

 private:
  //! Offset of ranges input in the input vector
  int getRangeInputOffset() const {
    return 1;
  }

  //! Iterator to the first range inputs
  auto getRangeInputBegin() const {
    return inputs().cbegin() + getRangeInputOffset();
  }

  //! Iterator to the end of the range inputs
  auto getRangeInputEnd() const {
    return inputs().cend();
  }
};

class CatOp : public Expr {
 public:
  using Expr::Expr;

  CatOp(
      IrBuilderPasskey passkey,
      Val* out,
      const std::vector<Val*>& inputs,
      int64_t concatenated_dim);

  //! Create a cat op with the index and predicates for codegen. Only
  //! used for the Kernel container
  CatOp(
      IrBuilderPasskey passkey,
      Val* out,
      const std::vector<Val*>& inputs,
      int64_t concatenated_dim,
      Val* concatenated_domain_index,
      const std::vector<Val*>& preds);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "CatOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      std::unordered_map<const Val*, PolymorphicValue>& known_values)
      const override;

  int64_t concatenatedDim() const {
    return attribute<int64_t>(0);
  }

  //! The index val that determines which input tensor should be used
  //! to fill the particular output position of this expression. Only
  //! valid after indexing
  Val* getConcatenatedDomainIndex() const;

  //! Gets a Bool indicating if the input tensor specified by
  //! tensor_idx should be used to fill the output tensor. Only valid
  //! with the Kernel container
  Val* getPred(int input_idx) const;
};

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
      Val* query,
      Val* key,
      Val* value,
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

  Val* dropout_p() const {
    return input(3);
  }

  Val* is_causal() const {
    return input(4);
  }

  Val* scale() const {
    if (inputs().size() > 5) {
      return input(5);
    }
    return nullptr;
  }

  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;
};

class Scope {
 public:
  using ExprList = std::list<Expr*>;
  using Iterator = ExprList::const_iterator;

  explicit Scope(Expr* owner) : owner_(owner) {}

  std::string toString(int indent_size = 0) const;

  const ExprList& exprs() const {
    return exprs_;
  }

  // Used only by MultiDeviceExecutor. Should generally be avoided in favor of
  // other modifying methods.
  ExprList& mutableExprs() {
    return exprs_;
  }

  Expr* front() const {
    NVF_ERROR(
        !exprs_.empty(), "Attempting to access the front of an empty Scope");
    return exprs_.front();
  }

  Expr* back() const {
    NVF_ERROR(
        !exprs_.empty(), "Attempting to access the back of an empty Scope");
    return exprs_.back();
  }

  bool empty() const {
    return exprs_.empty();
  }

  int64_t size() const {
    return std::ssize(exprs_);
  }

  Iterator insert(Iterator pos, Expr* expr);

  Iterator push_back(Expr* e) {
    return insert(exprs_.end(), e);
  }

  void clear();

  Expr* owner() const {
    return owner_;
  }

  // The following methods perform linear searches over exprs_. Use them only
  // when necessary, as they do not scale well with large scopes.
  Iterator insert_before(Expr* ref, Expr* expr);
  Iterator insert_after(Expr* ref, Expr* expr);
  void erase(Expr* ref);
  bool contains(Expr* expr) const;

 private:
  void erase(Iterator pos);

  ExprList exprs_;

  //! Owner exprssion of this scope, e.g., IfThenElse
  Expr* owner_ = nullptr;
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

//! NOTE -- [ Grouped Matrix Multiplication semantics ]
//!
//! This operation performs a grouped matrix multiplication.
//!
//! Parameters:
//! - out_mat: output tensor
//! - out_scale: output block scaling factor tensor
//! - out_gamma: output global scaling factor tensor
//! - matrix1: first input tensor matrix
//! - matrix2: second input tensor matrix
//! - offsets: 1D offsets tensor, specifying the ending index of each group
//! - scale1: scale tensor for matrix1 (optional)
//! - scale2: scale tensor for matrix2 (optional)
//! - alpha: alpha tensor (optional)
//! - bias: bias tensor (optional)
//! - beta: beta tensor (optional)
//!
//! The math operation is roughly two steps:
//! out =
//!   alpha * grouped_mm(dequant(mat1, scale1), dequant(mat2, scale2), offsets)
//!   + beta * bias
//! out_mat, out_scale, out_gamma = Quantization(out)
//!
//! Post quantization only applies when out_scale / out_gamma is not nullptr;
//!
//! Regarding the grouped mm semantics:
//!
//! The offsets tensor is a vector tensor of length `num_groups` that specifies
//! the ending index of each group in the matrix1 and matrix2 tensors.
//!
//! Given the number of groups as G, the operation conceptually runs G matmuls.
//! There are three configurations of grouping, reflected by ranks of input
//! matrices:
//!
//! Note 0: f(0) = 0 : offset[0]
//!         f(i) = offset(i-1) : offset(i), when i >= 1;
//!         f(i) is a slice with length equal to offsets[i] - offsets[i-1]
//! Note 1: scales don't need to follow broadcast rules against corresponding
//!         matrices on the k-dimension. Hardware uses blocked scale factors.
//!         so the corresponding scale factor on k-dimension is shared by fixed
//!         number of consecutive elements.
//!         e.g. Given k as the size of k-dimension on input matrices, and the
//!         scale factor has size k' on k-dimension.
//!         For mxfp8, k' = k // 32. Each scale factor is shared by 32
//!         consecutive elements.
//! Note 2: output could have a reduction axis rk if k is not broadcast on
//! inputs.
//!
//!     Case 1: grouped k-dimension:
//!       inputs: mat1[ m, k ] @ mat2[ k, n ] , offsets[ g ]
//!               scale1[ g, m, k' ], scale2[ g, k', n]
//!       requires: offsets[g-1] == k
//!       output: out[ g, m, n, [rk]]
//!
//!       math:
//!       for i in range(g):
//!         out[ i, 0:m, 0:n ] = (mat1[ 0:m, f(i) ] * scale1[ i, 0:m, 0:k' ])
//!                             @(mat2[ f(i), 0:n ] * scale2[ i, 0:k', 0:n ])
//!
//!     Case 2: grouped m-dimension:
//!       inputs: mat1[ m, k ] @ mat2[ g, k, n ] , offsets[ g ]
//!               scale1[ m, k' ], scale2[ g, k', n ]
//!       requires: offsets[g-1] == m
//!       output: out[ m, n, [rk]]
//!
//!       math:
//!       for i in range(g):
//!         out[ f(i), 0:n ] = (mat1[ f(i), 0:k ] * scale1[ f(i), 0:k' ])
//!                           @(mat2[ i, 0:k, 0:n ] * scale2[ i, 0:k', 0:n ])
//!
//!     Case 3: grouped n-dimension:
//!       inputs: mat1[ g, m, k ] @ mat2[ k, n ] , offsets[ g ]
//!               scale1[ g, m, k' ], scale2[ k', n ]
//!       requires: offsets[g-1] == n
//!       output: out[ m, n, [rk]]
//!
//!       math:
//!       for i in range(g):
//!         out[ 0:m, f(i) ] = (mat1[ i, 0:m, 0:k ] * scale1[ i, 0:m, 0:k' ])
//!                           @(mat2[ 0:k, f(i) ] * scale2[ 0:k', f(i) ])
//!
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

//! NOTE -- [ Scaled Matrix Multiplication semantics ]
//!
//! This operation performs a matrix multiplication.
//!
//! Parameters:
//! - out_mat: [output] tensor matrix
//! - out_scale: [output] block scaling factor tensor
//! - out_gamma: [output] global scaling factor tensor
//! - matrix1: first input tensor matrix
//! - matrix2: second input tensor matrix
//! - scale1: scale tensor for matrix1
//! - scale2: scale tensor for matrix2
//! - alpha: alpha tensor (optional)
//! - bias: bias tensor (optional)
//! - beta: beta tensor (optional)
//!
//! The math operation is conceptually:
//!   out =
//!       alpha * (dequant(mat1, scale1) @ dequant(mat2, scale2), offsets)
//!       + beta * bias
//!   out_mat, out_scale, out_gamma = Quantization(out)
//!
//! TODO: This operation is here to support non-codegen kernel. The block
//! semantics in scaling factor have implementation/hardware-specific padding
//! and alignment requirement. We haven't mapped it to out_scale/out_gamma yet.
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

//! TopK operation that finds the k largest or smallest elements
//! along a specified dimension.
//!
//! This operation returns two outputs:
//! - values: the k largest/smallest values along the specified dimension
//! - indices: the indices of those values in the original tensor
//!
//! Parameters:
//! - dim: dimension along which to find top-k elements
//! - largest: if true, return largest elements; if false, return smallest
//! - sorted: if true, return elements in sorted order
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

//! NOTE -- [ PreprocessGroupedMatmulInputSf ]
//!
//! This operation performs a layout change on the input, it's currently used
//! for block scaling factor accompanying narrow precision inputs.
//!
//! PreprocessGroupedMatmulInputSf(TensorView* output, TensorView* input, ...)
//!
//!   input:  logical domain:   (i0, i1)
//!   output: root domain:      (i0, i1)
//!           logical domain:   (i2, i3)
//!           loop domain:      (i0, i1)
//!
//! 1. This can be viewed as a point-wise operation, since output loop domain
//! matches the input logical domain.
//!
//! 2. Because of the potential padding/swizzle, the logical domain of the
//! output does not map to input. We don't rely on codegen for indexing, so we
//! don't care about mapping the logical/allocation of output to anything else.
//! Indexing will be done in runtime function, utilizing `input_offsets` and
//! `output_offsets`.
//!
//! 3. Output has a root domain that matches the logical domain of the input.
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

// This operations launches dependent grid in programmatic dependent launch.
class LaunchDependentGridOp : public Expr {
 public:
  using Expr::Expr;

  // The inputs are any computation expressions that need to finish before
  // launching any dependent grids. The output is a FusionIR-only TensorView
  // that becomes an input to any consumer expressions. The inputs and outputs
  // encode its position in the Fusion IR DAG.
  LaunchDependentGridOp(
      IrBuilderPasskey,
      Val* output,
      std::vector<Val*> inputs);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "pdl::launchDependentGrid";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;
};

// This operations waits for primary grid to finish in programmatic dependent
// launch.
class WaitForPriorGridOp : public Expr {
 public:
  using Expr::Expr;

  // The inputs are any global inputs generated by an upstream kernel that this
  // kernel must wait for. The output is a FusionIR-only TensorView that becomes
  // an input to any downstream computation expressions. The inputs and outputs
  // encode its position in the Fusion IR DAG.
  WaitForPriorGridOp(IrBuilderPasskey, Val* output, std::vector<Val*> inputs);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    return "pdl::waitForPriorGrid";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;
};

} // namespace nvfuser
