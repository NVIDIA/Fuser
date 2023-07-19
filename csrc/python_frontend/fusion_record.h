// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <c10/util/complex.h>
#include <debug.h>
#include <ir/interface_nodes.h>
#include <ops/all_ops.h>
#include <options.h>
#include <python_frontend/fusion_definition.h>
#include <python_frontend/fusion_state.h>
#include <serde/fusion_cache_generated.h>
#include <serde/utils.h>
#include <utils.h>

#include <algorithm>
#include <complex>
#include <variant>

namespace nvfuser::python_frontend {

//! RecordFunctor is the base class record for operations recorded by
//! the FusionState.  It is, in essence, a node in the graph with
//! input edges, args, and output edges where the stored
//! values are indices into the recorded state.
//!
//! The virtual functor operator is executed on a cache miss to build the
//! appropriate part of the nvFuser Fusion IR for a given record.
//!
//! The hash and equality operators are used to facilitate the hashing of
//! RecordFunctors in a hash map given those operators need to be
//! specified for custom objects.
//!
//! The print function is used to print the given Record as a statement
//! in a python formated function.

struct RecordFunctor {
  RecordFunctor(
      std::vector<State> _args,
      std::vector<State> _outputs,
      std::string _name,
      serde::RecordType _record_type)
      : args_(std::move(_args)),
        arg_names_(args_.size()),
        outputs_(std::move(_outputs)),
        name_(std::move(_name)),
        record_type_(_record_type) {}
  virtual ~RecordFunctor() = default;
  //! Allows for copying of Child Class objects with RecordFunctor pointers.
  virtual RecordFunctor* clone() = 0;

  //! The base class is placing the type, outputs, and args hashed as follows:
  //! | 63 - 56 | 55 - 48 | 47 ----------- 32 | 32 ------------------------  0 |
  //! | Type    | Outputs | Args              | Child Class Specified          |
  virtual size_t hash() const {
    size_t arg_hash = 0;
    for (auto arg : args_) {
      arg_hash ^= ((arg.index << 1) ^ static_cast<size_t>(arg.stype));
    }
    size_t output_hash = 0;
    for (auto output : outputs_) {
      output_hash ^= ((output.index << 1) ^ static_cast<size_t>(output.stype));
    }
    return ((static_cast<size_t>(record_type_) & 0xff) << 56) |
        ((output_hash & 0xff) << 48) | ((arg_hash & 0xffff) << 32);
  }

  //! The base virtual equality operator is defined so all child
  //! classes can utilize the check for the same args and outputs.
  virtual bool operator==(const RecordFunctor& other) const {
    auto result = (record_type_ == other.record_type_);
    result = result && (args_.size() == other.args_.size()) &&
        (outputs_.size() == other.outputs_.size());
    result = result && (arg_names_ == other.arg_names_);
    if (result) {
      for (size_t i = 0; i < args_.size(); ++i) {
        if ((args_[i].index != other.args_[i].index) ||
            (args_[i].stype != other.args_[i].stype)) {
          result = false;
          break;
        }
      }
    }
    if (result) {
      for (size_t i = 0; i < outputs_.size(); ++i) {
        if ((outputs_[i].index != other.outputs_[i].index) ||
            (outputs_[i].stype != other.outputs_[i].stype)) {
          result = false;
          break;
        }
      }
    }
    return result;
  }

  //! Abstraction for an operation to build this record's nvFuser Fusion IR
  //! piece if the recording has a cache miss.
  virtual void operator()(FusionState& fd) = 0;

  //! Abstraction for storing data specific to a record functor.
  virtual std::pair<serde::RecordData, flatbuffers::Offset<void>> recordData(
      flatbuffers::FlatBufferBuilder& builder) const {
    return {serde::RecordData_NONE, flatbuffers::Offset<void>()};
  }

  //! The base serialize function that handles args, outputs, name and
  //! recordType. Child recordFunctors should overload the recordData function
  //! if has supplementary attributes.
  virtual flatbuffers::Offset<serde::RecordFunctor> serialize(
      flatbuffers::FlatBufferBuilder& builder) const {
    // table RecordFunctor {
    //     args: [State];
    //     outputs: [State];
    //     name: string;
    //     type: RecordType;
    //     data: RecordData;
    // }

    std::vector<serde::State> fb_args;
    fb_args.reserve(args_.size());
    for (auto& it : args_) {
      fb_args.emplace_back(it.index, it.stype);
    }
    auto args_fb =
        builder.CreateVectorOfStructs(fb_args.data(), fb_args.size());

    std::vector<serde::State> fb_outputs;
    fb_outputs.reserve(outputs_.size());
    for (auto& it : outputs_) {
      fb_outputs.emplace_back(it.index, it.stype);
    }
    auto outputs_fb =
        builder.CreateVectorOfStructs(fb_outputs.data(), fb_outputs.size());

    auto&& [record_data_type, record_data] = recordData(builder);

    return serde::CreateRecordFunctor(
        builder,
        args_fb,
        outputs_fb,
        builder.CreateString(name_),
        recordType(),
        record_data_type,
        record_data);
  }

  //! The base print function when printing Record for a given FusionState
  //! in python formated code.
  virtual void print(std::ostream& os, bool close_function = true) const {
    bool first_output = true;
    for (auto& output : outputs_) {
      if (first_output) {
        first_output = false;
      } else {
        os << ", ";
      }
      os << output;
    }
    if (always_returns_tuple_) {
      os << ",";
    }
    if (!outputs_.empty()) {
      os << " = "
         << "fd." << name_ << "(";
    } else {
      os << "fd." << name_ << "(";
    }
    bool first_arg = true;
    size_t idx = 0;
    for (auto& arg : args_) {
      if (first_arg) {
        first_arg = false;
      } else {
        os << ", ";
      }
      if (!arg_names_[idx].empty()) {
        os << arg_names_[idx] << "=";
      }
      ++idx;
      os << arg;
    }
    if (close_function) {
      os << ")";
    }
  }

  size_t numOutputs() const {
    return outputs_.size();
  }

  serde::RecordType recordType() const {
    return record_type_;
  }

  //! Set the name of an argument. If given, it will be listed as a keyword
  //! argument during printing using the given name as the key. Unnamed
  //! arguments are the default, and are listed as positional arguments before
  //! any named arguments.
  void setArgName(size_t pos, std::string name) {
    arg_names_.at(pos) = name;
  }

 protected:
  //! Inputs that are indices into the FusionState's Recorded State.
  std::vector<State> args_;
  //! String name to print for arg in Python, if any. Defaults to empty.
  std::vector<std::string> arg_names_;
  //! Outputs that are indices into the FusionState's Recorded State.
  std::vector<State> outputs_;
  //! Record Name
  std::string name_;
  //! Record Type of child class used for hashing
  //! enum class RecordType is defined in flatbuffer schema
  serde::RecordType record_type_;
  //! Whether this record type returns a tuple of unknown length. This is only
  //! used for TensorSizesRecord.
  bool always_returns_tuple_ = false;
};

//! The OpRecord RecordFunctor is the most widely used child class because
//! it utilizes varidiac template arguments to represent unary, binary,
//! ternary, and other similar flavors of operations in nvFuser that have
//! a mix of Tensor and Scalar arguments only.
//!
//! The additional data memeber of this child class records the function
//! signature of the nvFuser Arith Operation to be replayed upon a cache
//! miss by the functor operator() call.

template <class OutType, class... ArgTypes>
struct OpRecord : RecordFunctor {
  OpRecord(
      std::vector<State> _args,
      std::vector<State> _outputs,
      std::string _name,
      serde::RecordType record_type,
      std::function<OutType(ArgTypes...)> fusion_op)
      : RecordFunctor(
            std::move(_args),
            std::move(_outputs),
            _name,
            record_type),
        fusion_op_(fusion_op) {}
  ~OpRecord() override = default;
  RecordFunctor* clone() final {
    return new OpRecord(*this);
  }

  //! Child specific hash function in lower 32 bits.= at::Symbol
  //! | 31 -------------------------------------  0 |
  //! | Arith Function Sigs hash code               |
  size_t hash() const final {
    auto result = RecordFunctor::hash();
    return result | (fusion_op_.target_type().hash_code() & 0xffffffff);
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    // A succesfull cast indicates a RecordFunctor of the same child class
    if (auto child_ptr = dynamic_cast<const OpRecord*>(&other)) {
      result = RecordFunctor::operator==(other);
      if (result) {
        // Match the nvFuser arith function types
        result = result &&
            (fusion_op_.target_type() == child_ptr->fusion_op_.target_type());
        if (isDebugDumpEnabled(DebugDumpOption::PythonFrontendDebug)) {
          debug() << "\nOpRecord: " << name_ << " Target Type [self: 0x"
                  << fusion_op_.target_type().name() << "] [other: 0x"
                  << child_ptr->fusion_op_.target_type().name() << "] ";
        }
        // Match the nvFuser arith function pointers
        // IMPORTANT! you need to dereference the target pointer in order
        // to match the function
        result = result &&
            (*fusion_op_.template target<OutType (*)(ArgTypes...)>() ==
             *child_ptr->fusion_op_
                  .template target<OutType (*)(ArgTypes...)>());
        if (isDebugDumpEnabled(DebugDumpOption::PythonFrontendDebug)) {
          debug()
              << "Target  Ptr [self: 0x" << std::hex
              << (size_t)*fusion_op_.template target<OutType (*)(ArgTypes...)>()
              << "] [other: 0x" << std::hex
              << (size_t)*child_ptr->fusion_op_
                     .template target<OutType (*)(ArgTypes...)>()
              << "]\n";
        }
      }
    }
    return result;
  }

  //! The variadic set of indices for the number of args for this op are
  //! deduced by providing the index_sequence as a parameter.  Similarly,
  //! the tuple type is also deduced.
  //!
  //! The tuple type is used to decide whether to cast the input argument
  //! to a Fusion IR TensorView or leave it as a Fusion IR Val (Scalar).
  //!
  //! A deduced binary op could look like:
  //!   OutType opFunc<std::tuple<TensorView*, TensorView*>, 0, 1>
  //! A deduced ternary op could look like:
  //!   OutTupe opFunc<std::tuple<TensorView*, Val*, Val*>, 0, 1, 2>
  template <class TupleType, std::size_t... Is>
  OutType opFunc(FusionState& fd, TupleType& tp, std::index_sequence<Is...>) {
    return fusion_op_(
        dynamic_cast<typename std::tuple_element<Is, TupleType>::type>(
            fd.getFusionState(args_.at(Is).index))...);
  }

  void operator()(FusionState& fd) final {
    using arg_tuple_t = std::tuple<ArgTypes...>;
    auto indices =
        std::make_index_sequence<std::tuple_size<arg_tuple_t>::value>();
    // The tuple variable is never populated, it is passed for its type.
    arg_tuple_t inputs;
    auto output = opFunc(fd, inputs, indices);
    fd.setFusionState(outputs_.at(0).index, output);
  }

 private:
  //! An nvFuser Arith Operation function signature
  std::function<OutType(ArgTypes...)> fusion_op_;
};

struct ReshapeOpRecord : RecordFunctor {
  ReshapeOpRecord(
      std::vector<State> _args,
      std::vector<State> _outputs,
      std::vector<int64_t> original_shape,
      std::vector<int64_t> new_shape)
      : RecordFunctor(
            std::move(_args),
            std::move(_outputs),
            "ops.reshape",
            serde::RecordType_ReshapeOp),
        original_shape_(std::move(original_shape)),
        new_shape_(std::move(new_shape)) {}
  ~ReshapeOpRecord() override = default;
  RecordFunctor* clone() final {
    return new ReshapeOpRecord(*this);
  }

  //! Child specific hash function in lower 32 bits.
  //! | 31 -------------- 16 | 15 --------------  0 |
  //! | original_shape hash  | new_shape hash       |
  size_t hash() const final {
    auto result = RecordFunctor::hash();
    size_t new_shape_hash = 0;
    for (auto shape : new_shape_) {
      new_shape_hash ^= static_cast<size_t>(shape);
    }
    size_t original_shape_hash = 0;
    for (auto shape : original_shape_) {
      original_shape_hash |= 1 << ((new_shape_.size() - 1) - shape);
    }
    original_shape_hash = (original_shape_hash & 0xffff) << 16;
    return result | original_shape_hash | (new_shape_hash & 0xffff);
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto child_ptr = dynamic_cast<const ReshapeOpRecord*>(&other)) {
      result = RecordFunctor::operator==(other);
      result &= std::equal(
          original_shape_.begin(),
          original_shape_.end(),
          child_ptr->original_shape_.begin());
      result &= std::equal(
          new_shape_.begin(), new_shape_.end(), child_ptr->new_shape_.begin());
    }
    return result;
  }

  void operator()(FusionState& fd) final {
    auto arg = fd.getFusionState(args_.at(0).index)->template as<TensorView>();
    auto output = reshape(arg, original_shape_, new_shape_);
    fd.setFusionState(outputs_.at(0).index, output);
  }

  void print(std::ostream& os, bool close_function = true) const final {
    RecordFunctor::print(os, false);
    os << ", original_shape=[";
    bool first_arg = true;
    for (auto shape : original_shape_) {
      if (first_arg) {
        first_arg = false;
      } else {
        os << ", ";
      }
      os << shape;
    }
    os << "]";
    os << ", new_shape=[";
    first_arg = true;
    for (auto shape : new_shape_) {
      if (first_arg) {
        first_arg = false;
      } else {
        os << ", ";
      }
      os << shape;
    }
    os << "]";
    if (close_function) {
      os << ")";
    }
  }

  std::pair<serde::RecordData, flatbuffers::Offset<void>> recordData(
      flatbuffers::FlatBufferBuilder& builder) const final {
    return {
        serde::RecordData_Reshape,
        serde::CreateReshapeDirect(builder, &original_shape_, &new_shape_)
            .Union()};
  }

 private:
  //! Represents the tensor dimensions of the input tensor.
  std::vector<int64_t> original_shape_;
  //! Represents the tensor dimensions of the output tensor.
  std::vector<int64_t> new_shape_;
};

struct PadOpRecord : RecordFunctor {
  PadOpRecord(
      std::vector<State> _args,
      std::vector<State> _outputs,
      std::vector<int64_t>&& pad_widths)
      : RecordFunctor(
            std::move(_args),
            std::move(_outputs),
            "ops.pad",
            serde::RecordType_PadOp),
        pad_widths_(std::move(pad_widths)) {}
  ~PadOpRecord() override = default;
  RecordFunctor* clone() final {
    return new PadOpRecord(*this);
  }

  //! Child specific hash function in lower 32 bits.
  //! | 31 ------------------------------ 0 |
  //! |          pad_widths                 |
  size_t hash() const final {
    auto result = RecordFunctor::hash();
    size_t widths_hash = 0;
    for (size_t i = 0; i < pad_widths_.size(); ++i) {
      auto w = pad_widths_.at(i);
      // Circular shift the lower 32 bits of w by 4 * i
      // This reduces collisions by only directly xor-ing every 8th argument.
      // Since many shifts will occupy less than 4 bits, we will have no
      // collisions for most pads of up to 4 trailing dimensions.
      size_t shift = (i * 4) % 32;
      w = w << shift | w >> (32 - shift);
      widths_hash ^= w << i;
    }
    return result | (widths_hash & 0xffffffff);
  }

  bool operator==(const RecordFunctor& other) const final {
    if (auto child_ptr = dynamic_cast<const PadOpRecord*>(&other)) {
      if (!RecordFunctor::operator==(other)) {
        return false;
      }
      if (pad_widths_.size() != child_ptr->pad_widths_.size()) {
        return false;
      }
      for (size_t i = 0; i < pad_widths_.size(); ++i) {
        if (pad_widths_.at(i) != child_ptr->pad_widths_.at(i)) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

  void operator()(FusionState& fd) final {
    auto arg = fd.getFusionState(args_.at(0).index)->template as<TensorView>();
    std::vector<Val*> val_widths;
    val_widths.reserve(pad_widths_.size());
    for (auto p : pad_widths_) {
      auto pval = IrBuilder::create<nvfuser::Val>(p);
      val_widths.push_back(pval);
    }

    TensorView* output = nullptr;
    if (args_.at(1).stype == serde::StateType_Scalar) {
      output = pad(arg, val_widths, fd.getFusionState(args_.at(1).index));
    } else { // default: None
      output = pad(arg, val_widths);
    }

    fd.setFusionState(outputs_.at(0).index, output);
  }

  void print(std::ostream& os, bool close_function = true) const final {
    // pad_widths is the second (required) argument, but the fill value is a
    // Scalar, so it would be printed first by the default printer, so we
    // implement our own print() here
    os << outputs_.at(0);
    os << " = "
       << "fd." << name_ << "(";
    os << args_.at(0); // unpadded tensor
    os << ", [";
    bool first_arg = true;
    for (auto w : pad_widths_) {
      if (first_arg) {
        first_arg = false;
      } else {
        os << ", ";
      }
      os << w;
    }
    os << "]";
    if (args_.at(1).stype == serde::StateType_Scalar) {
      // fill value was given
      os << ", " << args_.at(1);
    }
    os << ")";
  }

  std::pair<serde::RecordData, flatbuffers::Offset<void>> recordData(
      flatbuffers::FlatBufferBuilder& builder) const final {
    return {
        serde::RecordData_Pad,
        serde::CreatePadDirect(builder, &pad_widths_).Union()};
  }

 private:
  //! Pairs of non-negative integers indicating the amount to pad the front and
  //! back of each dimension.
  std::vector<int64_t> pad_widths_;
};

struct PermuteOpRecord : RecordFunctor {
  PermuteOpRecord(
      std::vector<State> _args,
      std::vector<State> _outputs,
      std::vector<int64_t> dims)
      : RecordFunctor(
            std::move(_args),
            std::move(_outputs),
            "ops.permute",
            serde::RecordType_PermuteOp),
        dims_(std::move(dims)) {}
  ~PermuteOpRecord() override = default;
  RecordFunctor* clone() final {
    return new PermuteOpRecord(*this);
  }

  size_t hash() const final {
    auto result = RecordFunctor::hash();
    size_t dims_hash = 0;
    for (auto dim : dims_) {
      dims_hash ^= static_cast<size_t>(dim);
    }
    return result | (dims_hash & 0xffff);
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto child_ptr = dynamic_cast<const PermuteOpRecord*>(&other)) {
      result = RecordFunctor::operator==(other);
      if (result) {
        result = (dims_.size() == child_ptr->dims_.size());
        if (result) {
          for (size_t i = 0; i < dims_.size(); ++i) {
            if (dims_[i] != child_ptr->dims_[i]) {
              result = false;
              break;
            }
          }
        }
      }
    }
    return result;
  }

  void operator()(FusionState& fd) final {
    auto arg = fd.getFusionState(args_.at(0).index)->template as<TensorView>();
    auto output = permute(arg, dims_);
    fd.setFusionState(outputs_.at(0).index, output);
  }

  void print(std::ostream& os, bool close_function = true) const final {
    RecordFunctor::print(os, false);
    os << ", dims=[";
    bool first_arg = true;
    for (auto dim : dims_) {
      if (first_arg) {
        first_arg = false;
      } else {
        os << ", ";
      }
      os << dim;
    }
    os << "]";
    if (close_function) {
      os << ")";
    }
  }

  std::pair<serde::RecordData, flatbuffers::Offset<void>> recordData(
      flatbuffers::FlatBufferBuilder& builder) const final {
    return {
        serde::RecordData_Permute,
        serde::CreatePermuteDirect(builder, &dims_).Union()};
  }

 private:
  //! Represents the mapping from the original shape to the new shape
  std::vector<int64_t> dims_;
};

struct SqueezeOpRecord : RecordFunctor {
  SqueezeOpRecord(
      std::vector<State> _args,
      std::vector<State> _outputs,
      std::vector<int64_t> original_shape,
      std::vector<int64_t> dims)
      : RecordFunctor(
            std::move(_args),
            std::move(_outputs),
            "ops.squeeze",
            serde::RecordType_SqueezeOp),
        original_shape_(std::move(original_shape)),
        dims_(std::move(dims)) {}
  ~SqueezeOpRecord() override = default;
  RecordFunctor* clone() final {
    return new SqueezeOpRecord(*this);
  }

  //! Child specific hash function in lower 32 bits.
  //! | 31 -------------- 16 | 15 --------------  0 |
  //! | Squeeze Dim hash     | original_shape hash  |
  size_t hash() const final {
    auto result = RecordFunctor::hash();
    size_t original_shape_hash = 0;
    for (auto shape : original_shape_) {
      original_shape_hash ^= static_cast<size_t>(shape);
    }
    size_t squeeze_dims_hash = 0;
    for (auto dim : dims_) {
      squeeze_dims_hash ^= static_cast<size_t>(dim);
    }
    squeeze_dims_hash = (squeeze_dims_hash & 0xffff) << 16;
    return result | squeeze_dims_hash | (original_shape_hash & 0xffff);
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto child_ptr = dynamic_cast<const SqueezeOpRecord*>(&other)) {
      result = RecordFunctor::operator==(other);
      if (result) {
        result = (original_shape_.size() == child_ptr->original_shape_.size());
        if (result) {
          for (size_t i = 0; i < dims_.size(); ++i) {
            if (dims_[i] != child_ptr->dims_[i]) {
              result = false;
              break;
            }
          }
        }
        if (result) {
          for (size_t i = 0; i < original_shape_.size(); ++i) {
            if (original_shape_[i] != child_ptr->original_shape_[i]) {
              result = false;
              break;
            }
          }
        }
      }
    }
    return result;
  }

  void operator()(FusionState& fd) final {
    auto arg = fd.getFusionState(args_.at(0).index)->template as<TensorView>();
    auto output = squeeze(arg, original_shape_, dims_);
    fd.setFusionState(outputs_.at(0).index, output);
  }

  void print(std::ostream& os, bool close_function = true) const final {
    RecordFunctor::print(os, false);
    os << ", original_shape=[";
    bool first_arg = true;
    for (auto shape : original_shape_) {
      if (first_arg) {
        first_arg = false;
      } else {
        os << ", ";
      }
      os << shape;
    }
    os << "], dims=[";
    first_arg = true;
    for (auto dim : dims_) {
      if (first_arg) {
        first_arg = false;
      } else {
        os << ", ";
      }
      os << dim;
    }
    os << "]";
    if (close_function) {
      os << ")";
    }
  }

  std::pair<serde::RecordData, flatbuffers::Offset<void>> recordData(
      flatbuffers::FlatBufferBuilder& builder) const final {
    return {
        serde::RecordData_Squeeze,
        serde::CreateSqueezeDirect(builder, &original_shape_, &dims_).Union()};
  }

 private:
  //! Represents the tensor dimensions of the input tensor.
  std::vector<int64_t> original_shape_;
  //! Dimension to squeeze.
  std::vector<int64_t> dims_;
};

//! Specialized Record Functor for the FusionState's broadcast_in_dim op.

template <typename OutputShapeType>
struct BroadcastInDimOpRecord : RecordFunctor {
  BroadcastInDimOpRecord(
      std::vector<State> _args,
      std::vector<State> _outputs,
      std::string _name,
      serde::RecordType record_type,
      std::vector<OutputShapeType> output_shape,
      std::vector<int64_t> broadcast_dims)
      : RecordFunctor(
            std::move(_args),
            std::move(_outputs),
            _name,
            record_type),
        output_shape_(std::move(output_shape)),
        broadcast_dims_(std::move(broadcast_dims)) {}
  ~BroadcastInDimOpRecord() override = default;
  RecordFunctor* clone() final {
    return new BroadcastInDimOpRecord(*this);
  }

  inline size_t outputShapeHash(
      const std::vector<OutputShapeType>& shape) const;

  //! Child specific hash function in lower 32 bits.
  //! | 31 -------------- 16 | 15 --------------  0 |
  //! | broadcast_dims hash  | output_shape hash    |
  //!
  //! The output_shape hash is specialized in 2 ways using the method
  //! outputShapeHash:
  //! 1. int64_t - hashes dimension sizes.
  //! 2. State - hashes number of dimensions
  size_t hash() const final {
    auto result = RecordFunctor::hash();
    size_t broadcast_dims_hash = 0;
    for (auto dim : broadcast_dims_) {
      broadcast_dims_hash |= 1 << ((output_shape_.size() - 1) - dim);
    }
    broadcast_dims_hash = (broadcast_dims_hash & 0xffff) << 16;
    return result | broadcast_dims_hash |
        (outputShapeHash(output_shape_) & 0xffff);
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto child_ptr = dynamic_cast<const BroadcastInDimOpRecord*>(&other)) {
      result = RecordFunctor::operator==(other);
      if (result) {
        result =
            ((output_shape_.size() == child_ptr->output_shape_.size()) &&
             (broadcast_dims_.size() == child_ptr->broadcast_dims_.size()));
        if (result) {
          for (size_t i = 0; i < output_shape_.size(); ++i) {
            if (output_shape_[i] != child_ptr->output_shape_[i]) {
              result = false;
              break;
            }
          }
        }
        if (result) {
          for (size_t i = 0; i < broadcast_dims_.size(); ++i) {
            if (broadcast_dims_[i] != child_ptr->broadcast_dims_[i]) {
              result = false;
              break;
            }
          }
        }
      }
    }
    return result;
  }

  inline std::optional<std::vector<Val*>> expandShape(
      const FusionState& fd,
      const std::vector<bool>& expand_dim,
      const std::vector<OutputShapeType>& shape) const;

  //! The operator() call is specialize with th expandShape() method based on
  //! the OutputShapeType template parameter
  void operator()(FusionState& fd) final {
    auto arg = fd.getFusionState(args_.at(0).index)->template as<TensorView>();

    const auto& arg_domains_nr = arg->domain()->noReductions();
    const auto arg_ndims = arg_domains_nr.size();
    TORCH_CHECK(
        output_shape_.size() >= arg_ndims,
        "The new shape is expected to be greater-then-or-equal to the input",
        output_shape_.size(),
        arg_ndims);
    TORCH_CHECK(
        arg_ndims == broadcast_dims_.size(),
        "The broadcast dimensions should match the input dimensions.",
        arg_ndims,
        broadcast_dims_.size());

    std::vector<bool> is_broadcast_dim(output_shape_.size(), true);
    std::vector<bool> is_expand_dim(output_shape_.size(), true);
    for (const auto idx : c10::irange(broadcast_dims_.size())) {
      if (idx > 0) {
        TORCH_CHECK(
            broadcast_dims_[idx - 1] < broadcast_dims_[idx],
            "Broadcast dimension is not greater than the previous value.");
      }
      TORCH_CHECK(
          broadcast_dims_[idx] < static_cast<int>(output_shape_.size()),
          "Invalid broadcast_dims value.");
      is_broadcast_dim.at(broadcast_dims_[idx]) = false;
      // Note: when we expand a broadcasted dimension, we need to expand it
      // to a concrete size, hence the need for `is_expand_dim` flag and the
      // expand operation following the broadcast.
      is_expand_dim.at(broadcast_dims_[idx]) =
          arg_domains_nr[idx]->isBroadcast();
    }

    auto output = broadcast(arg, is_broadcast_dim);

    std::optional<std::vector<Val*>> expand_shape =
        expandShape(fd, is_expand_dim, output_shape_);
    if (expand_shape.has_value()) {
      output = expand(output, expand_shape.value());
    }
    fd.setFusionState(outputs_.at(0).index, output);
  }

  void print(std::ostream& os, bool close_function = true) const final {
    RecordFunctor::print(os, false);
    os << ", output_shape=[";
    bool first_arg = true;
    for (auto shape : output_shape_) {
      if (first_arg) {
        first_arg = false;
      } else {
        os << ", ";
      }
      os << shape;
    }
    os << "]";
    os << ", broadcast_dims=[";
    first_arg = true;
    for (auto dim : broadcast_dims_) {
      if (first_arg) {
        first_arg = false;
      } else {
        os << ", ";
      }
      os << dim;
    }
    os << "]";
    if (close_function) {
      os << ")";
    }
  }

  std::pair<serde::RecordData, flatbuffers::Offset<void>> recordData(
      flatbuffers::FlatBufferBuilder& builder) const final {
    return outputShapeRecordData(builder, output_shape_);
  };

  inline std::pair<serde::RecordData, flatbuffers::Offset<void>>
  outputShapeRecordData(
      flatbuffers::FlatBufferBuilder& builder,
      const std::vector<OutputShapeType>& shape) const;

 private:
  //! Represents the tensor dimensions of the output tensor.
  std::vector<OutputShapeType> output_shape_;
  //! Communicates which dimensions of the output the input tensor maps.
  //! For instance, for output [2, 3, 4] and input [3]. This vector would
  //! contain [1].
  std::vector<int64_t> broadcast_dims_;
};

//! ouputShapeHash Specializations used by hash()

template <>
inline size_t BroadcastInDimOpRecord<int64_t>::outputShapeHash(
    const std::vector<int64_t>& shape) const {
  size_t shape_hash = 0;
  for (auto size : shape) {
    shape_hash ^= static_cast<size_t>(size);
  }
  return shape_hash;
}

template <>
inline size_t BroadcastInDimOpRecord<State>::outputShapeHash(
    const std::vector<State>& shape) const {
  return shape.size();
}

//! expandShape Specializations used by operator()

template <>
inline std::optional<std::vector<Val*>> BroadcastInDimOpRecord<int64_t>::
    expandShape(
        const FusionState& fd,
        const std::vector<bool>& expand_dim,
        const std::vector<int64_t>& shape) const {
  std::vector<Val*> expand_shape(shape.size(), nullptr);
  bool has_expand = false;
  for (const auto idx : c10::irange(shape.size())) {
    if (expand_dim[idx] && shape[idx] != 1 && shape[idx] != -1) {
      expand_shape[idx] = IrBuilder::create<nvfuser::Val>(shape[idx]);
      has_expand = true;
    } else {
      expand_shape[idx] = IrBuilder::create<nvfuser::Val>(-1L);
    }
  }

  if (has_expand) {
    return std::optional<std::vector<Val*>>(expand_shape);
  } else {
    return std::nullopt;
  }
}

template <>
inline std::optional<std::vector<Val*>> BroadcastInDimOpRecord<State>::
    expandShape(
        const FusionState& fd,
        const std::vector<bool>& expand_dim,
        const std::vector<State>& shape) const {
  std::vector<Val*> expand_shape(shape.size(), nullptr);
  std::transform(
      shape.begin(),
      shape.end(),
      expand_shape.begin(),
      [&fd](const State& state) {
        return fd.getFusionState(state.index)->template as<Val>();
      });
  return std::optional<std::vector<Val*>>(expand_shape);
}

//! outputShapeRecordData Specializations used by recordData()

template <>
inline std::pair<serde::RecordData, flatbuffers::Offset<void>>
BroadcastInDimOpRecord<int64_t>::outputShapeRecordData(
    flatbuffers::FlatBufferBuilder& builder,
    const std::vector<int64_t>& shape) const {
  return {
      serde::RecordData_BroadcastInDim,
      serde::CreateBroadcastInDimDirect(builder, &shape, &broadcast_dims_)
          .Union()};
}

template <>
inline std::pair<serde::RecordData, flatbuffers::Offset<void>>
BroadcastInDimOpRecord<State>::outputShapeRecordData(
    flatbuffers::FlatBufferBuilder& builder,
    const std::vector<State>& shape) const {
  std::vector<serde::State> fb_output_shape;
  fb_output_shape.reserve(shape.size());
  for (auto& it : shape) {
    fb_output_shape.emplace_back(it.index, it.stype);
  }
  auto output_shape_fb = builder.CreateVectorOfStructs(
      fb_output_shape.data(), fb_output_shape.size());

  auto bcast_dims_fb = builder.CreateVector(broadcast_dims_);

  serde::BroadcastInDimSymbolicBuilder bcast_builder(builder);
  bcast_builder.add_output_shape(output_shape_fb);
  bcast_builder.add_broadcast_dims(bcast_dims_fb);
  auto bcast_in_dim_data = bcast_builder.Finish();

  return {serde::RecordData_BroadcastInDimSymbolic, bcast_in_dim_data.Union()};
}

//! Specialized Record Functor for the FusionState's broadcast op.

struct BroadcastOpRecord : RecordFunctor {
  BroadcastOpRecord(
      std::vector<State> _args,
      std::vector<State> _outputs,
      std::string _name,
      std::vector<bool> is_broadcast_dim)
      : RecordFunctor(
            std::move(_args),
            std::move(_outputs),
            _name,
            serde::RecordType_BroadcastOp),
        is_broadcast_dim_(std::move(is_broadcast_dim)) {}
  ~BroadcastOpRecord() override = default;
  RecordFunctor* clone() final {
    return new BroadcastOpRecord(*this);
  }

  size_t hash() const final {
    auto result = RecordFunctor::hash();
    size_t is_broadcast_dim_hash = 0;
    for (size_t i = 0; i < is_broadcast_dim_.size(); ++i) {
      is_broadcast_dim_hash |=
          (is_broadcast_dim_[i] << (is_broadcast_dim_.size() - 1 - i));
    }
    return result | (is_broadcast_dim_hash & 0xfff);
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto child_ptr = dynamic_cast<const BroadcastOpRecord*>(&other)) {
      result = RecordFunctor::operator==(other);
      result &= std::equal(
          is_broadcast_dim_.begin(),
          is_broadcast_dim_.end(),
          child_ptr->is_broadcast_dim_.begin());
    }
    return result;
  }

  void operator()(FusionState& fd) final {
    auto arg = fd.getFusionState(args_.at(0).index)->template as<TensorView>();
    auto output = broadcast(arg, is_broadcast_dim_);
    fd.setFusionState(outputs_.at(0).index, output);
  }

  void print(std::ostream& os, bool close_function = true) const final {
    RecordFunctor::print(os, false);
    os << ", is_broadcast_dim=[";
    bool first_arg = true;
    for (auto dim : is_broadcast_dim_) {
      if (first_arg) {
        first_arg = false;
      } else {
        os << ", ";
      }
      os << (dim ? "True" : "False");
    }
    os << "]";
    if (close_function) {
      os << ")";
    }
  }

  std::pair<serde::RecordData, flatbuffers::Offset<void>> recordData(
      flatbuffers::FlatBufferBuilder& builder) const final {
    auto fb_broadcast_dims = builder.CreateVector(is_broadcast_dim_);

    serde::BroadcastBuilder bcast_builder(builder);
    bcast_builder.add_broadcast_dims(fb_broadcast_dims);
    auto expr_data = bcast_builder.Finish();
    return {serde::RecordData_Broadcast, expr_data.Union()};
  }

 private:
  //! Communicates which dimensions in the output are broadcasted.
  std::vector<bool> is_broadcast_dim_;
};

template <class OutType, class ArgType>
struct CastOpRecord : RecordFunctor {
  CastOpRecord(
      std::vector<State> _args,
      std::vector<State> _outputs,
      std::string _name,
      serde::RecordType record_type,
      std::function<OutType(DataType, ArgType)> fusion_op,
      PrimDataType dtype)
      : RecordFunctor(
            std::move(_args),
            std::move(_outputs),
            _name,
            record_type),
        fusion_op_(fusion_op),
        dtype_(dtype) {}
  ~CastOpRecord() override = default;
  RecordFunctor* clone() final {
    return new CastOpRecord(*this);
  }

  //! Child specific hash function in lower 32 bits.
  //! | 31 --- 24 | 23 --------------------------  0 |
  //! | Dtype     | Arith Function Sig hash code     |
  size_t hash() const final {
    auto result = RecordFunctor::hash();
    result |= ((static_cast<size_t>(dtype_) & 0xff) << 24);
    result |= (fusion_op_.target_type().hash_code() & 0xffffff);
    return result;
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto child_ptr = dynamic_cast<const CastOpRecord*>(&other)) {
      result = RecordFunctor::operator==(other);
      if (result) {
        result = result &&
            (fusion_op_.target_type() == child_ptr->fusion_op_.target_type());
        if (isDebugDumpEnabled(DebugDumpOption::PythonFrontendDebug)) {
          debug() << "\nCastOpRecord: " << name_ << " Target Type [self: 0x"
                  << fusion_op_.target_type().name() << "] [other: 0x"
                  << child_ptr->fusion_op_.target_type().name() << "]";
        }
        // IMPORTANT! you need to dereference the target pointer in order
        // to match the function
        result = result &&
            (*fusion_op_.template target<OutType (*)(DataType, ArgType)>() ==
             *child_ptr->fusion_op_
                  .template target<OutType (*)(DataType, ArgType)>());
        if (isDebugDumpEnabled(DebugDumpOption::PythonFrontendDebug)) {
          debug() << " Target  Ptr [self: 0x" << std::hex
                  << (size_t)*fusion_op_
                         .template target<OutType (*)(DataType, ArgType)>()
                  << "] [other: 0x" << std::hex
                  << (size_t)*child_ptr->fusion_op_
                         .template target<OutType (*)(DataType, ArgType)>()
                  << "]\n";
        }
        result = result && (dtype_ == child_ptr->dtype_);
      }
    }
    return result;
  }

  void operator()(FusionState& fd) final {
    auto arg = dynamic_cast<ArgType>(fd.getFusionState(args_.at(0).index));
    auto output = fusion_op_(dtype_, arg);
    fd.setFusionState(outputs_.at(0).index, output);
  }

  void print(std::ostream& os, bool close_function = true) const final {
    RecordFunctor::print(os, false);
    os << ", dtype=" << dtypeToPyString(dtype_);
    if (close_function) {
      os << ")";
    }
  }

  std::pair<serde::RecordData, flatbuffers::Offset<void>> recordData(
      flatbuffers::FlatBufferBuilder& builder) const final {
    return {
        serde::RecordData_Dtype,
        serde::CreateDtype(builder, serde::mapToSerdeDtype(dtype_)).Union()};
  }

 private:
  //! nvFuser arith function signature
  std::function<OutType(DataType, ArgType)> fusion_op_;
  //! Type to cast to.
  PrimDataType dtype_;
};

struct CatOpRecord : RecordFunctor {
  CatOpRecord(
      std::vector<State> _args,
      std::vector<State> _outputs,
      int64_t dim)
      : RecordFunctor(
            std::move(_args),
            std::move(_outputs),
            "ops.cat",
            serde::RecordType_CatOp),
        dim_(dim) {}
  ~CatOpRecord() override = default;
  RecordFunctor* clone() final {
    return new CatOpRecord(*this);
  }

  size_t hash() const final {
    auto result = RecordFunctor::hash();
    return result | (static_cast<size_t>(dim_) & 0xffff);
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto child_ptr = dynamic_cast<const CatOpRecord*>(&other)) {
      result = RecordFunctor::operator==(other) && dim_ == child_ptr->dim_;
    }
    return result;
  }

  void operator()(FusionState& fd) final {
    std::vector<TensorView*> input_tvs;
    input_tvs.reserve(args_.size());
    for (auto& a : args_) {
      input_tvs.push_back(
          fd.getFusionState(a.index)->template as<TensorView>());
    }
    auto output = cat(input_tvs, dim_);
    fd.setFusionState(outputs_.at(0).index, output);
  }

  void print(std::ostream& os, bool close_function = true) const final {
    // Similar to RecordFunctor::print(os, false), but don't print args
    bool first_output = true;
    for (auto& output : outputs_) {
      if (first_output) {
        first_output = false;
      } else {
        os << ", ";
      }
      os << output;
    }
    if (always_returns_tuple_) {
      os << ",";
    }
    if (!outputs_.empty()) {
      os << " = "
         << "fd." << name_ << "(";
    } else {
      os << "fd." << name_ << "(";
    }
    os << "[";
    bool first_arg = true;
    for (auto& arg : args_) {
      if (first_arg) {
        first_arg = false;
      } else {
        os << ", ";
      }
      os << arg;
    }
    os << "], dim=" << dim_;
    if (close_function) {
      os << ")";
    }
  }

  std::pair<serde::RecordData, flatbuffers::Offset<void>> recordData(
      flatbuffers::FlatBufferBuilder& builder) const final {
    return {
        serde::RecordData_Dimension,
        serde::CreateDimension(builder, dim_).Union()};
  }

 private:
  //! The dimension along which we will concatenate
  int64_t dim_;
};

//! Specialized Record Functor for recording FusionState End.
//! The accompanying Fusion Cache Entry holds a Fusion Object.

struct EndRecord : RecordFunctor {
  EndRecord() : RecordFunctor({}, {}, "end", serde::RecordType_End) {}
  ~EndRecord() override = default;
  RecordFunctor* clone() final {
    return new EndRecord(*this);
  }

  //! Child specific hash function in lower 32 bits.
  //! | 31 ---------------------------------------  0 |
  //! | None                                          |
  size_t hash() const final {
    return RecordFunctor::hash();
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (dynamic_cast<const EndRecord*>(&other)) {
      result = RecordFunctor::operator==(other);
    }
    return result;
  }

  void operator()(FusionState& fd) final {}
};

//! Specialized Record Functor for recording FusionState input tensors.

struct TensorRecord : RecordFunctor {
  TensorRecord(
      std::vector<State> _outputs,
      std::vector<int64_t> _symbolic_sizes,
      std::vector<std::optional<bool>> _contiguity,
      PrimDataType _dtype,
      bool _is_cpu = false)
      : RecordFunctor(
            {},
            std::move(_outputs),
            "define_tensor",
            serde::RecordType_Tensor),
        symbolic_sizes_(std::move(_symbolic_sizes)),
        contiguity_(std::move(_contiguity)),
        dtype_(_dtype),
        is_cpu_(_is_cpu) {}
  ~TensorRecord() override = default;
  RecordFunctor* clone() final {
    return new TensorRecord(*this);
  }

  //! Child specific hash function in lower 32 bits.
  //! |  31  | 30 --- 24 | 23 --------- 12 | 11 ---------  0 |
  //! | CPU? | Dtype     | Symbolic Sizes  | Contiguous Info |
  size_t hash() const final {
    auto result = RecordFunctor::hash();
    size_t ssize_hash = 0;
    for (size_t i = 0; i < symbolic_sizes_.size(); ++i) {
      size_t ssize = 0;
      if (symbolic_sizes_[i] == -1) {
        ssize = 1;
      }
      ssize_hash |= (ssize << (symbolic_sizes_.size() - 1 - i));
    }
    size_t contig_hash = 0;
    for (size_t i = 0; i < contiguity_.size(); ++i) {
      auto contiguity_value = contiguity_[i];
      contig_hash |=
          ((contiguity_value.has_value() && contiguity_value.value())
           << (contiguity_.size() - 1 - i));
    }

    result |= ((static_cast<size_t>(is_cpu_) & 0x1) << 31);
    result |= ((static_cast<size_t>(dtype_) & 0x7f) << 24);
    return result | ((ssize_hash & 0xfff) << 12) | (contig_hash & 0xfff);
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto child_ptr = dynamic_cast<const TensorRecord*>(&other)) {
      result = RecordFunctor::operator==(other);
      result = result && (dtype_ == child_ptr->dtype_);
      result = result && (is_cpu_ == child_ptr->is_cpu_);
      if (result) {
        result =
            ((symbolic_sizes_.size() == child_ptr->symbolic_sizes_.size()) &&
             (contiguity_.size() == child_ptr->contiguity_.size()));
        if (result) {
          for (size_t i = 0; i < symbolic_sizes_.size(); ++i) {
            if (symbolic_sizes_[i] != child_ptr->symbolic_sizes_[i]) {
              result = false;
              break;
            }
          }
        }
        if (result) {
          for (size_t i = 0; i < contiguity_.size(); ++i) {
            if (contiguity_[i] != child_ptr->contiguity_[i]) {
              result = false;
              break;
            }
          }
        }
      }
    }
    return result;
  }

  void operator()(FusionState& fd) final {
    auto rank = symbolic_sizes_.size();
    std::vector<bool> is_expand(rank);

    for (const auto index : c10::irange(rank)) {
      bool is_broadcast = !contiguity_[index].has_value();
      bool has_symbolic_size = (symbolic_sizes_[index] == -1);
      is_expand[index] = is_broadcast && has_symbolic_size;
    }

    auto tv = TensorViewBuilder()
                  .ndims(symbolic_sizes_.size())
                  .contiguity(contiguity_)
                  .shape(symbolic_sizes_)
                  .dtype(dtype_)
                  .expanded(std::move(is_expand))
                  .build();

    if (symbolic_sizes_.empty() && is_cpu_) {
      tv->setCpuScalar(true);
    } else {
      TORCH_CHECK(!is_cpu_, "CPU non-scalar tensor is not supported!");
    }

    fd.setFusionState(outputs_.at(0).index, tv);
    fd.addInput(tv);
  }

  void print(std::ostream& os, bool close_function = true) const final {
    RecordFunctor::print(os, false);
    os << "symbolic_sizes=[";
    bool first_arg = true;
    for (auto ss : symbolic_sizes_) {
      if (first_arg) {
        first_arg = false;
      } else {
        os << ", ";
      }
      os << ss;
    }
    os << "], contiguity=[";
    first_arg = true;
    for (auto ci : contiguity_) {
      if (first_arg) {
        first_arg = false;
      } else {
        os << ", ";
      }
      if (!ci.has_value()) {
        os << "None";
      } else {
        if (*ci) {
          os << "True";
        } else {
          os << "False";
        }
      }
    }
    os << "], dtype=" << dtypeToPyString(dtype_);
    os << ", is_cpu=" << (is_cpu_ ? "True" : "False");
    if (close_function) {
      os << ")";
    }
  }

  std::pair<serde::RecordData, flatbuffers::Offset<void>> recordData(
      flatbuffers::FlatBufferBuilder& builder) const final {
    auto fb_sizes = builder.CreateVector(symbolic_sizes_);

    auto mapOptionalToEnum = [](std::optional<bool> v) -> int {
      if (!v.has_value()) {
        return serde::Contiguity_None;
      } else if (v.value()) {
        return serde::Contiguity_Contiguous;
      } else {
        return serde::Contiguity_Strided;
      }
    };
    std::vector<int> contiguity_enum;
    std::transform(
        contiguity_.cbegin(),
        contiguity_.cend(),
        std::back_inserter(contiguity_enum),
        mapOptionalToEnum);
    auto fb_contiguity_enum = builder.CreateVector(contiguity_enum);

    serde::TensorBuilder tensor_builder(builder);
    tensor_builder.add_sizes(fb_sizes);
    tensor_builder.add_contiguity(fb_contiguity_enum);
    tensor_builder.add_dtype(serde::mapToSerdeDtype(dtype_));
    tensor_builder.add_is_cpu(is_cpu_);
    auto expr_data = tensor_builder.Finish();
    return {serde::RecordData_Tensor, expr_data.Union()};
  }

 private:
  //! A vector of tensor dimension sizes.
  //! This vector only captures sizes of -1 or 1 to indicate a symbolic
  //! dimension (-1) or a broadcast dimension (1).
  std::vector<int64_t> symbolic_sizes_;
  //! A vector to indicate whether the a tensor dimension is contiguous
  //! with the dimension just to its right.
  std::vector<std::optional<bool>> contiguity_;
  //! Tensor data type.
  PrimDataType dtype_;
  //! Notes a scalar CPU Tensor
  bool is_cpu_;
};

//! Specialized Record Functor for recording FusionState outputs.

template <class OutputType>
struct OutputRecord : RecordFunctor {
  OutputRecord(
      std::vector<State> _args,
      serde::RecordType record_type,
      std::vector<int64_t> stride_order = {})
      : RecordFunctor(std::move(_args), {}, "add_output", record_type) {
    if (!stride_order.empty()) {
      bool requires_permutation = false;
      for (const auto i : c10::irange(stride_order.size())) {
        if (stride_order[i] != (int64_t)i) {
          requires_permutation = true;
          break;
        }
      }
      if (requires_permutation) {
        stride_order_ = stride_order;
      }
    }
  }
  ~OutputRecord() override = default;
  RecordFunctor* clone() final {
    return new OutputRecord(*this);
  }

  //! Nothing extra necessary in hash
  //! Child specific hash function in lower 32 bits.
  //! | 31 ----------------------------------------  0 |
  //! | stride_order hash                              |
  size_t hash() const final {
    size_t stride_order_hash = 0;
    for (auto i : c10::irange(stride_order_.size())) {
      stride_order_hash = (stride_order_hash << 4) | stride_order_[i];
    }
    return RecordFunctor::hash() | (stride_order_hash & 0xffffffff);
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto child_ptr = dynamic_cast<const OutputRecord*>(&other)) {
      result = RecordFunctor::operator==(other);
      if (result) {
        result = (stride_order_.size() == child_ptr->stride_order_.size());
        if (result) {
          for (size_t i = 0; i < stride_order_.size(); ++i) {
            if (stride_order_[i] != child_ptr->stride_order_[i]) {
              result = false;
              break;
            }
          }
        }
      }
    }
    return result;
  }

  void operator()(FusionState& fd) final {
    auto output = fd.getFusionState(args_.at(0).index);
    Val* alias_input = nullptr;
    if (args_.size() == 2) {
      alias_input = fd.getFusionState(args_.at(1).index);
    }

    if (alias_input) {
      TORCH_CHECK(
          stride_order_.empty(),
          "stride_order can't be dictated for aliased outputs.");
      if (std::is_same<OutputType, TensorView>::value) {
        fd.aliasOutputToInput(output, alias_input);
      } else {
        TORCH_INTERNAL_ASSERT(false, "Scalar outputs should not alias inputs.");
      }
    } else {
      // With C++17, this statement should be "if constexpr"
      if (std::is_same<OutputType, TensorView>::value) {
        auto tv_output = output->template as<TensorView>();

        if (!stride_order_.empty()) {
          std::vector<int64_t> reverse_perm(stride_order_.size());
          int64_t duplicate_check = 0;
          for (const auto i : c10::irange((int64_t)stride_order_.size())) {
            TORCH_CHECK(
                stride_order_[i] >= 0 &&
                    stride_order_[i] < (int64_t)reverse_perm.size(),
                "stride_order elements need to be within [0, stride_order.size())!");
            reverse_perm[stride_order_[i]] = i;
            duplicate_check |= 1 << stride_order_[i];
          }
          TORCH_CHECK(
              duplicate_check == (1 << reverse_perm.size()) - 1,
              "duplicated elements in stride_order detected!");
          tv_output = permute(tv_output, reverse_perm);
          fd.addOutput(tv_output, stride_order_);
        } else {
          fd.addOutput(tv_output);
        }
      } else {
        TORCH_CHECK(
            stride_order_.empty(),
            "stride_order can't be dictated for scalar outputs.");
        fd.addOutput(output);
      }
    }
  }

  void print(std::ostream& os, bool close_function = true) const final {
    RecordFunctor::print(os, false);
    if (!stride_order_.empty()) {
      os << ", stride_order=[";
      bool first_arg = true;
      for (auto item : stride_order_) {
        if (first_arg) {
          first_arg = false;
        } else {
          os << ", ";
        }
        os << item;
      }
      os << "]";
    }
    if (close_function) {
      os << ")";
    }
  }

  std::pair<serde::RecordData, flatbuffers::Offset<void>> recordData(
      flatbuffers::FlatBufferBuilder& builder) const final {
    return {
        serde::RecordData_Output,
        serde::CreateOutputDirect(builder, &stride_order_).Union()};
  }

 private:
  //! The tensor dimensions to reduce
  std::vector<int64_t> stride_order_;
};

//! Specialized Record Functor for the FusionState's sum/min/max ops.

struct ReductionOpRecord : RecordFunctor {
  ReductionOpRecord(
      std::vector<State> _args,
      std::vector<State> _outputs,
      std::string _name,
      serde::RecordType record_type,
      std::function<
          TensorView*(TensorView*, const std::vector<int>&, bool, DataType)>
          fusion_op,
      std::vector<int> axes,
      bool keep_dim,
      PrimDataType dtype)
      : RecordFunctor(
            std::move(_args),
            std::move(_outputs),
            _name,
            record_type),
        fusion_op_(std::move(fusion_op)),
        axes_(std::move(axes)),
        keep_dim_(keep_dim),
        dtype_(dtype) {}
  ~ReductionOpRecord() override = default;
  RecordFunctor* clone() final {
    return new ReductionOpRecord(*this);
  }

  //! Child specific hash function in lower 32 bits.
  //! | 31 -- 28 | 27 --- 20 | 19 -----------------  0 |
  //! | keep_dim | Dtype     | Axes Hash               |
  size_t hash() const final {
    auto result = RecordFunctor::hash();
    size_t axes_hash = 0;
    // Normally I would make a little endian hash of the axes but I do not
    // know the size of the tensor based on just the record information.
    for (auto i : c10::irange(axes_.size())) {
      axes_hash |= (1 << axes_[i]);
    }

    return result | (static_cast<size_t>(keep_dim_) << 28) |
        ((static_cast<size_t>(dtype_) & 0xff) << 20) | (axes_hash & 0xfffff);
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto child_ptr = dynamic_cast<const ReductionOpRecord*>(&other)) {
      result = RecordFunctor::operator==(other);
      if (result) {
        result = result &&
            (fusion_op_.target_type() == child_ptr->fusion_op_.target_type());
        if (isDebugDumpEnabled(DebugDumpOption::PythonFrontendDebug)) {
          debug() << "\nReductionOpRecord: " << name_
                  << " Target Type [self: 0x" << fusion_op_.target_type().name()
                  << "] [other: 0x"
                  << child_ptr->fusion_op_.target_type().name() << "]";
        }
        // IMPORTANT! you need to dereference the target pointer in order
        // to match the function
        result = result &&
            (*fusion_op_.template target<

                 TensorView* (*)(TensorView*,
                                 const std::vector<int>&,
                                 bool,
                                 DataType)>() ==
             *child_ptr->fusion_op_.template target<

                 TensorView* (*)(TensorView*,
                                 const std::vector<int>&,
                                 bool,
                                 DataType)>());
        if (isDebugDumpEnabled(DebugDumpOption::PythonFrontendDebug)) {
          debug() << " Target  Ptr [self: 0x" << std::hex
                  << (size_t)*fusion_op_.template target<

                         TensorView* (*)(TensorView*,
                                         const std::vector<int>&,
                                         bool,
                                         DataType)>()
                  << "] [other: 0x" << std::hex
                  << (size_t)*child_ptr->fusion_op_.template target<

                         TensorView* (*)(TensorView*,
                                         const std::vector<int>&,
                                         bool,
                                         DataType)>()
                  << "]\n";
        }
        result = result && (keep_dim_ == child_ptr->keep_dim_);
        result = result && (dtype_ == child_ptr->dtype_);
        if (result) {
          result = (axes_.size() == child_ptr->axes_.size());
          if (result) {
            for (size_t i = 0; i < axes_.size(); ++i) {
              if (axes_[i] != child_ptr->axes_[i]) {
                result = false;
                break;
              }
            }
          }
        }
      }
    }
    return result;
  }

  void operator()(FusionState& fd) final {
    auto arg = fd.getFusionState(args_.at(0).index)->template as<TensorView>();
    auto output = fusion_op_(arg, axes_, keep_dim_, dtype_);
    fd.setFusionState(outputs_.at(0).index, output);
  }

  void print(std::ostream& os, bool close_function = true) const final {
    RecordFunctor::print(os, false);
    os << ", axes=[";
    bool first_arg = true;
    for (auto axis : axes_) {
      if (first_arg) {
        first_arg = false;
      } else {
        os << ", ";
      }
      os << axis;
    }
    os << "]";
    os << ", keepdim=" << (keep_dim_ ? "True" : "False");
    os << ", dtype=" << dtypeToPyString(dtype_);
    if (close_function) {
      os << ")";
    }
  }

  std::pair<serde::RecordData, flatbuffers::Offset<void>> recordData(
      flatbuffers::FlatBufferBuilder& builder) const final {
    // TODO add dtype
    return {
        serde::RecordData_Reduction,
        serde::CreateReductionDirect(
            builder, &axes_, keep_dim_, serde::mapToSerdeDtype(dtype_))
            .Union()};
  }

 private:
  //! nvFuser arith function signature for a given reduction operation
  std::function<
      TensorView*(TensorView*, const std::vector<int>&, bool, DataType)>
      fusion_op_;
  //! The tensor dimensions to reduce
  std::vector<int> axes_;
  //! Indicates whether to keep the reduced dimension(s).
  bool keep_dim_;
  //! The output data type.
  PrimDataType dtype_;
};

struct IndexSelectOpRecord : RecordFunctor {
  IndexSelectOpRecord(
      std::vector<State> _args,
      std::vector<State> _outputs,
      int64_t dim)
      : RecordFunctor(
            std::move(_args),
            std::move(_outputs),
            "ops.index_select",
            serde::RecordType_IndexSelectOp),
        dim_(dim) {}
  ~IndexSelectOpRecord() override = default;
  RecordFunctor* clone() final {
    return new IndexSelectOpRecord(*this);
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto child_ptr = dynamic_cast<const IndexSelectOpRecord*>(&other)) {
      result = RecordFunctor::operator==(other) && dim_ == child_ptr->dim_;
    }
    return result;
  }

  void operator()(FusionState& fd) final {
    auto arg1 = fd.getFusionState(args_.at(0).index)->template as<TensorView>();
    auto arg3 = fd.getFusionState(args_.at(1).index)->template as<TensorView>();

    Val* output = index_select(arg1, (int)dim_, arg3);
    fd.setFusionState(outputs_.at(0).index, output);
  }

  void print(std::ostream& os, bool close_function = true) const final {
    RecordFunctor::print(os, false);
    os << ", dim=" << dim_;
    if (close_function) {
      os << ")";
    }
  }

  std::pair<serde::RecordData, flatbuffers::Offset<void>> recordData(
      flatbuffers::FlatBufferBuilder& builder) const final {
    return {
        serde::RecordData_Dimension,
        serde::CreateDimension(builder, dim_).Union()};
  }

 private:
  //! Dimension to select.
  int64_t dim_;
};

struct TorchGatherOpRecord : RecordFunctor {
  TorchGatherOpRecord(
      std::vector<State> _args,
      std::vector<State> _outputs,
      int64_t dim)
      : RecordFunctor(
            std::move(_args),
            std::move(_outputs),
            "ops.gather",
            serde::RecordType_TorchGatherOp),
        dim_(dim) {}
  ~TorchGatherOpRecord() override = default;
  RecordFunctor* clone() final {
    return new TorchGatherOpRecord(*this);
  }

  void operator()(FusionState& fd) final {
    auto arg1 = fd.getFusionState(args_.at(0).index)->template as<TensorView>();
    auto arg3 = fd.getFusionState(args_.at(1).index)->template as<TensorView>();

    Val* output = torch_gather(arg1, (int)dim_, arg3);
    fd.setFusionState(outputs_.at(0).index, output);
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto child_ptr = dynamic_cast<const TorchGatherOpRecord*>(&other)) {
      result = RecordFunctor::operator==(other) && dim_ == child_ptr->dim_;
    }
    return result;
  }

  void print(std::ostream& os, bool close_function = true) const final {
    RecordFunctor::print(os, false);
    os << ", dim=" << dim_;
    if (close_function) {
      os << ")";
    }
  }

  std::pair<serde::RecordData, flatbuffers::Offset<void>> recordData(
      flatbuffers::FlatBufferBuilder& builder) const final {
    return {
        serde::RecordData_Dimension,
        serde::CreateDimension(builder, dim_).Union()};
  }

 private:
  //! Dimension to select.
  int64_t dim_;
};

//! Similar to TorchGatherOpRecord but enforces that non-index dimension
//! extents match between index tensor and value tensor.
struct TakeAlongAxisOpRecord : RecordFunctor {
  TakeAlongAxisOpRecord(
      std::vector<State> _args,
      std::vector<State> _outputs,
      int64_t dim)
      : RecordFunctor(
            std::move(_args),
            std::move(_outputs),
            "ops.take_along_axis",
            serde::RecordType_TakeAlongAxisOp),
        dim_(dim) {}
  ~TakeAlongAxisOpRecord() override = default;
  RecordFunctor* clone() final {
    return new TakeAlongAxisOpRecord(*this);
  }

  void operator()(FusionState& fd) final {
    auto arg1 = fd.getFusionState(args_.at(0).index)->template as<TensorView>();
    auto arg3 = fd.getFusionState(args_.at(1).index)->template as<TensorView>();

    Val* output = take_along_axis(arg1, arg3, dim_);
    fd.setFusionState(outputs_.at(0).index, output);
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto child_ptr = dynamic_cast<const TakeAlongAxisOpRecord*>(&other)) {
      result = RecordFunctor::operator==(other) && dim_ == child_ptr->dim_;
    }
    return result;
  }

  void print(std::ostream& os, bool close_function = true) const final {
    RecordFunctor::print(os, false);
    os << ", dim=" << dim_;
    if (close_function) {
      os << ")";
    }
  }

  std::pair<serde::RecordData, flatbuffers::Offset<void>> recordData(
      flatbuffers::FlatBufferBuilder& builder) const final {
    return {
        serde::RecordData_Dimension,
        serde::CreateDimension(builder, dim_).Union()};
  }

 private:
  //! Dimension to select.
  int64_t dim_;
};

//! Specialized Record Functor for recording FusionState scalars for both
//! inputs and constants.

struct ScalarRecord : RecordFunctor {
  ScalarRecord(
      std::vector<State> _outputs,
      PolymorphicValue value,
      std::optional<PrimDataType> dtype)
      : RecordFunctor(
            {},
            std::move(_outputs),
            "define_scalar",
            serde::RecordType_Scalar),
        value_(
            dtype.has_value() ? castToDtype(std::move(value), dtype.value())
                              : std::move(value)),
        dtype_(
            dtype.has_value()
                ? dtype.value()
                : std::get<PrimDataType>(getDataType(value_).type)) {}
  ~ScalarRecord() override = default;
  RecordFunctor* clone() final {
    return new ScalarRecord(*this);
  }

  //! Child specific hash function in lower 32 bits.
  //! | 31 ---------------------------------------  0 |
  //! | Dtype                                         |
  size_t hash() const final {
    auto result = RecordFunctor::hash();
    return result | (static_cast<size_t>(dtype_) & 0xffffffff);
  }

  bool operator==(const RecordFunctor& other) const final {
    if (auto child_ptr = dynamic_cast<const ScalarRecord*>(&other)) {
      if (RecordFunctor::operator==(other)) {
        if (value_.hasValue() != child_ptr->value_.hasValue() ||
            dtype_ != child_ptr->dtype_) {
          return false;
        }
        if (value_.hasValue()) {
          if (value_.is<double>() && std::isnan(value_.as<double>()) &&
              std::isnan(child_ptr->value_.as<double>())) {
            return true;
          } else {
            return value_ == child_ptr->value_;
          }
        } else {
          return true;
        }
      }
    }
    return false;
  }

  void operator()(FusionState& fd) final {
    Val* output = IrBuilder::create<nvfuser::Val>(value_, dtype_);
    if (!value_.hasValue()) {
      fd.addInput(output);
    }
    fd.setFusionState(outputs_.at(0).index, output);
  }

  void print(std::ostream& os, bool close_function = true) const final {
    RecordFunctor::print(os, false);
    if (value_.hasValue()) {
      if (value_.is<bool>()) {
        os << ((bool)value_ ? "True" : "False");
      } else if (value_.is<std::complex<double>>()) {
        os << std::showpoint << std::real(value_.as<std::complex<double>>())
           << "+" << std::showpoint
           << std::imag(value_.as<std::complex<double>>()) << "j";
      } else if (value_.is<double>()) {
        if (std::isinf(value_.as<double>())) {
          if (std::signbit(value_.as<double>())) {
            os << "float(\"-inf\")";
          } else {
            os << "float(\"inf\")";
          }
        } else if (std::isnan(value_.as<double>())) {
          os << "float(\"nan\")";
        } else {
          os << std::showpoint << value_.as<double>();
        }
      } else if (value_.is<int64_t>()) {
        os << value_;
      } else {
        TORCH_CHECK(false, "Unsupported dtype.");
      }
    } else {
      os << "None";
    }

    os << ", dtype=" << dtypeToPyString(dtype_);

    if (close_function) {
      os << ")";
    }
  }

  std::pair<serde::RecordData, flatbuffers::Offset<void>> recordData(
      flatbuffers::FlatBufferBuilder& builder) const final {
    return {
        serde::RecordData_Scalar,
        serde::serializeScalar(builder, value_, dtype_).Union()};
  }

  inline std::pair<serde::RecordData, flatbuffers::Offset<void>> valueRecordData(
      flatbuffers::FlatBufferBuilder& builder,
      PolymorphicValue value) const;

 private:
  //! The scalar's value, an input is a nullopt
  PolymorphicValue value_;
  //! Scalar data type.
  PrimDataType dtype_;
};

struct SliceOpRecord : RecordFunctor {
  SliceOpRecord(
      std::vector<State> _args,
      std::vector<State> _outputs,
      std::vector<int64_t> start_indices,
      std::vector<int64_t> end_indices,
      std::vector<int64_t> strides)
      : RecordFunctor(
            std::move(_args),
            std::move(_outputs),
            "ops.slice",
            serde::RecordType_SliceOp),
        start_indices_(std::move(start_indices)),
        end_indices_(std::move(end_indices)),
        strides_(std::move(strides)) {}
  ~SliceOpRecord() override = default;
  RecordFunctor* clone() final {
    return new SliceOpRecord(*this);
  }

  //! Child specific hash function in lower 32 bits.
  //! | 31 -------- 20 | 19 --------  8 |  7 ------  0 |
  //! | start_indices  | end_indices    | strides      |
  size_t hash() const final {
    auto result = RecordFunctor::hash();
    size_t start_idx_hash = 0;
    for (auto i : start_indices_) {
      start_idx_hash ^= static_cast<size_t>(i);
    }
    size_t end_idx_hash = 0;
    for (auto i : end_indices_) {
      end_idx_hash ^= static_cast<size_t>(i);
    }
    size_t stride_hash = 0;
    for (auto i : strides_) {
      stride_hash ^= static_cast<size_t>(i);
    }

    result |= (start_idx_hash & 0xfff) << 20;
    result |= (end_idx_hash & 0xfff) << 8;
    return result | (stride_hash & 0xff);
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto child_ptr = dynamic_cast<const SliceOpRecord*>(&other)) {
      result = RecordFunctor::operator==(other) &&
          (start_indices_ == child_ptr->start_indices_) &&
          (end_indices_ == child_ptr->end_indices_) &&
          (strides_ == child_ptr->strides_);
    }
    return result;
  }

  void operator()(FusionState& fd) final {
    auto ndims = start_indices_.size();
    std::vector<Slice> ranges;
    ranges.reserve(ndims);
    for (const auto i : c10::irange(ndims)) {
      Slice tmp;
      tmp.start = IrBuilder::create<nvfuser::Val>(start_indices_[i]);
      tmp.stop = IrBuilder::create<nvfuser::Val>(end_indices_[i]);
      tmp.step = IrBuilder::create<nvfuser::Val>(strides_[i]);
      ranges.emplace_back(tmp);
    }

    auto arg = fd.getFusionState(args_.at(0).index)->as<TensorView>();
    auto output = slice(arg, ranges);
    fd.setFusionState(outputs_.at(0).index, output);
  }

  void print(std::ostream& os, bool close_function = true) const final {
    RecordFunctor::print(os, false);
    os << ", start_indices=[";
    bool first_arg = true;
    for (auto idx : start_indices_) {
      if (first_arg) {
        first_arg = false;
      } else {
        os << ", ";
      }
      os << idx;
    }
    os << "], end_indices=[";
    first_arg = true;
    for (auto idx : end_indices_) {
      if (first_arg) {
        first_arg = false;
      } else {
        os << ", ";
      }
      os << idx;
    }
    os << "], strides=[";
    first_arg = true;
    for (auto stride : strides_) {
      if (first_arg) {
        first_arg = false;
      } else {
        os << ", ";
      }
      os << stride;
    }
    os << "]";
    if (close_function) {
      os << ")";
    }
  }

  std::pair<serde::RecordData, flatbuffers::Offset<void>> recordData(
      flatbuffers::FlatBufferBuilder& builder) const final {
    return {
        serde::RecordData_Slice,
        serde::CreateSliceDirect(
            builder, &start_indices_, &end_indices_, &strides_)
            .Union()};
  }

 private:
  //! A slices beginning index for each dimension
  //! Values must be greater-than or equal to 0
  std::vector<int64_t> start_indices_;
  //! A slices end index for each dimension (excluded from the slice)
  //! Values are greater than or equal to the start index for a dimension
  std::vector<int64_t> end_indices_;
  //! For a dim, the step between start and end.
  //! NOTE: Strides are currently limited to steps of 1
  std::vector<int64_t> strides_;
};

//! Specialized Record Functor for recording FusionDefinition Start.
//! There should only ever be one instance of this Record in the
//! Fusion Cache.

struct StartRecord : RecordFunctor {
  StartRecord() : RecordFunctor({}, {}, "start", serde::RecordType_Start) {}
  ~StartRecord() override = default;
  RecordFunctor* clone() final {
    return new StartRecord(*this);
  }

  //! Child specific hash function in lower 32 bits.
  //! | 31 ---------------------------------------  0 |
  //! | None                                          |
  size_t hash() const final {
    return RecordFunctor::hash();
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (dynamic_cast<const StartRecord*>(&other)) {
      result = RecordFunctor::operator==(other);
    }
    return result;
  }

  void operator()(FusionState& fd) final {}
};

//! Specialized Record Functors for Normalization based ops.

struct NormOpRecord : RecordFunctor {
  NormOpRecord(
      std::vector<State> args,
      std::vector<State> outputs,
      std::string name,
      serde::RecordType type,
      std::vector<int> axes,
      int64_t correction,
      bool keep_dim)
      : RecordFunctor(std::move(args), std::move(outputs), name, type),
        axes_(std::move(axes)),
        correction_(correction),
        keep_dim_(keep_dim) {}
  ~NormOpRecord() override = default;
  RecordFunctor* clone() override = 0;

  // I am skipping the bassel's correction value in the hash because
  // I suspect we might change it to a bool from a 64-bit value
  //! Child specific hash function in lower 32 bits.
  //! | 31 -- 28 | 27 -----------------------------  0 |
  //! | keep_dim | Axes Hash                           |
  size_t hash() const final {
    auto result = RecordFunctor::hash();
    size_t axes_hash = 0;
    // Normally I would make a little endian hash of the axes but I do not
    // know the size of the tensor based on just the record information.
    for (auto i : c10::irange(axes_.size())) {
      axes_hash |= (1 << axes_[i]);
    }
    return result | (static_cast<size_t>(keep_dim_) << 28) |
        (axes_hash & 0xfffffff);
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto child_ptr = dynamic_cast<const NormOpRecord*>(&other)) {
      result = RecordFunctor::operator==(other);
      result = result && (correction_ == child_ptr->correction_);
      result = result && (keep_dim_ == child_ptr->keep_dim_);
      if (result) {
        result = (axes_.size() == child_ptr->axes_.size());
        if (result) {
          for (size_t i = 0; i < axes_.size(); ++i) {
            if (axes_[i] != child_ptr->axes_[i]) {
              result = false;
              break;
            }
          }
        }
      }
    }
    return result;
  }

  //! Each NormOp Child should define the operator() to build the IR
  void operator()(FusionState& fd) override = 0;

  void print(std::ostream& os, bool close_function = true) const final {
    RecordFunctor::print(os, false);
    os << ", axes=[";
    bool first_arg = true;
    for (auto axis : axes_) {
      if (first_arg) {
        first_arg = false;
      } else {
        os << ", ";
      }
      os << axis;
    }
    os << "]";
    os << ", correction=" << correction_;
    os << ", keepdim=" << (keep_dim_ ? "True" : "False");
    if (close_function) {
      os << ")";
    }
  }

  std::pair<serde::RecordData, flatbuffers::Offset<void>> recordData(
      flatbuffers::FlatBufferBuilder& builder) const final {
    return {
        serde::RecordData_Norm,
        serde::CreateNormDirect(builder, &axes_, correction_, keep_dim_)
            .Union()};
  }

 protected:
  //! Dimensions of tensor to reduce for variance calculation
  std::vector<int> axes_;
  //! Bessel's correction value
  int64_t correction_;
  //! Indicates whether to keep the reduced dimension(s).
  bool keep_dim_;
};

struct VarianceOpRecord : NormOpRecord {
  VarianceOpRecord(
      std::vector<State> args,
      std::vector<State> outputs,
      std::vector<int> axes,
      int64_t correction,
      bool keep_dim)
      : NormOpRecord(
            std::move(args),
            std::move(outputs),
            "ops.var",
            serde::RecordType_VarianceOp,
            std::move(axes),
            correction,
            keep_dim) {}
  ~VarianceOpRecord() override = default;
  RecordFunctor* clone() final {
    return new VarianceOpRecord(*this);
  }

  void operator()(FusionState& fd) final {
    auto arg = fd.getFusionState(args_.at(0).index)->as<TensorView>();
    auto output = variance(arg, axes_, correction_, keep_dim_);
    fd.setFusionState(outputs_.at(0).index, output);
  }
};

//! VarianceMean requires a separate Record because nvFuser defines the output
//! of var_mean as a custom struct.
struct VarianceMeanOpRecord : NormOpRecord {
  VarianceMeanOpRecord(
      std::vector<State> args,
      std::vector<State> outputs,
      std::vector<int> axes,
      int64_t correction,
      bool keep_dim)
      : NormOpRecord(
            std::move(args),
            std::move(outputs),
            "ops.var_mean",
            serde::RecordType_VarianceMeanOp,
            std::move(axes),
            correction,
            keep_dim) {}
  ~VarianceMeanOpRecord() override = default;
  RecordFunctor* clone() final {
    return new VarianceMeanOpRecord(*this);
  }

  void operator()(FusionState& fd) final {
    auto arg = fd.getFusionState(args_.at(0).index)->as<TensorView>();
    auto output = variance_mean(arg, axes_, correction_, keep_dim_);
    fd.setFusionState(outputs_.at(0).index, output.var);
    fd.setFusionState(outputs_.at(1).index, output.mean);
  }
};

struct BatchNormOpRecord : RecordFunctor {
  BatchNormOpRecord(
      std::vector<State> args,
      std::vector<State> outputs,
      bool training,
      bool channels_last)
      : RecordFunctor(
            std::move(args),
            std::move(outputs),
            "ops.batch_norm",
            serde::RecordType_BatchNormOp),
        training_(training),
        channels_last_(channels_last) {}
  ~BatchNormOpRecord() override = default;
  RecordFunctor* clone() final {
    return new BatchNormOpRecord(*this);
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto child_ptr = dynamic_cast<const BatchNormOpRecord*>(&other)) {
      result = RecordFunctor::operator==(other);
      result = result && (training_ == child_ptr->training_);
      result = result && (channels_last_ == child_ptr->channels_last_);
    }
    return result;
  }

  size_t hash() const final {
    auto result = RecordFunctor::hash();
    return result | (static_cast<size_t>(training_) << 28) |
        (static_cast<size_t>(channels_last_) << 29);
  }

  void operator()(FusionState& fd) final {
    auto x = fd.getFusionState(args_.at(0).index)->as<TensorView>();
    auto weight = (args_.at(1).stype == serde::StateType_Tensor)
        ? fd.getFusionState(args_.at(1).index)->as<TensorView>()
        : nullptr;
    auto bias = (args_.at(2).stype == serde::StateType_Tensor)
        ? fd.getFusionState(args_.at(2).index)->as<TensorView>()
        : nullptr;
    auto running_mean = (args_.at(3).stype == serde::StateType_Tensor)
        ? fd.getFusionState(args_.at(3).index)->as<TensorView>()
        : nullptr;
    auto running_var = (args_.at(4).stype == serde::StateType_Tensor)
        ? fd.getFusionState(args_.at(4).index)->as<TensorView>()
        : nullptr;
    auto momentum = fd.getFusionState(args_.at(5).index)->as<Val>();
    auto eps = fd.getFusionState(args_.at(6).index)->as<Val>();
    auto output = batch_norm(
        x,
        weight,
        bias,
        running_mean,
        running_var,
        training_,
        momentum,
        eps,
        channels_last_);
    fd.setFusionState(outputs_.at(0).index, output.output);
    fd.setFusionState(outputs_.at(1).index, output.mean);
    fd.setFusionState(outputs_.at(2).index, output.invstd);
  }

  void print(std::ostream& os, bool close_function = true) const final {
    RecordFunctor::print(os, false);
    os << ", training=" << (training_ ? "True" : "False");
    os << ", channels_last=" << (channels_last_ ? "True" : "False");
    if (close_function) {
      os << ")";
    }
  }

  std::pair<serde::RecordData, flatbuffers::Offset<void>> recordData(
      flatbuffers::FlatBufferBuilder& builder) const final {
    return {
        serde::RecordData_BatchNorm,
        serde::CreateBatchNorm(builder, training_, channels_last_).Union()};
  }

 private:
  bool training_;
  bool channels_last_;
};

//! Specialized Record Functor for the FusionState's tensor_size op.
//! Uses the default hash() and print() methods of Record Functor

struct TensorSizesRecord : RecordFunctor {
  TensorSizesRecord(std::vector<State> args, std::vector<State> outputs)
      : RecordFunctor(
            std::move(args),
            std::move(outputs),
            "ops.tensor_sizes",
            serde::RecordType_TensorSizes) {
    always_returns_tuple_ = true;
  }
  ~TensorSizesRecord() override = default;
  RecordFunctor* clone() final {
    return new TensorSizesRecord(*this);
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (dynamic_cast<const TensorSizesRecord*>(&other)) {
      result = RecordFunctor::operator==(other);
    }
    return result;
  }

  void operator()(FusionState& fd) final {
    auto arg = fd.getFusionState(args_.at(0).index)->as<TensorView>();
    auto sizes = tensor_sizes(arg);
    for (const auto idx : c10::irange(sizes.size())) {
      fd.setFusionState(outputs_.at(idx).index, sizes[idx]);
    }
  }
};

struct FullOpRecord : RecordFunctor {
  FullOpRecord(
      std::vector<State> _args,
      std::vector<State> _outputs,
      std::vector<int64_t> shape,
      PrimDataType dtype)
      : RecordFunctor(
            std::move(_args),
            std::move(_outputs),
            "ops.full",
            serde::RecordType_FullOp),
        shape_(std::move(shape)),
        dtype_(dtype) {}
  ~FullOpRecord() override = default;
  RecordFunctor* clone() final {
    return new FullOpRecord(*this);
  }

  //! Child specific hash function in lower 32 bits.
  //! | 31 --- 24 | 23 --------------------------  0 |
  //! | Dtype     | Shape hash code                  |
  size_t hash() const final {
    auto result = RecordFunctor::hash();
    size_t shape_hash = 0;
    for (auto p : shape_) {
      shape_hash ^= static_cast<size_t>(p);
    }
    result |= ((static_cast<size_t>(dtype_) & 0xff) << 24);
    result |= (shape_hash & 0xffff);
    return result;
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto child_ptr = dynamic_cast<const FullOpRecord*>(&other)) {
      result = RecordFunctor::operator==(other) &&
          shape_ == child_ptr->shape_ && dtype_ == child_ptr->dtype_;
    }
    return result;
  }

  void operator()(FusionState& fd) final {
    auto arg = fd.getFusionState(args_.at(0).index);

    std::vector<Val*> nvf_shape(shape_.size(), nullptr);
    for (const auto idx : c10::irange(shape_.size())) {
      nvf_shape[idx] = IrBuilder::create<nvfuser::Val>(shape_.at(idx));
    }
    auto output = full(nvf_shape, arg, dtype_);
    fd.setFusionState(outputs_.at(0).index, output);
  }

  void print(std::ostream& os, bool close_function = true) const override {
    RecordFunctor::print(os, false);
    os << ", shape=[";
    bool first_arg = true;
    for (auto p : shape_) {
      if (first_arg) {
        first_arg = false;
      } else {
        os << ", ";
      }
      os << p;
    }
    os << "]";
    os << ", dtype=" << dtypeToPyString(dtype_);
    if (close_function) {
      os << ")";
    }
  }

  std::pair<serde::RecordData, flatbuffers::Offset<void>> recordData(
      flatbuffers::FlatBufferBuilder& builder) const final {
    return {
        serde::RecordData_TensorCreation,
        serde::CreateTensorCreationDirect(
            builder, &shape_, serde::mapToSerdeDtype(dtype_))
            .Union()};
  }

 private:
  //! Represents shape of new tensor
  std::vector<int64_t> shape_;
  //! Type of output
  PrimDataType dtype_;
};

struct IotaOpRecord : RecordFunctor {
  IotaOpRecord(
      std::vector<State> _args,
      std::vector<State> _outputs,
      PrimDataType dtype)
      : RecordFunctor(
            std::move(_args),
            std::move(_outputs),
            "ops.iota",
            serde::RecordType_IotaOp),
        dtype_(dtype) {}
  ~IotaOpRecord() override = default;
  RecordFunctor* clone() final {
    return new IotaOpRecord(*this);
  }

  //! Child specific hash function in lower 32 bits.
  //! | 31 --------------------------------------  0 |
  //! | Dtype                                        |
  size_t hash() const final {
    return RecordFunctor::hash() | static_cast<uint32_t>(dtype_);
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto child_ptr = dynamic_cast<const IotaOpRecord*>(&other)) {
      result = RecordFunctor::operator==(other) && dtype_ == child_ptr->dtype_;
    }
    return result;
  }

  void operator()(FusionState& fd) final {
    auto length = fd.getFusionState(args_.at(0).index);
    auto start = (args_.at(1).stype == serde::StateType_Scalar)
        ? fd.getFusionState(args_.at(1).index)->as<Val>()
        : nullptr;
    auto step = (args_.at(2).stype == serde::StateType_Scalar)
        ? fd.getFusionState(args_.at(2).index)->as<Val>()
        : nullptr;
    auto output = iota(length, start, step, dtype_);
    fd.setFusionState(outputs_.at(0).index, output);
  }

  void print(std::ostream& os, bool close_function = true) const override {
    RecordFunctor::print(os, false);
    os << ", dtype=" << dtypeToPyString(dtype_);
    if (close_function) {
      os << ")";
    }
  }

  std::pair<serde::RecordData, flatbuffers::Offset<void>> recordData(
      flatbuffers::FlatBufferBuilder& builder) const final {
    return {
        serde::RecordData_Dtype,
        serde::CreateDtype(builder, serde::mapToSerdeDtype(dtype_)).Union()};
  }

 private:
  //! Type of output
  PrimDataType dtype_;
};

//! Specialized Record Functors for random ops.
struct RandomOpRecord : RecordFunctor {
  RandomOpRecord(
      std::vector<State> _args,
      std::vector<State> _outputs,
      std::vector<State> output_shape,
      std::string _name,
      PrimDataType dtype)
      : RecordFunctor(
            std::move(_args),
            std::move(_outputs),
            _name,
            serde::RecordType_RandomOp),
        output_shape_(std::move(output_shape)),
        dtype_(dtype) {
    if (args_.size() == 4) {
      // seed and offset were provided in addition to the usual 2 arguments
      setArgName(2, "rng_seed");
      setArgName(3, "rng_offset");
    }
  }
  ~RandomOpRecord() override = default;
  RecordFunctor* clone() final {
    return new RandomOpRecord(*this);
  }

  //! Child specific hash function in lower 32 bits.
  //! | 31 -------------- 16 | 15 --------------  0 |
  //! |   distribution hash  | output_shape hash    |
  size_t hash() const final {
    auto result = RecordFunctor::hash();
    return result | (output_shape_.size() & 0xffff) |
        (std::hash<std::string>{}(name_.c_str()) & 0xffff << 16);
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto child_ptr = dynamic_cast<const RandomOpRecord*>(&other)) {
      result = RecordFunctor::operator==(other);
      if (result) {
        result = (output_shape_.size() == child_ptr->output_shape_.size());
        if (result) {
          for (size_t i = 0; i < output_shape_.size(); ++i) {
            if (output_shape_[i] != child_ptr->output_shape_[i]) {
              result = false;
              break;
            }
          }
        }
      }
    }
    return result;
  }

  void operator()(FusionState& fd) final {
    auto arg1 = fd.getFusionState(args_.at(0).index);
    auto arg2 = fd.getFusionState(args_.at(1).index);

    std::vector<Val*> output_shape(output_shape_.size(), nullptr);
    std::transform(
        output_shape_.begin(),
        output_shape_.end(),
        output_shape.begin(),
        [&fd](const State& state) {
          return fd.getFusionState(state.index)->template as<Val>();
        });
    Val* output = nullptr;
    if (name_.compare("ops.uniform") == 0) {
      if (args_.size() == 2) { // stochastic uniform
        output = uniform(output_shape, arg1, arg2, dtype_);
      } else if (args_.size() == 4) { // provided seed and offset
        auto seed = fd.getFusionState(args_.at(2).index);
        auto offset = fd.getFusionState(args_.at(3).index);
        output = uniform(output_shape, arg1, arg2, dtype_, seed, offset);
      }
    } else if (name_.compare("ops.normal") == 0) {
      if (args_.size() == 2) { // stochastic normal
        output = normal(output_shape, arg1, arg2, dtype_);
      } else if (args_.size() == 4) { // provided seed and offset
        auto seed = fd.getFusionState(args_.at(2).index);
        auto offset = fd.getFusionState(args_.at(3).index);
        output = normal(output_shape, arg1, arg2, dtype_, seed, offset);
      }
    } else {
      TORCH_INTERNAL_ASSERT(
          false, "random distribution not recognized:", name_);
    }
    fd.setFusionState(outputs_.at(0).index, output);
  }

  void print(std::ostream& os, bool close_function = true) const final {
    RecordFunctor::print(os, false);
    os << ", shape=[";
    bool first_arg = true;
    for (auto shape : output_shape_) {
      if (first_arg) {
        first_arg = false;
      } else {
        os << ", ";
      }
      os << shape;
    }
    os << "]";
    os << ", dtype=" << dtypeToPyString(dtype_);
    if (close_function) {
      os << ")";
    }
  }

  std::pair<serde::RecordData, flatbuffers::Offset<void>> recordData(
      flatbuffers::FlatBufferBuilder& builder) const final {
    std::vector<serde::State> fb_shape;
    fb_shape.reserve(output_shape_.size());
    for (auto& it : output_shape_) {
      fb_shape.emplace_back(it.index, it.stype);
    }
    return {
        serde::RecordData_TensorCreationSymbolic,
        serde::CreateTensorCreationSymbolicDirect(
            builder, &fb_shape, serde::mapToSerdeDtype(dtype_))
            .Union()};
  }

 private:
  //! Represents the tensor dimensions of the output tensor.
  std::vector<State> output_shape_;
  //! DataType of output
  PrimDataType dtype_;
};

//! Specialized Record Functor for recording Vector of Scalars

struct VectorRecord : RecordFunctor {
  VectorRecord(
      std::vector<State> _args,
      std::vector<State> _outputs,
      PrimDataType dtype)
      : RecordFunctor(
            std::move(_args),
            std::move(_outputs),
            "define_vector",
            serde::RecordType_Vector),
        dtype_(dtype) {}
  ~VectorRecord() override = default;
  RecordFunctor* clone() final {
    return new VectorRecord(*this);
  }

  //! Child specific hash function in lower 32 bits.
  //! | 31 ---------------------------------------  0 |
  //! | Dtype                                         |
  size_t hash() const final {
    auto result = RecordFunctor::hash();
    return result | (static_cast<size_t>(dtype_) & 0xffffffff);
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto child_ptr = dynamic_cast<const VectorRecord*>(&other)) {
      result = RecordFunctor::operator==(other);
      result = result && (dtype_ == child_ptr->dtype_);
    }
    return result;
  }

  void operator()(FusionState& fd) final {
    std::vector<Val*> output(args_.size(), nullptr);
    TORCH_CHECK(
        dtype_ == DataType::Int,
        "Only Int Dtype is not supported by a vector of sizes: ",
        dtype_);
    for (size_t i = 0; i < args_.size(); ++i) {
      TORCH_CHECK(
          args_.at(i).stype == serde::StateType_Scalar,
          "Unsupported State type!");
      output.at(i) = fd.getFusionState(args_.at(i).index);
    }
    fd.setFusionStateVector(outputs_.at(0).index, output);
  }

  void print(std::ostream& os, bool close_function = true) const final {
    bool first_output = true;
    for (auto& output : outputs_) {
      if (first_output) {
        first_output = false;
      } else {
        os << ", ";
      }
      os << output;
    }
    os << " = fd." << name_ << "([";
    bool first_arg = true;
    for (auto& arg : args_) {
      if (first_arg) {
        first_arg = false;
      } else {
        os << ", ";
      }
      os << arg;
    }
    os << "], dtype=" << dtypeToPyString(dtype_);
    if (close_function) {
      os << ")";
    }
  }

  std::pair<serde::RecordData, flatbuffers::Offset<void>> recordData(
      flatbuffers::FlatBufferBuilder& builder) const final {
    return {
        serde::RecordData_Vector,
        serde::CreateVector(builder, serde::mapToSerdeDtype(dtype_)).Union()};
  };

 private:
  //! Scalar data type.
  PrimDataType dtype_;
};

} // namespace nvfuser::python_frontend

//! Creating the template specialized hash and equal_to functions for a
//! RecordFunctor object in order to use hash maps (unordered_maps) in STL.
namespace std {
using namespace nvfuser::python_frontend;

template <>
struct hash<RecordFunctor*> {
  size_t operator()(const RecordFunctor* p) const {
    TORCH_CHECK(p, "The RecordFunctor Pointer for hashing is null!");
    return p->hash();
  }
};
template <>
struct equal_to<RecordFunctor*> {
  bool operator()(const RecordFunctor* p, const RecordFunctor* q) const {
    TORCH_CHECK(
        p,
        "The RecordFunctor Pointer on the lhs of an equality check is null!");
    TORCH_CHECK(
        q,
        "The RecordFunctor Pointer on the rhs of an equality check is null!");
    return p->operator==(*q);
  }
};
} // namespace std
