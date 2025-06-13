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
#include <exceptions.h>
#include <ir/interface_nodes.h>
#include <ops/all_ops.h>
#include <options.h>
#include <python_frontend/fusion_definition.h>
#include <python_frontend/fusion_state.h>
#include <python_utils.h>
#include <serde/fusion_cache_generated.h>
#include <serde/polymorphic_value.h>
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
      serde::RecordType _record_type,
      bool _inline_def = false)
      : args_(std::move(_args)),
        arg_names_(args_.size()),
        outputs_(std::move(_outputs)),
        name_(std::move(_name)),
        record_type_(_record_type),
        inline_def_(
            _inline_def &&
            !isOptionDisabled(DisableOption::PythonInlineDefinitions)) {
    // Set this Record as the parent of each output
    if (inline_def_) {
      for (auto& out : outputs_) {
        out.setInlineDefRecord(this);
      }
    }
  }
  RecordFunctor(const RecordFunctor& other)
      : args_(other.args_),
        arg_names_(other.arg_names_),
        outputs_(other.outputs_),
        name_(other.name_),
        record_type_(other.record_type_),
        inline_def_(other.inline_def_) {
    // Set this Record as the parent of each output
    if (inline_def_) {
      for (auto& out : outputs_) {
        out.setInlineDefRecord(this);
      }
    }
  }
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
    // NOTE: The inline_def is not part of the hash as it is not used for
    // comparison
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
    // NOTE: The inline_def is not part of the equality operator as it is not
    // used for comparison
    return result;
  }

  //! Abstraction for an operation to build this record's nvFuser Fusion IR
  //! piece if the recording has a cache miss.
  virtual void operator()(FusionState& fd) = 0;

  //! Abstraction for storing data specific to a record functor.
  virtual std::pair<serde::RecordData, flatbuffers::Offset<void>> recordData(
      flatbuffers::FlatBufferBuilder& builder) const {
    return {serde::RecordData::NONE, flatbuffers::Offset<void>()};
  }

  //! The base serialize function that handles args, outputs, name and
  //! recordType. Child recordFunctors should overload the recordData function
  //! if has supplementary attributes.
  virtual flatbuffers::Offset<serde::RecordFunctor> serialize(
      flatbuffers::FlatBufferBuilder& builder) const {
    // See table definition for RecordFunctor in serde/fusion_cache.fbs

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
    NVF_ERROR(
        !inline_def_,
        "The default print function does not handle inline definitions!");
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

  const std::vector<State>& outputs() const {
    return outputs_;
  }
  std::vector<State>& args() {
    return args_;
  }

  serde::RecordType recordType() const {
    return record_type_;
  }

  bool inlineDef() const {
    return inline_def_;
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
  //! Indicates if a record was defined inline with another record for printing
  bool inline_def_;
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

struct SliceOpRecord : RecordFunctor {
  SliceOpRecord(
      std::vector<State> _args,
      std::vector<State> _outputs,
      bool manual_normalization)
      : RecordFunctor(
            std::move(_args),
            std::move(_outputs),
            "ops.slice",
            serde::RecordType::SliceOp),
        manual_normalization_(manual_normalization) {
    arg_names_[1] = "start_indices";
    arg_names_[2] = "end_indices";
    arg_names_[3] = "strides";
  }
  ~SliceOpRecord() override = default;
  RecordFunctor* clone() final {
    return new SliceOpRecord(*this);
  }

  //! Child specific hash function in lower 32 bits.
  //! |       31              | 30  ------------------------  0 |
  //! | manual_normalization? |              other              |
  size_t hash() const final {
    auto result = RecordFunctor::hash();
    result |= ((static_cast<size_t>(manual_normalization_) & 0x1) << 31);
    return result;
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto child_ptr = dynamic_cast<const SliceOpRecord*>(&other)) {
      result = RecordFunctor::operator==(other);
      result =
          result && (manual_normalization_ == child_ptr->manual_normalization_);
    }
    return result;
  }

  void operator()(FusionState& fd) final {
    TensorView* arg = fd.getFusionState(args_.at(0).index)->as<TensorView>();
    const std::vector<Val*>& start = fd.getFusionStateVector(args_.at(1).index);
    const std::vector<Val*>& end = fd.getFusionStateVector(args_.at(2).index);
    const std::vector<Val*>& stride =
        fd.getFusionStateVector(args_.at(3).index);
    std::vector<Slice> vec_slice;
    for (const auto idx : arange(arg->domain()->noReductions().size())) {
      // NOTE: there's an extra move, we can use emplace_back if we go write
      // some constructors for Slice.
      Val* start_idx = start.at(idx);
      Val* end_idx = end.at(idx);
      Val* stride_idx = stride.at(idx);
      NVF_CHECK(
          !start_idx->isConstInt() || start_idx->evaluate().as<int64_t>() >= 0,
          "Slice operation start_indices must be greater than or equal to 0. "
          "Start Indices: ",
          start_idx->evaluate().as<int64_t>());
      NVF_CHECK(
          !start_idx->isConstInt() || !end_idx->isConstInt() ||
              end_idx->evaluate().as<int64_t>() >=
                  start_idx->evaluate().as<int64_t>(),
          "Slice operation end_indices must be greater than or equal to "
          "start_indices. Start Indices: ",
          start_idx->evaluate().as<int64_t>(),
          " End Indices: ",
          end_idx->evaluate().as<int64_t>());
      NVF_CHECK(
          stride_idx->isConstInt() && stride_idx->evaluate().as<int64_t>() == 1,
          "nvFuser Limitation: All slice operation strides must be of const "
          "size 1.");
      vec_slice.push_back({start_idx, end_idx, stride_idx});
    }
    auto output = slice(arg, vec_slice, manual_normalization_);
    fd.setFusionState(outputs_.at(0).index, output);
  }

  void print(std::ostream& os, bool close_function = true) const final {
    RecordFunctor::print(os, false);
    os << ", manual_normalization=" << manual_normalization_;
    if (close_function) {
      os << ")";
    }
  }

  std::pair<serde::RecordData, flatbuffers::Offset<void>> recordData(
      flatbuffers::FlatBufferBuilder& builder) const final {
    return {
        serde::RecordData::Slice,
        serde::CreateSlice(builder, manual_normalization_).Union()};
  }

 private:
  //! A flag to skip slice normalization step in composite operation.
  bool manual_normalization_;
};

struct ReshapeOpRecord : RecordFunctor {
  ReshapeOpRecord(std::vector<State> _args, std::vector<State> _outputs)
      : RecordFunctor(
            std::move(_args),
            std::move(_outputs),
            "ops.reshape",
            serde::RecordType::ReshapeOp) {
    arg_names_[1] = "new_shape";
  }
  ~ReshapeOpRecord() override = default;
  RecordFunctor* clone() final {
    return new ReshapeOpRecord(*this);
  }

  void operator()(FusionState& fd) final {
    TensorView* arg = fd.getFusionState(args_.at(0).index)->as<TensorView>();
    const std::vector<Val*>& new_shape =
        fd.getFusionStateVector(args_.at(1).index);
    auto output = reshape(arg, new_shape);
    fd.setFusionState(outputs_.at(0).index, output);
  }
};

struct PadOpRecord : RecordFunctor {
  PadOpRecord(std::vector<State> _args, std::vector<State> _outputs)
      : RecordFunctor(
            std::move(_args),
            std::move(_outputs),
            "ops.pad",
            serde::RecordType::PadOp) {}
  ~PadOpRecord() override = default;
  RecordFunctor* clone() final {
    return new PadOpRecord(*this);
  }

  void operator()(FusionState& fd) final {
    auto arg = fd.getFusionState(args_.at(0).index)->template as<TensorView>();
    const std::vector<Val*>& val_widths =
        fd.getFusionStateVector(args_.at(1).index);

    TensorView* output = nullptr;
    if (args_.at(2).stype == serde::StateType::Scalar) {
      output = pad(arg, val_widths, fd.getFusionState(args_.at(2).index));
    } else { // default: None
      NVF_ERROR(args_.at(2).stype == serde::StateType::None);
      output = pad(arg, val_widths);
    }

    fd.setFusionState(outputs_.at(0).index, output);
  }
};

template <serde::RecordType op_type>
struct DimsOpRecord : RecordFunctor {
  DimsOpRecord(
      std::vector<State> _args,
      std::vector<State> _outputs,
      std::vector<int64_t> dims,
      std::string name)
      : RecordFunctor(std::move(_args), std::move(_outputs), name, op_type) {
    int64_t rank = (int64_t)dims.size();
    dims_.reserve(rank);
    std::unordered_set<int64_t> dims_set;
    for (auto dim : dims) {
      dims_set.insert(dim);
      if (dim < 0) {
        NVF_CHECK(
            dim >= -rank,
            name + " dims argument is out of range, expects >= -" +
                std::to_string(rank) + ", but got: " + std::to_string(dim));
        dim += rank;
      } else {
        NVF_CHECK(
            dim < rank,
            name + " dims argument is out of range, expects < " +
                std::to_string(rank) + ", but got: " + std::to_string(dim));
      }
      dims_.push_back(dim);
    }
    NVF_CHECK(
        dims_set.size() == dims.size(),
        name + " got duplicated dimension entries: " + toDelimitedString(dims));
  }
  ~DimsOpRecord() override = default;
  RecordFunctor* clone() final {
    return new DimsOpRecord(*this);
  }

  size_t hash() const final {
    auto result = RecordFunctor::hash();
    size_t dims_hash = 0;
    for (auto dim : dims_) {
      hashCombine(dims_hash, static_cast<size_t>(dim));
    }
    return result | (dims_hash & 0xffff);
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto child_ptr = dynamic_cast<const DimsOpRecord*>(&other)) {
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
    if constexpr (op_type == serde::RecordType::PermuteOp) {
      auto arg =
          fd.getFusionState(args_.at(0).index)->template as<TensorView>();
      auto output = permute(arg, dims_);
      fd.setFusionState(outputs_.at(0).index, output);
    } else if constexpr (op_type == serde::RecordType::StrideOrderOp) {
      auto arg =
          fd.getFusionState(args_.at(0).index)->template as<TensorView>();
      auto output = set(arg);
      std::vector<IterDomain*> allocation_domain =
          ir_utils::strideOrderToAllocation(output->getLogicalDomain(), dims_);
      output->setAllocationDomain(allocation_domain, true);
      fd.setFusionState(outputs_.at(0).index, output);
    } else {
      NVF_THROW("op_type is not recognized by dims operator.");
    }
  }

  void print(std::ostream& os, bool close_function = true) const final {
    RecordFunctor::print(os, false);
    if constexpr (op_type == serde::RecordType::PermuteOp) {
      os << ", dims=[";
    } else if constexpr (op_type == serde::RecordType::StrideOrderOp) {
      os << ", stride_order=[";
    } else {
      NVF_THROW("op_type is not recognized by dims operator.");
    }
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
        serde::RecordData::Dims,
        serde::CreateDimsDirect(builder, &dims_).Union()};
  }

 private:
  //! Represents the mapping from the original shape to the new shape
  std::vector<int64_t> dims_;
};

struct SqueezeOpRecord : RecordFunctor {
  SqueezeOpRecord(
      std::vector<State> _args,
      std::vector<State> _outputs,
      std::vector<int64_t> dims,
      bool squeeze_expanded = false)
      : RecordFunctor(
            std::move(_args),
            std::move(_outputs),
            "ops.squeeze",
            serde::RecordType::SqueezeOp),
        dims_(std::move(dims)),
        squeeze_expanded_(squeeze_expanded) {}
  ~SqueezeOpRecord() override = default;
  RecordFunctor* clone() final {
    return new SqueezeOpRecord(*this);
  }

  //! Child specific hash function in lower 32 bits.
  //! | 31 | 30 --------------------------------  0 |
  //! | squeeze_expanded? | Squeeze Dim hash        |
  size_t hash() const final {
    auto result = RecordFunctor::hash();
    size_t squeeze_dims_hash = 0;
    for (auto dim : dims_) {
      squeeze_dims_hash ^= static_cast<size_t>(dim);
    }
    result = result | (squeeze_dims_hash & 0x7fffffff);
    result |= ((static_cast<size_t>(squeeze_expanded_) & 0x1) << 31);
    return result;
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto child_ptr = dynamic_cast<const SqueezeOpRecord*>(&other)) {
      result = RecordFunctor::operator==(other) && (dims_ == child_ptr->dims_);
    }
    return result;
  }

  void operator()(FusionState& fd) final {
    auto arg = fd.getFusionState(args_.at(0).index)->template as<TensorView>();
    // In pytorch, the squeeze operation cannot remove expanded dimensions.
    // In nvfuser, for reduction operations, we apply squeeze to remove
    // broadcast and expanded iterDomains. The squeeze_expanded_ flag bypasses
    // assertion used to match pytorch's behavior.
    auto output = squeeze(arg, dims_, squeeze_expanded_);
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
    os << "], squeeze_expanded=" << (squeeze_expanded_ ? "True" : "False");
    if (close_function) {
      os << ")";
    }
  }

  std::pair<serde::RecordData, flatbuffers::Offset<void>> recordData(
      flatbuffers::FlatBufferBuilder& builder) const final {
    return {
        serde::RecordData::Squeeze,
        serde::CreateSqueezeDirect(builder, &dims_, squeeze_expanded_).Union()};
  }

 private:
  //! Dimension to squeeze.
  std::vector<int64_t> dims_;
  //! Option to remove expanded dimensions
  bool squeeze_expanded_;
};

//! Specialized Record Functor for the FusionState's broadcast_in_dim op.
// NOTE: output_ndims gives the rank of the output tensor.  This size can be
// found from the State after the definition is read and the Fusion IR is in the
// process of being created.  However, pior to that point, the size is needed
// for matching a Fusion Record node in the Trie used to cache definitions.
struct BroadcastInDimOpRecord : RecordFunctor {
  BroadcastInDimOpRecord(
      std::vector<State> _args,
      std::vector<State> _outputs,
      size_t output_ndims,
      std::vector<int64_t> broadcast_dims)
      : RecordFunctor(
            std::move(_args),
            std::move(_outputs),
            "ops.broadcast_in_dim",
            serde::RecordType::BroadcastInDim),
        output_ndims_(output_ndims),
        broadcast_dims_(std::move(broadcast_dims)) {
    arg_names_[1] = "shape";
  }
  ~BroadcastInDimOpRecord() override = default;
  RecordFunctor* clone() final {
    return new BroadcastInDimOpRecord(*this);
  }

  //! Child specific hash function in lower 32 bits.
  //! | 31 -------------------------------------  0 |
  //! | broadcast_dims hash                         |
  size_t hash() const final {
    auto result = RecordFunctor::hash();
    size_t broadcast_dims_hash = 0;
    for (auto dim : broadcast_dims_) {
      broadcast_dims_hash |= 1 << ((output_ndims_ - 1) - dim);
    }
    return result | (broadcast_dims_hash & 0xffffffff);
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto child_ptr = dynamic_cast<const BroadcastInDimOpRecord*>(&other)) {
      result = RecordFunctor::operator==(other);
      if (result) {
        result =
            ((output_ndims_ == child_ptr->output_ndims_) &&
             (broadcast_dims_.size() == child_ptr->broadcast_dims_.size()));
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

  void operator()(FusionState& fd) final {
    auto arg = fd.getFusionState(args_.at(0).index)->template as<TensorView>();
    const std::vector<Val*>& output_shape =
        fd.getFusionStateVector(args_.at(1).index);

    const auto& arg_domains_nr = arg->domain()->noReductions();
    const auto arg_ndims = arg_domains_nr.size();
    NVF_CHECK(
        output_ndims_ >= arg_ndims,
        "The new shape is expected to be greater-then-or-equal to the input: ",
        output_ndims_,
        " vs ",
        arg_ndims);
    NVF_CHECK(
        arg_ndims == broadcast_dims_.size(),
        "The broadcast dimensions should match the input dimensions: ",
        arg_ndims,
        " vs ",
        broadcast_dims_.size(),
        ". arg = ",
        arg->toString());

    std::vector<bool> is_broadcast_dim(output_ndims_, true);
    for (const auto idx : arange(broadcast_dims_.size())) {
      if (idx > 0) {
        NVF_CHECK(
            broadcast_dims_[idx - 1] < broadcast_dims_[idx],
            "Broadcast dimension is not greater than the previous value.");
      }
      NVF_CHECK(
          broadcast_dims_[idx] < static_cast<int>(output_ndims_),
          "Invalid broadcast_dims value.");
      is_broadcast_dim.at(broadcast_dims_[idx]) = false;
    }

    auto output = broadcast(arg, is_broadcast_dim);
    auto expanded_output = expand(output, output_shape);

    fd.setFusionState(outputs_.at(0).index, expanded_output);
  }

  void print(std::ostream& os, bool close_function = true) const final {
    RecordFunctor::print(os, false);
    os << ", broadcast_dims=[";
    bool first_arg = true;
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
    return {
        serde::RecordData::BroadcastInDim,
        serde::CreateBroadcastInDimDirect(
            builder, output_ndims_, &broadcast_dims_)
            .Union()};
  };

 private:
  //! Number of dims of shape Vector used to communicate the output tensor shape
  size_t output_ndims_;
  //! Communicates which dimensions of the output the input tensor maps.
  //! For instance, for output [2, 3, 4] and input [3]. This vector would
  //! contain [1].
  std::vector<int64_t> broadcast_dims_;
};

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
            serde::RecordType::BroadcastOp),
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
      result = result &&
          std::equal(
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
    return {serde::RecordData::Broadcast, expr_data.Union()};
  }

 private:
  //! Communicates which dimensions in the output are broadcasted.
  std::vector<bool> is_broadcast_dim_;
};

//! Specialized Record Functor for the FusionState's expand op.
struct ExpandOpRecord : RecordFunctor {
  ExpandOpRecord(std::vector<State> _args, std::vector<State> _outputs)
      : RecordFunctor(
            std::move(_args),
            std::move(_outputs),
            "ops.expand",
            serde::RecordType::ExpandOp) {
    arg_names_[1] = "shape";
  }
  ~ExpandOpRecord() override = default;
  RecordFunctor* clone() final {
    return new ExpandOpRecord(*this);
  }

  //! Child specific hash function in lower 32 bits.
  //! | 31 ---------------------------------------  0 |
  //! | None                                          |
  size_t hash() const final {
    return RecordFunctor::hash();
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (dynamic_cast<const ExpandOpRecord*>(&other)) {
      result = RecordFunctor::operator==(other);
    }
    return result;
  }

  void operator()(FusionState& fd) final {
    auto arg = fd.getFusionState(args_.at(0).index)->template as<TensorView>();
    const std::vector<Val*>& output_shape =
        fd.getFusionStateVector(args_.at(1).index);

    size_t arg_ndims = arg->domain()->noReductions().size();
    NVF_CHECK(
        output_shape.size() == arg_ndims,
        "The new shape is expected to be equal to the input: ",
        output_shape.size(),
        " vs ",
        arg_ndims);
    auto expanded_output = expand(arg, output_shape);

    fd.setFusionState(outputs_.at(0).index, expanded_output);
  }

  void print(std::ostream& os, bool close_function = true) const final {
    RecordFunctor::print(os, false);
    if (close_function) {
      os << ")";
    }
  }
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
        serde::RecordData::Dtype,
        serde::CreateDtype(builder, nvfuser::toUnderlying(dtype_)).Union()};
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
      int64_t dim,
      bool manual_padding)
      : RecordFunctor(
            std::move(_args),
            std::move(_outputs),
            "ops.cat",
            serde::RecordType::CatOp),
        dim_(dim),
        manual_padding_(manual_padding) {}
  ~CatOpRecord() override = default;
  RecordFunctor* clone() final {
    return new CatOpRecord(*this);
  }

  //! Child specific hash function in lower 32 bits.
  //! |       31        | 30  ------------------------  0 |
  //! | manual_padding? |              dim                |
  size_t hash() const final {
    auto result = RecordFunctor::hash();
    result |= ((static_cast<size_t>(manual_padding_) & 0x1) << 31);
    result |= (static_cast<size_t>(dim_) & 0x7fff);
    return result;
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto child_ptr = dynamic_cast<const CatOpRecord*>(&other)) {
      result = RecordFunctor::operator==(other);
      result = result && (dim_ == child_ptr->dim_);
      result = result && (manual_padding_ == child_ptr->manual_padding_);
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
    auto output =
        cat(input_tvs, dim_, /*iter_type_opt=*/std::nullopt, manual_padding_);
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
    os << ", manual_padding=" << manual_padding_;
    if (close_function) {
      os << ")";
    }
  }

  std::pair<serde::RecordData, flatbuffers::Offset<void>> recordData(
      flatbuffers::FlatBufferBuilder& builder) const final {
    return {
        serde::RecordData::Cat,
        serde::CreateCat(builder, dim_, manual_padding_).Union()};
  }

 private:
  //! The dimension along which we will concatenate
  int64_t dim_;
  //! A flag to skip the pad operation in the cat composite operation.
  bool manual_padding_;
};

//! Specialized Record Functor for recording FusionState End.
//! The accompanying Fusion Cache Entry holds a Fusion Object.

struct EndRecord : RecordFunctor {
  EndRecord() : RecordFunctor({}, {}, "end", serde::RecordType::End) {}
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
      std::vector<int64_t> _shape,
      std::vector<std::optional<bool>> _contiguity,
      PrimDataType _dtype,
      bool _is_cpu = false,
      std::vector<int64_t> _stride_order = {})
      : RecordFunctor(
            {},
            std::move(_outputs),
            "define_tensor",
            serde::RecordType::Tensor),
        shape_(std::move(_shape)),
        contiguity_(std::move(_contiguity)),
        stride_order_(std::move(_stride_order)),
        dtype_(_dtype),
        is_cpu_(_is_cpu) {
    normalizeStrideOrder(stride_order_);
  }
  ~TensorRecord() override = default;
  RecordFunctor* clone() final {
    return new TensorRecord(*this);
  }

  //! Child specific hash function in lower 32 bits.
  //! |  31  | 30 --- 24 | 23 --------- 12 | 11 ------------------------  0 |
  //! | CPU? | Dtype     | Symbolic Sizes  | Contiguous Info & stride_order |
  size_t hash() const final {
    auto result = RecordFunctor::hash();
    size_t ssize_hash = 0;
    for (size_t i = 0; i < shape_.size(); ++i) {
      size_t ssize = 0;
      if (shape_[i] == -1) {
        ssize = 1;
      }
      ssize_hash |= (ssize << (shape_.size() - 1 - i));
    }
    size_t contig_stride_hash = 0;
    for (size_t i = 0; i < contiguity_.size(); ++i) {
      auto contiguity_value = contiguity_[i];
      contig_stride_hash |=
          ((contiguity_value.has_value() && contiguity_value.value())
           << (contiguity_.size() - 1 - i));
    }
    for (size_t i = 0; i < stride_order_.size(); ++i) {
      contig_stride_hash ^= (stride_order_[i] << i);
    }

    result |= ((static_cast<size_t>(is_cpu_) & 0x1) << 31);
    result |= ((static_cast<size_t>(dtype_) & 0x7f) << 24);
    return result | ((ssize_hash & 0xfff) << 12) | (contig_stride_hash & 0xfff);
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto child_ptr = dynamic_cast<const TensorRecord*>(&other)) {
      result = RecordFunctor::operator==(other);
      result = result && (dtype_ == child_ptr->dtype_);
      result = result && (is_cpu_ == child_ptr->is_cpu_);
      if (result) {
        result =
            ((shape_.size() == child_ptr->shape_.size()) &&
             (stride_order_.size() == child_ptr->stride_order_.size()) &&
             (contiguity_.size() == child_ptr->contiguity_.size()));
        if (result) {
          for (size_t i = 0; i < shape_.size(); ++i) {
            if (shape_[i] != child_ptr->shape_[i]) {
              result = false;
              break;
            }
          }
        }
        if (result) {
          for (size_t i = 0; i < stride_order_.size(); ++i) {
            if (stride_order_[i] != child_ptr->stride_order_[i]) {
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
    TensorView* tv =
        TensorViewBuilder()
            .contiguity(contiguity_)
            .shape(shape_)
            .dtype(dtype_)
            .expanded(getExpanded(shape_, contiguity_, stride_order_))
            .strideOrder(stride_order_)
            .build();

    if (shape_.empty() && is_cpu_) {
      tv->setCpuScalar(true);
    } else {
      NVF_CHECK(!is_cpu_, "CPU non-scalar tensor is not supported!");
    }

    fd.setFusionState(outputs_.at(0).index, tv);
    fd.addInput(tv, outputs_.at(0).index);
  }

  void print(std::ostream& os, bool close_function = true) const final {
    RecordFunctor::print(os, false);
    os << "shape=[";
    bool first_arg = true;
    for (auto ss : shape_) {
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
    auto fb_sizes = builder.CreateVector(shape_);

    auto mapOptionalToEnum = [](std::optional<bool> v) -> serde::Contiguity {
      if (!v.has_value()) {
        return serde::Contiguity::None;
      } else if (v.value()) {
        return serde::Contiguity::Contiguous;
      } else {
        return serde::Contiguity::Strided;
      }
    };
    std::vector<serde::Contiguity> contiguity_enum;
    std::transform(
        contiguity_.cbegin(),
        contiguity_.cend(),
        std::back_inserter(contiguity_enum),
        mapOptionalToEnum);
    auto fb_contiguity_enum = builder.CreateVector(contiguity_enum);
    auto fb_stride_order = builder.CreateVector(stride_order_);

    serde::TensorBuilder tensor_builder(builder);
    tensor_builder.add_sizes(fb_sizes);
    tensor_builder.add_contiguity(fb_contiguity_enum);
    tensor_builder.add_stride_order(fb_stride_order);
    tensor_builder.add_dtype(toUnderlying(dtype_));
    tensor_builder.add_is_cpu(is_cpu_);
    auto expr_data = tensor_builder.Finish();
    return {serde::RecordData::Tensor, expr_data.Union()};
  }

 private:
  //! A vector of tensor dimension sizes.
  //! This vector only captures sizes of -1 or 1 to indicate a symbolic
  //! dimension (-1) or a broadcast dimension (1).
  std::vector<int64_t> shape_;
  //! A vector to indicate whether the a tensor dimension is contiguous
  //! with the dimension just to its right.
  std::vector<std::optional<bool>> contiguity_;
  //! A vector to indicate stride order of tensor
  std::vector<int64_t> stride_order_;
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
      stride_order_ = stride_order;
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
    for (auto i : arange(stride_order_.size())) {
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
      NVF_CHECK(
          stride_order_.empty(),
          "stride_order can't be dictated for aliased outputs.");
      if constexpr (std::is_same_v<OutputType, TensorView>) {
        fd.aliasOutputToInput(output, alias_input);
      } else {
        NVF_THROW("Scalar outputs should not alias inputs.");
      }
    } else {
      if constexpr (std::is_same_v<OutputType, TensorView>) {
        auto tv_output = output->template as<TensorView>();
        if (!stride_order_.empty()) {
          auto logical_domain = tv_output->getLogicalDomain();
          std::vector<IterDomain*> allocation_domain =
              ir_utils::strideOrderToAllocation(logical_domain, stride_order_);
          tv_output->setAllocationDomain(allocation_domain, true);
        }
        fd.addOutput(tv_output, args_.at(0).index);
      } else {
        NVF_CHECK(
            stride_order_.empty(),
            "stride_order can't be dictated for scalar outputs.");
        fd.addOutput(output, args_.at(0).index);
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
        serde::RecordData::Output,
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
          TensorView*(TensorView*, const std::vector<int64_t>&, bool, DataType)>
          fusion_op,
      std::vector<int64_t> axes,
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
    for (auto i : arange(axes_.size())) {
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
                                 const std::vector<int64_t>&,
                                 bool,
                                 DataType)>() ==
             *child_ptr->fusion_op_.template target<

                 TensorView* (*)(TensorView*,
                                 const std::vector<int64_t>&,
                                 bool,
                                 DataType)>());
        if (isDebugDumpEnabled(DebugDumpOption::PythonFrontendDebug)) {
          debug() << " Target  Ptr [self: 0x" << std::hex
                  << (size_t)*fusion_op_.template target<

                         TensorView* (*)(TensorView*,
                                         const std::vector<int64_t>&,
                                         bool,
                                         DataType)>()
                  << "] [other: 0x" << std::hex
                  << (size_t)*child_ptr->fusion_op_.template target<

                         TensorView* (*)(TensorView*,
                                         const std::vector<int64_t>&,
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
    os << ", dims=[";
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
        serde::RecordData::Reduction,
        serde::CreateReductionDirect(
            builder, &axes_, keep_dim_, toUnderlying(dtype_))
            .Union()};
  }

 private:
  //! nvFuser arith function signature for a given reduction operation
  std::function<
      TensorView*(TensorView*, const std::vector<int64_t>&, bool, DataType)>
      fusion_op_;
  //! The tensor dimensions to reduce
  std::vector<int64_t> axes_;
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
            serde::RecordType::IndexSelectOp),
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

    Val* output = indexSelect(arg1, dim_, arg3);
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
        serde::RecordData::Dimension,
        serde::CreateDimension(builder, dim_).Union()};
  }

 private:
  //! Dimension to select.
  int64_t dim_;
};

// TODO Merge IndexSelectOpRecord and SelectOpRecord for cleaner interface.
// If the index TensorView is a scalar, then use select operation.
struct SelectOpRecord : RecordFunctor {
  SelectOpRecord(
      std::vector<State> _args,
      std::vector<State> _outputs,
      int64_t dim)
      : RecordFunctor(
            std::move(_args),
            std::move(_outputs),
            "ops.select",
            serde::RecordType::SelectOp),
        dim_(dim) {}
  ~SelectOpRecord() override = default;
  RecordFunctor* clone() final {
    return new SelectOpRecord(*this);
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto child_ptr = dynamic_cast<const SelectOpRecord*>(&other)) {
      result = RecordFunctor::operator==(other) && dim_ == child_ptr->dim_;
    }
    return result;
  }

  void operator()(FusionState& fd) final {
    auto arg1 = fd.getFusionState(args_.at(0).index)->template as<TensorView>();
    auto arg3 = fd.getFusionState(args_.at(1).index);

    Val* output = select(arg1, dim_, arg3);
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
        serde::RecordData::Dimension,
        serde::CreateDimension(builder, dim_).Union()};
  }

 private:
  //! Dimension to select.
  int64_t dim_;
};

struct ScatterOpRecord : RecordFunctor {
  ScatterOpRecord(
      std::vector<State> _args,
      std::vector<State> _outputs,
      int64_t dim)
      : RecordFunctor(
            std::move(_args),
            std::move(_outputs),
            "ops.scatter",
            serde::RecordType::ScatterOp),
        dim_(dim) {}
  ~ScatterOpRecord() override = default;
  RecordFunctor* clone() final {
    return new ScatterOpRecord(*this);
  }

  void operator()(FusionState& fd) final {
    auto arg1 = fd.getFusionState(args_.at(0).index)->template as<TensorView>();
    auto arg3 = fd.getFusionState(args_.at(1).index)->template as<TensorView>();
    auto arg4 = fd.getFusionState(args_.at(2).index)->template as<TensorView>();

    Val* output = scatter(arg1, dim_, arg3, arg4);
    fd.setFusionState(outputs_.at(0).index, output);
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto child_ptr = dynamic_cast<const ScatterOpRecord*>(&other)) {
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
        serde::RecordData::Dimension,
        serde::CreateDimension(builder, dim_).Union()};
  }

 private:
  //! Dimension to select.
  int64_t dim_;
};

struct GatherOpRecord : RecordFunctor {
  GatherOpRecord(
      std::vector<State> _args,
      std::vector<State> _outputs,
      int64_t dim)
      : RecordFunctor(
            std::move(_args),
            std::move(_outputs),
            "ops.gather",
            serde::RecordType::GatherOp),
        dim_(dim) {}
  ~GatherOpRecord() override = default;
  RecordFunctor* clone() final {
    return new GatherOpRecord(*this);
  }

  void operator()(FusionState& fd) final {
    auto arg1 = fd.getFusionState(args_.at(0).index)->template as<TensorView>();
    auto arg3 = fd.getFusionState(args_.at(1).index)->template as<TensorView>();

    Val* output = gather(arg1, dim_, arg3);
    fd.setFusionState(outputs_.at(0).index, output);
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto child_ptr = dynamic_cast<const GatherOpRecord*>(&other)) {
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
        serde::RecordData::Dimension,
        serde::CreateDimension(builder, dim_).Union()};
  }

 private:
  //! Dimension to select.
  int64_t dim_;
};

//! Similar to GatherOpRecord but enforces that non-index dimension
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
            serde::RecordType::TakeAlongAxisOp),
        dim_(dim) {}
  ~TakeAlongAxisOpRecord() override = default;
  RecordFunctor* clone() final {
    return new TakeAlongAxisOpRecord(*this);
  }

  void operator()(FusionState& fd) final {
    auto arg1 = fd.getFusionState(args_.at(0).index)->template as<TensorView>();
    auto arg3 = fd.getFusionState(args_.at(1).index)->template as<TensorView>();

    Val* output = takeAlongAxis(arg1, arg3, dim_);
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
        serde::RecordData::Dimension,
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
      std::optional<PrimDataType> dtype,
      bool inline_def = false)
      : RecordFunctor(
            {},
            std::move(_outputs),
            "define_scalar",
            serde::RecordType::Scalar,
            inline_def),
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
      fd.addInput(output, outputs_.at(0).index);
    }
    fd.setFusionState(outputs_.at(0).index, output);
  }

  void print(std::ostream& os, bool close_function = true) const final {
    if (inline_def_) {
      NVF_CHECK(
          value_.hasValue(),
          "Only ScalarRecords with values support inline definitions!");
      if (value_.is<bool>()) {
        NVF_CHECK(
            dtype_ == PrimDataType::Bool,
            "A ScalarRecord for Bool inline definition not have a matching "
            "data type!");
        os << ((bool)value_ ? "True" : "False");
      } else if (value_.is<double>()) {
        NVF_CHECK(
            dtype_ == PrimDataType::Double,
            "A ScalarRecord for Double inline definition not have a matching "
            "data type!");
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
        NVF_CHECK(
            dtype_ == PrimDataType::Int,
            "A ScalarRecord for Int inline definition not have a matching data "
            "type!");
        os << value_;
      } else {
        NVF_THROW("A ScalarRecord with an unsupported inline definition type!");
      }
      // NOTE: close_function is not relevant for the inline definition as the
      // printing is specific to each operator and not partially done with the
      // base class print method.
    } else {
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
          NVF_CHECK(false, "Unsupported dtype.");
        }
      } else {
        os << "None";
      }

      os << ", dtype=" << dtypeToPyString(dtype_);

      if (close_function) {
        os << ")";
      }
    }
  }

  std::pair<serde::RecordData, flatbuffers::Offset<void>> recordData(
      flatbuffers::FlatBufferBuilder& builder) const final {
    return {
        serde::RecordData::Scalar,
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

//! Specialized Record Functor for recording FusionDefinition Start.
//! There should only ever be one instance of this Record in the
//! Fusion Cache.

struct StartRecord : RecordFunctor {
  StartRecord() : RecordFunctor({}, {}, "start", serde::RecordType::Start) {}
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
      std::vector<int64_t> axes,
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
    for (auto i : arange(axes_.size())) {
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
    os << ", dims=[";
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
        serde::RecordData::Norm,
        serde::CreateNormDirect(builder, &axes_, correction_, keep_dim_)
            .Union()};
  }

 protected:
  //! Dimensions of tensor to reduce for variance calculation
  std::vector<int64_t> axes_;
  //! Bessel's correction value
  int64_t correction_;
  //! Indicates whether to keep the reduced dimension(s).
  bool keep_dim_;
};

struct VarianceOpRecord : NormOpRecord {
  VarianceOpRecord(
      std::vector<State> args,
      std::vector<State> outputs,
      std::vector<int64_t> axes,
      int64_t correction,
      bool keep_dim)
      : NormOpRecord(
            std::move(args),
            std::move(outputs),
            "ops.var",
            serde::RecordType::VarianceOp,
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
      std::vector<int64_t> axes,
      int64_t correction,
      bool keep_dim)
      : NormOpRecord(
            std::move(args),
            std::move(outputs),
            "ops.var_mean",
            serde::RecordType::VarianceMeanOp,
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

struct WelfordOpRecord : RecordFunctor {
  WelfordOpRecord(
      std::vector<State> _args,
      std::vector<State> _outputs,
      std::vector<int64_t> axes)
      : RecordFunctor(
            std::move(_args),
            std::move(_outputs),
            "ops.welford",
            serde::RecordType::WelfordOp),
        axes_(std::move(axes)) {}
  ~WelfordOpRecord() override = default;
  RecordFunctor* clone() final {
    return new WelfordOpRecord(*this);
  }

  size_t hash() const final {
    auto result = RecordFunctor::hash();
    size_t axes_hash = 0;
    for (auto axis : axes_) {
      hashCombine(axes_hash, static_cast<size_t>(axis));
    }
    return result | (axes_hash & 0xffff);
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto child_ptr = dynamic_cast<const WelfordOpRecord*>(&other)) {
      result = RecordFunctor::operator==(other);
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

  void operator()(FusionState& fd) final {
    auto arg = fd.getFusionState(args_.at(0).index)->template as<TensorView>();
    auto output = WelfordRaw(arg, axes_);
    fd.setFusionState(outputs_.at(0).index, output.avg);
    fd.setFusionState(outputs_.at(1).index, output.var_sum);
    fd.setFusionState(outputs_.at(2).index, output.n);
  }

  void print(std::ostream& os, bool close_function = true) const final {
    RecordFunctor::print(os, false);
    os << ", dims=[";
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
    if (close_function) {
      os << ")";
    }
  }

  std::pair<serde::RecordData, flatbuffers::Offset<void>> recordData(
      flatbuffers::FlatBufferBuilder& builder) const final {
    return {
        serde::RecordData::Welford,
        serde::CreateWelfordDirect(builder, &axes_).Union()};
  }

 private:
  //! The tensor dimensions to reduce
  std::vector<int64_t> axes_;
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
            serde::RecordType::BatchNormOp),
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
    auto weight = (args_.at(1).stype == serde::StateType::Tensor)
        ? fd.getFusionState(args_.at(1).index)->as<TensorView>()
        : nullptr;
    auto bias = (args_.at(2).stype == serde::StateType::Tensor)
        ? fd.getFusionState(args_.at(2).index)->as<TensorView>()
        : nullptr;
    auto running_mean = (args_.at(3).stype == serde::StateType::Tensor)
        ? fd.getFusionState(args_.at(3).index)->as<TensorView>()
        : nullptr;
    auto running_var = (args_.at(4).stype == serde::StateType::Tensor)
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
        serde::RecordData::BatchNorm,
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
            serde::RecordType::TensorSizes) {
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
    auto sizes = shape(arg);
    for (const auto idx : arange(sizes.size())) {
      fd.setFusionState(outputs_.at(idx).index, sizes[idx]);
    }
  }
};

//! Specialized Record Functor for the shape op.
//! Uses the default hash() and print() methods of Record Functor

struct ShapeOpRecord : RecordFunctor {
  ShapeOpRecord(std::vector<State> args, std::vector<State> outputs)
      : RecordFunctor(
            std::move(args),
            std::move(outputs),
            "ops.shape",
            serde::RecordType::ShapeOp) {}
  ~ShapeOpRecord() override = default;
  RecordFunctor* clone() final {
    return new ShapeOpRecord(*this);
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (dynamic_cast<const ShapeOpRecord*>(&other)) {
      result = RecordFunctor::operator==(other);
    }
    return result;
  }

  void operator()(FusionState& fd) final {
    auto arg = fd.getFusionState(args_.at(0).index)->as<TensorView>();
    auto result = shape(arg);
    fd.setFusionStateVector(outputs_.at(0).index, result);
  }
};

//! Specialized Record Functor for the size op.
//! Uses the default hash() and print() methods of Record Functor

struct SizeOpRecord : RecordFunctor {
  SizeOpRecord(std::vector<State> args, std::vector<State> outputs, int64_t dim)
      : RecordFunctor(
            std::move(args),
            std::move(outputs),
            "ops.size",
            serde::RecordType::SizeOp),
        dim_(dim) {}
  ~SizeOpRecord() override = default;
  RecordFunctor* clone() final {
    return new SizeOpRecord(*this);
  }

  //! Child specific hash function in lower 32 bits.
  //! | 31 --------------------------------------  0 |
  //! | dim                                          |
  size_t hash() const final {
    auto result = RecordFunctor::hash();
    return result | (static_cast<size_t>(dim_) & 0xffffffff);
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto child_ptr = dynamic_cast<const SizeOpRecord*>(&other)) {
      result = RecordFunctor::operator==(other);
      result = result && (dim_ == child_ptr->dim_);
    }
    return result;
  }

  void operator()(FusionState& fd) final {
    auto arg = fd.getFusionState(args_.at(0).index)->as<TensorView>();
    auto result = size(arg, dim_);
    fd.setFusionState(outputs_.at(0).index, result);
  }

  std::pair<serde::RecordData, flatbuffers::Offset<void>> recordData(
      flatbuffers::FlatBufferBuilder& builder) const final {
    return {serde::RecordData::Size, serde::CreateSize(builder, dim_).Union()};
  }

  void print(std::ostream& os, bool close_function = true) const final {
    RecordFunctor::print(os, false);
    os << ", dim=" << dim_;
    if (close_function) {
      os << ")";
    }
  }

 private:
  int64_t dim_;
};

//! Specialized Record Functor for the at() op.
//! Uses the default hash() and print() methods of Record Functor

struct AtOpRecord : RecordFunctor {
  AtOpRecord(std::vector<State> args, std::vector<State> outputs, int64_t index)
      : RecordFunctor(
            std::move(args),
            std::move(outputs),
            "ops.at",
            serde::RecordType::AtOp),
        index_(index) {}
  ~AtOpRecord() override = default;
  RecordFunctor* clone() final {
    return new AtOpRecord(*this);
  }

  //! Child specific hash function in lower 32 bits.
  //! | 31 --------------------------------------  0 |
  //! | index                                        |
  size_t hash() const final {
    auto result = RecordFunctor::hash();
    return result | (static_cast<size_t>(index_) & 0xffffffff);
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto child_ptr = dynamic_cast<const AtOpRecord*>(&other)) {
      result = RecordFunctor::operator==(other);
      result = result && (index_ == child_ptr->index_);
    }
    return result;
  }

  void operator()(FusionState& fd) final {
    NVF_CHECK(
        args_.at(0).stype == serde::StateType::Vector,
        "Expected Vector State!");
    const std::vector<Val*>& arg = fd.getFusionStateVector(args_.at(0).index);
    auto result = at(arg, index_);
    fd.setFusionState(outputs_.at(0).index, result);
  }

  std::pair<serde::RecordData, flatbuffers::Offset<void>> recordData(
      flatbuffers::FlatBufferBuilder& builder) const final {
    return {serde::RecordData::At, serde::CreateAt(builder, index_).Union()};
  }

  void print(std::ostream& os, bool close_function = true) const final {
    RecordFunctor::print(os, false);
    os << ", index=" << index_;
    if (close_function) {
      os << ")";
    }
  }

 private:
  int64_t index_;
};

struct FullOpRecord : RecordFunctor {
  FullOpRecord(
      std::vector<State> _args,
      std::vector<State> _outputs,
      PrimDataType dtype)
      : RecordFunctor(
            std::move(_args),
            std::move(_outputs),
            "ops.full",
            serde::RecordType::FullOp),
        dtype_(dtype) {
    setArgName(0, "shape");
    setArgName(1, "fill_value");
  }
  ~FullOpRecord() override = default;
  RecordFunctor* clone() final {
    return new FullOpRecord(*this);
  }

  //! Child specific hash function in lower 32 bits.
  //! | 31 --------------------------------------  0 |
  //! | Dtype                                        |
  size_t hash() const final {
    auto result = RecordFunctor::hash();
    result |= (static_cast<size_t>(dtype_) & 0xffffffff);
    return result;
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto child_ptr = dynamic_cast<const FullOpRecord*>(&other)) {
      result = RecordFunctor::operator==(other) && dtype_ == child_ptr->dtype_;
    }
    return result;
  }

  void operator()(FusionState& fd) final {
    const std::vector<Val*>& shape = fd.getFusionStateVector(args_.at(0).index);
    auto fill_value = fd.getFusionState(args_.at(1).index);

    auto output = full(shape, fill_value, dtype_);
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
        serde::RecordData::TensorCreationSymbolic,
        serde::CreateTensorCreationSymbolic(builder, toUnderlying(dtype_))
            .Union()};
  }

 private:
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
            serde::RecordType::IotaOp),
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
    auto start = (args_.at(1).stype == serde::StateType::Scalar)
        ? fd.getFusionState(args_.at(1).index)->as<Val>()
        : nullptr;
    auto step = (args_.at(2).stype == serde::StateType::Scalar)
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
        serde::RecordData::Dtype,
        serde::CreateDtype(builder, nvfuser::toUnderlying(dtype_)).Union()};
  }

 private:
  //! Type of output
  PrimDataType dtype_;
};

//! Specialized Record Functors for random ops.
template <serde::RecordType RType>
struct RandomDistOpRecord : RecordFunctor {
  RandomDistOpRecord(
      std::vector<State> _args,
      std::vector<State> _outputs,
      PrimDataType dtype)
      : RecordFunctor(std::move(_args), std::move(_outputs), "", RType),
        dtype_(dtype) {
    if constexpr (RType == serde::RecordType::UniformDistOp) {
      name_ = "ops.uniform";
    } else if constexpr (RType == serde::RecordType::NormalDistOp) {
      name_ = "ops.normal";
    } else {
      static_assert(
          (RType == serde::RecordType::NormalDistOp) ||
          (RType == serde::RecordType::UniformDistOp));
    }
    setArgName(2, "shape");
    if (args_.size() == 5) {
      setArgName(3, "rng_seed");
      setArgName(4, "rng_offset");
    }
  }
  ~RandomDistOpRecord() override = default;
  RecordFunctor* clone() final {
    return new RandomDistOpRecord(*this);
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
    if (auto child_ptr = dynamic_cast<const RandomDistOpRecord*>(&other)) {
      result = RecordFunctor::operator==(other);
      result = result && (dtype_ == child_ptr->dtype_);
    }
    return result;
  }

  void operator()(FusionState& fd) final {
    auto arg1 = fd.getFusionState(args_.at(0).index);
    auto arg2 = fd.getFusionState(args_.at(1).index);
    const std::vector<Val*>& output_shape =
        fd.getFusionStateVector(args_.at(2).index);

    Val* output = nullptr;
    if constexpr (RType == serde::RecordType::UniformDistOp) {
      if (args_.size() == 3) { // stochastic uniform
        output = uniform(output_shape, arg1, arg2, dtype_);
      } else if (args_.size() == 5) { // provided seed and offset
        auto seed = fd.getFusionState(args_.at(3).index);
        auto offset = fd.getFusionState(args_.at(4).index);
        output = uniform(output_shape, arg1, arg2, dtype_, seed, offset);
      }
    } else if constexpr (RType == serde::RecordType::NormalDistOp) {
      if (args_.size() == 3) { // stochastic normal
        output = normal(output_shape, arg1, arg2, dtype_);
      } else if (args_.size() == 5) { // provided seed and offset
        auto seed = fd.getFusionState(args_.at(3).index);
        auto offset = fd.getFusionState(args_.at(4).index);
        output = normal(output_shape, arg1, arg2, dtype_, seed, offset);
      }
    } else {
      static_assert(
          (RType == serde::RecordType::NormalDistOp) ||
          (RType == serde::RecordType::UniformDistOp));
    }

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
        serde::RecordData::TensorCreationSymbolic,
        serde::CreateTensorCreationSymbolic(builder, toUnderlying(dtype_))
            .Union()};
  }

 private:
  //! DataType of output
  PrimDataType dtype_;
};

//! Specialized Record Functor for recording Vector of Scalars

struct VectorRecord : RecordFunctor {
  VectorRecord(
      std::vector<State> _args,
      std::vector<State> _outputs,
      PrimDataType dtype,
      bool inline_def = false)
      : RecordFunctor(
            std::move(_args),
            std::move(_outputs),
            "define_vector",
            serde::RecordType::Vector,
            inline_def),
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
    NVF_CHECK(
        dtype_ == DataType::Int,
        "Only Int Dtype is not supported by a vector of sizes: ",
        dtype_);
    for (size_t i = 0; i < args_.size(); ++i) {
      NVF_CHECK(
          args_.at(i).stype == serde::StateType::Scalar,
          "Unsupported State type!");
      output.at(i) = fd.getFusionState(args_.at(i).index);
    }
    fd.setFusionStateVector(outputs_.at(0).index, output);
  }

  void print(std::ostream& os, bool close_function = true) const final {
    if (inline_def_) {
      bool first_arg = true;
      NVF_CHECK(outputs_.size() == 1, "VectorRecord's does not have 1 output!");
      os << "[";
      for (auto& arg : args_) {
        if (first_arg) {
          first_arg = false;
        } else {
          os << ", ";
        }
        os << arg;
      }
      os << "]";
    } else {
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
  }

  std::pair<serde::RecordData, flatbuffers::Offset<void>> recordData(
      flatbuffers::FlatBufferBuilder& builder) const final {
    return {
        serde::RecordData::Vector,
        serde::CreateVector(builder, nvfuser::toUnderlying(dtype_)).Union()};
  };

 private:
  //! Scalar data type.
  PrimDataType dtype_;
};

struct SdpaFwdOpRecord : RecordFunctor {
  SdpaFwdOpRecord(std::vector<State> args, std::vector<State> outputs)
      : RecordFunctor(
            std::move(args),
            std::move(outputs),
            "ops.sdpfa_fwd",
            serde::RecordType::SdpaFwdOp) {}
  ~SdpaFwdOpRecord() override = default;
  RecordFunctor* clone() final {
    return new SdpaFwdOpRecord(*this);
  }

  void operator()(FusionState& fd) final {
    auto query = fd.getFusionState(args_.at(0).index)->as<TensorView>();
    auto key = fd.getFusionState(args_.at(1).index)->as<TensorView>();
    auto value = fd.getFusionState(args_.at(2).index)->as<TensorView>();
    auto dropout_p = (args_.at(3).stype == serde::StateType::Scalar)
        ? fd.getFusionState(args_.at(3).index)->as<Val>()
        : nullptr;
    auto is_causal = (args_.at(4).stype == serde::StateType::Scalar)
        ? fd.getFusionState(args_.at(4).index)->as<Val>()
        : nullptr;
    auto scale = (args_.at(5).stype == serde::StateType::Scalar)
        ? fd.getFusionState(args_.at(5).index)->as<Val>()
        : nullptr;
    auto output = sdpfa_fwd(query, key, value, dropout_p, is_causal, scale);
    fd.setFusionState(outputs_.at(0).index, output.output);
    fd.setFusionState(outputs_.at(1).index, output.log_sumexp);
    fd.setFusionState(outputs_.at(2).index, output.philox_seed);
    fd.setFusionState(outputs_.at(3).index, output.philox_offset);
  }
};

struct SdpaBwdOpRecord : RecordFunctor {
  SdpaBwdOpRecord(std::vector<State> args, std::vector<State> outputs)
      : RecordFunctor(
            std::move(args),
            std::move(outputs),
            "ops.sdpfa_bwd",
            serde::RecordType::SdpaBwdOp) {}
  ~SdpaBwdOpRecord() override = default;
  RecordFunctor* clone() final {
    return new SdpaBwdOpRecord(*this);
  }

  void operator()(FusionState& fd) final {
    auto grad_output = fd.getFusionState(args_.at(0).index)->as<TensorView>();
    auto query = fd.getFusionState(args_.at(1).index)->as<TensorView>();
    auto key = fd.getFusionState(args_.at(2).index)->as<TensorView>();
    auto value = fd.getFusionState(args_.at(3).index)->as<TensorView>();
    auto output = fd.getFusionState(args_.at(4).index)->as<TensorView>();
    auto log_sumexp = fd.getFusionState(args_.at(5).index)->as<TensorView>();

    auto dropout_p = (args_.at(6).stype == serde::StateType::Scalar)
        ? fd.getFusionState(args_.at(6).index)->as<Val>()
        : nullptr;
    auto is_causal = (args_.at(7).stype == serde::StateType::Scalar)
        ? fd.getFusionState(args_.at(7).index)->as<Val>()
        : nullptr;

    auto philox_seed = fd.getFusionState(args_.at(8).index)->as<TensorView>();
    auto philox_offset = fd.getFusionState(args_.at(9).index)->as<TensorView>();

    auto scale = (args_.at(10).stype == serde::StateType::Scalar)
        ? fd.getFusionState(args_.at(10).index)->as<Val>()
        : nullptr;

    auto grad = sdpfa_bwd(
        grad_output,
        query,
        key,
        value,
        output,
        log_sumexp,
        dropout_p,
        is_causal,
        philox_seed,
        philox_offset,
        scale);
    fd.setFusionState(outputs_.at(0).index, grad.grad_query);
    fd.setFusionState(outputs_.at(1).index, grad.grad_key);
    fd.setFusionState(outputs_.at(2).index, grad.grad_value);
  }
};

struct EmbeddingFwdOpRecord : RecordFunctor {
  EmbeddingFwdOpRecord(std::vector<State> args, std::vector<State> outputs)
      : RecordFunctor(
            std::move(args),
            std::move(outputs),
            "ops.embedding_fwd",
            serde::RecordType::EmbeddingFwdOp) {}
  ~EmbeddingFwdOpRecord() override = default;
  RecordFunctor* clone() final {
    return new EmbeddingFwdOpRecord(*this);
  }

  void operator()(FusionState& fd) final {
    auto input = fd.getFusionState(args_.at(0).index)->as<TensorView>();
    auto weight = fd.getFusionState(args_.at(1).index)->as<TensorView>();
    auto padding_idx = (args_.at(2).stype == serde::StateType::Scalar)
        ? fd.getFusionState(args_.at(2).index)->as<Val>()
        : nullptr;
    auto max_norm = (args_.at(3).stype == serde::StateType::Scalar)
        ? fd.getFusionState(args_.at(3).index)->as<Val>()
        : nullptr;
    auto norm_type = (args_.at(4).stype == serde::StateType::Scalar)
        ? fd.getFusionState(args_.at(4).index)->as<Val>()
        : nullptr;
    auto scale_grad_by_freq = (args_.at(5).stype == serde::StateType::Scalar)
        ? fd.getFusionState(args_.at(5).index)->as<Val>()
        : nullptr;
    auto sparse = (args_.at(6).stype == serde::StateType::Scalar)
        ? fd.getFusionState(args_.at(6).index)->as<Val>()
        : nullptr;

    auto output = embedding_fwd(
        input,
        weight,
        padding_idx,
        max_norm,
        norm_type,
        scale_grad_by_freq,
        sparse);
    fd.setFusionState(outputs_.at(0).index, output);
  }
};

struct IndexPutAccumulateOpRecord : RecordFunctor {
  IndexPutAccumulateOpRecord(
      std::vector<State> args,
      std::vector<State> outputs)
      : RecordFunctor(
            std::move(args),
            std::move(outputs),
            "ops.index_put_accumulate",
            serde::RecordType::IndexPutAccumulateOp) {}
  ~IndexPutAccumulateOpRecord() override = default;
  RecordFunctor* clone() final {
    return new IndexPutAccumulateOpRecord(*this);
  }

  void operator()(FusionState& fd) final {
    auto acc = fd.getFusionState(args_.at(0).index)->as<TensorView>();
    auto index = fd.getFusionState(args_.at(1).index)->as<TensorView>();
    auto value = fd.getFusionState(args_.at(2).index)->as<TensorView>();

    auto output = indexPutAccumulate(acc, index, value);
    fd.setFusionState(outputs_.at(0).index, output);
  }
};

struct ArgsortOpRecord : RecordFunctor {
  ArgsortOpRecord(
      std::vector<State> _args,
      std::vector<State> _outputs,
      int64_t dim,
      bool descending,
      bool stable)
      : RecordFunctor(
            std::move(_args),
            std::move(_outputs),
            "ops.argsort",
            serde::RecordType::ArgsortOp),
        dim_(dim),
        descending_(descending),
        stable_(stable) {}
  ~ArgsortOpRecord() override = default;
  RecordFunctor* clone() final {
    return new ArgsortOpRecord(*this);
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto other_argsort = dynamic_cast<const ArgsortOpRecord*>(&other)) {
      result = RecordFunctor::operator==(other) &&
          dim_ == other_argsort->dim_ &&
          descending_ == other_argsort->descending_ &&
          stable_ == other_argsort->stable_;
    }
    return result;
  }

  void operator()(FusionState& fd) final {
    auto arg = fd.getFusionState(args_.at(0).index)->template as<TensorView>();
    Val* output = argsort(arg, dim_, descending_, stable_);
    fd.setFusionState(outputs_.at(0).index, output);
  }

  void print(std::ostream& os, bool close_function = true) const final {
    RecordFunctor::print(os, false);
    os << ", dim=" << dim_
       << ", descending=" << (descending_ ? "True" : "False")
       << ", stable=" << (stable_ ? "True" : "False");
    if (close_function) {
      os << ")";
    }
  }

  std::pair<serde::RecordData, flatbuffers::Offset<void>> recordData(
      flatbuffers::FlatBufferBuilder& builder) const final {
    return {
        serde::RecordData::Sort,
        serde::CreateSort(builder, dim_, descending_, stable_).Union()};
  }

 private:
  int64_t dim_;
  bool descending_;
  bool stable_;
};

//! Record for TopK operation in fusion cache and Python frontend
//!
//! Stores the parameters needed to recreate a TopK operation:
//! - dim: dimension along which to find top-k elements
//! - largest: whether to find largest (true) or smallest (false) elements
//! - sorted: whether the output should be sorted
//!
//! The operation takes two inputs: the tensor and k (number of elements)
//! and produces two outputs: values and indices tensors.
struct TopKOpRecord : RecordFunctor {
  TopKOpRecord(
      std::vector<State> _args,
      std::vector<State> _outputs,
      int64_t dim,
      bool largest,
      bool sorted)
      : RecordFunctor(
            std::move(_args),
            std::move(_outputs),
            "ops.topk",
            serde::RecordType::TopKOp),
        dim_(dim),
        largest_(largest),
        sorted_(sorted) {}
  ~TopKOpRecord() override = default;
  RecordFunctor* clone() final {
    return new TopKOpRecord(*this);
  }

  bool operator==(const RecordFunctor& other) const final {
    auto result = false;
    if (auto other_topk = dynamic_cast<const TopKOpRecord*>(&other)) {
      result = RecordFunctor::operator==(other) && dim_ == other_topk->dim_ &&
          largest_ == other_topk->largest_ && sorted_ == other_topk->sorted_;
    }
    return result;
  }

  void operator()(FusionState& fd) final {
    auto arg = fd.getFusionState(args_.at(0).index)->template as<TensorView>();
    auto k = fd.getFusionState(args_.at(1).index);
    auto output = topk(arg, k, dim_, largest_, sorted_);
    fd.setFusionState(outputs_.at(0).index, output.values);
    fd.setFusionState(outputs_.at(1).index, output.indices);
  }

  void print(std::ostream& os, bool close_function = true) const final {
    RecordFunctor::print(os, false);
    os << ", dim=" << dim_ << ", largest=" << (largest_ ? "True" : "False")
       << ", sorted=" << (sorted_ ? "True" : "False");
    if (close_function) {
      os << ")";
    }
  }

  std::pair<serde::RecordData, flatbuffers::Offset<void>> recordData(
      flatbuffers::FlatBufferBuilder& builder) const final {
    return {
        serde::RecordData::TopK,
        serde::CreateTopK(builder, dim_, largest_, sorted_).Union()};
  }

 private:
  int64_t dim_;
  bool largest_;
  bool sorted_;
};

struct ScaledGroupedMmaOpRecord : RecordFunctor {
  ScaledGroupedMmaOpRecord(
      std::vector<State> _args,
      std::vector<State> _outputs)
      : RecordFunctor(
            std::move(_args),
            std::move(_outputs),
            "ops.scaled_grouped_mm",
            serde::RecordType::ScaledGroupedMmaOp) {}
  ~ScaledGroupedMmaOpRecord() override = default;
  RecordFunctor* clone() final {
    return new ScaledGroupedMmaOpRecord(*this);
  }

  void operator()(FusionState& fd) final {
    auto mat1 = fd.getFusionState(args_.at(0).index)->template as<TensorView>();
    auto mat2 = fd.getFusionState(args_.at(1).index)->template as<TensorView>();
    auto offsets = fd.getFusionState(args_.at(2).index)->template as<TensorView>();
    auto scale1 = fd.getFusionState(args_.at(3).index)->template as<TensorView>();
    auto scale2 = fd.getFusionState(args_.at(4).index)->template as<TensorView>();
    auto output = scaled_grouped_mm(mat1, mat2, offsets, scale1, scale2);
    fd.setFusionState(outputs().at(0).index, output);
  }
};

} // namespace nvfuser::python_frontend

//! Creating the template specialized hash and equal_to functions for a
//! RecordFunctor object in order to use hash maps (unordered_maps) in STL.
namespace std {
using namespace nvfuser::python_frontend;

template <>
struct hash<RecordFunctor*> {
  size_t operator()(const RecordFunctor* p) const {
    NVF_CHECK(p, "The RecordFunctor Pointer for hashing is null!");
    return p->hash();
  }
};
template <>
struct equal_to<RecordFunctor*> {
  bool operator()(const RecordFunctor* p, const RecordFunctor* q) const {
    NVF_CHECK(
        p,
        "The RecordFunctor Pointer on the lhs of an equality check is null!");
    NVF_CHECK(
        q,
        "The RecordFunctor Pointer on the rhs of an equality check is null!");
    return p->operator==(*q);
  }
};
} // namespace std
