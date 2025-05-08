// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ATen/core/ivalue.h>
#include <exceptions.h>

#include <debug.h>
#include <fusion_guard.h>
#include <ir/base_nodes.h>
#include <ir/cloner.h>
#include <ir/container.h>
#include <iter_visitor.h>
#include <runtime/executor_params.h>
#include <visibility.h>

#include <any>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nvfuser {

//! Usage: FusionGuard (defined in fusion_guard.h) and Fusion are required user
//! interfaces for any operation underlying the code generator. In order to
//! create values, expressions, and generate code a Fusion instance must be
//! active. It is the responsibility of the user to create a Fusion instance and
//! register it with the fusion guard. The simplest example of this is:
//!
//!     Fusion fusion;
//!     FusionGuard fg(&fusion);
//!
//! Once a fusion is active all values and operations will be registered with
//! it.
//!
//! FusionGuard and Fusion are critical to the lifetime model of the IR system.
//! FusionGuard is a convenient way to set what base container instance holds
//! the defined IR. Statements that are defined are registered through the
//! FusionGuard with a particular Fusion. FusionGuard provides convenient
//! methods to access the active fusion so it doesn't need to be passed around
//! constantly. Any IR node derived classes from Statement must register with
//! Fusion to avoid memory leaks.
//!
//! Fusion is generally thought of as a translated fusion group from the JIT. It
//! is likely a single kernel, although, we don't have to stick to this in the
//! future and could in theory generate multiple kernels with an executor to run
//! them.
//!
//! Fusion also allows users to set input/output values that will allow us to
//! figure out how to hook up runtime data to and from the JIT as well as
//! provide us mechanisms for dependency analysis and DCE including safety
//! checks.

class Fusion;
class TensorView;
class WelfordResult;

class SegmentCandidateFinder;
class SegmentedFusion;
class KernelArgumentHolder;

class DynamicTransformConcretizationInfo;

// Set the enum base to `int` so it can be safely serialized as a part of
// serde::InputOutputAlias.
enum class AllocationType : int {
  New, // Allocate a new buffer
  // Reuse the buffer allocated to `aliased_io`. For example, the tensor storing
  // BatchNorm's running mean. The output EMA is updated in place.
  ReuseBuffer,
  // This is used to cheaply compute the output tensor using
  // `ExpressionEvaluator` (instead of a kernel) for:
  // 1. PointerArithmetics: For example, the output of a ViewOp is merely a
  // pointer arithmetic of the input.  In this case, aliased_io is a non-null
  // tensor.
  // 2. To evaluate output tensors which are not aliases. For example, default
  // scheduling for MatmulOp/LinearOp in ExprEval scheduler.
  Evaluate,
};

struct AliasInfo {
  AllocationType type;
  Val* aliased_io;
  // Whether integration should hide the output from users. This is currently
  // only used for ReuseBuffer.
  bool hide_output;

  bool operator==(const AliasInfo& other) const {
    return type == other.type && aliased_io == other.aliased_io &&
        hide_output == other.hide_output;
  }

  bool operator!=(const AliasInfo& other) const {
    return !(*this == other);
  }

  std::string toString() const {
    std::stringstream ss;
    ss << "AliasInfo{\n";
    ss << "  type = ";
    switch (type) {
      case AllocationType::Evaluate:
        ss << "Evaluate";
        break;
      case AllocationType::New:
        ss << "New";
        break;
      case AllocationType::ReuseBuffer:
        ss << "ReuseBuffer";
        break;
    }
    ss << ",\n  aliased_io = "
       << (aliased_io == nullptr ? "nullptr" : aliased_io->toString()) << ",\n";
    ss << "  hide_output = " << (hide_output ? "true" : "false") << "\n";
    ss << "}\n";
    return ss.str();
  }
};

//! Fusion is mutable but unique. Nodes cannot be copied in any way from one
//! Fusion to another. If anything like that is desired, it would require
//! duplicating all associated values and exprs. Fusion is considered to be SSA,
//! though this could also change in the future if there is a good reason to do
//! so.
//!
//! The Fusion owns the whole IR graph (Vals and Exprs)
//!
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class NVF_API Fusion : public IrContainer {
  typedef std::unordered_map<int, std::vector<int64_t>> PermutationMap;

 public:
  Fusion() = default;

  Fusion(const Fusion& other);
  Fusion(Fusion&& other) noexcept;

  Fusion& operator=(const Fusion& other);
  Fusion& operator=(Fusion&& other) noexcept;

  ~Fusion() override;

  friend void swap(Fusion& a, Fusion& b) noexcept;

  void clear() noexcept;

  //! Break dependency chains associated with Expr, remove references to expr
  //! delete expr
  void removeExpr(Expr* expr) override;

  //! Completely remove val from the fusion, break all dependencies associated
  //! with it
  void removeVal(Val* val) override;

  //! Register input as an input of the fusion
  void addInput(Val* input);

  //! Add output to outputs_ without modifying hide_output
  void addOutputInternal(Val* output);

  //! Register output as an output of the fusion
  void addOutput(Val* output);

  //! Deregister input as an input of the fusion
  void removeInput(Val* input);

  //! Deregister output as an output of the fusion
  void removeOutput(Val* output);

  //! Replace output with another value
  void replaceOutput(Val* output, Val* replacement);

  //! Assert that all leaves found from outputs are registered as an input
  void validateInputs();

  //! Print this fusion to an output stream
  std::ostream& print(std::ostream& os, bool include_tensor_transforms = true)
      const;

  //! Print to default debugging output stream
  std::ostream& print() const {
    return print(debug());
  }

  //! Print Arith exprs
  //! \param from_outputs_only Only print exprs reachable from outputs
  void printMath(bool from_outputs_only = true);

  //! Print transformations used in fusion (can be very verbose)
  void printTransforms();

  //! Lower the fusion and print a kernel
  void printKernel(const CompileParams& compile_params = CompileParams());

  //! Returns if this fusion is noop, for example, trivially forwarding inputs,
  //! or all outputs are size-0 tensors, etc.
  bool isNoOp();

  //! Lower the fusion and evaluate bank conflict info
  //! Returns (tensor, read conflict ways, write conflict ways)
  //! Each tensor can be read/write by multiple expressions, so the ways are
  //! vectors.
  std::unordered_map<
      TensorView*,
      std::pair<std::vector<int64_t>, std::vector<int64_t>>>
  bankConflictInfo(const CompileParams& compile_params = CompileParams());

  //! Return a list of topologically sorted expressions. This only includes
  //! exprs required to generate registered outputs.
  std::vector<Expr*> exprs() const;

  //! Return a vector of fusion inputs that feed this Val
  std::vector<Val*> inputsOf(Val* val);

  //! Return all Vals in math expressions that cannot be eliminated.
  //!
  //! It is generally equivalent to vals that are used to generate
  //! outputs, however, when a multi-output expression exists, and only
  //! some of the outputs are used, the remaining unused outputs are
  //! also included as they must show up in the final code.
  std::vector<Val*> usedMathVals();

  //! Returns all vals that are produced by used math expressions and
  //!  also do not have further consumers.
  //!
  //! In the case of an active multi-output expressions, the returned vector
  //!  will include the expression outputs that did not lead to an fusion
  //!  output.
  std::vector<Val*> terminatingMathVals();

  //! Return all Exprs that use val
  std::unordered_set<Expr*> unordered_uses(const Val* val) const;

  //! Return the Expr that produces val
  Expr* definition(const Val* val) const;

  //! Run fusion segmentation algorithm to create a segmented fusion
  std::unique_ptr<SegmentedFusion> segment(const KernelArgumentHolder& args);

  const std::vector<Val*>& inputs() const {
    return inputs_;
  }

  std::vector<Val*> inputsAndCreated();

  const std::vector<Val*>& outputs() const {
    return outputs_;
  }

  std::vector<Val*> getTerminatingOutputs() const;

  // Aliasing output to input value, this is a WAR to allow inplace update on
  // input tensor.
  // Note: this is not always safe and should be used with extra caution.
  // Currently the only place it's used is in the running stats update for batch
  // normalization.
  //
  // TODO(wujingyue): Rename this method because `input` can be another fusion
  // output.
  //
  // TODO: alias should be made aware to segmentation, so we'll always include
  // the input tensor to the section where output is produced. Currently,
  // aliases of type `PointerArithmetics` are marked after segmentation, but
  // those of type `ReuseBuffer` are marked in fusion definitions.
  NVF_API void aliasOutputToInput(Val* output, Val* input, AllocationType type);

  //! Returns the aliased input of a given output along with an `AliasInfo`
  //! describing how they alias. Returns <nullptr,nullptr> when `output` is not
  //! aliased.
  const AliasInfo& getOutputAlias(const Val* output) const;

  bool isTVUseInfoValid() {
    return all_tv_uses_valid_;
  }

  bool isUpdatingTVUseInfo() {
    return is_during_update_uses_;
  }

  // NOTE: [Fusion managed data]
  //
  // Fusion-managed data is a mechanism to communicate data that survives fusion
  // clone. Managed data can be named or unnamed.
  //
  // For unnamed data, to let fusion manage that data, do the following：
  //   size_t index = fusion.manage(data);  // or
  //   size_t index = fusion.manage(data, clone_fn);
  // This function returns an index which can be used to retrieve the data back.
  // To retrieve the unnamed managed data, do
  //   T data = fusion.getManaged<T>(index); // rvalue
  //   T& data = fusion.getManaged<T>(index); // lvalue
  // To test if fusion have managed data with the given index, do:
  //   bool has_data = fusion.hasManaged(index);
  //
  // For named data, the usage is similar. To manage:
  //   std::string name = "interesting_tvs";
  //   fusion.manage(name, data);  // or
  //   fusion.manage(name, data, clone_fn);
  // To retrieve:
  //   T data = fusion.getManaged<T>(name); // rvalue
  //   T& data = fusion.getManaged<T>(name); // lvalue
  // To check existence:
  //   bool has_data = fusion.hasManaged(name);
  // Note that special names, such as "loop_rotation", are reserved as lowering
  // options.
  //
  // The managed data can be any type. To retrieve managed data, you always need
  // to specify the actual type of the data. For the data whose type already
  // have an overload of IrCloner::clone, fusion will automatically know how to
  // modify it when a fusion clone happens. For these type of data, you can just
  // use the overload of `manage` without the clone function. For example
  //   std::vector<TensorView*> interested_tvs;
  //   size_t index = fusion.manage(interested_tvs);
  // For the data whose type does not have an overload of IrCloner::clone, you
  // need to tell fusion how to transform the data to keep consistency during
  // fusion clone. For example:
  //   struct InputsOutputs {
  //     TensorView* input;
  //     TensorView* output;
  //     bool some_flag;
  //   };
  //   auto clone_fn = [](IrCloner& cloner, std::any data) -> std::any {
  //     InputsOutputs result;
  //     auto d = std::any_cast<InputsOutputs>(data);
  //     result.input = cloner.clone(d.input);
  //     result.output = cloner.clone(d.output);
  //     result.some_flag = d.some_flag;
  //     return result;
  //   };
  //   InputsOutputs data{...};
  //   size_t index = fusion.manage(data, clone_fn);
  //
  // See test FusionManagedData_CUDA for example use cases.
  using CloneFn = std::function<std::any(IrCloner&, std::any)>;

  inline size_t manage(std::any data, CloneFn clone) {
    managed_data_.emplace_back(data, clone);
    return managed_data_.size() - 1;
  }

  inline void manage(std::string key, std::any data, CloneFn clone) {
    managed_named_data_[key] = std::make_pair(data, clone);
  }

  template <typename T>
  inline size_t manage(T data);

  template <typename T>
  inline void manage(std::string key, T data);

  template <typename T>
  inline T getManaged(size_t index) const {
    return std::any_cast<T>(managed_data_.at(index).first);
  }

  template <typename T>
  inline T getManaged(std::string key) const {
    return std::any_cast<T>(managed_named_data_.at(key).first);
  }

  template <typename T>
  inline T& getManaged(size_t index) {
    return std::any_cast<T&>(managed_data_.at(index).first);
  }

  template <typename T>
  inline T& getManaged(std::string key) {
    return std::any_cast<T&>(managed_named_data_.at(key).first);
  }

  //! Try to get managed data by index, checking that we have an entry for it,
  //! and that the entry has not been reset (see stopManaging).
  template <typename T>
  inline std::optional<const T> getManagedSafe(size_t index) const {
    if (hasManaged(index)) {
      return std::any_cast<T>(managed_data_.at(index).first);
    }
    return std::nullopt;
  }

  //! Try to get managed data by key, checking that we have an entry for that
  //! key.
  template <typename T>
  inline std::optional<const T> getManagedSafe(std::string key) const {
    auto it = managed_named_data_.find(key);
    if (it == managed_named_data_.end()) {
      return std::nullopt;
    }
    return std::any_cast<T>(it->second.first);
  }

  //! Disables a piece of managed data. After this, there will still be an entry
  //! but .has_value() will return false. getManagedSafe() should be used in
  //! cases where the data management may have been stopped.
  inline void stopManaging(size_t index) {
    if (!hasManaged(index)) {
      return;
    }
    managed_data_.at(index).first.reset();
  }

  //! Disables a piece of managed data by removing the entry with this key.
  //! getManagedSafe() should be used in cases where the data management may
  //! have been stopped.
  inline void stopManaging(std::string key) {
    auto it = managed_named_data_.find(key);
    if (it == managed_named_data_.end()) {
      return;
    }
    managed_named_data_.erase(it);
  }

  inline bool hasManaged(size_t index) const {
    return index < managed_data_.size() &&
        managed_data_[index].first.has_value();
  }

  inline bool hasManaged(std::string key) const {
    return managed_named_data_.find(key) != managed_named_data_.end();
  }

  //! True if any of tensors has a symblic axis
  bool hasDynamicTransform();

  static IrCloner copy(const Fusion* from, Fusion* to);

  //! During scheduling, this can be set to a non-negative value. If done, then
  //! during execution by KernelExecutor, we will check that this value matches
  //! the corresponding value in LaunchParams.
  int64_t expectedDynamicSmemBytes() const {
    return expected_dynamic_smem_bytes_;
  }

  void setExpectedDynamicSmemBytes(int64_t bytes) {
    expected_dynamic_smem_bytes_ = bytes;
  }

  //! This is a cached version of ir_utils::allTvs that is invalidated. Return a
  //! copy of the vector instead of a reference as it can be invalidated by many
  //! operations. If we returned a reference and are iterating on it while
  //! making modifications to the fusion, it can easily cause a segfault.
  std::vector<TensorView*> allTvs();

  //! Specify id0 and id1 are mapped in the Exact graph. This should
  //! be used only when absolutely necessary.
  //!
  //! Currently, id0->sameAs(id1) needs to hold. It will be an error
  //! otherwise.
  void registerExactMapping(IterDomain* id0, IterDomain* id1);

  bool hasRegisteredExactMappings() const {
    return hasManaged(exact_mappings_key);
  }

  DisjointSets<IterDomain*> registeredExactMappings() const;

  void resetExactMappings();

 protected:
  friend SegmentCandidateFinder;
  friend SegmentedFusion;
  friend class TranslateApplicableWelford;
  friend Val;

  using IrContainer::registerExpr;
  using IrContainer::registerVal;

  //! Register the Val with this fusion
  void registerVal(Val* val) override;

  //! Register expr with this fusion.
  //! When we register an expression, we want to update the dependency tracking
  //! of Vals. If this container is a not a Kernel, it will remove previous
  //! definitions of outputs and register this Expr as the definition. Otherwise
  //! will update definition if not previously set, but will not remove old
  //! definitions.
  void registerExpr(Expr* expr) override;

  //! Clear Expr's from TV uses that are not required to produce outputs from
  //! inputs. Only other place this is used (other than Fusion) is in
  //! Val::uses()
  void resetTvUses();

  //! Declare that TensorView uses need to be updated (but don't actually do
  //! the update).
  void invalidateTvsAndUses() {
    all_tv_uses_valid_ = false;
    all_tvs_ptr_.reset();
  }

 private:
  // Fusion inputs and outputs
  std::vector<Val*> inputs_;
  std::vector<Val*> outputs_;

  // io alias pointing from output to input
  std::unordered_map<const Val*, AliasInfo> io_alias_;

  // Records if the current use data in the IR nodes are valid
  //  the states are either all valid or all invalid
  bool all_tv_uses_valid_ = false;
  bool is_during_update_uses_ = false;

  // See note [Fusion managed data]
  // Stores data of arbitrary type as std::any, and how the data will be cloned
  // when the fusion is cloned as CloneFn. The primary task that the clone
  // function does is to update pointers to IR nodes inside the data structure
  // being managed with new pointers. By default, the clone function will be
  // the `defaultCloneFunction` below, which just dispatch to IrCloner::clone.
  std::vector<std::pair<std::any, CloneFn>> managed_data_;
  std::unordered_map<std::string, std::pair<std::any, CloneFn>>
      managed_named_data_;

  // If set to a non-negative value during scheduling, this will be checked by
  // the executor.
  int64_t expected_dynamic_smem_bytes_ = -1LL;

  std::unique_ptr<std::vector<TensorView*>> all_tvs_ptr_ = nullptr;

  inline static const std::string exact_mappings_key = "exact_mappings";
};

template <typename T>
std::any defaultCloneFunction(IrCloner& cloner, std::any data) {
  auto cloned_data = cloner.clone(std::any_cast<T>(data));
  // Adding a static_assert to improve error message. Without this
  // static_assert, the following cast will still fail, but the error message
  // will be unreadable.
  static_assert(
      std::is_convertible_v<decltype(cloned_data), T>,
      "IrCloner::clone returns a data type that is not compatible with the original managed data type. "
      "Likely you will need to check IrCloner::clone for your data type.");
  // Convert the result of the clone back to T before assigning to std::any.
  // This ensures the type of the std::any does not change over the clone of
  // fusion.
  return std::any((T)cloned_data);
}

template <typename T>
size_t Fusion::manage(T data) {
  return manage(std::any(data), defaultCloneFunction<T>);
}

template <typename T>
void Fusion::manage(std::string key, T data) {
  return manage(key, std::any(data), defaultCloneFunction<T>);
}

} // namespace nvfuser
