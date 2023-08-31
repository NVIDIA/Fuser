// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ATen/core/ivalue.h>
#include <c10/macros/Export.h>
#include <c10/util/Exception.h>

#include <debug.h>
#include <executor_params.h>
#include <ir/base_nodes.h>
#include <ir/container.h>
#include <iter_visitor.h>

#include <any>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nvfuser {

//! Usage: FusionGuard and Fusion are required user interfaces for any operation
//! underlying the code generator. In order to create values, expressions, and
//! generate code a Fusion instance must be active. It is the responsibility of
//! the user to create a Fusion instance and register it with the fusion guard.
//! The simplest example of this is:
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

//! Fusion Guard is our "context manager". It holds the actrive fusion and
//! allows it to be accessed anywhere through FusionGuard::getCurFusion()
class TORCH_CUDA_CU_API FusionGuard {
 public:
  Fusion* prev_fusion;

  //! Set the active fusion so it can be manipulated.
  explicit FusionGuard(Fusion* fusion);

  ~FusionGuard();

  static Fusion* getCurFusion();
  static void setCurFusion(Fusion* fusion);
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
class TORCH_CUDA_CU_API Fusion : public IrContainer {
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
  std::ostream& print(std::ostream& os, bool include_tensor_transforms = true);

  //! Print to default debugging output stream
  std::ostream& print() {
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
  std::unordered_map<TensorView*, std::pair<std::vector<int>, std::vector<int>>>
  bankConflictInfo(const CompileParams& compile_params = CompileParams());

  //! Return a list of topologically sorted expressions. This only includes
  //! exprs required to genereate registered outputs.
  std::vector<Expr*> exprs();

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

  //! Indicate to kernel to set itself up to generate random numbers
  bool isStochastic();

  //! Run fusion segmentation algorithm to create a segmented fusion
  std::unique_ptr<SegmentedFusion> segment(const KernelArgumentHolder& args);

  const auto& inputs() const {
    return inputs_;
  }

  std::vector<Val*> inputsAndCreated();

  const auto& outputs() const {
    return outputs_;
  }

  std::vector<Val*> getTerminatingOutputs() const;

  // Aliasing output to input value, this is a WAR to allow inplace update on
  // input tensor.
  // Note: this is not always safe and should be used with extra caution.
  // Currently the only place it's used is in the running stats update for batch
  // normalization.
  // TODO: alias should be made aware to segmentation, so we'll always include
  // the input tensor to the section where output is produced.
  void aliasOutputToInput(Val* output, Val* input);

  //! Return the aliased input of a given output or nullptr if not aliased
  Val* getOutputAlias(Val* output);

  //! Get indices of aliased outputs
  std::unordered_set<int> getIndicesOfAliasedOutputs() const;

  //! Get alias mappings from fusion outputs to inputs
  std::vector<std::pair<int, int>> getOutputToInputAliasIndices() const;

  // mark input at index to be permuted by permutation
  void setPermutationOnInput(int index, std::vector<int64_t> permutation) {
    permuted_input_map_.insert({index, permutation});
  }

  // mark output at index to be restored by permutation
  void setPermutationOnOutput(int index, std::vector<int64_t> permutation) {
    permuted_output_map_.insert({index, permutation});
  }

  // return a map of indices to permutation, which indicates all input tensors
  // that needs to be permuted
  const PermutationMap& getPermutationInputMap() const {
    return permuted_input_map_;
  }

  // return a map of indices to permutation, which indicates all output tensors
  // that needs to be permuted
  const PermutationMap& getPermutationOutputMap() const {
    return permuted_output_map_;
  }

  bool isTVUseInfoValid() {
    return all_tv_uses_valid_;
  }

  bool isUpdatingTVUseInfo() {
    return is_during_update_uses_;
  }

  const auto& ioAlias() const {
    return io_alias_;
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

 protected:
  friend SegmentCandidateFinder;
  friend SegmentedFusion;
  friend class TranslateApplicableWelford;
  friend Val;

  static IrCloner copy(const Fusion* from, Fusion* to);

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
  void invalidateTvUses() {
    all_tv_uses_valid_ = false;
  }

 private:
  // Determine if the two values are compatible for aliasing
  // Same DataType, ValType, and number of dimensions
  bool isAliasCompatible(Val* left, Val* right);

 private:
  // Fusion inputs and outputs
  std::vector<Val*> inputs_;
  std::vector<Val*> outputs_;

  // io alias pointing from output to input
  std::unordered_map<Val*, Val*> io_alias_;

  // See Note [ Permutation support in nvfuser ]
  // map from indices of input tensor to permutation
  PermutationMap permuted_input_map_;
  // map from indices of output tensor to permutation
  PermutationMap permuted_output_map_;

  // Records if the current use data in the IR nodes are valid
  //  the states are either all valid or all invalid
  bool all_tv_uses_valid_ = false;
  bool is_during_update_uses_ = false;

  std::vector<std::pair<std::any, CloneFn>> managed_data_;
  std::unordered_map<std::string, std::pair<std::any, CloneFn>>
      managed_named_data_;
};

} // namespace nvfuser
