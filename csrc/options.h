// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/macros/Export.h>
#include <c10/util/Exception.h>
#include <exceptions.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace nvfuser {

//! Types of debug print-outs
//!
//! These can be set through the `NVFUSER_DUMP` environment variable
//!
enum class DebugDumpOption {
  FusionIr, //!< Dump the Fusion IR before lowering
  FusionIrMath, //!< Dump just the compute (math) part of the Fusion IR
  FusionIrPresched, //!< Dump the Fusion IR before it is scheduled.
  FusionIrConcretized, //!< Dump the Fusion IR after concretization
  FusionIrPreseg, //!< Dump the Fusion IR after pre-segmenter optimization
  KernelIr, //!< Dump the compiler Kernel IR
  ComputeAtMap, //!< Dump the computeAt map
  CudaKernel, //!< Dump the generated CUDA C++ kernel code
  CudaFull, //!< Dump the complete CUDA C++ code
  CudaToFile, //!< Dump CUDA Strings to File
  DebugInfo, //!< Embed line info and debug info to compiled kernel, and dump
             //!< the full CUDA C++ code
  LaunchParam, //!< Dump the Launch parameters of kernel
  FusionSegments, //!< Dump Segmented Fusion Graph
  FusionSegmenterLog, //!< Dump Detailed Segmenter Logging
  FusionArgs, //!< Print the runtime fusion arguments
  KernelArgs, //!< Print the runtime kernel arguments when launching kernels
  EffectiveBandwidth, //! Measure kernel performance and print effective
                      //! bandwidth
  FusionSegmentsDrawing, //!< Dump Segmented Fusion Graph
  PrintPtxasLog, //!< Print the ptxas verbose log including register usage
  BufferReuseInfo, //!< Dump the analysis details of local/shared buffer re-use
  SchedulerDebug, //! Dump scheduler heuristic parameters
  SchedulerVerbose, //! Dump detailed scheduler logging
  ParallelDimensions, //!< Dump known parallel dimensions
  Halo, //! Halo information of tensors
  PerfDebugVerbose, //! When running kernels, print verbose information
                    //! associated with what's running
  PreSegmenterLogging,
  PythonDefinition, //! Python Frontend Fusion Definition.
  PythonFrontendDebug, //! Python Frontend debug information.
  TransformPropagator, //! When running TransformPropagator, print propagation
                       //! path and replay result
  Cubin, //! Dump compiled CUBIN
  Sass, // Dump disassembled SASS
  Ptx, //! Dump compiled PTX
  BankConflictInfo, //! Dump bank confliction info
  SyncMap, //! RAW dependency info
  LowerVerbose, //! Print all passes' transform in GpuLower::lower
  ExprSimplification, //! Print all passes' transform in simplifyExpr
  ExprSort, //! Print merging decisions on expression sorting
  ExprSortVerbose, //! Print verbose debug info on expression sorting
  LoopRotation, //! Print loop rotation log
  Occupancy, // Dump occupancy
  IndexType, //! Print the index type of the launched kernel
  EndOfOption //! Placeholder for counting the number of elements
};

//! Types of features to enable
//!
//! These can be set through the `NVFUSER_ENABLE` environment variable
//!
enum class EnableOption {
  IdModel, //! Enable IdModel
  KernelDb, //! Enable Kernel Database
  KernelProfile, //! Enable intra-kernel performance profiling
  MemoryPromotion, //! Enable promotion of memory types for non-pointwise ops
  StaticFusionCount, //! Enable using single static count in kernel name
  WarnRegisterSpill, //! Enable warnings of register spill
  EndOfOption //! Placeholder for counting the number of elements
};

//! Types of features to disable
//!
//! These can be set through the `NVFUSER_DISABLE` environment variable
//!
enum class DisableOption {
  CompileToSass, //! Disable direct compilation to sass so the ptx can be
                 //! examined
  ExprSimplify, //! Disable expression simplifier
  Fallback, //! Disable fallback
  Fma, //! Disable FMA instructions
  GroupedGridWelfordOuterOpt, //! Disable use of outer-optimized
                              //! grouped grid welford kernel
  IndexHoist, //! Disable index hoisting
  MagicZero, //! Disable nvfuser_zero
  Nvtx, //! Disable NVTX instrumentation
  ParallelCompile, //! Disable compiling Fusion segments in parallel
  ParallelSerde, //! Disable deserializing FusionExecutorCache in parallel
  PredicateElimination, //! Disable predicate elimination
  KernelReuse, //! Disable re-using cached FusionKernelRuntimes with different
               //! input shapes
  VarNameRemapping, //! Disable variable name remapping
  WelfordVectorization, //! Disable vectorizaton of Welford ops
  ReuseMismatchedTypeRegisters, //! Disable explicitly re-using registers unless
                                //! types match
  EndOfOption //! Placeholder for counting the number of elements
};

//! Options to set for Fusion Profiling.  Whenever the profiler
//! is enabled, its output can be queried from the FusionProfile object.
//! All options enable the profiler.
//!
//! These can be set through the `NVFUSER_PROF` environment variable
//!
enum class ProfilerOption {
  Enable, //! Enables the profiler.
  EnableNocupti, //! Enables the profiler, but disables CUPTI specific
                 //! profiling inorder to measure true host time without
                 //! overhead.
  Print, //! Enables the profiler and prints the output to the console.
  PrintNocupti, //! Enables the profiler, disables CUPTI specific
                //! profiling inorder to measure true host time without
                //! overhead, and prints the output to the console.
  PrintVerbose, //! Enables the profiler and prints a complete set of columns
                //! to the console.  WARNING: The output is will wrap on small
                //! screens!
  EndOfOption //! Placeholder for counting the number of elements
};

//! The base template class for the options such as EnableOption
template <typename OptionEnum>
class Options {
 public:
  Options() : options_(getOptionsFromEnv()) {}

  bool has(OptionEnum option) const {
    return options_.count(option);
  }

  bool hasAny() const {
    return !options_.empty();
  }

  const std::vector<std::string>& getArgs(OptionEnum option) const {
    NVF_ERROR(has(option), "Option not set");
    return options_.at(option);
  }

  void set(OptionEnum option_type, std::vector<std::string> option = {}) {
    options_[option_type] = option;
  }

  void unset(OptionEnum option_type) {
    options_.erase(option_type);
  }

  static std::unordered_map<OptionEnum, std::vector<std::string>>
  getOptionsFromEnv();

 protected:
  std::unordered_map<OptionEnum, std::vector<std::string>> options_;
};

//! Utility class to temporarily overrride the Enable options,
//! including those provided by the environment variable
template <typename OptionEnum>
class OptionsGuard {
 public:
  OptionsGuard() : prev_options_(getCurOptions()) {}

  ~OptionsGuard() {
    getCurOptions() = prev_options_;
  }

  static Options<OptionEnum>& getCurOptions();

 private:
  Options<OptionEnum> prev_options_;
};

// DebugDump options
template <>
std::unordered_map<DebugDumpOption, std::vector<std::string>> Options<
    DebugDumpOption>::getOptionsFromEnv();

using DebugDumpOptions = Options<DebugDumpOption>;

template <>
Options<DebugDumpOption>& OptionsGuard<DebugDumpOption>::getCurOptions();

using DebugDumpOptionsGuard = OptionsGuard<DebugDumpOption>;

bool isDebugDumpEnabled(DebugDumpOption option);

const std::vector<std::string>& getDebugDumpArguments(DebugDumpOption option);

// Enable options
template <>
std::unordered_map<EnableOption, std::vector<std::string>> Options<
    EnableOption>::getOptionsFromEnv();

using EnableOptions = Options<EnableOption>;

bool isOptionEnabled(EnableOption option);

const std::vector<std::string>& getEnableOptionArguments(EnableOption option);

template <>
Options<EnableOption>& OptionsGuard<EnableOption>::getCurOptions();

using EnableOptionsGuard = OptionsGuard<EnableOption>;

// Disable options
template <>
std::unordered_map<DisableOption, std::vector<std::string>> Options<
    DisableOption>::getOptionsFromEnv();

using DisableOptions = Options<DisableOption>;

bool isOptionDisabled(DisableOption option);

const std::vector<std::string>& getDisableOptionArguments(DisableOption option);

template <>
Options<DisableOption>& OptionsGuard<DisableOption>::getCurOptions();

using DisableOptionsGuard = OptionsGuard<DisableOption>;

// Profiler Options

template <>
std::unordered_map<ProfilerOption, std::vector<std::string>> Options<
    ProfilerOption>::getOptionsFromEnv();

using ProfilerOptions = Options<ProfilerOption>;

// Specific queries for the Profiler Options
bool isProfilerEnabled();
bool isProfilerEnabledWithoutCupti();
bool isProfilerPrintingEnabled();
bool isProfilerPrintingVerbose();

const std::vector<std::string>& getProfilerOptionArguments(
    ProfilerOption option);

template <>
Options<ProfilerOption>& OptionsGuard<ProfilerOption>::getCurOptions();

using ProfilerOptionsGuard = OptionsGuard<ProfilerOption>;

} // namespace nvfuser
