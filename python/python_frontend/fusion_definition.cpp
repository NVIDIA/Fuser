// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <debug.h>
#include <distributed_tensor.h>
#include <fusion_profiler.h>
#include <host_ir/pass/stream_parallel_type.h>
#include <instrumentation.h>
#include <multidevice/utils.h>
#include <options.h>
#include <preseg_passes/pre_segmenter.h>
#include <python_frontend/fusion_cache.h>
#include <python_frontend/fusion_definition.h>
#include <python_frontend/translation.h>
#include <runtime/executor_kernel_arg.h>
#include <runtime/fusion_kernel_runtime.h>
#include <scheduler/compile_time_info.h>
#include <scheduler/scheduler_types.h>
#include <utils.h>
#include <validator_utils.h>

// Require namespace for perf scope instrumentation
using namespace nvfuser::inst;

namespace nvfuser::python_frontend {

FusionDefinition::FusionDefinition(
    std::optional<size_t> id,
    size_t max_length,
    bool use_multidevice_executor,
    CommunicatorBackend backend_type)
    : FusionState(),
      max_length_(max_length),
      fusion_id_(id),
      fusion_cache_(FusionCache::get()),
      trie_node_(nullptr),
      prev_fusion_(nullptr),
      user_sched_(nullptr),
      ops(this),
      sched(this),
      use_multidevice_executor_(use_multidevice_executor),
      backend_type_(backend_type) {}

FusionCache* FusionDefinition::fusionCache() const {
  NVF_ERROR(fusion_cache_ != nullptr, "FusionCache pointer is null!");
  return fusion_cache_;
}

FusionDefinition* FusionDefinition::setupDefinition() {
  NVF_CHECK(max_length_ > 0, "Can't make a FusionDefinition with 0 records!");
  NVF_CHECK(!id().has_value(), "Fusion Schedule is already found!");
  trie_node_ = fusionCache()->rootTriePtr();
  return this;
}

void FusionDefinition::finalizeDefinition() {
  FUSER_PERF_SCOPE("FusionDefinition::finalizeDefinition");
  auto child_node = fusionCache()->queryChildren(trie_node_, end_record_.get());
  if (!child_node.has_value()) {
    if (isDebugDumpEnabled(DebugDumpOption::PythonFrontendDebug)) {
      debug() << "\nFusionDefinition: Terminal Node not found.\n";
    }
    trie_node_ = fusionCache()->createChild(trie_node_, end_record_.get());
    fusion_id_ = std::optional<size_t>(trie_node_->fusion_id);
    try {
      NVF_CHECK(id().has_value(), "Invalid fusion id!");

      if (isDebugDumpEnabled(DebugDumpOption::PythonDefinition)) {
        print(debug());
      }

      buildFusionIr(preschedFusion());
      verifyTensorDimensions();
    } catch (const std::exception& e) {
      // Exception thrown after fusionCache()->createChild wouldn't be visible
      // by fusion cache, if the exception is suppressed on the python side. We
      // explicitly set the exception message on the terminal trie node, so
      // we'll be able to throw the same exception again when user tries to
      // create the same fusion entry.
      trie_node_->setException(e.what());
      fusion_id_ = std::nullopt;
      throw;
    }

    // The FusionState creates a mapping from CPP Fusion to its State objects.
    // Since the CPP Fusion is cached in FusionCache and the FusionState is
    // temporary, the information linking CPP Fusion and Python
    // FusionDefinition is stored in FusionCache.
    FusionSchedules* fs =
        fusionCache()->queryFusionSchedules(fusion_id_.value());
    fs->inputs_fid_ = inputs();
    fs->outputs_fid_ = outputs();
    fs->extents_fid_ = extents();
    fs->map_value_to_fid_ = getValueMap();

    if (isDebugDumpEnabled(DebugDumpOption::FusionIrOriginal)) {
      printIr();
    }
  } else {
    if (isDebugDumpEnabled(DebugDumpOption::PythonFrontendDebug)) {
      debug() << "\nFusionDefinition: Terminal Node found!\n";
    }
    trie_node_ = child_node.value();
    std::optional<std::string> opt_e = trie_node_->getException();
    // rethrow the exception message if the cached FusionDefinition fails to
    // build a proper fusion earlier.
    NVF_CHECK(!opt_e.has_value(), opt_e.value());
    fusion_id_ = std::optional<size_t>(trie_node_->fusion_id);

    // A CPP fusion already exists in the FusionCache for this FusionDefinition.
    // In this case, a new CPP Fusion is not created, so the mapping from CPP
    // fusion to Python FusionDefinition is not initialized. This state is
    // stored within FusionSchedules and is retrieved for this FusionDefinition.
    FusionSchedules* fs =
        fusionCache()->queryFusionSchedules(fusion_id_.value());
    inputs_fid_ = fs->inputs_fid_;
    outputs_fid_ = fs->outputs_fid_;
    extents_fid_ = fs->extents_fid_;
    map_value_to_fid_ = fs->map_value_to_fid_;
  }

  NVF_ERROR(
      num_recording_states_presched_ == 0,
      "Expected number of recording states for prescheduled fusion to be "
      "uninitialized.");
  num_recording_states_presched_ = (int64_t)recording_state_.size();
}

void FusionDefinition::findHiddenTensorViews(Fusion* fusion) {
  NVF_ERROR(fusion != nullptr);

  // Filter Tensor states
  std::vector<State> tensor_states;
  std::copy_if(
      recording_state_.begin(),
      recording_state_.end(),
      std::back_inserter(tensor_states),
      [](const State& s) { return s.stype == serde::StateType::Tensor; });

  // Get corresponding CPP values and add to set for membership check.
  std::unordered_set<Val*> known_tensor_vals;
  std::transform(
      tensor_states.begin(),
      tensor_states.end(),
      std::inserter(known_tensor_vals, known_tensor_vals.end()),
      [this](State s) { return getFusionState(s.index); });

  // Get set difference between CPP Fusion and Python FusionDefinition
  std::vector<Val*> all_vals = fusion->usedMathVals();
  std::vector<Val*> new_fusion_tvs;
  std::copy_if(
      all_vals.begin(),
      all_vals.end(),
      std::back_inserter(new_fusion_tvs),
      [&](Val* v) {
        return v->isA<TensorView>() && known_tensor_vals.count(v) == 0;
      });

  // Short-Circuit: No new TensorViews found
  if (new_fusion_tvs.empty()) {
    return;
  }

  // Add missing TensorViews to FusionDefinition
  for (Val* v : new_fusion_tvs) {
    addTensor(v->as<TensorView>());
  }
}

void FusionDefinition::updateSymbolicStates(
    const std::unordered_map<Val*, Val*>& symbolic_to_concretized_map) {
  for (const State& s : recording_state_) {
    // Only update Tensor and Scalar states
    if (s.stype != serde::StateType::Tensor &&
        s.stype != serde::StateType::Scalar) {
      continue;
    }

    Val* old_value = getFusionState(s.index);

    // Skip replacement if unnecessary
    if (symbolic_to_concretized_map.count(old_value) == 0) {
      continue;
    }

    // Update symbolic states with new concretized values
    setFusionState(s.index, symbolic_to_concretized_map.at(old_value));
  }
}

void FusionDefinition::verifyTensorDimensions() {
  NVF_CHECK(id().has_value(), "Invalid fusion id!");

  std::vector<Tensor> all_tensors = tensors();
  for (const Tensor& t : all_tensors) {
    Val* v = getFusionState(t.index);
    NVF_ERROR(v->isA<TensorView>(), v->toString());
    const int64_t tv_ndims = v->as<TensorView>()->nDims();
    NVF_ERROR(
        tv_ndims == (int64_t)t.dims,
        "Expected TensorView to have same number of dimensions as Tensor but "
        "got: ",
        tv_ndims,
        " and ",
        t.dims);
  }
}

bool FusionDefinition::existSchedule(const KernelArgumentHolder& args) {
  FUSER_PERF_SCOPE("FusionDefinition::existsSchedule");
  NVF_CHECK(id().has_value(), "FusionDefinition definition does not exist!");
  FusionSchedules* scheds = fusionCache()->queryFusionSchedules(id().value());
  int8_t device = getCommonDeviceCUDA(args);
  NVF_CHECK(
      args.empty() || device > -1, "Inputs are not all on the same device!");
  return fusionCache()->existUserSchedule(scheds, args, device);
}

void FusionDefinition::setupSchedule(
    const KernelArgumentHolder& args,
    bool overwrite_existing_schedule) {
  FUSER_PERF_SCOPE("FusionDefinition::setupSchedule");
  NVF_CHECK(id().has_value(), "FusionDefinition definition does not exist!");
  FusionSchedules* scheds = fusionCache()->queryFusionSchedules(id().value());
  int8_t device = getCommonDeviceCUDA(args);
  NVF_CHECK(
      args.empty() || (device > -1 && device == args.getDeviceIndex()),
      "Inputs are not all on the same device!");

  // NOTE: Clear user schedule state in setupSchedule.
  // Scheduling the fusion can add states to recording_state.
  // Remove any schedule-only states before applying new schedule.
  size_t num_states_to_remove =
      recording_state_.size() - num_recording_states_presched_;
  for (size_t rnd = 0; rnd < num_states_to_remove; ++rnd) {
    recording_state_.pop_back();
  }

  user_sched_ = fusionCache()->createUserSchedule(
      scheds, args, device, overwrite_existing_schedule);

  // Create scheduler data cache
  user_sched_->data_cache = std::make_unique<HeuristicDataCache>();

  // Building a new Fusion container for scheduling with definition such that
  // the definition's tensor data members refer to the corresponding IR objects
  // needed for scheduling. A simple copy of the container would mean the data
  // members that represent tensors would refer to the IR objects in the
  // original and not the copy needed for scheduling.
  buildFusionIr(user_sched_->scheduled_fusion.get());

  // Add TensorViews created by composite operations to Python FusionDefinition.
  findHiddenTensorViews(user_sched_->scheduled_fusion.get());

  // Concretize fusion
  std::unordered_map<Val*, Val*> symbolic_to_concrete_map =
      DynamicTransform::concretizeFusion(
          user_sched_->scheduled_fusion.get(), args);

  // Update symbolic values to their new concretized values.
  // Users will access concretized values in schedule function.
  updateSymbolicStates(symbolic_to_concrete_map);

  // Create runtime info for schedulers
  Fusion* user_schedule_fusion = user_sched_->scheduled_fusion.get();
  user_sched_->runtime_info = std::make_unique<SchedulerRuntimeInfo>(
      user_schedule_fusion,
      args,
      /*precomuted_values=*/nullptr,
      user_schedule_fusion->allTvs());

  // Manually setting the fusion guard as there is not a good way of using a
  // guard in a local scope across the schedule function
  prev_fusion_ = FusionGuard::getCurFusion();
  FusionGuard::setCurFusion(user_sched_->scheduled_fusion.get());
}

void FusionDefinition::finalizeSchedule(const KernelArgumentHolder& args) {
  FUSER_PERF_SCOPE("FusionDefinition::finalizeSchedule");

  FusionGuard::setCurFusion(prev_fusion_);
  user_sched_->runtime_info.reset();
  prev_fusion_ = nullptr;

  // NOTE: Clear user schedule state in setupSchedule.
  // Users can access schedule objects after scheduling the fusion.
}

void FusionDefinition::setupMultideviceSchedule() {
  // FusionDefinition.multidevice_schedule may create new Exprs (e.g. DID
  // splits), which will be added to the presched fusion.
  prev_fusion_ = FusionGuard::getCurFusion();
  FusionGuard::setCurFusion(preschedFusion());
}

void FusionDefinition::finalizeMultideviceSchedule() {
  FusionGuard::setCurFusion(prev_fusion_);
}

void FusionDefinition::print(std::ostream& os) const {
  if (id().has_value()) {
    os << "\ndef nvfuser_fusion_id" << id().value();
  } else {
    os << "\ndef nvfuser_incomplete_fusion";
  }
  os << "(fd : FusionDefinition) -> None :\n";
  os << std::dec;
  for (auto& rec : recording_) {
    // Skip inline defined records
    if (!rec.get()->inlineDef()) {
      os << "    ";
      rec->print(os);
      os << "\n";
    }
  }
  os << std::endl;
}

std::pair<KernelArgumentHolder, std::vector<Sharding>> FusionDefinition::
    execute(
        KernelArgumentHolder args,
        std::optional<int8_t> selected_device,
        bool override_user_schedule,
        bool capture_debug_output,
        bool profile,
        std::vector<std::string> _enable_options,
        std::vector<std::string> _disable_options) const {
  debug_output_ = std::nullopt;
  std::stringstream debug_ss;
  DebugStreamGuard dsg(capture_debug_output ? debug_ss : std::cout);
  args.setDeviceIndex(selected_device);
  NVF_CHECK(id().has_value(), "Valid fusion schedule is not available!");

  auto scheds = fusionCache()->queryFusionSchedules(id().value());

  if (profile) {
    ProfilerOptionsGuard::getCurOptions().set(ProfilerOption::Enable);
  }

  EnableOptionsGuard enable_opt_guard;
  for (const auto& _enable_option : _enable_options) {
    std::optional<EnableOption> opt = stringToEnableOption(_enable_option);
    NVF_CHECK(opt.has_value(), "Unrecognized enable_option: ", _enable_option);
    EnableOptionsGuard::getCurOptions().set(opt.value());
  }

  DisableOptionsGuard disable_opt_guard;
  for (const auto& _disable_option : _disable_options) {
    std::optional<DisableOption> opt = stringToDisableOption(_disable_option);
    NVF_CHECK(
        opt.has_value(), "Unrecognized disable_option: ", _disable_option);
    DisableOptionsGuard::getCurOptions().set(opt.value());
  }

  auto find_user_schedule = [&]() -> const UserSchedule* {
    if (override_user_schedule) {
      return nullptr;
    }

    auto user_sched_id = fusionCache()->queryUserScheduleId(scheds, args);
    if (!user_sched_id.has_value()) {
      return nullptr;
    }

    NVF_CHECK(
        args.empty() || args.getDeviceIndex() > -1,
        "Inputs are not all on the same device or don't match selection!");
    const UserSchedule& user_sched = fusionCache()->queryUserSchedule(
        scheds, user_sched_id.value(), args.getDeviceIndex());
    return &user_sched;
  };
  const auto* user_sched = find_user_schedule();

  KernelArgumentHolder outputs;
  if (user_sched == nullptr) {
    if (use_multidevice_executor_) {
      if (scheds->multi_device_executor == nullptr) {
        MultiDeviceExecutorParams params;
        params.lower.communicator_backend = backend_type_;
        scheds->multi_device_executor = std::make_unique<MultiDeviceExecutor>(
            std::make_unique<Fusion>(*scheds->preschedFusion()),
            Communicator::getInstance(),
            std::move(params));
      }
      std::cout << "MultiDeviceExecutor" << std::endl;
      outputs = scheds->multi_device_executor->runWithInput(args);
    } else {
      std::cout << "FEC" << std::endl;
      scheds->createExecutorIfNotExists();
      outputs = scheds->auto_gen_schedules->runFusionWithInputs(
          args, std::nullopt, args.getDeviceIndex());
    }
  } else {
    NVF_ERROR(
        !use_multidevice_executor_,
        "multidevice_executor is not supported "
        "for user-defined schedules.");
    if (isProfilerEnabledWithCupti()) {
      FusionProfiler::start();
      FusionProfiler::createSegments(1);
    }

    scheds->last_user_def_scheduled_ir = user_sched->scheduled_fusion.get();
    scheds->last_user_def_executor = user_sched->executor.get();

    if (user_sched->heuristic_params == nullptr) {
      // Manual schedule
      if (!user_sched->executor->isCompiled()) {
        user_sched->executor->compile(user_sched->scheduled_fusion.get(), args);
      }
      outputs = user_sched->executor->run(args);
    } else {
      // Automatic scheduler was used for UserSchedule.
      // Pass launch and compile params to compileFusion and runFusion.
      if (!user_sched->executor->isCompiled()) {
        user_sched->executor->compile(
            user_sched->scheduled_fusion.get(),
            args,
            user_sched->heuristic_params->lparams,
            user_sched->heuristic_params->cparams,
            user_sched->heuristic_params->scheduler_type);
      }
      outputs = user_sched->executor->run(
          args,
          {},
          user_sched->heuristic_params->lparams,
          user_sched->heuristic_params->cparams);
    }

    if (isProfilerEnabledWithCupti()) {
      FusionProfiler::segment(0).scheduler("user");
      FusionProfiler::stop();
      if (isProfilerPrintingEnabled()) {
        debug() << FusionProfiler::profile();
      }
    }
  }

  if (profile) {
    ProfilerOptionsGuard::getCurOptions().unset(ProfilerOption::Enable);
  }

  if (capture_debug_output) {
    debug_output_ = debug_ss.str();
  }

  std::vector<Sharding> output_shardings;
  if (user_sched == nullptr) {
    Fusion* fusion = use_multidevice_executor_
        ? scheds->preschedFusion()
        : scheds->auto_gen_schedules->getMostRecentKernelRuntime()
              ->fusionSegments()
              ->completeFusion();
    output_shardings = getOutputShardings(fusion);
    NVF_ERROR(
        output_shardings.empty() ||
            std::ssize(output_shardings) == outputs.size(),
        "Found ",
        std::ssize(output_shardings),
        " output shardings but expected ",
        outputs.size(),
        " or 0.");
  }

  return std::make_pair(std::move(outputs), std::move(output_shardings));
}

std::string FusionDefinition::fusionIr() {
  NVF_CHECK(id().has_value(), "Invalid fusion definition!");
  std::stringstream ss;
  preschedFusion()->print(ss, false);
  return ss.str();
}

UserSchedule* FusionDefinition::userSchedule() {
  NVF_CHECK(id().has_value(), "Invalid fusion definition!");

  if (user_sched_ == nullptr) {
    NVF_THROW("User schedule is not defined.");
  }
  return user_sched_;
}

std::string FusionDefinition::userScheduleIr() {
  NVF_CHECK(id().has_value(), "Invalid fusion definition!");

  if (user_sched_ == nullptr) {
    return "User schedule is not defined.";
  }

  std::stringstream ss;
  user_sched_->scheduled_fusion->print(ss, false);
  return ss.str();
}

std::string FusionDefinition::lastCudaCode(
    bool intrinsic_code,
    bool override_user_schedule) const {
  std::string result;
  NVF_CHECK(id().has_value(), "Invalid fusion definition!");
  auto scheds = fusionCache()->queryFusionSchedules(id().value());
  auto user_exec = scheds->last_user_def_executor;

  if (!override_user_schedule && (user_exec != nullptr)) {
    if (intrinsic_code) {
      result = user_exec->compiledKernel()->getStructuredCode();
    } else {
      result = user_exec->compiledKernel()->kernelString();
    }
  } else {
    NVF_CHECK(
        scheds->auto_gen_schedules != nullptr,
        "Fusion ",
        *id(),
        " has never been executed via FusionExecutorCache.");
    result = scheds->auto_gen_schedules->getMostRecentCode(intrinsic_code);
  }
  return result;
}

std::string FusionDefinition::cudaCodeFor(
    KernelArgumentHolder args,
    bool intrinsic_code,
    bool override_user_schedule) const {
  NVF_CHECK(id().has_value(), "Invalid fusion definition!");
  auto scheds = fusionCache()->queryFusionSchedules(id().value());

  if (!override_user_schedule) {
    auto device = getCommonDeviceCUDA(args);
    NVF_CHECK(
        args.empty() || device > -1, "Inputs are not all on the same device!");
    auto user_sched_id = fusionCache()->queryUserScheduleId(scheds, args);
    if (user_sched_id.has_value()) {
      auto& user_sched = fusionCache()->queryUserSchedule(
          scheds, user_sched_id.value(), device);
      auto user_exec = user_sched.executor.get();
      if (intrinsic_code) {
        return user_exec->compiledKernel()->getStructuredCode();
      } else {
        return user_exec->compiledKernel()->kernelString();
      }
    }
  }
  NVF_CHECK(
      scheds->auto_gen_schedules != nullptr,
      "Fusion ",
      *id(),
      " has never been executed via FusionExecutorCache.");
  return scheds->auto_gen_schedules->getCodeFor(args, intrinsic_code);
}

std::string FusionDefinition::lastScheduledFusionIr(
    bool tensor_transforms,
    bool override_user_schedule) const {
  std::string result;
  NVF_CHECK(id().has_value(), "Invalid fusion definition!");
  auto scheds = fusionCache()->queryFusionSchedules(id().value());
  auto user_sched_ir = scheds->last_user_def_scheduled_ir;

  if (!override_user_schedule && (user_sched_ir != nullptr)) {
    std::stringstream ss;
    user_sched_ir->print(ss, tensor_transforms);
    result = ss.str();
  } else {
    NVF_CHECK(
        scheds->auto_gen_schedules != nullptr,
        "Fusion ",
        *id(),
        " has never been executed via FusionExecutorCache.");
    result =
        scheds->auto_gen_schedules->getMostRecentScheduledIr(tensor_transforms);
  }
  return result;
}

std::string FusionDefinition::scheduledFusionIrFor(
    const KernelArgumentHolder& args,
    bool tensor_transforms,
    bool override_user_schedule) const {
  NVF_CHECK(id().has_value(), "Invalid fusion definition!");
  auto scheds = fusionCache()->queryFusionSchedules(id().value());

  if (!override_user_schedule) {
    auto device = getCommonDeviceCUDA(args);
    NVF_CHECK(
        args.empty() || (device > -1 && device == args.getDeviceIndex()),
        "Inputs are not all on the same device!");
    auto user_sched_id = fusionCache()->queryUserScheduleId(scheds, args);
    if (user_sched_id.has_value()) {
      auto& user_sched = fusionCache()->queryUserSchedule(
          scheds, user_sched_id.value(), device);
      auto user_sched_ir = user_sched.scheduled_fusion.get();
      std::stringstream ss;
      user_sched_ir->print(ss, tensor_transforms);
      return ss.str();
    }
  }
  NVF_CHECK(
      scheds->auto_gen_schedules != nullptr,
      "Fusion ",
      *id(),
      " has never been executed via FusionExecutorCache.");
  return scheds->auto_gen_schedules->getScheduledIrFor(args, tensor_transforms);
}

std::optional<size_t> FusionDefinition::id() const {
  return fusion_id_;
}

Scalar FusionDefinition::defineScalar() {
  FUSER_PERF_SCOPE("FusionDefinition::defineScalar");
  NVF_CHECK(
      trie_node_ != nullptr,
      "define_scalar() must be called from an initialized definition via a "
      "python context manager or a child class' definition() method.");
  Scalar out(recording_state_.size(), this);
  recording_state_.emplace_back(out(), serde::StateType::Scalar);
  return out;
}

Tensor FusionDefinition::addTensor(TensorView* tv) {
  FUSER_PERF_SCOPE("FusionDefinition::addTensor");
  NVF_CHECK(
      trie_node_ != nullptr,
      "addTensor() must be called from an initialized definition via a python "
      "context manager or a child class' definition() method.");
  Tensor output = defineTensor(tv->nDims());
  NVF_CHECK(
      output.index == numFusionStates(),
      "Fusion State index does not match the size!");
  addFusionState(tv);
  return output;
}

Tensor FusionDefinition::defineTensor(size_t dims) {
  FUSER_PERF_SCOPE("FusionDefinition::defineTensor");
  NVF_CHECK(
      trie_node_ != nullptr,
      "define_tensor() must be called from an initialized definition via a "
      "python context manager or a child class' definition() method.");
  Tensor out(recording_state_.size(), dims, this);
  recording_state_.emplace_back(out(), serde::StateType::Tensor);
  return out;
}

Vector FusionDefinition::defineVector(size_t size) {
  FUSER_PERF_SCOPE("FusionDefinition::defineVector");
  NVF_CHECK(
      trie_node_ != nullptr,
      "define_vector() must be called from an initialized definition via a "
      "python context manager or a child class' definition() method.");
  Vector out(recording_state_.size(), size, this);
  recording_state_.emplace_back(out(), serde::StateType::Vector);
  return out;
}

void FusionDefinition::defineRecord(RecordFunctor* record) {
  FUSER_PERF_SCOPE("FusionDefinition::defineRecord");
  NVF_CHECK(
      trie_node_ != nullptr,
      "defineRecord() must be called from an initialized definition via a "
      "python context manager or a child class' definition() method.");
  NVF_CHECK(
      (recording_.size() + 1) <= max_length_,
      "The fusion definition has exceeded ",
      max_length_,
      "operations.  The max_length for FusionDefintion's might need to be ",
      "increased if the definition is created as expected.");
  addRecord(record);
  auto child_node =
      fusionCache()->queryChildren(trie_node_, recording_.back().get());
  // If the Record is found in the cache, the FusionDefinition and the Cache
  // will not share Record given the Record had to be created in order to
  // match it but it also already existed in the cache.
  if (child_node.has_value()) {
    if (isDebugDumpEnabled(DebugDumpOption::PythonFrontendDebug)) {
      debug() << "\nFusionDefinition: Record (hash: 0x" << std::hex
              << record->hash() << ") hit in Fusion Cache.\n";
    }
    trie_node_ = child_node.value();
    // The FusionDefinition and the Cache will share the Record
  } else {
    if (isDebugDumpEnabled(DebugDumpOption::PythonFrontendDebug)) {
      debug() << "\nFusionDefinition: Record (hash: 0x" << std::hex
              << record->hash() << ") missed in Fusion Cache.\n";
    }
    trie_node_ =
        fusionCache()->createChild(trie_node_, recording_.back().get());
  }
}

Fusion* FusionDefinition::preschedFusion() {
  NVF_CHECK(
      fusion_id_.has_value(),
      "FusionDefinition does not contain a definition, yet!");
  return fusionCache()
      ->queryFusionSchedules(fusion_id_.value())
      ->preschedFusion();
}

void FusionDefinition::printMathIr() {
  return preschedFusion()->printMath();
}

State FusionDefinition::recordingState(size_t index) const {
  return recording_state_.at(index);
}

std::vector<Tensor> FusionDefinition::tensors() {
  // Filter TensorView states
  std::vector<State> tensor_states;
  std::copy_if(
      recording_state_.begin(),
      recording_state_.end(),
      std::back_inserter(tensor_states),
      [](const State& s) { return s.stype == serde::StateType::Tensor; });

  // Reconstruct Tensors
  std::vector<Tensor> all_tensors;
  all_tensors.reserve(tensor_states.size());
  std::transform(
      tensor_states.begin(),
      tensor_states.end(),
      std::back_inserter(all_tensors),
      [this](const State& s) {
        return Tensor(
            s.index, getFusionState(s.index)->as<TensorView>()->nDims(), this);
      });
  return all_tensors;
}

std::vector<std::pair<double, double>> FusionDefinition::getValTolerances(
    const KernelArgumentHolder& args) {
  return get_val_constants(preschedFusion(), args);
}

int64_t FusionDefinition::setupSegmentation(const KernelArgumentHolder& args) {
  NVF_CHECK(id().has_value(), "FusionDefinition definition does not exist!");
  NVF_ERROR(
      segmentation_state_ == nullptr, "SegmentationState already exists!");
  segmentation_state_ = std::make_unique<SegmentationState>();
  return segmentation_state_->setupSegmentation(
      preschedFusion(), map_value_to_fid_, args);
}

std::unordered_map<int64_t, int64_t> FusionDefinition::buildSegment(
    FusionDefinition& segment_fd,
    int64_t segment_id) {
  NVF_CHECK(id().has_value(), "FusionDefinition does not exist!");
  NVF_CHECK(
      segmentation_state_ != nullptr,
      "Run setupSegmentation first before trying to build segments!");
  return segmentation_state_->buildSegment(segment_fd, segment_id);
}

void FusionDefinition::finalizeSegmentation() {
  // Destroy SegmentedState
  segmentation_state_.reset();
}

} // namespace nvfuser::python_frontend
