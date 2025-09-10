// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/lower2device.h>
#include <device_lower/utils.h>
#include <expr_evaluator.h>
#include <expr_simplifier.h>
#include <host_ir/container.h>
#include <ir/builder.h>
#include <ir/cloner.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <kernel.h>
#include <kernel_ir.h>
#include <type.h>

#include <cctype>
#include <iostream>
#include <regex>

namespace nvfuser {
namespace kir {

namespace {

inline const char* boolLiteral(bool value) {
  return value ? "true" : "false";
}

inline const char* optionalBoolLiteral(std::optional<bool> optional_value) {
  if (!optional_value.has_value()) {
    return "std::nullopt";
  }
  return boolLiteral(optional_value.value());
}

} // namespace

Predicate::Predicate(
    IrBuilderPasskey passkey,
    PredicateType ptype,
    const Expr* expr,
    Val* thread_pred)
    : Val(passkey, ValType::Predicate, DataType::Bool),
      ptype_(ptype),
      expr_(expr),
      thread_pred_(thread_pred) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  NVF_ERROR(ptype != PredicateType::Unswitch && ptype != PredicateType::Manual);
}

Predicate::Predicate(
    IrBuilderPasskey passkey,
    PredicateType ptype,
    const Expr* tma_1d_load_expr,
    std::vector<ForLoop*> tma_1d_load_loops)
    : Val(passkey, ValType::Predicate, DataType::Bool),
      ptype_(ptype),
      expr_(tma_1d_load_expr),
      tma_1d_load_loops_(std::move(tma_1d_load_loops)) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  NVF_ERROR(ptype == PredicateType::OneDimTmaLoadExpectArrive);
  NVF_ERROR(!tma_1d_load_loops_.empty());
}

Predicate::Predicate(IrBuilderPasskey passkey, ForLoop* unrolled_loop)
    : Val(passkey, ValType::Predicate, DataType::Bool),
      ptype_(PredicateType::Unswitch),
      unrolled_loop_(unrolled_loop) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  NVF_ERROR(unrolled_loop != nullptr);
}

Predicate::Predicate(IrBuilderPasskey passkey, Val* value)
    : Val(passkey, ValType::Predicate, DataType::Bool),
      ptype_(PredicateType::Manual),
      value_(value) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      (passkey.ir_container_->isOneOf<kir::Kernel, hir::HostIrContainer>()),
      "IR type only valid for Kernel or HostIr container.");
  NVF_ERROR(value != nullptr);
}

std::string Predicate::toString(int indent_size) const {
  std::stringstream ss;
  ss << predicate_type2string(predicate_type());
  if (hasValue()) {
    ss << " " << value()->toInlineString();
  }
  return ss.str();
}

std::string Predicate::toInlineString(int indent_size) const {
  return toString(indent_size);
}

TensorIndex::TensorIndex(
    IrBuilderPasskey passkey,
    const TensorView* view,
    Val* index,
    DataType dtype)
    : Val(passkey,
          ValType::TensorIndex,
          dtype != DataType::Null ? dtype : view->getDataType().value()),
      view_(view),
      index_(index) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  auto uint16x2 = ArrayType{std::make_shared<DataType>(DataType::UInt16), 2};
  NVF_ERROR(
      isPointerType(index->dtype()) || index->dtype() == DataType::Index ||
          isStructType(index->dtype()) ||
          index->dtype() ==
              DataType::UInt64 /*For matrix descriptor for hopper MMA*/
          || index->dtype() == uint16x2 /*For tensor memory tensor*/,
      "Cannot index with a value other than an int/pointer/struct.");
}

std::string TensorIndex::toString(int indent_size) const {
  std::stringstream ss;
  ss << ir_utils::varName(this);
  switch (view()->getMemoryType()) {
    case MemoryType::Global:
      ss << "_g";
      break;
    case MemoryType::Shared:
      ss << "_s";
      break;
    case MemoryType::Local:
      ss << "_l";
      break;
    case MemoryType::Tensor:
      ss << "_t";
      break;
    default:
      NVF_THROW("Unknown tensor memory type.");
  }
  ss << "[";
  ss << index()->toInlineString(indent_size);
  ss << "]";
  ss << " view( " << ir_utils::varName(view()) << " )";
  return ss.str();
}

std::string TensorIndex::toInlineString(int indent_size) const {
  return toString(indent_size);
}

Allocate::Allocate(
    IrBuilderPasskey passkey,
    Val* buffer,
    MemoryType memory_type,
    std::vector<Val*> shape,
    bool zero_init,
    bool resets_to_zero,
    Allocate* alias)
    : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      (passkey.ir_container_->isOneOf<kir::Kernel, hir::HostIrContainer>()),
      "IR type only valid for Kernel or HostIr container.");
  if (!shape.empty()) {
    NVF_ERROR(
        (shape.size() == 1 && shape[0]->isOneInt()) ||
        buffer->isA<TensorView>());
  } else {
    NVF_ERROR(buffer->isA<TensorView>());
    NVF_ERROR_EQ(buffer->as<TensorView>()->getMemoryType(), memory_type);
    const auto domain = buffer->as<TensorView>()->domain();
    for (auto axis : TensorDomain::noReductions(domain->maybeAllocation())) {
      shape.push_back(axis->extent());
    }
  }

  Val* size = nullptr;
  for (auto s : shape) {
    if (size == nullptr) {
      size = s;
    } else {
      size = IrBuilder::mulExpr(size, s);
    }
  }

  if (size == nullptr) {
    size = FusionGuard::getCurFusion()->oneVal();
  }

  if (alias != nullptr) {
    NVF_ERROR(alias != this, "Invalid alias");
    NVF_ERROR(alias->memoryType() == memory_type, "Invalid alias");
  }

  size = simplifyExpr(size);

  addInput(size);
  addAttribute(buffer);
  addDataAttribute(memory_type);
  addDataAttribute(zero_init);
  addDataAttribute(resets_to_zero);
  addAttribute(alias);
  // Always initialize smem/tmem addresses to nullptr
  addAttribute(nullptr);
  addAttribute(nullptr);
  addAttribute(nullptr);

  for (auto s : shape) {
    addAttribute(s);
  }
}

Allocate::Allocate(
    IrBuilderPasskey passkey,
    Val* buffer,
    MemoryType memory_type,
    Val* size,
    bool zero_init,
    bool resets_to_zero)
    : Allocate(
          passkey,
          buffer,
          memory_type,
          size == nullptr ? std::vector<Val*>{} : std::vector<Val*>{size},
          zero_init,
          resets_to_zero) {}

std::string Allocate::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << buffer()->toString();
  ss << " = ALLOCATE("
     << "buffer=" << buffer()->toString() << ", "
     << "mem_type=" << memoryType() << ", "
     << "size=" << size()->toInlineString() << ", "
     << "zero_init=" << boolLiteral(zeroInit()) << ", "
     << "resets_to_zero=" << boolLiteral(resetsToZero()) << ")\n";
  if (alias() != nullptr) {
    indent(ss, indent_size) << kTab << ".alias=";
    ss << alias()->buffer()->toString() << "\n";
  }
  return ss.str();
}

std::string Allocate::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(Allocate)

Asm::Asm(
    IrBuilderPasskey passkey,
    const std::string& code,
    const std::vector<Val*>& outputs,
    const std::vector<Val*>& inputs,
    const Options& options)
    : Expr(passkey) {
  addDataAttribute(code);
  addDataAttribute(options);
  for (auto out : outputs) {
    addOutput(out);
  }
  for (auto in : inputs) {
    addInput(in);
  }
}

// Reference:
// https://docs.nvidia.com/cuda/inline-ptx-assembly/#constraints

namespace {

// If value is a kir::TensorIndex, and its index is a pointer type, then
// return the pointer type. Otherwise return the value's dtype.
DataType getTypeOrIndexType(Val* value) {
  if (auto ti = dynamic_cast<kir::TensorIndex*>(value)) {
    if (isPointerType(ti->index()->dtype())) {
      return ti->index()->dtype();
    }
  }
  return value->dtype();
}

const char* getPTXConstraints(Val* value) {
  DataType dt = getTypeOrIndexType(value);
  if (dt == DataType::Bool) {
    return "r";
  }
  if (auto ti = dynamic_cast<kir::TensorIndex*>(value)) {
    // If the index type is a pointer type, then we directly uses the pointer in
    // the generated code, instead of generating something like T0[i]. For this
    // case we should use the pointer type as the constraint.
    if (isPointerType(ti->index()->dtype())) {
      dt = ti->index()->dtype();
    }
  }
  if (std::holds_alternative<ArrayType>(dt.type)) {
    dt = *std::get<ArrayType>(dt.type).type;
  }
  auto size = dataTypeSizeByte(dt);
  switch (size) {
    case 2:
      return "h";
    case 4:
      if (isFloatingPointType(dt)) {
        return "f";
      } else {
        return "r";
      }
    case 8:
      if (isFloatingPointType(dt)) {
        return "d";
      } else {
        return "l";
      }
    default:
      NVF_THROW("Unsupported data type ", dt, " for inline PTX assembly.");
  }
}

} // namespace

std::vector<std::pair<std::string, Val*>> Asm::constraintsAndOutputs() const {
  std::vector<std::pair<std::string, Val*>> result;
  for (auto i : arange((int64_t)(outputs().size()))) {
    std::string prefix;
    if (options().readable_outputs.count(i) > 0) {
      prefix = "+";
    } else {
      prefix = "=";
    }
    auto out = output(i);
    NVF_ERROR(!out->isConst());
    result.emplace_back(prefix + getPTXConstraints(out), out);
  }
  return result;
}
std::vector<std::pair<std::string, Val*>> Asm::constraintsAndInputs() const {
  std::vector<std::pair<std::string, Val*>> result;
  for (int64_t i : arange((int64_t)inputs().size())) {
    auto in = input(i);
    const char* constraint = nullptr;
    if (options().immediate_inputs.count(i) > 0) {
      constraint = "n";
    } else {
      constraint = getPTXConstraints(in);
    }
    result.emplace_back(constraint, in);
  }
  return result;
}

std::string Asm::parameters() const {
  int64_t counter = 0;
  int64_t bool_counter = 0;
  std::stringstream ss;
  auto gen = [&counter, &bool_counter, &ss](Val* v) {
    DataType dtype = getTypeOrIndexType(v);
    if (counter > 0) {
      ss << ", ";
    }
    if (isPointerType(dtype)) {
      ss << "[%" << counter++ << "]";
    } else if (dtype == DataType::Bool) {
      ss << "p" << bool_counter++;
      counter++;
    } else if (std::holds_alternative<PrimDataType>(dtype.type)) {
      ss << "%" << counter++;
    } else if (std::holds_alternative<ArrayType>(dtype.type)) {
      auto type = std::get<ArrayType>(dtype.type);
      ss << "{";
      for (auto i : arange(type.size)) {
        if (i > 0) {
          ss << ", ";
        }
        ss << "%" << counter++;
      }
      ss << "}";
    } else {
      NVF_THROW("Unsupported data type ", dtype);
    }
  };
  for (auto out : outputs()) {
    gen(out);
  }
  for (auto in : inputs()) {
    gen(in);
  }
  return ss.str();
}

std::string Asm::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "asm";
  if (volatile_()) {
    ss << " volatile";
  }
  ss << "(\n";
  indent(ss, indent_size + 1) << "\"" << code() << "\"\n:";
  bool first = true;
  for (auto out : outputs()) {
    if (!first) {
      ss << ",\n";
    }
    first = false;
    indent(ss, indent_size + 1) << out->toString();
  }
  ss << "\n:";
  first = true;
  for (auto in : inputs()) {
    if (!first) {
      ss << ",\n";
    }
    first = false;
    indent(ss, indent_size + 1) << in->toString();
  }
  if (memory()) {
    ss << "\n:";
    indent(ss, indent_size + 1) << "memory";
  }
  ss << ");";
  return ss.str();
}

std::string Asm::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Asm op can not be printed inline");
}

std::string Asm::utility() const {
  static const std::unordered_map<std::string, std::string> ptx_to_utility{
      {"tcgen05.wait::ld.sync.aligned", "tcgen05::waitLoad"},
      {"tcgen05.wait::st.sync.aligned", "tcgen05::waitStore"},
      {"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32",
       "tcgen05::alloc"},
      {"tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned",
       "tcgen05::relinquishAllocPermit"},
      {"tcgen05.dealloc.cta_group::1.sync.aligned.b32", "tcgen05::dealloc"},
      {"tcgen05.mma.cta_group::1.kind::f16", "tcgen05::mma_f16"},
      {"tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64",
       "tcgen05::commit"},
      {"wgmma.fence.sync.aligned", "wgmma::fence"},
      {"fence.proxy.async", "fenceAsyncProxy"},
      {"wgmma.commit_group.sync.aligned", "wgmma::commit"},
      {"wgmma.wait_group.sync.aligned", "wgmma::wait"},
      {"ldmatrix.sync.aligned.x1.m8n8.shared.b16", "ldmatrix1"},
      {"ldmatrix.sync.aligned.x2.m8n8.shared.b16", "ldmatrix2"},
      {"ldmatrix.sync.aligned.x4.m8n8.shared.b16", "ldmatrix4"},
      {"stmatrix.sync.aligned.x1.m8n8.shared.b16", "stmatrix1"},
      {"stmatrix.sync.aligned.x2.m8n8.shared.b16", "stmatrix2"},
      {"stmatrix.sync.aligned.x4.m8n8.shared.b16", "stmatrix4"},
      {"cp.async.bulk.commit_group", "cpAsyncBulkCommitGroup"},
      {"cp.async.bulk.wait_group.read", "cpAsyncBulkWaitGroup"},
      {"setmaxnreg.inc.sync.aligned.u32", "increaseRegisters"},
      {"setmaxnreg.dec.sync.aligned.u32", "decreaseRegisters"}};
  const std::string& code = this->code();
  auto it = ptx_to_utility.find(code);
  if (it != ptx_to_utility.end()) {
    return it->second;
  }

  // Match patterns like tcgen05.{ld,st}.sync.aligned.32x32b.x1.b32
  {
    std::regex ld_pattern(R"(tcgen05\.ld\.sync\.aligned\.([^.]+)\.x\d+\.b32)");
    std::smatch match;
    if (std::regex_match(code, match, ld_pattern)) {
      std::string result = "tcgen05::load";
      result.append(match[1]);
      return result;
    }
  }
  {
    std::regex st_pattern(R"(tcgen05\.st\.sync\.aligned\.([^.]+)\.x\d+\.b32)");
    std::smatch match;
    if (std::regex_match(code, match, st_pattern)) {
      std::string result = "tcgen05::store";
      result.append(match[1]);
      return result;
    }
  }

  // Match wgmma. Example:
  // instruction: wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16
  // utility: wgmmaM64N256K16Half
  {
    // Half
    std::regex pattern(
        R"(wgmma\.mma_async\.sync\.aligned\.(m\d+n\d+k\d+)\.f32\.f16\.f16)");
    std::smatch match;
    if (std::regex_match(code, match, pattern)) {
      std::string extracted = match[1];
      return "wgmma::" + extracted + "Half";
    }
  }
  {
    // BFloat16
    std::regex pattern(
        R"(wgmma\.mma_async\.sync\.aligned\.(m\d+n\d+k\d+)\.f32\.bf16\.bf16)");
    std::smatch match;
    if (std::regex_match(code, match, pattern)) {
      std::string extracted = match[1];
      return "wgmma::" + extracted + "BF16";
    }
  }
  return "";
}

std::string Asm::signature() const {
  std::string utility = this->utility();
  if (utility.empty()) {
    return "";
  }
  std::stringstream ss;
  ss << "void " << utility << "(";
  bool first = true;
  for (auto operand : outputs()) {
    if (!first) {
      ss << ", ";
    }
    ss << operand->dtype() << "&";
    first = false;
  }
  for (auto operand : inputs()) {
    if (!first) {
      ss << ", ";
    }
    ss << operand->dtype();
    first = false;
  }
  ss << ")";
  return ss.str();
}

NVFUSER_DEFINE_CLONE_AND_CREATE(Asm)

AllocTMem::AllocTMem(IrBuilderPasskey passkey, Val* address, Val* num_columns)
    : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  NVF_ERROR(
      ir_utils::getTv(address)->getMemoryType() == MemoryType::Shared,
      "AllocTMem address must be a shared memory tensor");
  addOutput(address);
  NVF_ERROR(
      num_columns->dtype() == DataType::UInt32,
      "AllocTMem num_columns must be a uint32_t");
  addInput(num_columns);
}

std::string AllocTMem::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << output(0)->toString() << " = AllocTMem("
                          << input(0)->toString() << ")\n";
  return ss.str();
}

std::string AllocTMem::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(AllocTMem)

BlockSync::BlockSync(
    IrBuilderPasskey passkey,
    bool war_sync,
    std::optional<bool> optional_compute_or_load_sync)
    : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  addDataAttribute(war_sync);
  addDataAttribute(optional_compute_or_load_sync);
}

std::string BlockSync::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "BLOCKSYNC(war_hazard="
                          << boolLiteral(isWarHazardSync())
                          << ", optional_compute_or_load_sync="
                          << optionalBoolLiteral(warpSpecializedState())
                          << ")\n";
  return ss.str();
}

std::string BlockSync::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(BlockSync)

GridSync::GridSync(
    IrBuilderPasskey passkey,
    ParallelTypeBitmap sync_dims,
    Val* sync_buffer)
    : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  addDataAttribute(sync_dims);
  addAttribute(sync_buffer);
}

std::string GridSync::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "GRIDSYNC(" << syncDims().toString() << ", "
                          << syncBuffer()->toString() << ")\n";
  return ss.str();
}

std::string GridSync::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GridSync)

FenceAsyncProxy::FenceAsyncProxy(IrBuilderPasskey passkey) : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

std::string FenceAsyncProxy::toString(int indent_size) const {
  return "fence.proxy.async\n";
}

std::string FenceAsyncProxy::toInlineString(int indent_size) const {
  NVF_CHECK(false, "FenceAsyncProxy can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(FenceAsyncProxy)

WgMmaFence::WgMmaFence(IrBuilderPasskey passkey) : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

std::string WgMmaFence::toString(int indent_size) const {
  return "fence.proxy.async\n";
}

std::string WgMmaFence::toInlineString(int indent_size) const {
  NVF_CHECK(false, "WgMmaFence can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(WgMmaFence)

SetMaxNReg::SetMaxNReg(
    IrBuilderPasskey passkey,
    Val* number_of_registers,
    bool increase_registers)
    : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  addInput(number_of_registers);
  addDataAttribute(increase_registers);
}

std::string SetMaxNReg::toString(int indent_size) const {
  std::stringstream ss;
  if (increaseRegisters()) {
    indent(ss, indent_size) << "setmaxnreg.inc.sync.aligned.u32\n";
  } else {
    indent(ss, indent_size) << "setmaxnreg.dec.sync.aligned.u32\n";
  }
  return ss.str();
}

std::string SetMaxNReg::toInlineString(int indent_size) const {
  NVF_CHECK(false, "SetMaxNReg can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(SetMaxNReg)

Continue::Continue(IrBuilderPasskey passkey) : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

std::string Continue::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "continue\n";
  return ss.str();
}

std::string Continue::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Continue can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(Continue)

Return::Return(IrBuilderPasskey passkey) : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

std::string Return::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "return\n";
  return ss.str();
}

std::string Return::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Return can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(Return)

MBarrierInit::MBarrierInit(
    IrBuilderPasskey passkey,
    Val* mbarrier,
    Val* thread_count)
    : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_CHECK(thread_count->dtype() == DataType::UInt32);
  addInput(mbarrier);
  addInput(thread_count);
}

std::string MBarrierInit::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "MBarrierInit(" << mbarrier()->toString() << ", "
                          << threadCount()->toString() << ")\n";
  return ss.str();
}

std::string MBarrierInit::toInlineString(int indent_size) const {
  NVF_CHECK(false, "MBarrierInit can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(MBarrierInit)

MBarrierInvalidate::MBarrierInvalidate(IrBuilderPasskey passkey, Val* mbarrier)
    : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  addInput(mbarrier);
}

std::string MBarrierInvalidate::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "MBarrierInvalidate(" << mbarrier()->toString()
                          << ")\n";
  return ss.str();
}

std::string MBarrierInvalidate::toInlineString(int indent_size) const {
  NVF_CHECK(false, "MBarrierInvalidate can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(MBarrierInvalidate)

MBarrierArrive::MBarrierArrive(
    IrBuilderPasskey passkey,
    Val* state,
    Val* mbarrier)
    : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  addInput(mbarrier);
  if (state != nullptr) {
    NVF_CHECK(state->dtype() == DataType::UInt64);
    addOutput(state);
  }
}

std::string MBarrierArrive::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "MBarrierArrive(" << mbarrier()->toString()
                          << ")\n";
  return ss.str();
}

std::string MBarrierArrive::toInlineString(int indent_size) const {
  NVF_CHECK(false, "MBarrierArrive can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(MBarrierArrive)

MBarrierArriveExpectTx::MBarrierArriveExpectTx(
    IrBuilderPasskey passkey,
    Val* state,
    Val* mbarrier,
    Val* tx_count)
    : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_CHECK(tx_count->dtype() == DataType::UInt32);
  addInput(mbarrier);
  addInput(tx_count);
  if (state != nullptr) {
    NVF_CHECK(state->dtype() == DataType::UInt64);
    addOutput(state);
  }
}

std::string MBarrierArriveExpectTx::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "MBarrierArriveExpectTx(" << mbarrier()->toString()
                          << ", " << txCount()->toString() << ")\n";
  return ss.str();
}

std::string MBarrierArriveExpectTx::toInlineString(int indent_size) const {
  NVF_CHECK(false, "MBarrierArriveExpectTx can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(MBarrierArriveExpectTx)

MBarrierWait::MBarrierWait(IrBuilderPasskey passkey, Val* mbarrier, Val* state)
    : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_CHECK(state->dtype() == DataType::UInt64);
  addInput(mbarrier);
  addInput(state);
}

std::string MBarrierWait::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "MBarrierWait(" << mbarrier()->toString() << ", "
                          << state()->toString() << ")\n";
  return ss.str();
}

std::string MBarrierWait::toInlineString(int indent_size) const {
  NVF_CHECK(false, "MBarrierWait can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(MBarrierWait)

MBarrierWaitParity::MBarrierWaitParity(
    IrBuilderPasskey passkey,
    Val* mbarrier,
    Val* parity)
    : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_CHECK(parity->dtype() == DataType::UInt32);
  addInput(mbarrier);
  addInput(parity);
}

std::string MBarrierWaitParity::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "MBarrierWaitParity(" << mbarrier()->toString()
                          << ", " << parity()->toString() << ")\n";
  return ss.str();
}

std::string MBarrierWaitParity::toInlineString(int indent_size) const {
  NVF_CHECK(false, "MBarrierWaitParity can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(MBarrierWaitParity)

BlockSerializeWait::BlockSerializeWait(
    IrBuilderPasskey passkey,
    ParallelTypeBitmap sync_dims,
    Val* sync_buffer)
    : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  addDataAttribute(sync_dims);
  addAttribute(sync_buffer);
}

std::string BlockSerializeWait::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "BLOCKSERIALIZEWAIT(" << syncDims().toString()
                          << ", " << syncBuffer()->toString() << ")\n";
  return ss.str();
}

std::string BlockSerializeWait::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Serial reduction pre sync can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(BlockSerializeWait)

BlockSerializeRelease::BlockSerializeRelease(
    IrBuilderPasskey passkey,
    ParallelTypeBitmap sync_dims,
    Val* sync_buffer)
    : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  addDataAttribute(sync_dims);
  addAttribute(sync_buffer);
}

std::string BlockSerializeRelease::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "BLOCKSERIALIZERELEASE(" << syncDims().toString()
                          << ", " << syncBuffer()->toString() << ")\n";
  return ss.str();
}

std::string BlockSerializeRelease::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Serial reduction post sync can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(BlockSerializeRelease)

AsyncWait::AsyncWait(
    IrBuilderPasskey passkey,
    AsyncOpType async_op_type,
    int64_t keep_stages)
    : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  addDataAttribute(async_op_type);
  addDataAttribute(keep_stages);
}

std::string AsyncWait::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << ptx() << " " << keepStages() << "\n";
  return ss.str();
}

std::string AsyncWait::toInlineString(int indent_size) const {
  NVF_CHECK(false, "AsyncWait can not be printed inline");
}

const char* AsyncWait::ptx() const {
  switch (asyncOpType()) {
    case AsyncOpType::CpAsync:
      if (keepStages() == 0) {
        return "cp.async.wait_all";
      } else {
        return "cp.async.wait_group";
      }
    case AsyncOpType::CpAsyncBulk:
      return "cp.async.bulk.wait_group.read";
    case AsyncOpType::WgMma:
      return "wgmma.wait_group.sync.aligned";
    default:
      NVF_THROW("Unsupported async op type.");
  }
}

bool AsyncWait::memory() const {
  switch (asyncOpType()) {
    case AsyncOpType::CpAsync:
      return false;
    case AsyncOpType::CpAsyncBulk:
    case AsyncOpType::WgMma:
      return true;
    default:
      NVF_THROW("Unsupported async op type.");
  }
}

NVFUSER_DEFINE_CLONE_AND_CREATE(AsyncWait)

AsyncCommit::AsyncCommit(IrBuilderPasskey passkey, AsyncOpType async_op_type)
    : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  addDataAttribute(async_op_type);
}

std::string AsyncCommit::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << ptx() << ";\n";
  return ss.str();
}

std::string AsyncCommit::toInlineString(int indent_size) const {
  NVF_CHECK(false, "AsyncCommit can not be printed inline");
}

const char* AsyncCommit::ptx() const {
  switch (asyncOpType()) {
    case AsyncOpType::CpAsync:
      return "cp.async.commit_group";
    case AsyncOpType::CpAsyncBulk:
      return "cp.async.bulk.commit_group";
    case AsyncOpType::WgMma:
      return "wgmma.commit_group.sync.aligned";
    default:
      NVF_THROW("Unsupported async op type.");
  }
}

bool AsyncCommit::memory() const {
  switch (asyncOpType()) {
    case AsyncOpType::CpAsync:
    case AsyncOpType::CpAsyncBulk:
      return false;
    case AsyncOpType::WgMma:
      return true;
    default:
      NVF_THROW("Unsupported async op type.");
  }
}

NVFUSER_DEFINE_CLONE_AND_CREATE(AsyncCommit)

InitMagicZero::InitMagicZero(IrBuilderPasskey passkey) : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

std::string InitMagicZero::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "NVFUSER_DEFINE_MAGIC_ZERO;\n";
  return ss.str();
}

std::string InitMagicZero::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(InitMagicZero)

UpdateMagicZero::UpdateMagicZero(IrBuilderPasskey passkey) : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

std::string UpdateMagicZero::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "NVFUSER_UPDATE_MAGIC_ZERO;\n";
  return ss.str();
}

std::string UpdateMagicZero::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(UpdateMagicZero)

IfThenElse::IfThenElse(IrBuilderPasskey passkey, Predicate* cond)
    : Expr(passkey) {
  setPredicate(cond);
  addInput(cond);
  // Storing IR nodes as Attribute is not safe with IrCloner, but fortunately
  // kernel IR does not need this feature.
  addDataAttribute(Scope(this));
  addDataAttribute(Scope(this));
}

std::string IfThenElse::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "IF " << predicate()->toString() << ":\n"
                          << thenBody().toString(indent_size + 1);
  if (hasElse()) {
    indent(ss, indent_size) << "ELSE:\n"
                            << elseBody().toString(indent_size + 1);
  }
  return ss.str();
}

std::string IfThenElse::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(IfThenElse)

GridReduction::GridReduction(
    IrBuilderPasskey passkey,
    BinaryOpType reduction_op_type,
    Val* init,
    Val* out,
    Val* in,
    Allocate* reduction_buffer,
    Allocate* sync_buffer,
    Val* entrance_index,
    Val* entrances,
    bool is_allreduce,
    TensorIndex* serial_reduction_tensor)
    : ReductionOp(passkey, reduction_op_type, init, out, in, is_allreduce) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  NVF_ERROR(
      attributes().size() == num_reduction_op_attr,
      "The num_reduction_op_attr does not match the number of attributes "
      "ReductionOp has."
      "If you changed ReductionOp, please change num_reduction_op_attr "
      "accordingly.");
  addAttribute(reduction_buffer);
  addAttribute(sync_buffer);
  addAttribute(entrance_index);
  addAttribute(entrances);
  addDataAttribute(ParallelTypeBitmap{});
  addAttribute(serial_reduction_tensor);
}

std::string GridReduction::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << " = reduction( "
                          << in()->toString()
                          << ", op = " << getReductionOpType()
                          << ", initial value = " << init()->toString()
                          << ",\n";
  ++indent_size;
  indent(ss, indent_size) << "reduction buffer = "
                          << reduction_buffer()->buffer()->toString() << ",\n";
  indent(ss, indent_size) << "sync buffer = "
                          << sync_buffer()->buffer()->toString() << ",\n";
  indent(ss, indent_size) << "read predicate = ";
  if (predicate() != nullptr) {
    ss << predicate()->toString();
  } else {
    ss << "nullptr";
  }
  ss << ",\n";
  indent(ss, indent_size) << "write predicate = ";
  if (writePredicate() != nullptr) {
    ss << writePredicate()->toString();
  } else {
    ss << "nullptr";
  }
  ss << ",\n";
  indent(ss, indent_size) << "thread predicate = "
                          << threadPredicate().toString() << ",\n";
  indent(ss, indent_size) << "allreduce = "
                          << (isAllreduce() ? "true" : "false") << " )\n";
  indent(ss, indent_size) << "serial reduction = "
                          << (isSerial() ? "true" : "false") << " )\n";
  if (isSerial()) {
    indent(ss, indent_size)
        << "serial reduction tensor = " << serialReductionTensor()->toString()
        << " )\n";
  }
  return ss.str();
}

std::string GridReduction::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GridReduction)

GroupedGridReduction::GroupedGridReduction(
    IrBuilderPasskey passkey,
    std::vector<BinaryOpType> reduction_op_types,
    std::vector<Val*> init_vals,
    std::vector<Val*> outputs,
    std::vector<Val*> inputs,
    std::vector<Allocate*> reduction_buffers,
    Allocate* sync_buffer,
    Val* entrance_index,
    Val* entrances,
    Val* buffer_stride,
    bool is_allreduce)
    : GroupedReductionOp(
          passkey,
          std::move(reduction_op_types),
          std::move(init_vals),
          std::move(outputs),
          std::move(inputs),
          is_allreduce) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  NVF_ERROR(
      attributes().size() == numGroupedReductionOpAttr(),
      "The numGroupedReductionOpAttr() does not match the number of attributes "
      "GroupedReductionOp has."
      "If you changed GroupedReductionOp, please change "
      "numGroupedReductionOpAttr() accordingly.");
  addAttribute(sync_buffer);
  addAttribute(entrance_index);
  addAttribute(entrances);
  addAttribute(buffer_stride);
  addDataAttribute(ParallelTypeBitmap{});
  for (auto buffer : reduction_buffers) {
    addAttribute(buffer);
  }
}

std::string GroupedGridReduction::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "GroupedGridReduction(\n";
  ++indent_size;
  for (const auto i : arange(numHorizontallyGroupedExprs())) {
    indent(ss, indent_size)
        << output(i)->toString() << " = reduction( " << input(i)->toString()
        << ", op = " << getReductionOpType(i)
        << ", initial value = " << initVal(i)->toString()
        << ", reduction buffer = "
        << reduction_buffers().at(i)->buffer()->toString() << " )\n";
  }
  indent(ss, indent_size) << "sync buffer = "
                          << sync_buffer()->buffer()->toString() << ",\n";
  indent(ss, indent_size) << "read predicate = ";
  if (predicate() != nullptr) {
    ss << predicate()->toString();
  } else {
    ss << "nullptr";
  }
  ss << ",\n";
  indent(ss, indent_size) << "write predicate = ";
  if (writePredicate() != nullptr) {
    ss << writePredicate()->toString();
  } else {
    ss << "nullptr";
  }
  ss << ",\n";
  indent(ss, indent_size) << "thread predicate = "
                          << threadPredicate().toString() << ",\n";
  indent(ss, indent_size) << "allreduce = "
                          << (isAllreduce() ? "true" : "false") << " )\n";
  --indent_size;
  return ss.str();
}

std::string GroupedGridReduction::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GroupedGridReduction)

GridBroadcast::GridBroadcast(
    IrBuilderPasskey passkey,
    BroadcastOp* broadcast_op,
    Allocate* broadcast_buffer,
    Allocate* sync_buffer)
    : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  addAttribute(broadcast_op);
  addAttribute(broadcast_buffer);
  addAttribute(sync_buffer);
}

std::string GridBroadcast::toString(int indent_size) const {
  std::stringstream ss;
  const auto* broadcast_op = this->broadcast_op();
  indent(ss, indent_size) << broadcast_op->out()->toString() << " = "
                          << "GRID_BROADCAST(in="
                          << broadcast_op->in()->toString() << ")\n";
  indent(ss, indent_size) << kTab << ".broadcast_buffer="
                          << broadcast_buffer()->buffer()->toString() << "\n";
  indent(ss, indent_size) << kTab << ".sync_buffer="
                          << sync_buffer()->buffer()->toString() << "\n";
  return ss.str();
}

std::string GridBroadcast::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GridBroadcast)

GridWelford::GridWelford(
    IrBuilderPasskey passkey,
    WelfordOp* welford_op,
    Allocate* var_buffer,
    Allocate* avg_buffer,
    Allocate* n_buffer,
    Allocate* sync_buffer,
    Val* entrance_index,
    Val* entrances)
    : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  addAttribute(welford_op);
  addAttribute(var_buffer);
  addAttribute(avg_buffer);
  addAttribute(n_buffer);
  addAttribute(sync_buffer);
  addAttribute(entrance_index);
  addAttribute(entrances);
  addDataAttribute(ParallelTypeBitmap{});
}

std::string GridWelford::toString(int indent_size) const {
  std::stringstream ss;
  const auto* welford_op = this->welford_op();
  indent(ss, indent_size) << welford_op->outAvg()->toString() << " (Avg),\n";
  indent(ss, indent_size) << welford_op->outVar()->toString() << " (Var),\n";
  indent(ss, indent_size) << welford_op->outN()->toString() << " (Count)\n";
  indent(ss, indent_size) << " = Welford (\n";
  ++indent_size;
  indent(ss, indent_size) << welford_op->inAvg()->toString() << " (Avg),\n";
  indent(ss, indent_size) << welford_op->inVar()->toString() << " (Var),\n";
  indent(ss, indent_size) << welford_op->inN()->toString() << " (Count)\n";
  indent(ss, indent_size) << "initial value =\n";
  ++indent_size;
  indent(ss, indent_size) << welford_op->initAvg()->toString() << " (Avg),\n";
  indent(ss, indent_size) << welford_op->initVar()->toString() << " (Var),\n";
  indent(ss, indent_size) << welford_op->initN()->toString() << " (Count),\n";
  --indent_size;
  indent(ss, indent_size) << "reduction buffer =\n";
  ++indent_size;
  indent(ss, indent_size) << avg_buffer()->buffer()->toString() << " (Avg),\n";
  indent(ss, indent_size) << var_buffer()->buffer()->toString() << " (Var),\n";
  indent(ss, indent_size) << N_buffer()->buffer()->toString() << " (Count),\n";
  --indent_size;
  indent(ss, indent_size) << "sync buffer = "
                          << sync_buffer()->buffer()->toString() << ",\n";
  indent(ss, indent_size) << "read predicate = ";
  if (welford_op->predicate() != nullptr) {
    ss << welford_op->predicate();
  } else {
    ss << "nullptr";
  }
  ss << ",\n";
  indent(ss, indent_size) << "write predicate = ";
  if (welford_op->writePredicate() != nullptr) {
    ss << welford_op->writePredicate()->toString();
  } else {
    ss << "nullptr";
  }
  ss << ",\n";
  indent(ss, indent_size) << "grid read predicate = ";
  if (predicate() != nullptr) {
    ss << predicate()->toString();
  } else {
    ss << "nullptr";
  }
  ss << ",\n";
  indent(ss, indent_size) << "grid write predicate = ";
  if (writePredicate() != nullptr) {
    ss << writePredicate()->toString();
  } else {
    ss << "nullptr";
  }
  ss << ",\n";
  indent(ss, indent_size) << "thread predicate = "
                          << threadPredicate().toString() << ",\n";
  indent(ss, indent_size) << "allreduce = "
                          << (welford_op->isAllreduce() ? "true" : "false")
                          << " )\n";
  return ss.str();
}

std::string GridWelford::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GridWelford)

GroupedGridWelford::GroupedGridWelford(
    IrBuilderPasskey passkey,
    std::vector<WelfordTriplet> output_vals,
    std::vector<WelfordTriplet> input_vals,
    std::vector<WelfordTriplet> init_vals,
    std::array<std::vector<Allocate*>, 3> reduction_buffers,
    Allocate* sync_buffer,
    Val* entrance_index,
    Val* entrances,
    Val* buffer_stride,
    bool is_allreduce,
    bool use_outer_opt)
    : GroupedWelfordOp(
          passkey,
          std::move(output_vals),
          std::move(input_vals),
          std::move(init_vals),
          is_allreduce) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  NVF_ERROR(
      attributes().size() == numGroupedWelfordOpAttr(),
      "The numGroupedWelfordOpAttr() does not match the number of attributes "
      "GroupedWelfordOp has."
      "If you changed GroupedReductionOp, please change "
      "numGroupedWelfordOpAttr() accordingly.");
  addAttribute(sync_buffer);
  addAttribute(entrance_index);
  addAttribute(entrances);
  addAttribute(buffer_stride);
  addDataAttribute(ParallelTypeBitmap{});
  NVF_ERROR(reduction_buffers[0].size() == reduction_buffers[1].size());
  NVF_ERROR(reduction_buffers[0].size() == reduction_buffers[2].size());
  for (auto i : arange(reduction_buffers[0].size())) {
    addAttribute(reduction_buffers[0].at(i));
    addAttribute(reduction_buffers[1].at(i));
    addAttribute(reduction_buffers[2].at(i));
  }

  addDataAttribute(use_outer_opt);
}

int64_t GroupedGridWelford::getSmemBufferSize(
    int64_t bdimx,
    int64_t bdimy,
    int64_t bdimz) const {
  auto out_tv = ir_utils::getTvOutput(this);
  auto kernel = dynamic_cast<kir::Kernel*>(container());
  NVF_ERROR(kernel != nullptr);

  // By default, the required size is the same as the normal Welford reduction
  if (!useOuterOpt()) {
    return bdimx * bdimy * bdimz *
        dataTypeSizeByte(out_tv->getDataType().value()) * 2 +
        bdimx * bdimy * bdimz *
        dataTypeSizeByte(DataType::Index, kernel->indexType());
  }

  // In the outer-reduction version, the size is blockDim.x * NumberOfWarps *
  // GroupCount

  int64_t group_count = 1;
  for (auto axis : out_tv->getLoopDomain()) {
    auto pt = axis->getParallelType();
    if (pt == ParallelType::Group) {
      auto extent_int = axis->extent()->value().as<int64_t>();
      group_count *= extent_int;
    }
  }

  NVF_ERROR(group_count > 1);

  int64_t num_warps = bdimx * bdimy / 32;
  NVF_ERROR((bdimx * bdimy) % 32 == 0);

  int64_t buf_size_for_avg_var = bdimx * num_warps * group_count *
      dataTypeSizeByte(out_tv->getDataType().value());
  int64_t buf_size_for_N =
      num_warps * dataTypeSizeByte(DataType::Index, kernel->indexType());

  return buf_size_for_avg_var * 2 + buf_size_for_N;
}

std::string GroupedGridWelford::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "GroupedGridWelford(\n";
  ++indent_size;
  for (const auto i : arange(numHorizontallyGroupedExprs())) {
    indent(ss, indent_size) << outAvg(i)->toString() << " (Avg),\n";
    indent(ss, indent_size) << outVar(i)->toString() << " (Var),\n";
    indent(ss, indent_size) << outN(i)->toString() << " (Count)\n";
    indent(ss, indent_size) << " = Welford (\n";
    ++indent_size;
    indent(ss, indent_size) << inAvg(i)->toString() << " (Avg),\n";
    indent(ss, indent_size) << inVar(i)->toString() << " (Var),\n";
    indent(ss, indent_size) << inN(i)->toString() << " (Count)\n";
    indent(ss, indent_size) << "initial value =\n";
    ++indent_size;
    indent(ss, indent_size) << initAvg(i)->toString() << " (Avg),\n";
    indent(ss, indent_size) << initVar(i)->toString() << " (Var),\n";
    indent(ss, indent_size) << initN(i)->toString() << " (Count),\n";
    --indent_size;
    indent(ss, indent_size) << "reduction buffer =\n";
    ++indent_size;
    indent(ss, indent_size)
        << reduction_buffers()[0].at(i)->buffer()->toString() << " (Avg),\n";
    indent(ss, indent_size)
        << reduction_buffers()[1].at(i)->buffer()->toString() << " (Var),\n";
    indent(ss, indent_size)
        << reduction_buffers()[2].at(i)->buffer()->toString() << " (Count) )\n";
    indent_size -= 2;
  }
  indent(ss, indent_size) << "sync buffer = "
                          << sync_buffer()->buffer()->toString() << ",\n";
  indent(ss, indent_size) << "read predicate = ";
  if (predicate() != nullptr) {
    ss << predicate()->toString();
  } else {
    ss << "nullptr";
  }
  ss << ",\n";
  indent(ss, indent_size) << "write predicate = ";
  if (writePredicate() != nullptr) {
    ss << writePredicate()->toString();
  } else {
    ss << "nullptr";
  }
  ss << ",\n";
  indent(ss, indent_size) << "thread predicate = "
                          << threadPredicate().toString() << ",\n";
  indent(ss, indent_size) << "allreduce = "
                          << (isAllreduce() ? "true" : "false") << " )\n";
  return ss.str();
}

std::string GroupedGridWelford::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GroupedGridWelford)

VectorizedWelfordOp::VectorizedWelfordOp(
    IrBuilderPasskey passkey,
    const WelfordTriplet& output,
    const WelfordTriplet& input,
    const WelfordTriplet& init,
    Val* count,
    Val* reciprocal_of_count,
    Val* hoisted_predicate)
    : WelfordOp(passkey, output, input, init, false) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  addAttribute(count);
  addAttribute(reciprocal_of_count);
  addAttribute(hoisted_predicate);
}

NVFUSER_DEFINE_CLONE_AND_CREATE(VectorizedWelfordOp)

AllocateFusedReduction::AllocateFusedReduction(
    IrBuilderPasskey passkey,
    Expr* grid_expr)
    : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  addAttribute(grid_expr);
}

std::string AllocateFusedReduction::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "AllocateFusedReduction(reduction buffer="
                          << out()->toString() << ")\n";
  return ss.str();
}

std::string AllocateFusedReduction::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Tensor op can not be printed inline");
}

TensorIndex* AllocateFusedReduction::out() const {
  NVF_ERROR(gridExpr() != nullptr);
  if (gridExpr()->isA<GridReduction>() ||
      gridExpr()->isA<GroupedGridReduction>()) {
    return gridExpr()->outputs().at(0)->as<kir::TensorIndex>();
  } else if (auto grid_welford = dynamic_cast<GridWelford*>(gridExpr())) {
    return grid_welford->welford_op()->out()->as<kir::TensorIndex>();
  } else if (
      auto grouped_grid_welford =
          dynamic_cast<GroupedGridWelford*>(gridExpr())) {
    return grouped_grid_welford->out(0)->as<kir::TensorIndex>();
  } else {
    NVF_THROW("Invalid grid expression: ", gridExpr()->toString());
  }
}

const ParallelTypeBitmap& AllocateFusedReduction::threadPredicate() const {
  NVF_ERROR(gridExpr() != nullptr);
  if (auto grid_reduction = dynamic_cast<GridReduction*>(gridExpr())) {
    return grid_reduction->threadPredicate();
  } else if (auto grid_welford = dynamic_cast<GridWelford*>(gridExpr())) {
    return grid_welford->threadPredicate();
  } else if (
      auto grouped_grid_reduction =
          dynamic_cast<GroupedGridReduction*>(gridExpr())) {
    return grouped_grid_reduction->threadPredicate();
  } else if (
      auto grouped_grid_welford =
          dynamic_cast<GroupedGridWelford*>(gridExpr())) {
    return grouped_grid_welford->threadPredicate();
  } else {
    NVF_THROW("Invalid grid expression: ", gridExpr()->toString());
  }
}

NVFUSER_DEFINE_CLONE_AND_CREATE(AllocateFusedReduction)

GetRNGSeedAndOffsetFromHost::GetRNGSeedAndOffsetFromHost(
    IrBuilderPasskey passkey,
    Val* seed_ptr,
    Val* seed_val,
    Val* first_offset_ptr,
    Val* first_offset_val,
    int64_t offsets)
    : Expr(passkey) {
  addOutput(seed_ptr);
  addOutput(seed_val);
  addOutput(first_offset_ptr);
  addOutput(first_offset_val);
  addDataAttribute(offsets);
}

std::string GetRNGSeedAndOffsetFromHost::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "(" << output(0)->toString() << ", "
                          << output(1)->toString() << ", "
                          << output(2)->toString() << ", "
                          << output(3)->toString() << ") = " << getOpString()
                          << "()\n";
  return ss.str();
}

std::string GetRNGSeedAndOffsetFromHost::toInlineString(int indent_size) const {
  return std::string(getOpString()) + "()";
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GetRNGSeedAndOffsetFromHost)

EncodeTensorMapTiled::EncodeTensorMapTiled(
    IrBuilderPasskey passkey,
    Val* output,
    DataType data_type,
    Val* global_address,
    Val* global_dim,
    Val* global_strides,
    Val* box_dim,
    Val* element_strides,
    tma::TensorMapInterleave interleave,
    MmaInputSmemSwizzle swizzle,
    tma::TensorMapL2Promotion l2_promotion,
    tma::TensorMapFloatOOBFill oob_fill)
    : Expr(passkey) {
  auto out_dtype = output->dtype();
  NVF_CHECK(std::holds_alternative<OpaqueType>(out_dtype.type));
  addOutput(output);

  NVF_CHECK(
      global_address->dtype() ==
      PointerType{std::make_shared<DataType>(data_type)});
  addInput(global_address);

  NVF_CHECK(std::holds_alternative<ArrayType>(global_dim->dtype().type));
  size_t tensor_rank = std::get<ArrayType>(global_dim->dtype().type).size;
  ArrayType expect_global_dim_type{
      std::make_shared<DataType>(DataType::Index), tensor_rank};
  NVF_CHECK(global_dim->dtype() == expect_global_dim_type);
  addInput(global_dim);

  ArrayType expect_global_strides_type{
      std::make_shared<DataType>(DataType::Index), tensor_rank - 1};
  NVF_CHECK(global_strides->dtype() == expect_global_strides_type);
  addInput(global_strides);

  ArrayType expect_box_dim_type{
      std::make_shared<DataType>(DataType::Index), tensor_rank};
  NVF_CHECK(box_dim->dtype() == expect_box_dim_type);
  addInput(box_dim);

  ArrayType expect_element_strides_type{
      std::make_shared<DataType>(DataType::Index), tensor_rank};
  NVF_CHECK(element_strides->dtype() == expect_element_strides_type);
  addInput(element_strides);

  addDataAttribute(data_type);
  addDataAttribute((int64_t)tensor_rank);
  addDataAttribute(interleave);
  addDataAttribute(swizzle);
  addDataAttribute(l2_promotion);
  addDataAttribute(oob_fill);
}

std::string EncodeTensorMapTiled::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << output(0)->toString() << " = " << getOpString()
                          << "(dtype=" << dataType()
                          << ", global_address=" << globalAddress()->toString()
                          << ", global_dim=" << globalDim()->toString()
                          << ", global_strides=" << globalStrides()
                          << ", box_dim=" << boxDim()->toString()
                          << ", element_strides="
                          << elementStrides()->toString()
                          << ", interleave=" << interleave()
                          << ", swizzle=" << nvfuser::toString(swizzle())
                          << ", l2_promotion=" << l2Promotion()
                          << ", oob_fill=" << oobFill() << ")\n";
  return ss.str();
}

std::string EncodeTensorMapTiled::toInlineString(int indent_size) const {
  std::stringstream ss;
  ss << getOpString() << "(dtype=" << dataType()
     << ", global_address=" << globalAddress()->toInlineString()
     << ", global_dim=" << globalDim()->toInlineString()
     << ", global_strides=" << globalStrides()->toInlineString()
     << ", box_dim=" << boxDim()->toInlineString()
     << ", element_strides=" << elementStrides()->toInlineString()
     << ", interleave=" << interleave()
     << ", swizzle=" << nvfuser::toString(swizzle())
     << ", l2_promotion=" << l2Promotion() << ", oob_fill=" << oobFill() << ")";
  return ss.str();
}

NVFUSER_DEFINE_CLONE_AND_CREATE(EncodeTensorMapTiled)

RNGOp::RNGOp(
    IrBuilderPasskey passkey,
    Val* out,
    Val* rng_result,
    Val* rng_component,
    DataType dtype,
    RNGOpType rng_type,
    // range high and low, or avg and std dev
    std::vector<Val*> parameters)
    : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  NVF_ERROR(out->isA<kir::TensorIndex>());
  NVF_ERROR(rng_result->isA<TensorView>());
  NVF_ERROR(rng_result->as<TensorView>()->getMemoryType() == MemoryType::Local);
  NVF_ERROR(rng_result->as<TensorView>()->dtype() == DataType::UInt32);
  NVF_ERROR(rng_result->as<TensorView>()->nDims() == 1);
  NVF_ERROR(rng_result->as<TensorView>()->axis(0)->extent()->isConstInt());
  NVF_ERROR(
      rng_result->as<TensorView>()
          ->axis(0)
          ->extent()
          ->evaluate()
          .as<int64_t>() == 4);
  NVF_ERROR(rng_component->dtype() == DataType::Index);
  NVF_ERROR(rng_component->isA<NamedScalar>());
  addInput(rng_result);
  addInput(rng_component);
  addOutput(out);
  for (auto v : parameters) {
    addInput(v);
  }
  addDataAttribute(rng_type);
  addDataAttribute(dtype);
}

std::string RNGOp::toString(int indent_size) const {
  std::stringstream ss;
  ss << output(0)->toString() << " = " << getRNGOpType() << "("
     << input(0)->toString();
  for (auto inp_i : arange(1, inputs().size())) {
    ss << ", " << input(inp_i)->toString();
  }
  ss << ")\n";
  return ss.str();
}

std::string RNGOp::toInlineString(int indent_size) const {
  std::stringstream ss;
  ss << getRNGOpType() << "(" << input(0)->toString();
  for (auto inp_i : arange(1, inputs().size())) {
    ss << ", " << input(inp_i)->toString();
  }
  ss << ")";
  return ss.str();
}

NVFUSER_DEFINE_CLONE_AND_CREATE(RNGOp)

} // namespace kir
} // namespace nvfuser
