// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <host_ir/container.h>
#include <host_ir/host_ir.h>
#include <ir/builder.h>
#include <ir/builder_passkey.h>
#include <ir/cloner.h>
#include <ir/iostream.h>
#include <ir/printer.h>
#include <ir/utils.h>
#include <kernel_ir.h>
#include <multidevice/communication.h>
#include <ops/all_ops.h>
#include <utils.h>

namespace nvfuser {

namespace hir {

HostUnit::HostUnit(IrBuilderPasskey passkey, std::unique_ptr<Fusion> fusion)
    : Expr(passkey), fusion_(std::make_unique<Fusion>(*fusion)) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(passkey.ir_container_->isA<HostIrContainer>());
}

HostUnit::HostUnit(const HostUnit* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      fusion_(std::make_unique<Fusion>(*src->fusion_to_execute())) {}

NVFUSER_DEFINE_CLONE_AND_CREATE(HostUnit)

std::string HostUnit::toString(int indent_size) const {
  std::stringstream ss;
  ss << toInlineString(indent_size) << ": Inputs={";
  std::for_each(
      fusion_to_execute()->inputs().begin(),
      fusion_to_execute()->inputs().end(),
      [&ss](auto input) { ss << input->toString(0) << ", "; });
  ss << "} -> Outputs={";
  std::for_each(
      fusion_to_execute()->outputs().begin(),
      fusion_to_execute()->outputs().end(),
      [&ss](auto output) { ss << output->toString(0) << ", "; });
  ss << "}";
  fusion_->print(ss, false);
  return ss.str();
}

// TODO: implement better ?
std::string HostUnit::toInlineString(int indent_size) const {
  std::stringstream ss;
  ss << "HostUnit" << name();
  return ss.str();
}

// TODO: implement
bool HostUnit::sameAs(const Statement* other) const {
  return false;
}

PostOnStream::PostOnStream(
    IrBuilderPasskey passkey,
    Expr* host_op,
    std::vector<Val*> inputs,
    std::vector<Val*> outputs)
    : Expr(passkey, std::move(inputs), std::move(outputs), {host_op}) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<HostIrContainer>(),
      this,
      "must be registered in a HostIrContainer");
  NVF_ERROR(
      (host_op->isA<HostUnit>()), "wrong host op type: ", host_op->toString());
  if (host_op->isA<HostUnit>()) {
    NVF_ERROR(
        this->inputs().size() ==
        host_op->as<HostUnit>()->fusion_to_execute()->inputs().size());
    NVF_ERROR(
        this->outputs().size() ==
        host_op->as<HostUnit>()->fusion_to_execute()->outputs().size());
    // TODO: harden the assert checks with smth like
    // for (int i : arange(inputs.size())) {
    //     NVF_ERROR(inputs.at(i)->sameAs(executable_fusion->inputs().at(i)));
    // }
    // for (int i : arange(outputs.size())) {
    //     NVF_ERROR(outputs.at(i)->sameAs(executable_fusion->outputs().at(i)));
    // }
  }
}

NVFUSER_DEFINE_CLONE_AND_CREATE(PostOnStream)

std::string PostOnStream::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "PostOnStream ("
                          << hostOpToPost()->toInlineString(0) << ", "
                          << "Inputs:{";
  std::for_each(inputs().begin(), inputs().end(), [&ss](auto input) {
    ss << input->toString(0) << ", ";
  });
  ss << "}, Outputs:{";
  std::for_each(outputs().begin(), outputs().end(), [&ss](auto output) {
    ss << output->toString(0) << ", ";
  });
  ss << "})" << std::endl;
  return ss.str();
}

std::string PostOnStream::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Can not be printed inline");
}

// TODO: implement
bool PostOnStream::sameAs(const Statement* other) const {
  return false;
}

LaunchKernel::LaunchKernel(
    IrBuilderPasskey passkey,
    int64_t group_id,
    const LaunchParams& launch_constraints,
    const CompileParams& compile_params,
    const std::vector<Val*>& inputs,
    const std::vector<Val*>& outputs,
    Val* cache_id)
    : Expr(passkey, inputs, outputs, {}) {
  addDataAttribute(group_id);
  addDataAttribute(launch_constraints);
  addDataAttribute(compile_params);
  addAttribute(cache_id);
}

NVFUSER_DEFINE_CLONE_AND_CREATE(LaunchKernel)

std::string LaunchKernel::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "LaunchKernel(" << std::endl;
  indent(ss, indent_size + 1) << "Group ID: " << groupId() << "," << std::endl;
  indent(ss, indent_size + 1)
      << "Inputs: {" << toDelimitedString(inputs()) << "}," << std::endl;
  indent(ss, indent_size + 1)
      << "Outputs: {" << toDelimitedString(outputs()) << "}," << std::endl;
  indent(ss, indent_size) << ")" << std::endl;
  return ss.str();
}

std::string LaunchKernel::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Can not be printed inline");
}

Deallocate::Deallocate(IrBuilderPasskey passkey, TensorView* tv)
    : Expr(passkey) {
  addAttribute(tv);
}

NVFUSER_DEFINE_CLONE_AND_CREATE(Deallocate)

TensorView* Deallocate::buffer() const {
  return attributes_.at(0)->as<TensorView>();
}

std::string Deallocate::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "Deallocate {" << std::endl;
  ss << buffer()->toString(indent_size + 1) << std::endl;
  indent(ss, indent_size) << "}" << std::endl;
  return ss.str();
}

std::string Deallocate::toInlineString(int indent_size) const {
  return std::string("Deallocate ") + buffer()->toInlineString();
}

Stream::Stream(IrBuilderPasskey passkey, Val* index)
    : Val(passkey, ValType::Stream), index_(index) {}

Stream::Stream(const Stream* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner), index_(src->index()) {}

NVFUSER_DEFINE_CLONE(Stream)

std::string Stream::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "Stream ";
  if (index() == nullptr) {
    ss << name();
  } else {
    ss << index()->toInlineString();
  }
  return ss.str();
}

std::string Stream::toInlineString(int indent_size) const {
  return toString(indent_size);
}

bool Stream::sameAs(const Statement* other) const {
  if (other == this) {
    return true;
  }
  if (!other->isA<Stream>()) {
    return false;
  }

  const auto* other_stream = other->as<Stream>();
  return index() != nullptr && index() == other_stream->index();
}

SetCurrentStream::SetCurrentStream(IrBuilderPasskey passkey, Stream* stream)
    : Expr(passkey, {stream}, {}, {stream}) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(passkey.ir_container_->isA<HostIrContainer>());
}

NVFUSER_DEFINE_CLONE_AND_CREATE(SetCurrentStream)

std::string SetCurrentStream::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "SetCurrentStream to " << stream()->toString()
                          << std::endl;
  return ss.str();
}

// TODO: implement better ?
std::string SetCurrentStream::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Cannot be printed inline");
}

// TODO: implement
bool SetCurrentStream::sameAs(const Statement* other) const {
  return false;
}

GetCurrentStream::GetCurrentStream(IrBuilderPasskey passkey) : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(passkey.ir_container_->isA<HostIrContainer>());
  auto stream = IrBuilder::createInContainer<Stream>(passkey.ir_container_);
  addAttribute(stream);
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GetCurrentStream)

std::string GetCurrentStream::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "GetCurrentStream into " << stream()->toString()
                          << std::endl;
  return ss.str();
}

Wait::Wait(IrBuilderPasskey passkey, Expr* expr)
    : Expr(passkey, {}, {}, {expr}) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<HostIrContainer>(),
      this,
      "must be registered in a HostIrContainer");
  NVF_ERROR(
      (expr->isOneOf<Communication, P2PCommunication, EndCoalescing>()),
      expr,
      "must be a Communication, a P2PCommunication, or a EndCoalescing");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(Wait)

std::string Wait::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "Wait Communication " << communication()->name()
                          << std::endl;
  return ss.str();
}

// TODO: implement better ?
std::string Wait::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Cannot be printed inline");
}

// TODO: implement
bool Wait::sameAs(const Statement* other) const {
  return false;
}

Synchronize::Synchronize(IrBuilderPasskey passkey, Stream* stream)
    : Expr(passkey, {}, {}, {stream}) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<HostIrContainer>(),
      this,
      "must be registered in a HostIrContainer");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(Synchronize)

std::string Synchronize::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "Synchronize " << stream() << std::endl;
  return ss.str();
}

std::string Synchronize::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Cannot be printed inline");
}

// TODO: implement
bool Synchronize::sameAs(const Statement* other) const {
  return false;
}

StartCoalescing::StartCoalescing(IrBuilderPasskey passkey) : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<HostIrContainer>(),
      this,
      "must be registered in a HostIrContainer");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(StartCoalescing)

std::string StartCoalescing::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "StartCoalescing" << std::endl;
  return ss.str();
}

std::string StartCoalescing::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Cannot be printed inline");
}

EndCoalescing::EndCoalescing(IrBuilderPasskey passkey) : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<HostIrContainer>(),
      this,
      "must be registered in a HostIrContainer");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(EndCoalescing)

std::string EndCoalescing::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "EndCoalescing " << name() << std::endl;
  return ss.str();
}

std::string EndCoalescing::toInlineString(int indent_size) const {
  NVF_CHECK(false, "Cannot be printed inline");
}

ShareMemHandles::ShareMemHandles(
    IrBuilderPasskey passkey,
    std::vector<P2PCommunication*> communications)
    : Expr(passkey) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<HostIrContainer>(),
      this,
      "must be registered in a HostIrContainer");
  addDataAttribute(std::move(communications));
}

NVFUSER_DEFINE_CLONE_AND_CREATE(ShareMemHandles)

std::string ShareMemHandles::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "ShareMemHandles(";
  for (auto communication : communications()) {
    ss << communication->toInlineString() << ", ";
  }
  ss << std::endl;
  return ss.str();
}

std::string ShareMemHandles::toInlineString(int indent_size) const {
  NVF_THROW("Cannot be printed inline");
}

HirAliasSelect::HirAliasSelect(
    IrBuilderPasskey passkey,
    TensorView* in,
    TensorView* out,
    int64_t axis,
    Val* index)
    : Expr(passkey, {in, index}, {}, {}) {
  NVF_ERROR(passkey.ir_container_ != nullptr);
  NVF_ERROR(
      passkey.ir_container_->isA<HostIrContainer>(),
      this,
      "must be registered in a HostIrContainer");
  NVF_ERROR(
      static_cast<int64_t>(in->getLogicalDomain().size()) > axis,
      "Select axis ",
      axis,
      " is out of bounds for tensor ",
      in->toString(),
      " with ",
      in->getLogicalDomain().size(),
      " dimensions");
  // "out" is not added as an output because the current op doesn't "define" it,
  // but rather sets its allocation. Since "out" will be used in another
  // producing expression, this avoids unnecessary cyclic dependencies. This
  // ressembles how kir::Allocate treats its allocated TensorView.
  addAttribute(out);
  addDataAttribute(axis);
}

NVFUSER_DEFINE_CLONE_AND_CREATE(HirAliasSelect)

std::string HirAliasSelect::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << "\n";
  indent_size++;
  indent(ss, indent_size) << " = HirAliasSelect( " << in()->toString()
                          << ", axis = " << in()->getLogicalDomain().at(axis())
                          << ", index = " << index()->toString() << " )\n";
  return ss.str();
}

std::string HirAliasSelect::toInlineString(int indent_size) const {
  NVF_THROW("Cannot be printed inline");
}

} // namespace hir

} // namespace nvfuser
