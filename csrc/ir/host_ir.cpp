// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <host_ir_container.h>
#include <ir/builder.h>
#include <ir/cloner.h>
#include <ir/host_ir.h>
#include <ir/printer.h>
#include <ir/utils.h>
#include <kernel_ir.h>
#include <ops/all_ops.h>

namespace nvfuser {

namespace hir {

HostUnit::HostUnit(IrBuilderPasskey passkey, std::unique_ptr<Fusion> fusion)
    : Expr(passkey), fusion_(std::make_unique<Fusion>(*fusion)) {}

HostUnit::HostUnit(const HostUnit* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      fusion_(std::make_unique<Fusion>(*src->fusion_to_execute())) {}

NVFUSER_DEFINE_CLONE_AND_CREATE(HostUnit)

std::string HostUnit::toString(int indent_size) const {
  int indent_increment = 2;
  std::stringstream ss;
  indent(ss, indent_size) << "Execute the following kernel, taking inputs :{\n";
  for (auto input : fusion_->inputs()) {
    indent(ss, indent_size + indent_increment)
        << input->toString(indent_size + indent_increment) << "\n";
  }
  indent(ss, indent_size) << "} and outputs: {\n";
  for (auto output : fusion_->outputs()) {
    indent(ss, indent_size + indent_increment)
        << output->toString(indent_size + indent_increment) << "\n";
  }
  indent(ss, indent_size) << "}. Kernel:{";
  fusion_->print(ss, false, indent_size + indent_increment);
  indent(ss, indent_size) << "\n";
  indent(ss, indent_size) << "}" << std::endl;
  return ss.str();
}

// TODO: implement better ?
std::string HostUnit::toInlineString(int indent_size) const {
  return toString(indent_size);
}

// TODO: implement
bool HostUnit::sameAs(const Statement* other) const {
  return false;
}

PostOnStream::PostOnStream(
    IrBuilderPasskey passkey,
    HostUnit* hu,
    std::vector<Val*> inputs,
    std::vector<Val*> outputs)
    : Expr(passkey, std::move(inputs), std::move(outputs), {hu}) {
  NVF_ERROR(this->inputs().size() == hu->fusion_to_execute()->inputs().size());
  NVF_ERROR(
      this->outputs().size() == hu->fusion_to_execute()->outputs().size());
  // TODO: harden the assert checks with smth like
  // for (int i : c10::irange(inputs.size())) {
  //     // NVF_ERROR(inputs.at(i)->sameAs(executable_fusion->inputs().at(i)));
  // }
  // for (int i : c10::irange(outputs.size())) {
  //     //
  //     NVF_ERROR(outputs.at(i)->sameAs(executable_fusion->outputs().at(i)));
  // }
}

NVFUSER_DEFINE_CLONE_AND_CREATE(PostOnStream)

std::string PostOnStream::toString(int indent_size) const {
  std::stringstream ss;
  return ss.str();
}

// TODO: implement better ?
std::string PostOnStream::toInlineString(int indent_size) const {
  return toString(indent_size);
}

// TODO: implement
bool PostOnStream::sameAs(const Statement* other) const {
  return false;
}

} // namespace hir

} // namespace nvfuser
