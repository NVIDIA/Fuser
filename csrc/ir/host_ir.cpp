// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ops/all_ops.h>
#include <kernel_ir.h>
#include <ir/builder.h>
#include <ir/printer.h>
#include <ir/cloner.h>
#include <ir/utils.h>
#include <ir/host_ir.h>
#include <host_ir_container.h>


namespace nvfuser {

namespace hir {

HostUnit::HostUnit(IrBuilderPasskey passkey,
                             std::unique_ptr<Fusion> fusion)
    : Expr(passkey), fusion_(std::make_unique<Fusion>(*fusion))
{

    // NVF_ERROR(passkey.ir_container_->isA<hir::HostIrContainer>());
    // std::for_each(inputs.begin(),
    //               inputs.end(),
    //               [this](auto val) {this->addInput(val);});
    // std::for_each(outputs.begin(),
    //               outputs.end(),
    //               [this](auto val) {this->addOutput(val);});
}

HostUnit::HostUnit(const HostUnit* src, IrCloner* ir_cloner) : Expr(src, ir_cloner), fusion_(std::make_unique<Fusion>(*src->fusion_to_execute())) {}


NVFUSER_DEFINE_CLONE_AND_CREATE(HostUnit)

std::string HostUnit::toString(int indent_size) const {
    int indent_increment = 2;
    std::stringstream ss;
    indent(ss, indent_size) << "Execute the following kernel, taking inputs :{\n";
    for (auto input : fusion_->inputs()) {
        indent(ss, indent_size + indent_increment) << input->toString(indent_size + indent_increment) << "\n";
    }
    indent(ss, indent_size) << "} and outputs: {\n";
    for (auto output : fusion_->outputs()) {
        indent(ss, indent_size + indent_increment) <<  output->toString(indent_size + indent_increment) << "\n";
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

int StreamIr::running_counter_ = 0;

StreamIr::StreamIr(IrBuilderPasskey passkey): Val(passkey, ValType::StreamIr), idx_(running_counter_++) {};

StreamIr::StreamIr(const StreamIr* src, IrCloner* ir_cloner): Val(src, ir_cloner), idx_(src->idx_){};
NVFUSER_DEFINE_CLONE(StreamIr)

std::string StreamIr::toString(int indent_size) const {
    std::stringstream ss;
    indent(ss, indent_size) << "Stream " << idx_;
    return ss.str();
}

std::string StreamIr::toInlineString(int indent_size) const {
    return toString(indent_size);
}

bool StreamIr::sameAs(const Statement* other) const {
    return false;
}


PostOnStream::PostOnStream(IrBuilderPasskey passkey,
                            HostUnit* hu,
                            std::vector<Val*> inputs,
                            std::vector<Val*> outputs)
    : Expr(passkey, std::move(inputs), std::move(outputs), {hu}) {
    NVF_ERROR(this->inputs().size() == hu->fusion_to_execute()->inputs().size());
    NVF_ERROR(this->outputs().size() == hu->fusion_to_execute()->outputs().size());
    // TODO: harden the assert checks
    // for (int i : c10::irange(inputs.size())) {
    //     // NVF_ERROR(inputs.at(i)->sameAs(executable_fusion->inputs().at(i)));
    //     std::cout << "host input " << i << ": " << inputs.at(i) << std::endl;
    //     std::cout << "fusion input " << i << ": " << executable_fusion->inputs().at(i) << std::endl;
    // }
    // for (int i : c10::irange(outputs.size())) {
    //     // NVF_ERROR(outputs.at(i)->sameAs(executable_fusion->outputs().at(i)));
    //     std::cout << "host output " << i << ": " << outputs.at(i) << std::endl;
    //     std::cout << "fusion output " << i << ": " << executable_fusion->outputs().at(i) << std::endl;
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
