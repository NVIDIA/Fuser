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
#include <host_ir.h>


namespace nvfuser {

namespace hir {

ExecuteFusion::ExecuteFusion(IrBuilderPasskey passkey,
                             std::unique_ptr<Fusion> fusion,
                           std::vector<Val*> inputs,
                           std::vector<Val*> outputs)
    : Expr(passkey), fusion_(std::make_unique<Fusion>(*fusion))
{
    std::for_each(inputs.begin(),
                  inputs.end(),
                  [this](auto val) {this->addInput(val);});
    std::for_each(outputs.begin(),
                  outputs.end(),
                  [this](auto val) {this->addOutput(val);});
}


NVFUSER_DEFINE_CLONE_AND_CREATE(ExecuteFusion)

std::string ExecuteFusion::toString(int indent_size) const {
    int indent_increment = 2;
    std::stringstream ss;
    indent(ss, indent_size) << "Execute the following kernel, taking inputs :{\n";
    for (auto input : inputs()) {
        indent(ss, indent_size + indent_increment) << input->toString(indent_size + indent_increment) << "\n";
    }
    indent(ss, indent_size) << "} and outputs: {\n";
    for (auto output : outputs()) {
        indent(ss, indent_size + indent_increment) <<  output->toString(indent_size + indent_increment) << "\n";
    }
    indent(ss, indent_size) << "}. Kernel:{";
    fusion_->print(ss, false, indent_size + indent_increment);
    indent(ss, indent_size) << "\n";
    indent(ss, indent_size) << "}" << std::endl;
    return ss.str();

}

// TODO: implement better ?
std::string ExecuteFusion::toInlineString(int indent_size) const {
    return toString(indent_size);
}

// TODO: implement
bool ExecuteFusion::sameAs(const Statement* other) const {
    return false;
}

} // namespace hir

} // namespace nvfuser
