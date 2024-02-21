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

using kir::ForLoop;

std::unique_ptr<HostFusion> makeHostFusionFromFusion(Fusion* fusion) {
    auto host_fusion = std::make_unique<HostFusion>();
    FusionGuard fg(host_fusion.get());
    host_fusion->gpu_fusion = fusion;

    auto ir_container = static_cast<IrContainer*>(host_fusion.get());
    auto tvs = ir_utils::allTvs(fusion);
    NVF_ERROR(std::all_of(tvs.begin(), tvs.end(),
                [](auto tv) {return tv->getLeafDomain().at(0)->isHostDim();}
                ), "Need the outmost of all tvs to be of parallel type host");

    IrCloner ir_cloner(host_fusion.get());
    // Input
    NVF_ERROR(fusion->inputs().size()==1, "there must be exactly one input");
    // TensorView* input_tv = IrBuilder::create<TensorView>(ir_container, fusion->inputs().at(0)->as<TensorView>(), ir_cloner);
    TensorView* input_tv = ir_cloner.clone(fusion->inputs().at(0)->as<TensorView>());
    host_fusion->addInput(input_tv);

    // Output
    NVF_ERROR(fusion->outputs().size()==1, "there must be exactly one output");
    // TensorView* input_tv = IrBuilder::create<TensorView>(ir_container, fusion->inputs().at(0)->as<TensorView>(), ir_cloner);
    TensorView* output_tv = ir_cloner.clone(fusion->outputs().at(0)->as<TensorView>());
    host_fusion->addOutput(output_tv);
    // bind host-extents to be one ?
    // auto vector_of_outputs = IrBuilder::create<VectorOfOuputs>(ir_container, output_tv);

    //For Loop
    // IterDomain* id = IrBuilder::create<IterDomain>(ir_container, fusion->inputs().at(0)->as<TensorView>()->getLeafDomain().at(0));
    IterDomain* id = input_tv->getLeafDomain().at(0);
    Val* index = IrBuilder::create<Val>(ir_container, 0, DataType::Index);
    Val* start = IrBuilder::create<Val>(ir_container, 0, DataType::Index);
    Val* stop = id->extent();
    Val* step = IrBuilder::create<Val>(ir_container, 1, DataType::Index);

    auto for_loop = IrBuilder::create<kir::ForLoop>(ir_container,
            id, index, start, stop, step, false, nullptr, false, DoubleBufferLoopStage::NotApplicable);

    // Slices
    std::vector<Slice> ranges_input(input_tv->getLeafDomain().size());
    Val* one = IrBuilder::create<Val>(ir_container, 1, DataType::Index);
    ranges_input.at(0).start = index;
    ranges_input.at(0).step = one;
    ranges_input.at(0).stop = add(index, one);

    std::vector<Slice> ranges_output(TensorDomain::noReductions(output_tv->getLeafDomain()).size());
    ranges_output.at(0).start = index;
    ranges_output.at(0).step = one;
    ranges_output.at(0).stop = add(index, one);

    // sliced I/O
    TensorView* sliced_input = slice(input_tv, ranges_input);
    TensorView* sliced_output = slice(output_tv, ranges_output);

    // launchFusion
    std::vector<Val*> inputs = {sliced_input};
    std::vector<Val*> outputs = {sliced_output};
    Expr* launch_fusion = IrBuilder::create<ExecuteFusion>(ir_container, fusion, inputs, outputs);

    // save sliced output
    Expr* save_sliced_output = IrBuilder::create<SaveSlicedOutput>(ir_container, sliced_output, output_tv, index);

    // populate for loop
    auto& scope = for_loop->body();
    for (auto expr : {sliced_input->definition(), launch_fusion, save_sliced_output}) {
        scope.push_back(expr);
    }

    host_fusion->top_level_exprs.push_back(for_loop);

    return host_fusion;
}

ExecuteFusion::ExecuteFusion(IrBuilderPasskey passkey,
                           Fusion* fusion,
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

SaveSlicedOutput::SaveSlicedOutput(IrBuilderPasskey passkey,
                           Val* src, Val* dst, Val* index)
    : Expr(passkey), src_(src), dst_(dst), index_(index)
{
    addInput(src_);
    addInput(index_);
    addOutput(dst_);
}

NVFUSER_DEFINE_CLONE_AND_CREATE(SaveSlicedOutput)

std::string SaveSlicedOutput::toString(int indent_size) const {
    std::stringstream ss;
    std::string indentation(" ", indent_size);
    indent(ss, indent_size)  << "save " << src_->toString()
       << " to " << dst_->toString()
       << " at index " << index_->toString() << std::endl;
    return ss.str();

}

// TODO: implement better ?
std::string SaveSlicedOutput::toInlineString(int indent_size) const {
    return toString(indent_size);
}

// TODO: implement
bool SaveSlicedOutput::sameAs(const Statement* other) const {
    return false;
}

std::ostream& HostFusion::print(std::ostream& os, bool include_tensor_transforms, int indent_size) const {
  os << "\n%HostFusion {\n";
  IrMathPrinter op_exprs(os, indent_size);
  op_exprs.handle(this);
  NVF_ERROR(!include_tensor_transforms, "not implemented for now");
//   if (include_tensor_transforms) {
//     os << "\nTransformPrinter : \n";
//     IrTransformPrinter t_exprs(os, indent_size);
//     t_exprs.handle(this);
//   }
  os << "}\n";

  return os;
}


} // namespace hir

} // namespace nvfuser



    // auto host_fusion = std::make_unique<HostFusion>();
    // IrCloner cloner (static_cast<IrContainer*>(host_fusion.get()));
    // for (auto input : fusion->inputs()) {
    //     auto new_input = IrBuilder::clone<TensorView>(input->as<TensorView>(), &cloner);
    //     host_fusion->addInput(new_input);
    // }

    // for (auto output : fusion->outputs()) {
    //     auto new_output = IrBuilder::clone<TensorView>(output->as<TensorView>(), &cloner);
    //     host_fusion->addOutput(new_output);
    // }

    // auto ca_map = ComputeAtMap(host_fusion.get());
    // std::unordered_map<IterDomain*, IterDomain*> concrete_to_reference_map;
    // for (auto tv : ir_utils::filterByType<TensorView>(host_fusion->vals())) {
    //     for (auto id : tv->getLeafDomain()) {
    //         auto ca_id = ca_map.getConcreteMappedID(id, IdMappingMode::PERMISSIVE_RESIZE);
    //         concrete_to_reference_map[ca_id] = id;
    //     }        
    // }

    // concat
//   CatOp(
//       IrBuilderPasskey passkey,
//       Val* out,
//       const std::vector<Val*>& inputs,
//       int64_t concatenated_dim);
//   auto tv4 = cat({tv0, tv1}, 0);




