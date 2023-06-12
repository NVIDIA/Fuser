// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ir/builder.h>
#include <ir/cloner.h>
#include <ir/printer.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <multidevice/pipeline.h>
#include <multidevice/pipeline_ir.h>

namespace nvfuser {

int PipelineStageDescriptor::running_unique_id_ = 0;

// Utility class used for Pipeline instantiation
class PipelineBuilder final {
 public:
  PipelineBuilder(Pipeline* pipeline) : pipeline_(pipeline) {
    validate();
    fillInfo();
    buildPipelineVals();
    buildPipelineStages();
    buildPipelineCommunications();
  }

 private:
  // Points to the pipeline to be constructed
  Pipeline* pipeline_;
  // Stores the Stages IR of the pipeline
  std::vector<PipelineStage*> stages;
  // stores the Vals that will be I/O of each stage
  std::unordered_map<const PipelineStageDescriptor*, ValSet> StageInputsDesc;
  std::unordered_map<const PipelineStageDescriptor*, ValSet> StageOutputsDesc;
  // Store the stages that consumes a given Val produced at another stage
  std::unordered_map<Val*, std::vector<const PipelineStageDescriptor*>>
      ValConsumerStageDesc;
  // maps Vals of the original fusion to their corresponding PipelineVals
  // which are inputs/outputs of a Stage. We need to differentiate those
  // two cases for when a stage contains only one Val, from which we create
  // two PipelineVal
  std::unordered_map<Val*, Val*> valToPipelineValInputOfStage;
  std::unordered_map<Val*, Val*> valToPipelineValOutputOfStage;

  // Check (partially) that the pipeline is valid and satisfies the assumptions
  // described in pipeline.h
  void validate() {
    std::unordered_set<Val*> tvInStages;
    // Check that each Val is a TensorView and that it belongs to at most one
    // stage
    for (auto& stage_desc : pipeline_->descriptor().stageDescriptors) {
      for (auto& val : stage_desc->vals()) {
        if (!val->isA<TensorView>()) {
          TORCH_INTERNAL_ASSERT(
              0,
              "the Val " + val->toString() +
                  " must be a TensorView to belong to a Pipeline");
        } else if (tvInStages.count(val)) {
          TORCH_INTERNAL_ASSERT(
              0,
              "the Val " + val->toString() + " belongs to more than one stage");
        } else {
          tvInStages.insert(val);
        }
      }
    }
    // Check that each TensorView belongs to at least one stage
    for (auto& val : pipeline_->originalFusion()->vals()) {
      TORCH_INTERNAL_ASSERT(
          (!val->isA<TensorView>()) || tvInStages.count(val),
          "the Val " + val->toString() + " must be added to a stage");
    }
  }

  // returns whether a Val is an input of the originalFusion
  bool isGlobalInput(Val* val) const {
    return std::count(
        pipeline_->originalFusion()->inputs().begin(),
        pipeline_->originalFusion()->inputs().end(),
        val);
  }

  // returns whether a Val is an output of the originalFusion
  bool isGlobalOutput(Val* val) const {
    return std::count(
        pipeline_->originalFusion()->outputs().begin(),
        pipeline_->originalFusion()->outputs().end(),
        val);
  }

  // populates StageInputsDesc, StageOutputsDesc, ValConsumerStageDesc
  void fillInfo() {
    for (auto& stage_desc : pipeline_->descriptor().stageDescriptors) {
      for (auto& val : stage_desc->vals()) {
        // Add global inputs of the original fusion
        if (isGlobalInput(val)) {
          StageInputsDesc[stage_desc].pushBack(val);
        }
        // Add global outputs of the original fusion
        if (isGlobalOutput(val)) {
          StageOutputsDesc[stage_desc].pushBack(val);
        }
        // Add Vals which are produced in-between stages
        if (val->definition()) {
          for (auto& val_producer : val->definition()->inputs()) {
            if (!stage_desc->vals().has(val_producer)) {
              StageInputsDesc[stage_desc].pushBack(val);
              ValConsumerStageDesc[val_producer].push_back(stage_desc);
            }
          }
        }
      }
    }
    // Add Vals which are consumed in-between stages
    for (auto& stage_desc : pipeline_->descriptor().stageDescriptors) {
      for (auto& val : stage_desc->vals()) {
        if (!ValConsumerStageDesc[val].empty()) {
          StageOutputsDesc[stage_desc].pushBack(val);
        }
      }
    }
  }

  /* Creates the PipelineVals IR in the Pipeline
     A PipelineVal is created for each input/output of each Stage of the
     Fusion. If a Val is an I/O of the original Fusion,
     the correspondings PipelineVal are also I/O of the Pipeline */
  void buildPipelineVals() {
    for (auto stage_desc : pipeline_->descriptor().stageDescriptors) {
      // Create a PipelineVal for each stage's input
      for (auto val : StageInputsDesc[stage_desc].vector()) {
        auto pVal =
            IrBuilder::create<PipelineVal>(pipeline_->as<IrContainer>(), val);
        valToPipelineValInputOfStage[val] = pVal;
        // If the Val is an Input of the original Fusion,
        // then add the newly created PipelineVal as an input of the pipeline
        if (isGlobalInput(val)) {
          pipeline_->addInput(pVal);
          TORCH_INTERNAL_ASSERT(
              stage_desc->mesh.size() == 1,
              "A global input must belong to a stage which mesh is of size 1");
        } else {
          // if the Val is a stage input but not a global input, it must be
          // defined by a "Set" operation
          TORCH_INTERNAL_ASSERT(
              (val->definition()->isA<LoadStoreOp>()) &&
                  (val->definition()->as<LoadStoreOp>()->opType() ==
                   LoadStoreOpType::Set),
              "A Val that is the input of a stage must be defined by a LoadStoreOp expression of type Set"
              "but here the definition is " +
                  val->definition()->toString());
        }
      }
      // Create a PipelineVal for each stage's output
      for (auto val : StageOutputsDesc[stage_desc].vector()) {
        auto pVal =
            IrBuilder::create<PipelineVal>(pipeline_->as<IrContainer>(), val);
        valToPipelineValOutputOfStage[val] = pVal;
        if (isGlobalOutput(val)) {
          pipeline_->addOutput(pVal);
        }
      }
    }
  }

  // Build the PipelineStage IR of the Pipeline
  // An PipelineStage is created for each Stage of the Fusion
  void buildPipelineStages() {
    for (auto& stage_desc : pipeline_->descriptor().stageDescriptors) {
      // containers for storing the I/O of the PipelineStage
      ValSet ins, outs;
      for (auto& val : StageInputsDesc[stage_desc]) {
        ins.pushBack(valToPipelineValInputOfStage[val]);
      }
      for (auto& val : StageOutputsDesc[stage_desc]) {
        outs.pushBack(valToPipelineValOutputOfStage[val]);
      }
      auto stage = IrBuilder::create<PipelineStage>(
          pipeline_->as<IrContainer>(), stage_desc, ins, outs);
      stages.push_back(stage);
    }
  }

  // Build the PipelineCommunication IR of the Pipeline
  // A PipelineCommunication is created for each stage's input
  void buildPipelineCommunications() {
    for (auto& stage : stages) {
      for (auto& pVal : stage->inputs()) {
        std::vector<Val*> ins, outs;
        auto val = pVal->as<PipelineVal>()->getOriginalVal();
        if (isGlobalInput(val)) {
          continue;
        }
        outs.push_back(pVal);
        for (auto& producer : val->definition()->inputs()) {
          ins.push_back(valToPipelineValOutputOfStage[producer]);
        }
        TORCH_INTERNAL_ASSERT(
            std::size(ins) == 1 && std::size(outs) == 1,
            "Pipeline Communications must involve one input and one output");
        IrBuilder::create<PipelineCommunication>(
            pipeline_->as<IrContainer>(), ins.at(0), outs.at(0));
      }
    }
  }
};

Pipeline::Pipeline(
    Fusion* fusion,
    PipelineDescriptor // NOLINT (pass fusion as value and use std::move)
        descriptor) // NOLINT (pass fusion as value and use std::move)
    : originalFusion_(fusion), descriptor_(descriptor) {
  PipelineBuilder{this};
}

std::unique_ptr<Fusion> Pipeline::stageToFusion(PipelineStage*& stage) const {
  std::unique_ptr<Fusion> fusion_copy = std::make_unique<Fusion>();
  /* WAR/TODO: copy the complete fusion and then change the inputs and outputs.
               This could be implemented in a more optimal way */
  auto original_to_copy_map = Fusion::copy(originalFusion(), fusion_copy.get());

  auto original_inputs = fusion_copy->inputs();
  auto original_outputs = fusion_copy->outputs();

  // Remove original inputs
  std::for_each(
      original_inputs.begin(), original_inputs.end(), [&](auto& input) {
        fusion_copy->removeInput(input);
      });
  // Remove original outputs
  std::for_each(
      original_outputs.begin(), original_outputs.end(), [&](auto& output) {
        fusion_copy->removeOutput(output);
      });

  // Add stage inputs
  std::for_each(
      stage->inputs().begin(), stage->inputs().end(), [&](Val* const& input) {
        fusion_copy->addInput(original_to_copy_map.clone(
            input->as<PipelineVal>()->getOriginalVal()));
      });

  // Add stage outputs
  std::for_each(
      stage->outputs().begin(),
      stage->outputs().end(),
      [&](Val* const& output) {
        fusion_copy->addOutput(original_to_copy_map.clone(
            output->as<PipelineVal>()->getOriginalVal()));
      });

  return fusion_copy;
}

// Printer for Pipeline
class PipelinePrinter : public IterVisitor {
 public:
  explicit PipelinePrinter(Pipeline* a) : IterVisitor(), pipeline_(a) {
    string_ << "Pipeline's inputs{:\n";
    for (auto input : pipeline_->inputs()) {
      string_ << " " << input << "\n";
    }
    string_ << "}\n";

    string_ << "Pipeline's Traversal inputs --> outputs {\n";
    traverseTo(pipeline_, pipeline_->outputs());
    string_ << "}\n";

    string_ << "Pipeline's outputs:{\n";
    for (auto output : pipeline_->outputs()) {
      string_ << " " << output << "\n";
    }
    string_ << "}";
  }

  std::string toString() const {
    return string_.str();
  }

 private:
  // Overriding IterVisitor
  void handle(Statement* stmt) override {
    if (std::count(
            pipeline_->inputs().begin(), pipeline_->inputs().end(), stmt) +
            std::count(
                pipeline_->outputs().begin(),
                pipeline_->outputs().end(),
                stmt) ==
        0) {
      string_ << "  " << stmt->toString() << "\n";
    }
  }

  Pipeline* pipeline_;
  std::stringstream string_;
};

std::string Pipeline::toString() {
  PipelinePrinter p(this);
  return p.toString();
}

} // namespace nvfuser
