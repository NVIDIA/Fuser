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

/* Utility class used for Pipeline instantiation called by the Pipeline's
  constructor. This class is responsible for:
  - checking that the parameters are valid
  - analyzing the data dependency from original_fusion_ and descriptor_ to infer
  what should be the I/O of each stage
  - creating the IR composing the pipeline:
       - each StageDescriptor will result in the creation a PipelineStage
       - each I/O of a PipelineStage will be represented by a PipelineVal
       - each communication of a PipelineVal between two PipelineStage will
         be represented by a PipelineCommunication
  - setting the global I/O of the Pipeline as the PipelineVals representing
    the global I/O of the original_fusion_
*/
class PipelineBuilder final {
  using ValSet = VectorOfUniqueEntries<Val*>;

 public:
  PipelineBuilder(Pipeline* pipeline) : pipeline_(pipeline) {
    validate();
    fillInfo();
    buildPipelineVals();
    buildPipelineStages();
    buildPipelineCommunications();
  }

 private:
  // Pointer to the pipeline to be constructed
  Pipeline* pipeline_;
  // Stores the Stages IR of the pipeline
  std::vector<PipelineStage*> stages_;
  // stores the Vals that will be I/O of each stage
  std::unordered_map<const PipelineStageDescriptor*, ValSet> stage_input_desc_;
  std::unordered_map<const PipelineStageDescriptor*, ValSet> stage_output_desc_;
  // Store the stages that consumes a given Val produced at another stage
  std::unordered_map<Val*, std::vector<const PipelineStageDescriptor*>>
      val_consumer_stage_desc_;
  // maps Vals of the original fusion to their corresponding PipelineVals
  // which are inputs/outputs of a Stage. We need to differentiate those
  // two cases for when a stage contains only one Val, from which we create
  // two PipelineVal
  std::unordered_map<Val*, PipelineVal*> val_to_pipeline_val_input_of_stage_;
  std::unordered_map<Val*, PipelineVal*> val_to_pipeline_val_output_of_stage_;

  // Check (partially) that the pipeline is valid and satisfies the assumptions
  // described in pipeline.h
  // TODO: For now we only check the TensorView for simplicity.
  //       Ideally we also need to perform the same check for Vals that are not
  //       TensorView, but this requires slight changes in the Pipeline
  //       interface
  void validate() {
    std::unordered_set<TensorView*> tv_in_stages;
    // Check that each TensorView belongs to at most one stage
    for (auto& stage_desc : pipeline_->descriptor().stage_descriptors) {
      for (auto val : stage_desc.vals()) {
        if (val->isA<TensorView>()) {
          NVF_ERROR(
              tv_in_stages.insert(val->as<TensorView>()).second,
              "the TensorView " + val->toString() +
                  " belongs to more than one stage");
        }
      }
    }
    // Check that each TensorView belongs to at least one stage
    for (auto tv : ir_utils::filterByType<TensorView>(
             pipeline_->originalFusion()->vals())) {
      NVF_ERROR(
          tv_in_stages.count(tv),
          "the Val " + tv->toString() + " must be added to a stage");
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

  // populates stage_input_desc_, stage_output_desc_, val_consumer_stage_desc_
  void fillInfo() {
    for (auto& stage_desc : pipeline_->descriptor().stage_descriptors) {
      for (auto& val : stage_desc.vals()) {
        // Add global inputs of the original fusion
        if (isGlobalInput(val)) {
          stage_input_desc_[&stage_desc].pushBack(val);
        }
        // Add global outputs of the original fusion
        if (isGlobalOutput(val)) {
          stage_output_desc_[&stage_desc].pushBack(val);
        }
        // Add Vals which are produced in-between stages
        if (val->definition()) {
          for (auto& val_producer : val->definition()->inputs()) {
            if (!stage_desc.vals().has(val_producer)) {
              stage_input_desc_[&stage_desc].pushBack(val);
              val_consumer_stage_desc_[val_producer].push_back(&stage_desc);
            }
          }
        }
      }
    }
    // Add Vals which are consumed in-between stages
    for (auto& stage_desc : pipeline_->descriptor().stage_descriptors) {
      for (auto& val : stage_desc.vals()) {
        if (!val_consumer_stage_desc_[val].empty()) {
          stage_output_desc_[&stage_desc].pushBack(val);
        }
      }
    }
  }

  /* Creates the PipelineVals IR in the Pipeline
     A PipelineVal is created for each input/output of each Stage of the
     Fusion. If a Val is an I/O of the original Fusion,
     the correspondings PipelineVal are also I/O of the Pipeline */
  void buildPipelineVals() {
    for (auto& stage_desc : pipeline_->descriptor().stage_descriptors) {
      // Create a PipelineVal for each stage's input
      for (auto val : stage_input_desc_[&stage_desc]) {
        auto p_val =
            IrBuilder::create<PipelineVal>(pipeline_->as<IrContainer>(), val);
        val_to_pipeline_val_input_of_stage_[val] = p_val;
        // If the Val is an Input of the original Fusion,
        // then add the newly created PipelineVal as an input of the pipeline
        if (isGlobalInput(val)) {
          pipeline_->addInput(p_val);
          NVF_ERROR(
              stage_desc.mesh.vector().size() == 1,
              "A global input must belong to a stage which mesh is of size 1");
        } else {
          // if the Val is a stage input but not a global input, it must be
          // defined by a "Set" operation
          NVF_ERROR(
              (val->definition()->isA<LoadStoreOp>()) &&
                  (val->definition()->as<LoadStoreOp>()->opType() ==
                   LoadStoreOpType::Set),
              "A Val that is the input of a stage must be defined by a LoadStoreOp expression of type Set"
              "but here the definition is " +
                  val->definition()->toString());
        }
      }
      // Create a PipelineVal for each stage's output
      for (auto val : stage_output_desc_[&stage_desc]) {
        auto p_val =
            IrBuilder::create<PipelineVal>(pipeline_->as<IrContainer>(), val);
        val_to_pipeline_val_output_of_stage_[val] = p_val;
        if (isGlobalOutput(val)) {
          pipeline_->addOutput(p_val);
        }
      }
    }
  }

  // Build the PipelineStage IR of the Pipeline
  // An PipelineStage is created for each Stage of the Fusion
  void buildPipelineStages() {
    for (auto& stage_desc : pipeline_->descriptor().stage_descriptors) {
      // containers for storing the I/O of the PipelineStage
      ValSet ins, outs;
      for (auto val : stage_input_desc_[&stage_desc]) {
        ins.pushBack(val_to_pipeline_val_input_of_stage_.at(val));
      }
      for (auto val : stage_output_desc_[&stage_desc]) {
        outs.pushBack(val_to_pipeline_val_output_of_stage_.at(val));
      }
      auto stage = IrBuilder::create<PipelineStage>(
          pipeline_->as<IrContainer>(), &stage_desc, ins, outs);
      stages_.push_back(stage);
    }
  }

  // Build the PipelineCommunication IR of the Pipeline
  // A PipelineCommunication is created for each stage's input
  void buildPipelineCommunications() {
    for (auto& stage : stages_) {
      for (auto& p_val : stage->inputs()) {
        std::vector<Val*> ins, outs;
        auto val = p_val->as<PipelineVal>()->getOriginalVal();
        if (isGlobalInput(val)) {
          continue;
        }
        outs.push_back(p_val);
        for (auto& producer : val->definition()->inputs()) {
          ins.push_back(val_to_pipeline_val_output_of_stage_.at(producer));
        }
        NVF_ERROR(
            std::size(ins) == 1 && std::size(outs) == 1,
            "Pipeline Communications must involve one input and one output");
        IrBuilder::create<PipelineCommunication>(
            pipeline_->as<IrContainer>(), ins.at(0), outs.at(0));
      }
    }
  }
};

Pipeline::Pipeline(Fusion* fusion, PipelineDescriptor descriptor)
    : original_fusion_(fusion), descriptor_(std::move(descriptor)) {
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
  void dispatch(Statement* stmt) override {
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
