// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <disjoint_set.h>
#include <fusion.h>
#include <ir/base_nodes.h>
#include <multidevice/device_mesh.h>

/*
This file implements the Pipeline interface.
A Pipeline represents a Fusion or a parent stage segmented into a series of
Stages, each Stage being thought as a portion of the original Fusion/stage that
will be treated as a task in task-parallelism.

The decomposition of the Pipeline into Stages is described through a
PipelineDescriptor, which is nothing but a vector of PipelineStageDescriptor.
Each PipelineStageDescriptor desribes a stage by listing all the
Vals that compose it.

During initialization, the system will figure out the data dependencies and
infer what is each stage's inputs and outputs, defined as follow: a Val is a
stage's input (resp. output) if it is either a global input (resp. output) of
the originalFusion or if it is a direct consumer (resp. producer) of a Val from
a another stage.

To be considered as valid, the PipelineDescriptor must ensure that the described
Pipeline satisfies the following assumption:
*) Each Val belongs to one and only one Stage
*) The directed graph composed of the Stages and their dependencies is acyclic
   (i.e., the Pipeline is a DAG (Directed Acyclic Graph))
   Remark: the word "Pipeline" can be misleading because it conveys the idea of
a linear dependency between stages -- however our Pipeline is allowed to have
any DAG structure
*) The inputs of a stage are TensorViews which are either global inputs of the
fusion OR are defined as copies (a "set" operation) of a Tv from another stage
    (Later we could add the case where they are defined as a reduction).
*) Global inputs of the Fusion belong to stage(s) whose mesh contains only one
device index. Note: Later, when we add a new parallel type for inter-device
sharding (dIdx, dIdy, etc...) we can loosen this condition by assuming only that
the input values are not replicated on different devices. This assumption is
natural because the input values should be defined only once.

The PipelineStageDescriptor are passed to the Pipeline (through a
PipelineDescriptor) by REFERENCE. As a consequence, after instantiation of the
Pipeline, it is not allowed to add Vals to the PipelineStageDescriptor nor to
free them. However, even after the Pipeline is instantiated, the user can still
set the mesh (see multidevice/device_mesh.h) which is basically an n-array of
devices indices on which the stage should be executed at runtime.
*/

namespace nvfuser {

class PipelineStage;

// Interface to describe the composition of a PipelineStage
class TORCH_CUDA_CU_API PipelineStageDescriptor final {
  using ValSet = VectorOfUniqueEntries<Val*>;

 public:
  PipelineStageDescriptor() : unique_id(running_unique_id_++) {}

  // The mesh on which the stage will be executed at runtime.
  DeviceMesh mesh;
  /* Unique identifier for the stage.
     Only used for printing/debugging to easily identify each stage.*/
  const int unique_id;

  // returns all the Vals belonging to the Stage
  auto vals() const {
    return vals_;
  }

  // add a Val to the Stage
  void addVal(std::vector<Val*> vals) {
    for (auto& val : vals) {
      vals_.pushBack(val);
    }
  }

 private:
  // stores the Vals belonging to the Stage
  ValSet vals_;
  // used to set the unique_id attribute
  static int running_unique_id_;
};

// Interface to describe the composition of a Pipeline
struct PipelineDescriptor {
  std::vector<PipelineStageDescriptor> stage_descriptors;
};

/*
The Pipeline Class is instantiated from a Fusion (called originalFusion)
and a PipelineDescriptor.

It itself inherits from Fusion, which is populated at initialization
with three types of IRs (see multidevice/pipeline_ir.h):
1) PipelineVal (inheriting from Val), representing Vals that are I/O of
   each Stage (including global I/O)
2) PipelineStage (inherinting from Expr), representing each Stage given
   in the PipelineDescriptor
3) PipelineCommunication (inherinting from Expr), representing the transfer of
   PipelineVals in-between stages

The pipeline forms a DAG with PipelineVals as nodes and PipelineStages &
PipelineCommunications as edges.
*/
class TORCH_CUDA_CU_API Pipeline : public Fusion {
 public:
  Pipeline(Fusion* fusion, PipelineDescriptor descriptor);

  std::string toString();

  const auto& descriptor() const {
    return descriptor_;
  }

  auto originalFusion() const {
    return original_fusion_;
  }

  /* returns a Fusion copied from the originalFusion but
     with I/O as the stage's I/O
     TODO: for now, we copy entirely the original fusion and then only change
     the inputs and outputs. Should be optimized */
  std::unique_ptr<Fusion> stageToFusion(PipelineStage*& stage) const;

 private:
  // utility class called at instantiation
  friend class PipelineBuilder;

  Fusion* original_fusion_ = nullptr;
  PipelineDescriptor descriptor_;
};

} // namespace nvfuser
