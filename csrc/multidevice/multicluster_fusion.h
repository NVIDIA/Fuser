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

namespace nvfuser {

using ProcessRankType = int;
using ValSet = VectorOfUniqueEntries<Val*>;
using ExprSet = VectorOfUniqueEntries<Expr*>;

class MultiClusterFusion;
class AggregateDag;

struct ClusterParams {
  // if the cluster should be auto-scheduled.
  bool auto_schedule{true};
  // The process's rank on which the cluster will be executed at runtime
  ProcessRankType process_rank;
};

// Class representing a segment in the fusion that will be compiled and executed
// on one single rank and device.
// It is represented by an AggregateExpr in the AggregateDag
class TORCH_CUDA_CU_API Cluster final {
 public:
  Cluster(const MultiClusterFusion* multi_cluster_fusion, ClusterParams params);

  std::string toString(int indent_size = 0) const;

  // Unique identifier for the cluster.
  const int unique_id;

  ClusterParams params() const {
    return params_;
  }

  // all the Vals belonging to the Cluster, including inputs/outputs
  const auto& vals() const {
    return vals_;
  }

  const auto& inputs() const {
    return inputs_;
  }

  const auto& outputs() const {
    return outputs_;
  }

  // add cluster input
  void addInput(Val* input) {
    vals_.pushBack(input);
    inputs_.pushBack(input);
  }

  // add cluster output
  void addOutput(Val* output) {
    vals_.pushBack(output);
    outputs_.pushBack(output);
  }

  // returns a Fusion representing the sub-DAG of the nodes present in the
  // Cluster
  std::unique_ptr<Fusion> toFusion() const;

 private:
  friend class MultiClusterFusion;

  const MultiClusterFusion* const multi_cluster_fusion_ = nullptr;
  ClusterParams params_;
  ValSet inputs_;
  ValSet outputs_;
  ValSet vals_;

  // stores the Exprs that appear in the Cluster
  // For now it is only used for printing. Could consider removing it
  ExprSet exprs_;
};

using ClusterPtr = std::shared_ptr<Cluster>;

// User interface for building multi-cluster fusion
class TORCH_CUDA_CU_API MultiClusterFusion : public Fusion {
 public:
  MultiClusterFusion() : Fusion() {}

  std::string toString(int indent_size = 0) const;

  // Returns list of all clusters from the fusion.
  const std::vector<ClusterPtr>& clusters() const {
    return clusters_;
  }

  // Creates a new Cluster and marks starting point of the newly created
  // cluster: Once this is called, all subsequent statement registered in the
  // fusion will belong to this cluster, until newCluster is called again.
  void newCluster(ClusterParams params);

  // Make the given tensor a cluster output
  void addClusterOutput(TensorView* tv);

  // overrides IrContainer
  // Called insided IrBuilder each time a new ir is created
  void registerStmt(IrBuilderPasskey, Statement* stmt) override;

  // return the current (which is basically last cluster created)
  ClusterPtr getCurrentCluster() {
    TORCH_INTERNAL_ASSERT(!clusters_.empty(), "call newCluster first.");
    return current_cluster_;
  }

  // set a cluster as the "current cluster"
  void setCurrentCluster(ClusterPtr cluster) {
    current_cluster_ = cluster;
  }

  int clusterCounter() const {
    return cluster_counter_;
  }

  // build the aggregateDag
  std::unique_ptr<AggregateDag> aggregateDag() const;

  // returns a Fusion representing the sub-DAG of the nodes present in a given
  // Cluster
  // TODO: for now, we copy the complete underlying fusion and then change
  // the inputs and outputs. Should be optimized
  std::unique_ptr<Fusion> ClusterToFusion(const Cluster* cluster) const;

 private:
  // Running counter to generate unique cluster id.
  int cluster_counter_ = 0;

  // Stores clusters of the fusion
  std::vector<ClusterPtr> clusters_;

  // Keep track of the current cluster, which either has been manually set
  // through setCurrentCluster method or is the latest cluster created
  ClusterPtr current_cluster_;
};

} // namespace nvfuser
