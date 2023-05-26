// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ir/builder.h>
#include <ir/cloner.h>
#include <ir/utils.h>
#include <multidevice/multicluster_fusion.h>

namespace nvfuser {

Cluster::Cluster(
    const MultiClusterFusion* multi_cluster_fusion,
    ClusterParams params)
    : unique_id(multi_cluster_fusion->clusterCounter()),
      multi_cluster_fusion_(multi_cluster_fusion),
      params_(params) {}

std::unique_ptr<Fusion> Cluster::toFusion() const {
  return multi_cluster_fusion_->ClusterToFusion(this);
}

std::string Cluster::toString(int indent_size) const {
  std::stringstream ss;
  std::string indent(indent_size, ' ');

  // print the Cluster params
  ss << indent << "g" << unique_id << " {"
     << "("
     << "auto_schedule=" << params().auto_schedule
     << ", process_rank=" << params().process_rank << ")\n";

  ss << indent << "  inputs:\n";
  for (auto input : inputs().vector()) {
    ss << indent << "    " << input->toString(indent_size) << "\n";
  }

  ss << indent << "  exprs:\n";
  for (auto expr : exprs_.vector()) {
    ss << indent << "    " << expr->toString(indent_size) << "\n";
  }

  ss << indent << "  outputs:\n";
  for (auto output : outputs().vector()) {
    ss << indent << "    " << output->toString(indent_size) << "\n";
  }

  ss << indent << "}";
  return ss.str();
}

void MultiClusterFusion::newCluster(ClusterParams params) {
  // Stores the new cluster into the fusion's container
  clusters_.push_back(std::make_shared<Cluster>(this, params));

  // increments the cluster counter
  cluster_counter_++;

  // Set the newly created cluster as the current cluster
  setCurrentCluster(clusters_.back());
}

void MultiClusterFusion::registerStmt(
    IrBuilderPasskey passkey,
    Statement* stmt) {
  Fusion::registerStmt(passkey, stmt);

  if (stmt->isA<Expr>()) {
    auto expr = stmt->as<Expr>();
    auto current_cluster = getCurrentCluster();

    current_cluster->exprs_.pushBack(expr);

    for (auto output_tv : ir_utils::filterByType<TensorView>(expr->outputs())) {
      current_cluster->vals_.pushBack(output_tv);
    }

    for (auto input_tv : ir_utils::filterByType<TensorView>(expr->inputs())) {
      // If an Expr's input is not in the cluster, mark it as a cluster input.
      if (!current_cluster->vals().has(input_tv)) {
        current_cluster->addInput(input_tv);
      }
    }
  }
}

void MultiClusterFusion::addClusterOutput(TensorView* tv) {
  auto cluster = getCurrentCluster();

  // Check that the given tensor is defined internally
  //  within the cluster's context.
  TORCH_INTERNAL_ASSERT(
      cluster->vals().has(tv), tv->toString(), "not in cluster");

  // Add the tv to the cluster outputs.
  cluster->addOutput(tv);
}

std::unique_ptr<AggregateDag> MultiClusterFusion::aggregateDag() const {
  return std::make_unique<AggregateDag>(this);
}

std::unique_ptr<Fusion> MultiClusterFusion::ClusterToFusion(
    const Cluster* cluster) const {
  std::unique_ptr<Fusion> fusion_copy = std::make_unique<Fusion>();
  // WAR: copy the complete fusion and then change the inputs and outputs.
  // TODO: This could be implemented in a better way
  auto original_to_copy_map = Fusion::copy(this, fusion_copy.get());

  auto original_inputs = fusion_copy->inputs();
  auto original_outputs = fusion_copy->outputs();

  // Remove original inputs
  std::for_each(
      original_inputs.begin(), original_inputs.end(), [&](auto input) {
        fusion_copy->removeInput(input);
      });
  // Remove original outputs
  std::for_each(
      original_outputs.begin(), original_outputs.end(), [&](auto output) {
        fusion_copy->removeOutput(output);
      });

  // // Add cluster inputs
  std::for_each(
      cluster->inputs().begin(), cluster->inputs().end(), [&](auto input) {
        fusion_copy->addInput(original_to_copy_map.clone(input));
      });

  // // Add cluster outputs
  std::for_each(
      cluster->outputs().begin(), cluster->outputs().end(), [&](auto output) {
        fusion_copy->addOutput(original_to_copy_map.clone(output));
      });

  return fusion_copy;
}

std::string MultiClusterFusion::toString(int indent_size) const {
  std::stringstream ss;
  ss << "MultiClusterFusion {\n";
  for (auto& cluster_ptr : clusters_) {
    ss << cluster_ptr->toString(indent_size + 2);
    ss << "\n";
  }
  ss << "} //MultiClusterFusion";

  return ss.str();
}

} // namespace nvfuser
