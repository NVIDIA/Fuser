// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <scheduler/mma_utils.h>
#include <visibility.h>

#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace nvfuser {

class Fusion;
class TensorView;

namespace cutlass_codegen {

//! This is a tree data structure that mimics the CUTLASS EVT we will generate
class EVTModel {
 public:
  EVTModel() = default;

  EVTModel(const EVTModel& m);

  EVTModel(EVTModel&& m) = default;

  EVTModel& operator=(const EVTModel& m) = default;

  ~EVTModel() = default;

  //! Nodes are generic cutlass templated classes. We do not distinguish tree
  //! visitors from topo visitors, or compute nodes from load or store nodes,
  //! etc. Each Node is defined by its type name and a list of template
  //! parameters, which we call inputs. These objects are owned by EVTModel and
  //! can be created using makeNode.
  struct Node {
    const std::string name;
    std::vector<Node*> inputs;
    // If an argument is required, provide it here
    Val* argument = nullptr;
  };

  Node* makeNode(const std::string& name) {
    nodes_up_.push_back(std::make_unique<Node>(name));
    return nodes_up_.back().get();
  }

  Node* root() const {
    return root_;
  }

  void setRoot(Node* new_root) {
    root_ = new_root;
  }

  //! Generate the C++ code used to define the EVT type
  std::string defString(Node* node = nullptr, int64_t indent = 2) const;

  //! Generate the arguments to be used as args.epilogue.thread
  std::string argString(Node* node = nullptr, int64_t indent = 2) const;

  //! Print all nodes in a topological order
  std::string toString() const;

 private:
  std::deque<std::unique_ptr<Node>> nodes_up_;
  Node* root_;
};

//! Convert a Fusion into an EVTModel. This includes creating nodes to
//! represent the default epilogue in ScaledMmaOp, alpha*acc + beta*bias, when
//! those arguments are provided.
mma_utils::DataWrapperOpt<EVTModel> extractEVTModel(Fusion* fusion);

} // namespace cutlass_codegen

} // namespace nvfuser
