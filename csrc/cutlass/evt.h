// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace nvfuser {

class Fusion

namespace cutlass_codegen {

//! This is a tree data structure that reflects the EVT we will generate
class EVTModel {
 public:
  EVTModel copy() const;

  struct Node {
    std::string name;
    std::vector<Node*> inputs;
  };

  Node* makeNode(const std::string& name) {
    Node* new_node = nodes_up_.emplace_back(new Node(name)).get();
    return new_node;
  }

  Node* root() const {
    return root_;
  }

  void setRoot(Node* new_root) {
    root_ = new_root;
  }

  // TODO: accept a "depth" argument and format the output prettily
  std::string defString(Node* node = nullptr) const {
    if (node == nullptr) {
      node = root_;
    }
    NVF_ERROR(node != nullptr);
    std::stringstream ss;
    ss << node->name;
    if (!node->inputs.empty()) {
      ss << "<";
      bool first = true;
      for (Node* input : node->inputs) {
        if (!first) {
          ss << ", ";
        }
        first = false;
        ss << defString(input);
      }
      ss << ">";
    }
    return ss.str();
  }

 private:
  std::deque<std::unique_ptr<Node>> nodes_up_;
  Node* root_;
};

EVTModel extractEVTModel(Fusion* fusion);

} // namespace cutlass_codegen

} // namespace nvfuser
