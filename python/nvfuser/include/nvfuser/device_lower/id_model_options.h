// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <id_model/utils.h>

#include <sstream>

namespace nvfuser {

class IdModelOptions {
 public:
  IdModelOptions()
      : build_id_model_(!isOptionDisabled(DisableOption::IdModel)),
        consumer_index_(
            isIdModelOptionEnabled(IdModelEnableOption::ConsumerIndex)),
        producer_index_(
            isIdModelOptionEnabled(IdModelEnableOption::ProducerIndex)),
        inline_predicate_(
            isIdModelOptionEnabled(IdModelEnableOption::InlinePredicate)),
        unswitch_predicate_(
            isIdModelOptionEnabled(IdModelEnableOption::UnswitchPredicate)),
        loop_(isIdModelOptionEnabled(IdModelEnableOption::Loop)) {
    ensureConsistency();
  }

  bool buildIdModel() const {
    return build_id_model_;
  }

  void setBuildIdModel(bool b) {
    build_id_model_ = b;
    ensureConsistency();
  }

  bool buildTensorIndexer() const {
    return build_tensor_indexer_;
  }

  void setBuildTensorIndexer(bool b) {
    build_tensor_indexer_ = b;
    ensureConsistency();
  }

  bool consumerIndex() const {
    return consumer_index_;
  }

  void setConsumerIndex(bool b) {
    consumer_index_ = b;
    ensureConsistency();
  }

  bool producerIndex() const {
    return producer_index_;
  }

  void setProducerIndex(bool b) {
    producer_index_ = b;
    ensureConsistency();
  }

  void setIndex(bool b) {
    setConsumerIndex(b);
    setProducerIndex(b);
  }

  bool inlinePredicate() const {
    return inline_predicate_;
  }

  void setInlinePredicate(bool b) {
    inline_predicate_ = b;
    ensureConsistency();
  }

  bool unswitchPredicate() const {
    return unswitch_predicate_;
  }

  void setUnswitchPredicate(bool b) {
    unswitch_predicate_ = b;
    ensureConsistency();
  }

  void setPredicate(bool b) {
    setInlinePredicate(b);
    setUnswitchPredicate(b);
  }

  bool loop() const {
    return loop_;
  }

  void setLoop(bool b) {
    loop_ = b;
    ensureConsistency();
  }

  std::string toString() const {
    auto bool2str = [](bool b) { return b ? "true" : "false"; };

    std::stringstream ss;
    ss << "build_id_model=" << bool2str(build_id_model_)
       << ", build_tensor_indexer=" << bool2str(build_tensor_indexer_)
       << ", consumer_index=" << bool2str(consumer_index_)
       << ", producer_index=" << bool2str(producer_index_)
       << ", inline_predicate=" << bool2str(inline_predicate_)
       << ", unswitch_predicate=" << bool2str(unswitch_predicate_)
       << ", loop=" << bool2str(loop_);
    return ss.str();
  }

 private:
  void ensureConsistency() {
    if (!build_id_model_) {
      build_tensor_indexer_ = false;
      consumer_index_ = false;
      producer_index_ = false;
      inline_predicate_ = false;
      unswitch_predicate_ = false;
      loop_ = false;
    } else {
      // TensorIndexer is required if these options are enabled
      build_tensor_indexer_ = build_tensor_indexer_ || consumer_index_ ||
          producer_index_ || inline_predicate_ || unswitch_predicate_ || loop_;
    }
  }

 private:
  // Build IdModel
  bool build_id_model_ = true;
  // Build TensorIndexer
  bool build_tensor_indexer_ = false;
  // Globally enables consumer indexing.
  bool consumer_index_ = false;
  // Globally enables producer indexing.
  bool producer_index_ = false;
  // Globally enables inline predicate
  bool inline_predicate_ = false;
  // Globally enables unswitch predicate
  bool unswitch_predicate_ = false;
  // Generate loops using IdModel
  bool loop_ = false;
};

} // namespace nvfuser
