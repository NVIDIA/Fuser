// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <fusion.h>
#include <ir_base_nodes.h>
#include <ir_container.h>

namespace nvfuser {

class Cluster;
class MultiClusterFusion;

// TODO: Revisit
using ClusterPtr = std::shared_ptr<Cluster>;

// Implements the AggregateDag. It is built from a multicluster fusion.
// Its traversal is what defines the runtime execution.
// The nodes of the dag are the Vals AggregateVal
// The edges are Expr of either type 1) AggregateExpr which represent the
// operations contained in a Cluster, or 2)SendRecv which represents the
// collective operation between two processes that needs to be done at runtime.

// AggregateDag is an IrContainer, built from a multicluster fusion.
// Its traversal is what orders the runtime execution.
class TORCH_CUDA_CU_API AggregateDag : public Fusion {
 public:
  AggregateDag(const MultiClusterFusion* MCFusion);

  std::string toString();

  const std::vector<Val*>& MCFusionInputs() const;
  const std::vector<Val*>& MCFusionOutputs() const;

 private:
  const MultiClusterFusion* MCFusion_ = nullptr;

  // The following methods are a set of helper functions called in the class
  // constructor

  // Creates the AggregateVals IR in the AggregateDag
  // An AggregateVal is created for each input/output of each Cluster of the
  // MultiClusterFusion If a Val is an I/O of the original MultiClusterFusion,
  // the correspondings AggregateVal are also I/O of the AggregateDag
  void buildAVals();

  // Build the AggregateExpr IR of the AggregateDag
  // An AggregateExpr is created for each Cluster of the MultiClusterFusion
  void buildAExpr();

  // Build the SendRecv IR of the AggregateDag
  // A SendRecv is created for each pair of AggregateVals which Vals match
  // and which are not inputs/outputs of the AggregateDag
  void buildSendRecv();
};

// An AggregateVal is basically a Val and a Cluster. At runtime, since a Cluster
// is typically associated with a rank, AggregateVal represents a Val which is
// located at a rank.
class TORCH_CUDA_CU_API AggregateVal : public Val {
 public:
  AggregateVal(IrBuilderPasskey passkey, Val* val, ClusterPtr cluster);
  AggregateVal(const AggregateVal* src, IrCloner* ir_cloner);

  NVFUSER_DECLARE_CLONE

  virtual std::string toString(int indent_size = 0) const override;
  virtual std::string toInlineString(int indent_size = 0) const override;

  // returns the Val from which this AggregateVal has been created
  Val* getOriginalVal() const {
    return original_val_;
  }

  // returns the Cluster from which this AggregateVal has been created
  ClusterPtr getCluster() const {
    return cluster_;
  }

  // returns true if the Val used in the constructor is an output (resp. input)
  // of the Cluster used in the constructor
  // Note that, currently, exactly one of the two predicates must be true.
  bool isOriginalValOutputOfCluster() const;
  bool isOriginalValInputOfCluster() const;

  bool sameAs(const Statement* other) const override;

 private:
  // stores the Val used in the constructor
  Val* original_val_;
  // stores the Cluster used in the constructor
  ClusterPtr cluster_;
};

// An AggregateExpr represents the Exprs that are defined inside a Cluster.
class TORCH_CUDA_CU_API AggregateExpr : public Expr {
 public:
  using Expr::Expr;
  AggregateExpr(IrBuilderPasskey, ClusterPtr cluster);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual std::string toString(int indent_size = 0) const override;
  virtual std::string toInlineString(int indent_size = 0) const override;

  const char* getOpString() const override {
    return "AggregateExpr";
  }

  bool sameAs(const Statement* other) const override;

  const ClusterPtr getCluster() const {
    return cluster_;
  }

 private:
  // stores the Cluster used in the constructor
  ClusterPtr cluster_;
};

// SendRecv is an Expr that represents at runtime the send/receive
// of a IValue between two processes.
class TORCH_CUDA_CU_API SendRecv : public Expr {
 public:
  using Expr::Expr;
  SendRecv(IrBuilderPasskey, AggregateVal* out, AggregateVal* in);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual std::string toString(int indent_size = 0) const override;
  virtual std::string toInlineString(int indent_size = 0) const override;

  const char* getOpString() const override {
    return "SendRecv";
  }

  AggregateVal* out() const {
    return out_;
  }
  AggregateVal* in() const {
    return in_;
  }

 private:
  AggregateVal* const out_ = nullptr;
  AggregateVal* const in_ = nullptr;
};

} // namespace nvfuser
