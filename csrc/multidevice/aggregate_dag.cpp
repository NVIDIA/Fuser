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
#include <iter_visitor.h>
#include <multidevice/aggregate_dag.h>
#include <multidevice/multicluster_fusion.h>

namespace nvfuser {

// Utility function that returns a copy of a given AggregateVal's container
// by removing the AggregateVals which do not match with Val or Cluster,
// or that do not satisfy a unary predicate
template <typename UnaryPredicate, typename containerType>
containerType FilterAggregateVals(
    containerType aVals,
    UnaryPredicate unaryPredicate,
    Val* original_val = nullptr,
    ClusterPtr cluster = nullptr) {
  containerType aVals_filtered;
  std::copy_if(
      aVals.cbegin(),
      aVals.cend(),
      std::inserter(aVals_filtered, aVals_filtered.end()),
      [=](Val* v) {
        auto aVal = v->as<AggregateVal>();
        return (cluster == nullptr || aVal->getCluster() == cluster) &&
            (original_val == nullptr ||
             aVal->getOriginalVal() == original_val) &&
            unaryPredicate(aVal);
      });
  return aVals_filtered;
}

template <typename containerType>
containerType FilterAggregateVals(
    containerType aVals,
    Val* val = nullptr,
    ClusterPtr cluster = nullptr) {
  return FilterAggregateVals(
      aVals, [](auto v) { return true; }, val, cluster);
}

const std::vector<Val*>& AggregateDag::MCFusionInputs() const {
  return MCFusion_->inputs();
}

const std::vector<Val*>& AggregateDag::MCFusionOutputs() const {
  return MCFusion_->outputs();
}

void AggregateDag::buildAVals() {
  for (const auto& cluster : MCFusion_->clusters()) {
    for (auto output_val : cluster->outputs().vector()) {
      auto av = IrBuilder::create<AggregateVal>(
          this->as<IrContainer>(), output_val, cluster);
      if (std::count(
              MCFusion_->outputs().begin(),
              MCFusion_->outputs().end(),
              output_val)) {
        addOutput(av);
      }
    }
    for (auto input_val : cluster->inputs().vector()) {
      auto av = IrBuilder::create<AggregateVal>(
          this->as<IrContainer>(), input_val, cluster);
      if (std::count(
              MCFusion_->inputs().begin(),
              MCFusion_->inputs().end(),
              input_val)) {
        addInput(av);
      }
    }
  }
}

void AggregateDag::buildAExpr() {
  for (const auto& cluster : MCFusion_->clusters()) {
    IrBuilder::create<AggregateExpr>(this->as<IrContainer>(), cluster);
  }
}

void AggregateDag::buildSendRecv() {
  // select all AggregateVal that are not global I/O of the AggregateDag
  auto internal_aVals = FilterAggregateVals(vals_, [&](auto val) {
    return std::count(inputs().begin(), inputs().end(), val) +
        std::count(outputs().begin(), outputs().end(), val) ==
        0;
  });

  // Builds the set of original Vals represented in internal_aVals
  std::unordered_set<Val*> internal_Vals;
  std::transform(
      internal_aVals.cbegin(),
      internal_aVals.cend(),
      std::inserter(internal_Vals, internal_Vals.end()),
      [](Val* val) { return val->as<AggregateVal>()->getOriginalVal(); });

  for (auto val : internal_Vals) {
    auto srcs = FilterAggregateVals(
        internal_aVals,
        [](auto val) { return val->isOriginalValOutputOfCluster(); },
        val);
    auto dsts = FilterAggregateVals(
        internal_aVals,
        [](auto val) { return val->isOriginalValInputOfCluster(); },
        val);
    TORCH_INTERNAL_ASSERT(
        std::size(dsts) > 0,
        "WARN: Try to define SendRecv IR with no destinations");
    TORCH_INTERNAL_ASSERT(
        std::size(srcs) == 1,
        "WARN: Try to define SendRecv IR with a number of sources not equal to 1");
    auto src = (*srcs.begin())->as<AggregateVal>();
    for (auto dst : dsts) {
      IrBuilder::create<SendRecv>(
          this->as<IrContainer>(), dst->as<AggregateVal>(), src);
    }
  }
}

AggregateDag::AggregateDag(const MultiClusterFusion* MCFusion)
    : Fusion(), MCFusion_(MCFusion) {
  buildAVals();
  buildAExpr();
  buildSendRecv();
}

// Printer for AggregateDag
class AggregateDagPrinter : public IterVisitor {
 public:
  explicit AggregateDagPrinter(AggregateDag* a) : IterVisitor(), aDag_(a) {
    string_ << "AggregateDag's inputs{:\n";
    for (auto input : aDag_->inputs()) {
      string_ << " " << input << "\n";
    }
    string_ << "}\n";

    string_ << "AggregateDag's Traversal inputs --> outputs {\n";
    traverseTo(aDag_, aDag_->outputs());
    string_ << "}\n";

    string_ << "AggregateDag's outputs:{\n";
    for (auto output : aDag_->outputs()) {
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
    if (std::count(aDag_->inputs().begin(), aDag_->inputs().end(), stmt) == 0 &&
        std::count(aDag_->outputs().begin(), aDag_->outputs().end(), stmt) ==
            0) {
      string_ << "  " << stmt->toString() << "\n";
    }
  }

  AggregateDag* aDag_;
  std::stringstream string_;
};

std::string AggregateDag::toString() {
  AggregateDagPrinter p(this);
  return p.toString();
}

AggregateExpr::AggregateExpr(IrBuilderPasskey passkey, ClusterPtr cluster)
    : Expr(passkey), cluster_(cluster) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<AggregateDag>(),
      "IR type only valid for AggregateDag container.");

  auto output_set = FilterAggregateVals(
      passkey.ir_container_->vals(),
      [](auto val) { return val->isOriginalValOutputOfCluster(); },
      nullptr,
      cluster);
  auto input_set = FilterAggregateVals(
      passkey.ir_container_->vals(),
      [](auto val) { return val->isOriginalValInputOfCluster(); },
      nullptr,
      cluster);
  for (auto v : output_set) {
    addOutput(v);
  }
  for (auto v : input_set) {
    addInput(v);
  }
}

std::string AggregateExpr::toString(int indent_size) const {
  std::stringstream ss;
  ss << "AggregateExpr representing Cluster " << getCluster()->unique_id << ".";
  ss << "Inputs={";
  for (auto input : inputs()) {
    // TODO: use dispatcher to print the Original Val (here, we assume it is a
    // TensorView)
    ss << input->as<AggregateVal>()->getOriginalVal()->toString(indent_size);
    ss << ", ";
  }
  ss << "}. Outputs={";
  for (auto output : outputs()) {
    ss << output->as<AggregateVal>()->getOriginalVal()->toString(indent_size);
    ss << ", ";
  }
  ss << "}.";
  return ss.str();
}

std::string AggregateExpr::toInlineString(int indent_size) const {
  return toString(indent_size);
}

NVFUSER_DEFINE_CLONE_AND_CREATE(AggregateExpr)

bool AggregateExpr::sameAs(const Statement* other) const {
  if (!Expr::sameAs(other)) {
    return false;
  }
  return cluster_ == other->as<AggregateExpr>()->getCluster();
}

SendRecv::SendRecv(
    IrBuilderPasskey passkey,
    AggregateVal* out,
    AggregateVal* in)
    : Expr(passkey), out_{out}, in_{in} {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<AggregateDag>(),
      "IR type only valid for AggregateDag container.");
  addOutput(out);
  addInput(in);
}

NVFUSER_DEFINE_CLONE_AND_CREATE(SendRecv)

std::string SendRecv::toString(int indent_size) const {
  std::stringstream ss;
  ss << "Send/Receive Val {" << in()->getOriginalVal()->toString(indent_size)
     << "} from cluster " << in()->getCluster()->unique_id << " to cluster "
     << out()->getCluster()->unique_id;
  return ss.str();
}

std::string SendRecv::toInlineString(int indent_size) const {
  return toString(indent_size);
}

AggregateVal::AggregateVal(
    IrBuilderPasskey passkey,
    Val* val,
    ClusterPtr cluster)
    : Val(passkey, ValType::AggregateVal, val->dtype()),
      original_val_(val),
      cluster_(cluster) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<AggregateDag>(),
      "IR type only valid for AggregateDag container.");
  TORCH_INTERNAL_ASSERT(
      std::count(cluster->outputs().begin(), cluster->outputs().end(), val) +
              std::count(
                  cluster->inputs().begin(), cluster->inputs().end(), val) >
          0,
      "When building an AggregateVal, the val must be one of the "
      "cluster's input or ouput");
}

AggregateVal::AggregateVal(const AggregateVal* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner),
      original_val_(src->original_val_),
      cluster_(src->cluster_) {}

bool AggregateVal::sameAs(const Statement* other) const {
  if (!Val::sameAs(other)) {
    return false;
  }
  const auto other_aggregate_val = other->as<AggregateVal>();
  return original_val_->sameAs(other_aggregate_val->original_val_) &&
      cluster_->unique_id == other_aggregate_val->cluster_->unique_id;
}

NVFUSER_DEFINE_CLONE(AggregateVal)

bool AggregateVal::isOriginalValOutputOfCluster() const {
  return std::find(
             cluster_->outputs().begin(),
             cluster_->outputs().end(),
             original_val_) != cluster_->outputs().end();
}

bool AggregateVal::isOriginalValInputOfCluster() const {
  return std::find(
             cluster_->inputs().begin(),
             cluster_->inputs().end(),
             original_val_) != cluster_->inputs().end();
}

std::string AggregateVal::toString(int indent_size) const {
  std::stringstream ss;
  ss << "AggregateVal representing Val "
     << getOriginalVal()->toString(indent_size) << " on cluster "
     << getCluster()->unique_id;
  return ss.str();
}

std::string AggregateVal::toInlineString(int indent_size) const {
  return toString(indent_size);
}

} // namespace nvfuser
