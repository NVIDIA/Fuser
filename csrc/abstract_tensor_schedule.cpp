// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <abstract_tensor.h>
#include <abstract_tensor_schedule.h>
#include <ir/internal_base_nodes.h>
#include <iter_visitor.h>
#include <val_graph.h>

namespace nvfuser {

namespace {

class AbstractTensorSchedule : public IterVisitor {
 public:
  static void apply(const AbstractTensor& abstract, TensorView* concrete) {
    AbstractTensorSchedule ats(abstract, concrete);
    ats.run();
  }

 private:
  AbstractTensorSchedule(const AbstractTensor& abstract, TensorView* concrete)
      : abstract_(abstract), concrete_(concrete) {}

  void run() {
    findNearestProducers();

    // Now, for each ValGroup in abstract, find the closest producer ValGroups
    // with entries in concrete_ids_ and replay the path from them. If none
    // exists, then do not include this dimension in the output
    std::vector<IterDomain*> loop_domain;
    for (const AbstractId& abs_id : abstract_.domain) {
      IterDomain* new_id = replayAbstractId(abs_id);
      if (new_id == nullptr) {
        continue;
      }
      loop_domain.push_back(new_id);
    }

    concrete_->setLoopDomain(loop_domain);
  }

  //! Work backward from each loop IterDomain in concrete. When we find an
  //! IterDomain* with a ValGroup in scheduled_val_groups, we map to it
  //! (including siblings) in concrete_ids_. Assert that we do find a
  //! scheduled ValGroup, otherwise we would not know where to place this
  //! IterDomain.
  void findNearestProducers() {
    // Record all ValGroups that are producers of those in abstract_. This lets
    // us find starting IterDomains for scheduling concrete's loop domain.
    // TODO: this should probably be a more general utility like DependencyCheck
    // in val_graph_visitor.h
    std::unordered_set<ValGroup> scheduled_val_groups;
    ValGraph* graph = nullptr;
    std::stack<ValGroup> vg_stack;
    for (const AbstractId& abs_id : abstract_.domain) {
      NVF_ERROR(
          abs_id.is<ValGroupAndItsGraph>(),
          "abstract tensor must contain only ValGroups");
      const ValGroupAndItsGraph& gg = abs_id.as<ValGroupAndItsGraph>();
      if (graph == nullptr) {
        graph = gg.graph;
      } else {
        NVF_ERROR(graph == gg.graph);
      }
      vg_stack.push(gg.group);
    }
    while (!vg_stack.empty()) {
      ValGroup vg = vg_stack.top();
      vg_stack.pop();
      scheduled_val_groups.insert(vg);
      // TODO: Do we need to check for cycles?
      for (const ExprGroup& eg : graph->getDefinitions(vg)) {
        for (Expr* e : *eg) {
          for (Val* inp : e->inputs()) {
            ValGroup vg_inp = graph->toGroup(inp);
            vg_stack.push(vg_inp);
          }
        }
      }
    }

    // Now traverse c2p from concrete loop domain, stopping when we find a
    // scheduled ValGroup
    std::stack<IterDomain*> id_stack;
    for (IterDomain* id : concrete_->getLoopDomain()) {
      id_stack.push(id);
    }
    while (!id_stack.empty()) {
      IterDomain* id = id_stack.top();
      id_stack.pop();
      ValGroup g = graph->toGroup(id);
      if (scheduled_val_groups.find(g) != scheduled_val_groups.end()) {
        concrete_ids_.emplace(g, id);
        continue;
      }
      NVF_ERROR(
          id->definition() != nullptr,
          "Root IterDomain ",
          id->toString(),
          " does not appear in the history of any ValGroups in abstract tensor");
      for (Val* inp : id->definition()->inputs()) {
        if (auto inp_id = dynamic_cast<IterDomain*>(inp)) {
          id_stack.push(inp_id);
        }
      }
    }
  }

  IterDomain* replayAbstractId(AbstractId abs_id) {
    NVF_ERROR(
        abs_id.is<ValGroupAndItsGraph>(),
        "abstract must contain only ValGroups");
    ValGroup g = abs_id.as<ValGroupAndItsGraph>().group;
    return nullptr;
  }

 private:
  const AbstractTensor& abstract_;
  TensorView* concrete_;

  std::unordered_map<ValGroup, IterDomain*> concrete_ids_;
};

} // namespace

void applyAbstractSchedule(
    const AbstractTensor& abstract,
    TensorView* concrete) {
  AbstractTensorSchedule::apply(abstract, concrete);
  /*
  if (concrete->nDims() == 0) {
    return;
  }

  VectorOfUniqueEntries<ValGroup> concrete_loop, abstract_loop;

  ValGraph* graph = nullptr;
  for (const AbstractId& abs_id : abstract.domain) {
    NVF_ERROR(abs_id.is<ValGroupAndItsGraph>());
    ValGroupAndItsGraph gg(abs_id.as<ValGroupAndItsGraph>());
    abstract_loop.pushBack(gg.group);
    if (graph == nullptr) {
      graph = gg.graph;
    } else {
      NVF_ERROR(graph == gg.graph);
    }
  }

  // Mapping from ValGroups to IterDomains corresponding to concrete
  std::unordered_map<ValGroup, IterDomain*> concrete_iter_map;

  NVF_ERROR(abstract.size() > 0);
  for (IterDomain* id : concrete->getLoopDomain()) {
    ValGroup g = graph->toGroup(id);
    concrete_loop.pushBack(g);
    concrete_iter_map.emplace(g, id);
  }

  for (auto [eg, dir] :
       ValGraphBFS::getExprsBetween(graph, concrete_loop, abstract_loop)) {
    NVF_ERROR(
        dir == Direction::Forward,
        "Only forward paths are allowed between concrete and abstract");
    NVF_ERROR(!eg->empty());
    if (auto s = dynamic_cast<Split*>(eg->front())) {
      ValGroup g_in = graph->toGroup(s->in());
      ValGroup g_outer = graph->toGroup(s->outer());
      ValGroup g_inner = graph->toGroup(s->inner());

      auto it = concrete_iter_map.find(g_in);
      if (it == concrete_iter_map.end()) {
        continue;
      }
      IterDomain* id_in = it->second;

      auto [id_outer, id_inner] = IterDomain::split(id_in, s->factor(),
  s->innerSplit());

      concrete_iter_map.emplace(g_outer, id_outer);
      concrete_iter_map.emplace(g_inner, id_inner);
    } else if (auto m = dynamic_cast<Merge*>(eg->front())) {
      ValGroup g_outer = graph->toGroup(m->outer());
      ValGroup g_inner = graph->toGroup(m->inner());
      ValGroup g_out = graph->toGroup(m->out());

      auto it = concrete_iter_map.find(g_outer);
      if (it == concrete_iter_map.end()) {
        continue;
      }
      IterDomain* id_outer = it->second;
      it = concrete_iter_map.find(g_inner);
      if (it == concrete_iter_map.end()) {
        continue;
      }
      IterDomain* id_inner = it->second;

      IterDomain* id_out = IterDomain::merge(id_outer, id_inner);

      concrete_iter_map.emplace(g_out, id_out);
    } else if (auto s = dynamic_cast<Swizzle*>(eg->front())) {
      NVF_ERROR(
          false,
          "Expr ",
          s->toString(),
          " not yet handled in applyAbstractSchedule");
    } else if (auto s = dynamic_cast<Swizzle2D*>(eg->front())) {
      NVF_ERROR(
          false,
          "Expr ",
          s->toString(),
          " not yet handled in applyAbstractSchedule");
    }
  }

  // Now create loop domain using concrete_iter_map
  std::vector<IterDomain*> new_loop;
  for (AbstractId abs_id : abstract.domain) {
    auto it = concrete_iter_map.find(abs_id.as<ValGroupAndItsGraph>().group);
    if (it != concrete_iter_map.end()) {
      new_loop.push_back(it->second);
    }
  }
  concrete->setLoopDomain(new_loop);
  */
}

} // namespace nvfuser
