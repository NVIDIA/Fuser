// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/analysis/tma.h>
#include <device_lower/lower2device.h>
#include <id_model/id_model.h>
#include <ir/utils.h>
#include <val_graph.h>
#include <val_graph_visitor.h>

#include <list>
#include <unordered_map>
#include <vector>

namespace nvfuser {

namespace {

int64_t getCpAsyncBulkTensorSwizzleSize(TensorView* smem_tv) {
  auto exprs = DependencyCheck::getAllExprsBetween(
      {smem_tv->getLogicalDomain().begin(), smem_tv->getLogicalDomain().end()},
      {smem_tv->getMaybeAllocationDomain().begin(),
       smem_tv->getMaybeAllocationDomain().end()});
  for (auto expr : exprs) {
    if (auto s = dynamic_cast<Swizzle*>(expr)) {
      return s->inX()->extent()->evaluate().as<int64_t>();
    }
  }
  return 1;
}

// Analyze the schedule of the TMA expression, find the ValGroups for each role.
// We first need to infer the TMA domain based on the schedule, which is done by
// finding tile ValGroups first and analyze their definitions.
TMAInfo getTMAInfo(LoadStoreOp* ldst) {
  TensorView* producer_tv = ldst->in()->as<TensorView>();
  TensorView* consumer_tv = ldst->out()->as<TensorView>();
  TensorView *smem_tv = nullptr, *gmem_tv = nullptr;
  if (producer_tv->getMemoryType() == MemoryType::Shared) {
    NVF_ERROR(consumer_tv->getMemoryType() == MemoryType::Global);
    smem_tv = producer_tv;
    gmem_tv = consumer_tv;
  } else {
    NVF_ERROR(producer_tv->getMemoryType() == MemoryType::Global);
    NVF_ERROR(consumer_tv->getMemoryType() == MemoryType::Shared);
    smem_tv = consumer_tv;
    gmem_tv = producer_tv;
  }

  int64_t itemsize = dataTypeSize(gmem_tv->dtype());

  const IdModel& id_model = GpuLower::current()->idModel();
  const ValGraph& exact_graph = id_model.idGraph(IdMappingMode::EXACT);

  auto gmem_alloc_dom = TensorDomain::noBroadcasts(
      TensorDomain::noReductions(gmem_tv->getMaybeAllocationDomain()));
  std::vector<ValGroup> gmem_alloc_groups_vec;
  std::transform(
      gmem_alloc_dom.begin(),
      gmem_alloc_dom.end(),
      std::back_inserter(gmem_alloc_groups_vec),
      [&](IterDomain* id) { return exact_graph.toGroup(id); });
  ValGroups gmem_alloc_groups(gmem_alloc_groups_vec);
  std::unordered_set<ValGroup> gmem_alloc_groups_set;
  gmem_alloc_groups_set.reserve(gmem_alloc_dom.size());
  std::transform(
      gmem_alloc_dom.begin(),
      gmem_alloc_dom.end(),
      std::inserter(gmem_alloc_groups_set, gmem_alloc_groups_set.end()),
      [&](IterDomain* id) { return exact_graph.toGroup(id); });

  // Step 1: Get all bulk ValGroups and tile ValGroups.
  // An ValGroup is considered "bulk" if it contains an IterDomain that has
  // parallel type "Bulk" or all its children are considered "bulk". A "tile"
  // ValGroup is a bulk ValGroup whose parents are not bulk.

  // Get all bulk ValGroups
  std::unordered_set<ValGroup> bulk_groups;
  // Bulk ValGroup that we need to check its definition to see if it is a
  // tile ValGroup.
  std::deque<ValGroup> pending;
  pending.push_back(nullptr); // use nullptr as a checkpoint
  // Start from loop domain, where all the bulk IterDomains in the loop domain
  // must be parallelized as ParallelType::Bulk.
  for (auto id : consumer_tv->getLoopDomain()) {
    if (id->getParallelType() == ParallelType::Bulk) {
      auto g = exact_graph.toGroup(id);
      bulk_groups.insert(g);
      pending.push_back(g);
    }
  }
  // Use a BFS-like (not exactly BFS) algorithm to propagate back to get all
  // bulk ValGroups
  bool updated = true;
  while (true) {
    auto g = pending.front();
    pending.pop_front();
    if (g == nullptr) {
      if (updated) {
        // We discovered new bulk ValGroups in the last round, so we need to
        // continue start a new round to see if we can discover more bulk
        // ValGroups.
        pending.push_back(nullptr);
        updated = false;
        continue;
      } else {
        // We have visited all ValGroups in pending for one round, but nothing
        // has changed. This means that all ValGroups in pending are
        // tile ValGroups, so we can no longer propagate further.
        break;
      }
    }

    auto defs = exact_graph.getDefinitions(g);
    NVF_ERROR(
        gmem_alloc_groups_set.count(g) || !defs.empty(),
        "Allocation domain of the gmem tensor is unreachable");
    if (defs.empty() || gmem_alloc_groups_set.count(g)) {
      pending.push_back(g);
    } else {
      for (const ExprGroup& def : defs) {
        // We only continue propagating if we have not reached the allocation
        // domain of the gmem tensor yet.
        if (bulk_groups.count(exact_graph.inputGroups(def)[0])) {
          // already processed from another path
          continue;
        }
        auto output_groups = exact_graph.outputGroups(def);
        bool should_propagate = std::all_of(
            output_groups.begin(),
            output_groups.end(),
            [&](const ValGroup& out) { return bulk_groups.count(out) > 0; });

        if (should_propagate) {
          updated = true;
          for (const auto& gg : exact_graph.inputGroups(def)) {
            if (bulk_groups.insert(gg).second) {
              pending.push_back(gg);
            }
          }
        } else {
          // Not all outputs of def are bulk ValGroups, this could be because:
          // 1. g is a tile ValGroup
          // 2. g is not a tile ValGroup, we just haven't visited def's other
          //    outputs yet.
          pending.push_back(g);
        }
      }
    }
  }

  // Get tile groups. Use VectorOfUniqueEntries instead of
  // std::unordered_set to make the algorithm deterministic. However, the order
  // here has no meaning, especially, is is not the order specifying which
  // ValGroup is inner and which is outer. The actual order must be determined
  // by propagating from the allocation domain of the gmem tensor.
  VectorOfUniqueEntries<ValGroup> tile_groups;
  for (const auto& g : pending) {
    if (g == nullptr) {
      continue;
    }
    tile_groups.pushBack(g);
  }

  // Step 2: Get the box, partitioned, and stride ValGroups from each tile
  // ValGroup. Similarily, the order of the `tma_groups` has no meaning.
  // So `tma_groups` contains the same set of ValGroups as the TMA domain, but
  // can be in different order. We are using a std::vector<Val*> just to make
  // the algorithm deterministic, not because we care about its order.

  // tma_groups contains ValGroups known to be in the TMA domain. These
  // ValGroups can be a box ValGroup or partitioned ValGroup. If a partitioned
  // ValGroup is in tma_groups, this means that there is a box dimension defined
  // by partitioning. If a box ValGroup is in tma_groups, this means that there
  // is a box dimension defined by compositing.
  std::vector<ValGroup> tma_groups;
  std::unordered_map<ValGroup, ValGroup> tma_g_to_box_g;
  std::unordered_map<ValGroup, ValGroup> tma_g_to_tile_g;
  std::unordered_map<ValGroup, ValGroup> tma_g_to_stride_g;
  std::unordered_map<ValGroup, ValGroup> tma_g_to_partitioned_g;
  for (const auto& tile_g : tile_groups) {
    const auto& defs = exact_graph.getDefinitions(tile_g);
    NVF_ERROR(
        defs.size() <= 1,
        "Having multiple definitions of tile group is not supported");
    ExprGroup striding_split = nullptr;
    if (!defs.empty() && exact_graph.outputGroups(defs.front())[0] == tile_g &&
        defs.front()->front()->isA<Split>()) {
      striding_split = defs.front();
    }
    ValGroup box_g =
        (striding_split != nullptr ? exact_graph.inputGroups(striding_split)[0]
                                   : tile_g);
    ValGroup stride_g =
        (striding_split != nullptr ? exact_graph.outputGroups(striding_split)[1]
                                   : nullptr);
    const ExprGroups& defs2 = exact_graph.getDefinitions(box_g);
    NVF_ERROR(
        defs2.size() <= 1,
        "Having multiple definitions of box group is not supported");
    ExprGroup boxing_split = nullptr;
    if (!defs2.empty() && defs2.front()->front()->isA<Split>()) {
      boxing_split = defs2.front();
    }
    ValGroup partitioned_g =
        (boxing_split != nullptr ? exact_graph.inputGroups(boxing_split)[0]
                                 : nullptr);
    ValGroup tma_g = (partitioned_g != nullptr ? partitioned_g : box_g);

    tma_groups.push_back(tma_g);
    tma_g_to_box_g[tma_g] = box_g;
    if (stride_g != nullptr) {
      tma_g_to_tile_g[tma_g] = tile_g;
      tma_g_to_stride_g[tma_g] = stride_g;
    }
    if (partitioned_g != nullptr) {
      tma_g_to_partitioned_g[tma_g] = partitioned_g;
    }
  }

  // Stpe 3: Propagate from the gmen tensor's allocation domain to the TMA
  // domain, compute the order, contiguity, and stride of partitioned
  // ValGroups. Note that this order is meaningful, and it is the order that
  // defines which is inner and which is outer. The strides are also meaningful,
  // and they are the `globalStrides` of the `cuTensorMapEncodeTiled`. After
  // propagation, `frontier` will be the TMA domain

  std::list<std::tuple<ValGroup, /*contiguity*/ bool, /*stride*/ Val*>>
      frontier;
  // Initialize frontier as the allocation domain
  auto metadata = IrBuilder::metadataExpr(gmem_tv);
  auto alloc_strides = IrBuilder::getAttrExpr(metadata, "alloc_stride");
  for (auto i : c10::irange((int64_t)gmem_alloc_dom.size())) {
    auto id = gmem_alloc_dom.at(i);
    // TODO: should I use i below, or should I instead use the position of id in
    // the allocation domain with broadcast? I don't remember the detail, but
    // I will just use i for now and leave the support for broadcast for future.
    auto stride = IrBuilder::getItemExpr(alloc_strides, i);
    frontier.emplace_back(
        exact_graph.toGroup(id),
        gmem_tv->getContiguity().at(i).value(),
        stride);
  }
  // Propagate forward from the gmem allocation domain to TMA ValGroups
  for (auto [expr, direction] : ValGraphBFS::getExprsBetween(
           exact_graph, gmem_alloc_groups, tma_groups)) {
    NVF_ERROR(!expr->empty());
    NVF_ERROR(
        direction == Direction::Forward,
        "Backward propagation from allocation domain to TMA domain is not supported yet.");
    if (expr->front()->isA<Split>()) {
      Split* split = expr->front()->as<Split>();
      auto in = exact_graph.inputGroups(expr)[0];
      auto in_it =
          std::find_if(frontier.begin(), frontier.end(), [in](auto tuple) {
            return std::get<0>(tuple) == in;
          });
      NVF_ERROR(
          in_it != frontier.end(),
          "The TMA domain must be equivalent to the allocation domain of the gmem tensor, but ",
          in->toString(),
          " is not on the path.");
      Val* is_divisible = SimplifyingIrBuilder::eqExpr(
          SimplifyingIrBuilder::modExpr(
              in->front()->as<IterDomain>()->extent(), split->factor()),
          gmem_tv->fusion()->zeroVal());
      GpuLower::current()->validate(
          is_divisible,
          "Invalid view in TMA: the extent of ",
          in,
          " must be divisible by ",
          split->factor());
      frontier.insert(
          in_it,
          std::make_tuple(
              exact_graph.outputGroups(expr)[0],
              true,
              SimplifyingIrBuilder::mulExpr(
                  std::get<2>(*in_it), split->factor())));
      std::get<0>(*in_it) = exact_graph.outputGroups(expr)[1];
    } else if (expr->front()->isA<Merge>()) {
      auto outer = exact_graph.inputGroups(expr)[0];
      auto outer_it =
          std::find_if(frontier.begin(), frontier.end(), [outer](auto tuple) {
            return std::get<0>(tuple) == outer;
          });
      NVF_ERROR(
          outer_it != frontier.end(),
          "The TMA domain must be equivalent to the allocation domain of the gmem tensor, but ",
          outer->toString(),
          " is not on the path.");
      auto inner = exact_graph.inputGroups(expr)[1];
      auto inner_it = std::next(outer_it);
      NVF_ERROR(
          inner_it != frontier.end(),
          "The TMA domain must be equivalent to the allocation domain, but ",
          inner->toString(),
          " is not on the path.");
      NVF_ERROR(
          std::get<0>(*inner_it) == inner && std::get<1>(*outer_it),
          "Can not merge discontiguous dimensions, but ",
          outer->toString(),
          " is merged with ",
          inner->toString());
      std::get<0>(*inner_it) = exact_graph.outputGroups(expr)[0];
      frontier.erase(outer_it);
    } else {
      NVF_ERROR(
          false,
          "Unsupported expression between the allocation domain and TMA domain",
          expr->toString());
    }
  }

  // Frontier is now the TMA domain
  const auto& tma_domain = frontier;

  NVF_ERROR(
      std::get<1>(tma_domain.back()),
      "The innermost dimension of the TMA domain must be contiguous");
  NVF_ERROR(
      tma_g_to_stride_g.count(std::get<0>(tma_domain.back())) == 0,
      "When interleave is CU_TENSOR_MAP_INTERLEAVE_NONE ",
      "(this is always the case for nvFuser now)",
      ", the first element of elementStrides must be one.");

  // Validate that tma_domain is a superset of tma_groups, otherwise there is
  // something wrong in the schedule.
  {
    std::unordered_set<ValGroup> seen;
    std::unordered_set<ValGroup> pending_tma_groups(
        tma_groups.begin(), tma_groups.end());
    for (auto tuple : tma_domain) {
      auto g = std::get<0>(tuple);
      NVF_ERROR(
          seen.insert(g).second,
          "Mistake in schedule. Duplicate ValGroup found: ",
          g->toString());
      pending_tma_groups.erase(g);
    }
    NVF_ERROR(
        pending_tma_groups.empty(),
        "Can not infer TMA domain from the schedule. The ValGroup ",
        ir_utils::toString(pending_tma_groups),
        " are expected to be in the TMA domain.");
  }

  // Step 4: Handle "defining box by compositing"

  // So far, we have infered the TMA domain. The size of TMA domain is not
  // necessarily the dimensionality of TMA because we support defining box
  // by compositing. We use a state machine to infer the dimensions of TMA.
  //
  // There can only be four types of ValGroups in the TMA domain:
  // -  P: partitioned ValGroup
  // -  C: coordinate ValGroup
  // - SB: strided box ValGroup
  // - CB: contiguous box ValGroup
  //
  // For the example of the Figure 6 in doc/dev/tma.md, the TMA domain is
  // [I1, I2, I3, I4, I5, I6, I7, I8, I9], and the types of these IDs are
  // [ C, CB,  P,  C, CB, CB,  C, CB, CB]
  //
  // The algorithm works as follows: We run a 3-state machine. The state machine
  // is initialized as START. After setting the initial state, we loop through
  // the TMA domain from inner to outer. During the loop, for each ValGroup we
  // see, we take an action and change the state of the machine. The action and
  // target state depend on the current state of the machine, and the type and
  // contiguity of the ValGroup we encounter. The actions and transition of
  // states are shown in the following diagram:
  //
  //                           P: create new dim
  //                            .-------------.
  //                            |             |
  //                            '-- [START] <-'
  //                      CB:     / ^  P:  ^ \     SB/C:
  //                    create   / / create \ \   create
  //                     new    / /  new dim \ \   new
  //                     dim   / /            \ \  dim
  //                          v /              \ v
  //              .--- [PENDING BOX] -----> [PENDING COORD] <--.
  //              |           ^ ^     SB/C:     | |            |
  //              '-----------' |    create     | '------------'
  //       CB: create new       |   new dim if  |       SB/C: create new
  // dim if discontiguous       | discontiguous |       dim if discontiguous
  // otherwise merge with       |     or SB     |       or SB, otherwise merge
  //            prev dim        |               |       with prev dim
  //                            '---------------'
  //                           CB: create new dim
  //
  // There are three states in the machine. The meaning of these states are:
  // - START: Everything clean, nothing pending merge.
  // - PENDING BOX: Is there another contiguous box ID? I can merge it into the
  //                current box.
  // - PENDING COORD: Is there another coordinate ID? I can merge it into the
  //                  current dimension.

  // As required by the hardware, tensors used by TMA must be in column major
  std::vector<TMADim> dims;
  enum { START, PENDING_BOX, PENDING_COORD } state = START;
  for (auto it = tma_domain.rbegin(); it != tma_domain.rend(); it++) {
    auto [g, contiguous, stride] = *it;
    auto partitioned_g_it = tma_g_to_partitioned_g.find(g);
    auto box_g_it = tma_g_to_box_g.find(g);
    auto stride_g_it = tma_g_to_stride_g.find(g);
    auto tile_g_it = tma_g_to_tile_g.find(g);
    enum IDType { P, C, SB, CB };
    IDType type =
        (partitioned_g_it != tma_g_to_partitioned_g.end()
             ? P
             : (box_g_it == tma_g_to_box_g.end()
                    ? C
                    : (stride_g_it != tma_g_to_stride_g.end() ? SB : CB)));
    bool should_create_new_dim =
        !(contiguous &&
          ((state == PENDING_BOX && (type == CB || type == C)) ||
           (state == PENDING_COORD && type == C)));

    if (should_create_new_dim) {
      dims.emplace_back();
      dims.back().gmem_stride_bytes =
          SimplifyingIrBuilder::mulExpr(stride, itemsize);
      if (type == CB) {
        dims.back().box = std::unique_ptr<Box>(new ContiguousBox{});
      } else if (type == SB) {
        dims.back().box = std::unique_ptr<Box>(new StridedBox(
            box_g_it->second, tile_g_it->second, stride_g_it->second));
      } else if (type == P) {
        if (stride_g_it != tma_g_to_stride_g.end()) {
          dims.back().box = std::unique_ptr<Box>(new StridedBox(
              box_g_it->second, tile_g_it->second, stride_g_it->second));
        } else {
          dims.back().box =
              std::unique_ptr<Box>(new ContiguousBox(box_g_it->second));
        }
      } else {
        NVF_ERROR(type == C);
        dims.back().box =
            std::unique_ptr<Box>(new ImplicitSizeOneBox(gmem_tv->fusion()));
      }
    }
    dims.back().partitioned.pushBack(g);
    if (type == C) {
      dims.back().coordinate.pushBack(g);
    } else if (type == CB) {
      ContiguousBox* box = dynamic_cast<ContiguousBox*>(dims.back().box.get());
      box->box_tile.pushBack(g);
    }

    state = (type == P ? START : (type == CB ? PENDING_BOX : PENDING_COORD));
  }
  return TMAInfo(
      std::move(dims),
      getSwizzleFromBytes(
          getCpAsyncBulkTensorSwizzleSize(smem_tv) * core_matrix_width_bytes),
      gmem_tv);
}

} // namespace

Val* TMAInfo::tensorMap() const {
  std::vector<Val*> tensor_sizes_inner_to_outer;
  std::transform(
      dims_.begin(),
      dims_.end(),
      std::back_inserter(tensor_sizes_inner_to_outer),
      [](const TMADim& d) { return d.tensorSize(); });

  std::vector<Val*> tensor_strides_inner_to_outer;
  std::transform(
      dims_.begin() + 1,
      dims_.end(),
      std::back_inserter(tensor_strides_inner_to_outer),
      [](const TMADim& d) { return d.gmem_stride_bytes; });

  std::vector<Val*> box_sizes_inner_to_outer;
  std::transform(
      dims_.begin(),
      dims_.end(),
      std::back_inserter(box_sizes_inner_to_outer),
      [](const TMADim& d) { return d.boxSize(); });

  std::vector<Val*> element_strides_inner_to_outer;
  std::transform(
      dims_.begin(),
      dims_.end(),
      std::back_inserter(element_strides_inner_to_outer),
      [](const TMADim& d) { return d.elementStride(); });

  int64_t dim = (int64_t)tensor_sizes_inner_to_outer.size();
  auto metadata = IrBuilder::metadataExpr(gmem_tv_);
  auto global_address = IrBuilder::getAttrExpr(metadata, "data");

  Val* global_stride =
      (dim > 1
           ? IrBuilder::arrayExpr(tensor_strides_inner_to_outer)
           : IrBuilder::create<Val>(
                 std::vector<int64_t>{},
                 ArrayType{std::make_shared<DataType>(DataType::Index), 0}));

  return tma::encodeTensorMapTiled(
      gmem_tv_->dtype(),
      global_address,
      IrBuilder::arrayExpr(tensor_sizes_inner_to_outer),
      global_stride,
      IrBuilder::arrayExpr(box_sizes_inner_to_outer),
      IrBuilder::arrayExpr(element_strides_inner_to_outer),
      tma::TensorMapInterleave::NoInterleave,
      swizzle_,
      tma::TensorMapL2Promotion::NoL2Promotion,
      tma::TensorMapFloatOOBFill::NoOOBFill);
}

std::unordered_map<TensorView*, const TMAInfo> getConsumerToTMAInfoMap(
    Fusion* fusion) {
  std::unordered_map<TensorView*, const TMAInfo> result;
  for (Expr* expr : fusion->exprs()) {
    if (auto ldst = dynamic_cast<LoadStoreOp*>(expr);
        ldst && ldst->opType() == LoadStoreOpType::CpAsyncBulkTensorTile) {
      NVF_ERROR(
          result.emplace(ir_utils::getTvOutput(ldst), getTMAInfo(ldst)).second,
          "Ambiguous TMA information, likely something is wrong in the Fusion IR");
    }
  }
  return result;
}

} // namespace nvfuser
