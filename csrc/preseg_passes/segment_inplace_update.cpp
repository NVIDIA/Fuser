// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/move_split_cat.h>

#include <vector>

#include <fusion.h>
#include <id_model/id_model.h>
#include <ir/builder.h>
#include <ir/interface_nodes.h>
#include <ir/internal_base_nodes.h>
#include <ir/utils.h>
#include <ops/alias.h>
#include <transform_replay.h>
#include <preseg_passes/segment_inplace_update.h>

namespace nvfuser::preseg_passes {
    namespace {
        void insertSegmentSet(
            Fusion* fusion
        ){
            std::vector<TensorView*> aliased_tvs;
            std::vector<BroadcastOp*> bcast;

            for (TensorView* tv : fusion->allTvs()){
                if (fusion->getOutputAlias(tv).aliased_io != nullptr){
                    aliased_tvs.push_back(tv);
                }
            }

            if (aliased_tvs.empty()){
                return;
            } 

            // fusion->exprs() is a topologically sorted list. Filter out the broadcast ops from the list.
            auto all_exprs = fusion->exprs();
            auto all_bcast_ops = ir_utils::filterByType<BroadcastOp>(all_exprs);

            // Traverse and store all direct/indirect consumer tensorviews of these broadcast nodes
            // If the tensorview has been visited, return --> this means that we have already traversed that branch
            std::unordered_set<TensorView*> visited_tvs;
            for (auto bcast_op: all_bcast_ops) {
                std::deque<TensorView*> tvs_to_visit;
                tvs_to_visit.push_back(bcast_op->output(0)->as<TensorView>());
                while (!tvs_to_visit.empty()){
                    TensorView* current_tv = tvs_to_visit.front();
                    tvs_to_visit.pop_front();
                    if (visited_tvs.count(current_tv)){
                        continue;
                    }
                    visited_tvs.insert(current_tv);
                    std::vector<Expr*> current_tv_uses = current_tv->uses();
                    for (Expr* use: current_tv_uses) {
                        for (auto output_tv: ir_utils::filterByType<TensorView>(use->outputs())){
                            tvs_to_visit.push_back(output_tv->as<TensorView>());
                        }
                    }
                }
            }

            // Traverse and store the direct/indirect producer tensorviews of these broadcast nodes
            // If that tensorview has been visited, return.
            for (auto bcast_op: all_bcast_ops) {
                std::deque<TensorView*> tvs_to_visit;
                tvs_to_visit.push_back(bcast_op->input(0)->as<TensorView>());
                while (!tvs_to_visit.empty()){
                    TensorView* current_tv = tvs_to_visit.front();
                    tvs_to_visit.pop_front();
                    if (visited_tvs.count(current_tv)){
                        continue;
                    }
                    visited_tvs.insert(current_tv);
                    auto definition = current_tv->definition();
                    if (definition != nullptr){
                        for (auto input_tv: ir_utils::filterByType<TensorView>(definition->inputs())){
                            tvs_to_visit.push_back(input_tv->as<TensorView>());
                        }
                    }
                }
            }

            for (auto aliased_tv: aliased_tvs){
                if (visited_tvs.count(aliased_tv)){
                    TensorView* alias_seg = segment_set(aliased_tv);
                    TensorView* alias_copy = set(alias_seg);
                    fusion->replaceOutput(aliased_tv, alias_copy);
                }
            }
        }
    } // namespace

    void InsertSegmentSetPass::runPass(Fusion* fusion) {
        insertSegmentSet(fusion);
    }  
} // namespace preseg_passes