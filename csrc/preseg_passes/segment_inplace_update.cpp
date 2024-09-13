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
        void findBroadcast(
            Fusion* fusion
        ){
            std::vector<TensorView*> aliased_tvs;
            std::vector<BroadcastOp*> bcast;

            for (TensorView* aliased_tv : fusion->allTvs()){
                if (fusion->getOutputAlias(aliased_tv).aliased_io != nullptr){
                    aliased_tvs.push_back(aliased_tv);
                    debug() << aliased_tv->toString() << std::endl;

                    TensorView* tv = aliased_tv;
                    while (auto expr = dynamic_cast<Expr*>(tv->definition())){
                        if (expr->isA<BroadcastOp>()){
                            bcast.push_back(expr->as<BroadcastOp>());
                            debug() << expr->toString() << std::endl;
                        }
                        tv = tv->definition()->inputs().front()->as<TensorView>();
                    }
                    tv = aliased_tv;
                    for (auto expr: tv->uses()){
                        if (expr->isA<BroadcastOp>()){
                            bcast.push_back(expr->as<BroadcastOp>());
                            debug() << expr->toString() << std::endl;
                        }
                        tv = expr->outputs().front()->as<TensorView>();
                    }
                }
            }
            debug() << aliased_tvs.front()->toString() << std::endl;
            debug() << bcast.front()->toString() << std::endl;
            for (auto expr: bcast){
                TensorView* bcast_output = expr->outputs().front()->as<TensorView>();
                TensorView* copy_output = segment_set(bcast_output);
                ir_utils::replaceValInAllExprInputsAndFusionOutputs(bcast_output, copy_output);
            }
        }
    } // namespace 
    
    void InsertSegmentSetPass::runPass(Fusion* fusion) {
        findBroadcast(fusion);
    }  
} // namespace preseg_passes
