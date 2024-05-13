// clang-format off
/*
* SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
* All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*/
// clang-format on
#include <gtest/gtest.h>

#include <codegen.h>
#include <device_lower/lower2device.h>
#include <disjoint_set.h>
#include <executor.h>
#include <executor_params.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <fusion_segmenter.h>
#include <ir/all_nodes.h>
#include <ir/graphviz.h>
#include <ir/iostream.h>
#include <ir/printer.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <kernel_cache.h>
#include <kernel_ir.h>
#include <mma_type.h>
#include <ops/all_ops.h>
#include <root_domain_map.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/utils.h>
#include <tests/cpp/multidevice.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <transform_replay.h>
#include <transform_rfactor.h>

#include <algorithm>
#include <iostream>

#include <host_ir/container.h>
#include <host_ir/executor.h>

namespace nvfuser {


namespace hir {

using MultiDeviceHostIrTestParams = std::tuple<bool>;

class MultiDeviceHostIrTest:
public MultiDeviceTest,
public testing::WithParamInterface<MultiDeviceHostIrTestParams> {};


TEST_P(MultiDeviceHostIrTest, SingleFusionSingleComm) {
    auto [use_fusion_executor_cache] = GetParam();

    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());

    std::vector<int64_t> input_sizes = {2, 8, 32};

    auto tv0 = makeConcreteTensor(input_sizes);
    auto tv1 = add(tv0, tv0);
    fusion->addInput(tv0);
    fusion->addOutput(tv1);

    DeviceMesh mesh({0,1});
    for (auto tv: {tv0,tv1}){
        tv->setDeviceMesh(mesh);
        tv->axis(0)->parallelize(ParallelType::DIDx);
    }

    auto hic = std::make_unique<HostIrContainer>();
    FusionGuard::setCurFusion(hic.get());
    auto hu = IrBuilder::create<HostUnit>(static_cast<IrContainer*>(hic.get()), std::move(fusion));

    IrCloner ir_cloner(hic.get());

    std::vector<Val*> post_inputs = {ir_cloner.clone(hu->fusion_to_execute()->inputs().back())};
    std::vector<Val*> post_outputs = {ir_cloner.clone(hu->fusion_to_execute()->outputs().back())};
    auto post_compute = IrBuilder::create<PostOnStream>(static_cast<IrContainer*>(hic.get()), hu, std::move(post_inputs), std::move(post_outputs));

    hic->pushBackTopLevelExprs(post_compute);

    std::vector<Val*> comm_inputs = post_outputs;
    auto tv2 = set(comm_inputs.back()->as<TensorView>());
    tv2->axis(0)->parallelize(ParallelType::Serial);
    std::vector<Val*> comm_outputs = {tv2};
    auto post_comm = IrBuilder::create<PostOnStream>(static_cast<IrContainer*>(hic.get()), hu, std::move(post_inputs), std::move(post_outputs));

    hic->pushBackTopLevelExprs(post_comm);

    // add global IO to the HostIrContainer. This step could potentially be infered automatically
    hic->addInput(post_compute->inputs().back());
    hic->addOutput(post_comm->outputs().back());
    hic->print(debug());

    HostIrExecutorParams params;
    params.use_fusion_executor_cache = use_fusion_executor_cache;
    HostIrExecutor hie(std::move(hic), std::move(params));

    auto options = at::TensorOptions().device(communicator->device());
    at::Tensor unsharded_input = at::randn(input_sizes, options);
    c10::IValue input = shardTensor(unsharded_input, tv0, communicator->deviceId());
    auto ref_output = unsharded_input * 2;

    auto outputs = hie.runWithInput({input});

    GTEST_EXPECT_TRUE(torch::allclose(ref_output, outputs.back()));
}

INSTANTIATE_TEST_SUITE_P(
    Manual,
    MultiDeviceHostIrTest,
    testing::Combine(testing::Bool()));

} // namespace hir

} // namespace nvfuser
