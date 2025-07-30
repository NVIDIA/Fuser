// clang-format off
/*
* SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
* All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*/
// clang-format on
#include <gmock/gmock-more-matchers.h>

#include <fusion.h>
#include <global_allocator.h>
#include <host_ir/container.h>
#include <host_ir/jit.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>
#include <random>

namespace nvfuser {

namespace hir {

using HostIrJitTest = NVFuserTest;
// Build with: python setup.py install --build-with-host-ir-jit
TEST_F(HostIrJitTest, PointwiseScheduler) {
  auto fusion_ptr = std::make_unique<Fusion>();
  {
    // defining simple fusion
    auto fusion = fusion_ptr.get();
    FusionGuard fg(fusion);

    TensorView* tv0 =
        TensorViewBuilder().ndims(2).contiguity({true, true}).build();
    fusion->addInput(tv0);
    auto tv1 = add(tv0, IrBuilder::create<Val>(1.0, DataType::Float));
    fusion->addOutput(tv1);
  }

  {
    Fusion fusion_clone = *fusion_ptr;
    Fusion* fusion = &fusion_clone;
    FusionGuard fg(fusion);
    // create inputs
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::empty_strided({1024, 128}, {128, 1}, options);

    KernelArgumentHolder runtime_inputs({t0});
    SchedulerType scheduler_type = SchedulerType::PointWise;

    std::cout << "before scheduling fusion" << std::endl;
    fusion->printMath();
    fusion->printTransforms();

    // scheduling fusion using pointwise scheduler
    auto heuristic_params = SchedulerEntry::scheduleWith(
        fusion, scheduler_type, runtime_inputs, /*validate_scheduler=*/true);

    std::cout << "after scheduling fusion" << std::endl;
    fusion->printMath();
    fusion->printTransforms();

    // print heuristic params
    std::cout << heuristic_params->toString() << std::endl;

    // compile and run
    auto ke = std::make_unique<KernelExecutor>();
    ke->compile(fusion, runtime_inputs, heuristic_params->lparams);
    auto cg_outputs = ke->run(runtime_inputs, {}, heuristic_params->lparams);

    testValidate(fusion, cg_outputs, {t0}, __LINE__, __FILE__);
  }

  {
    Fusion fusion_clone = *fusion_ptr;
    Fusion* fusion = &fusion_clone;
    FusionGuard fg(fusion);
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    // a different problem size
    at::Tensor t0 = at::empty_strided({2048, 128}, {128, 1}, options);

    KernelArgumentHolder runtime_inputs({t0});
    SchedulerType scheduler_type = SchedulerType::PointWise;
    // note: we probably would want to use the get heuristic function
    // instead of running schedule again.
    auto heuristic_params = SchedulerEntry::scheduleWith(
        fusion, scheduler_type, runtime_inputs, /*validate_scheduler=*/true);

    // print heuristic params
    std::cout << heuristic_params->toString() << std::endl;

    auto ke = std::make_unique<KernelExecutor>();
    ke->compile(fusion, runtime_inputs, heuristic_params->lparams);
    auto cg_outputs = ke->run(runtime_inputs, {}, heuristic_params->lparams);
    testValidate(fusion, cg_outputs, {t0}, __LINE__, __FILE__);
  }
}

} // namespace hir

} // namespace nvfuser
