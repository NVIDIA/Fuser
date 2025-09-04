// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>
#include <type.h>

namespace nvfuser {

using WelfordTest = NVFuserTest;

TEST_F(WelfordTest, SerialWelford) {
  int x = 128, y = 64, z = 64;

  std::string kernel = R"(
__global__ void kernel1(
    Tensor<float,3> inp,
    Tensor<float,1> out_var,
    Tensor<float,1> out_avg
){
    for(int i0=0;i0<inp.logical_size[0];i0++){
        float tmp_M2=0;
        float tmp_avg=0;
        long tmp_N=0;
        for(int i1=0;i1<inp.logical_size[1];i1++){
            for(int i2=0;i2<inp.logical_size[2];i2++){
                welfordCombine(
                    tmp_avg,
                    tmp_M2,
                    tmp_N,
                    inp[i0*inp.alloc_stride[0]+
                        i1*inp.alloc_stride[1]+
                        i2*inp.alloc_stride[2]],
                    0.f,
                    (long)1
                );
            }
        }
        out_var[i0*out_var.alloc_stride[0]]=
            tmp_M2/(tmp_N);
        out_avg[i0*out_avg.alloc_stride[0]]=
            tmp_avg;
    }
}
    )";
  RtcKernel rk;
  rk.compile(kernel, "kernel1", false, PrimDataType::Int);
  LaunchParams lp(
      1, // gdimx
      1, // gdimy
      1, // gdimz
      1, // bdimx
      1, // bdimy
      1 // bdimz
  );
  lp.setSmem(0);
  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const std::vector<int64_t> tensor_dims = {x, y, z};
  auto in0 = at::randn(tensor_dims, options);
  auto out_var = at::empty({x}, options);
  auto out_avg = at::empty({x}, options);
  rk.run(lp, {in0, out_var, out_avg}, PrimDataType::Int);

  NVF_CHECK(in0.var({1, 2}, false).allclose(out_var));
  NVF_CHECK(in0.mean({1, 2}).allclose(out_avg, /*rtol*/ 1e-5, /*atol*/ 1e-6));
}

TEST_F(WelfordTest, BlockWelford) {
  int x = 7, y = 8, z = 9;

  std::string kernel = R"(
__global__ void kernel1(
    Tensor<float,2> inp,
    Tensor<float,1> out_avg,
    Tensor<float,1> out_var,
    Tensor<float,1> init_avg,
    Tensor<float,1> init_var,
    Tensor<long,0> init_N
){
    //actual generated kernel will use dynamic shared mem,
    // here is just for prototype
    __shared__ float mem_avg[512];
    __shared__ float mem_M2[512];
    __shared__ long mem_N[512];
    float in=inp[threadIdx.x*inp.alloc_stride[0]+
                        threadIdx.y*inp.alloc_stride[1]];
    float tmp_avg=0;
    float tmp_M2=0;
    long tmp_N=0;
    blockWelford<false,true,false, true>(
        tmp_avg,
        tmp_M2,
        tmp_N,
        in,
        0.f,
        (long)1,
        (float*)mem_avg,
        (float*)mem_M2,
        (long*)mem_N,
        (bool)(threadIdx.x<inp.logical_size[0]),
        0.f,
        blockDim);
    __syncthreads();
    if(threadIdx.x<out_var.logical_size[0] && threadIdx.y==0){
        welfordCombine(
                    tmp_avg,
                    tmp_M2,
                    tmp_N,
                    init_avg[threadIdx.x*init_avg.alloc_stride[0]],
                    init_var[threadIdx.x*init_var.alloc_stride[0]]*init_N[0],
                    init_N[0]
                );
        out_avg[threadIdx.x*out_avg.alloc_stride[0]]=tmp_avg;
        out_var[threadIdx.x*out_var.alloc_stride[0]]=tmp_M2/(tmp_N);
    }
}
    )";
  RtcKernel rk;
  rk.compile(kernel, "kernel1", false, PrimDataType::Int);
  LaunchParams lp(
      1, // gdimx
      1, // gdimy
      1, // gdimz
      x, // bdimx
      y, // bdimy
      1 // bdimz
  );
  lp.setSmem(0);
  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const std::vector<int64_t> tensor_dims = {x, y};
  const std::vector<int64_t> init_dims = {x, z};

  // generate initial values
  auto init_in = at::randn(init_dims, options);
  auto init_var = init_in.var({1}, false);
  auto init_avg = init_in.mean({1});
  auto init_N =
      at::tensor(z, at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));

  auto in0 = at::randn(tensor_dims, options);

  // run kernel
  auto out_var = at::zeros({x}, options);
  auto out_avg = at::zeros({x}, options);
  rk.run(
      lp,
      {in0, out_avg, out_var, init_avg, init_var, init_N},
      PrimDataType::Int);

  // compare with reference output
  auto cat_tensor = at::cat({init_in, in0}, 1);
  NVF_CHECK(cat_tensor.var({1}, false).allclose(out_var));
  NVF_CHECK(
      cat_tensor.mean({1}).allclose(out_avg, /*rtol*/ 1e-5, /*atol*/ 1e-6));
}

TEST_F(WelfordTest, BlockWelfordNoInit) {
  int x = 7, y = 8, z = 9;

  // need support IValue for integer input as initial count
  std::string kernel = R"(
__global__ void kernel1(
    Tensor<float,3> inp,
    Tensor<float,1> out_avg,
    Tensor<float,1> out_var
){
    //actual generated kernel will use dynamic shared mem,
    // here is just for prototype
    __shared__ float mem_avg[512];
    __shared__ float mem_M2[512];
    __shared__ long mem_N[512];
    float in=inp[threadIdx.x*inp.alloc_stride[0]+
                        threadIdx.y*inp.alloc_stride[1]+
                        threadIdx.z*inp.alloc_stride[2]];
    float tmp_avg=0;
    float tmp_M2=0;
    long tmp_N=0;
    block_sync::init();
    blockWelford<false,true,true, true>(
        tmp_avg,
        tmp_M2,
        tmp_N,
        in,
        0.f,
        (long) 1,
        (float*)mem_avg,
        (float*)mem_M2,
        (long*)mem_N,
        (bool)(threadIdx.x<inp.logical_size[0]),
        0.f,
        blockDim);
    __syncthreads();
    if(threadIdx.x<out_var.logical_size[0] && threadIdx.y==0 && threadIdx.z==0){
        out_avg[threadIdx.x*out_var.alloc_stride[0]]=tmp_avg;
        out_var[threadIdx.x*out_var.alloc_stride[0]]=tmp_M2/(tmp_N);
    }
}
    )";
  RtcKernel rk;
  rk.compile(kernel, "kernel1", false, PrimDataType::Int);
  LaunchParams lp(
      1, // gdimx
      1, // gdimy
      1, // gdimz
      x, // bdimx
      y, // bdimy
      z // bdimz
  );
  lp.setSmem(0);
  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const std::vector<int64_t> tensor_dims = {x, y, z};
  auto in0 = at::randn(tensor_dims, options);
  auto out_var = at::empty({x}, options);
  auto out_avg = at::empty({x}, options);
  rk.run(lp, {in0, out_avg, out_var}, PrimDataType::Int);

  NVF_CHECK(in0.var({1, 2}, false).allclose(out_var));
  NVF_CHECK(in0.mean({1, 2}).allclose(out_avg, /*rtol*/ 1e-5, /*atol*/ 1e-6));
}

TEST_F(WelfordTest, GridWelfordNoInit) {
  KernelExecutor ke;
  int x = 128, y = 64, z = 128;

  std::string kernel = R"(
__global__ void kernel1(
    Tensor<float,3> inp,
    Tensor<float,1> out_avg,
    Tensor<float,1> out_var,
    Tensor<float,1> work_buf_avg,
    Tensor<float,1> work_buf_M2,
    Tensor<long,1> work_buf_N,
    Tensor<int64_t,1> sync_flag
){
    __shared__ float shared_buf_avg[512];
    __shared__ float shared_buf_M2[512];
    __shared__ long shared_buf_N[512];
    float tmp_avg=0;
    float tmp_M2=0;
    long tmp_N=0;
    float in = inp[ blockIdx.x  * inp.alloc_stride[0]+
                    blockIdx.y  * inp.alloc_stride[1]+
                    threadIdx.x * inp.alloc_stride[2]];
    block_sync::init();
    welford::gridWelford<
        true,true,false,
        true,false,false,
        false, true
    >(
        tmp_avg,
        tmp_M2,
        tmp_N,
        in,
        0.f,
        (long) 1,
        &work_buf_avg[0],
        &work_buf_M2[0],
        &work_buf_N[0],
        sync_flag,
        (float*)shared_buf_avg,
        (float*)shared_buf_M2,
        (long*)shared_buf_N,
        threadIdx.x<out_var.logical_size[0],
        threadIdx.x<out_var.logical_size[0],
        0.f,
        0,
        1,
        blockDim);
    if(blockIdx.x == gridDim.x - 1 && blockIdx.y == gridDim.y - 1){
        out_avg[threadIdx.x*out_avg.alloc_stride[0]]=tmp_avg;
        out_var[threadIdx.x*out_var.alloc_stride[0]]=tmp_M2/tmp_N;
    }
}
    )";
  RtcKernel rk;
  rk.compile(kernel, "kernel1", false, PrimDataType::Int);
  LaunchParams lp(
      x, // gdimx
      y, // gdimy
      1, // gdimz
      z, // bdimx
      1, // bdimy
      1 // bdimz
  );
  lp.setSmem(0);
  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const auto options_int =
      at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);

  const std::vector<int64_t> tensor_dims = {x, y, z};
  auto in0 = at::randn(tensor_dims, options);

  auto out_avg = at::empty({z}, options);
  auto out_var = at::empty({z}, options);
  auto work_buf_avg = at::empty({x * y * z}, options);
  auto work_buf_var = at::empty({x * y * z}, options);
  auto work_buf_N = at::empty({x * y * z}, options_int);
  auto sync_flag = at::zeros({1}, options_int);
  rk.run(
      lp,
      {in0,
       out_avg,
       out_var,
       work_buf_avg,
       work_buf_var,
       work_buf_N,
       sync_flag},
      PrimDataType::Int);
  std::vector<int64_t> dims{0, 1};

  NVF_CHECK(in0.mean(dims).allclose(out_avg, /*rtol*/ 1e-5, /*atol*/ 1e-6));
  NVF_CHECK(in0.var(dims, false).allclose(out_var));
}

TEST_F(WelfordTest, WelfordOp) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int M = 64, N = 128;

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = mul(tv0, IrBuilder::create<Val>(1.0));
  auto tvs = Welford(tv1, {1});
  auto tv_avg = tvs.avg;
  auto tv_M2 = tvs.var_sum;
  auto tv_N = tvs.n;
  fusion.addOutput(tv_avg);
  fusion.addOutput(tv_M2);
  fusion.addOutput(tv_N);

  tv_avg->split(1, 32);
  tv_avg->split(0, 32);
  tv_avg->split(0, 4);
  tv_avg->reorder({{-1, -3}, {-3, -1}});
  tv1->computeAt(tv_avg, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_int = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({M, N}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  // by default Welford outputs sum of square diff so need to divide to get var
  outputs[1] = outputs[1].as<at::Tensor>() / N;

  testValidate(
      ke.compiledKernel()->kernel(),
      outputs,
      {t0},
      {t0.mean({1}), t0.var({1}, false), at::ones({M}, options_int) * N},
      __LINE__,
      __FILE__);
}

TEST_F(WelfordTest, BlockWelfordOp) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int M = 64, N = 128;

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = mul(tv0, IrBuilder::create<Val>(1.0));
  auto tvs = Welford(tv1, {1});
  auto tv_avg = tvs.avg;
  auto tv_M2 = tvs.var_sum;
  auto tv_N = tvs.n;
  fusion.addOutput(tv_avg);
  fusion.addOutput(tv_M2);
  fusion.addOutput(tv_N);

  tv_avg->axis(-1)->parallelize(ParallelType::TIDx);

  tv1->computeAt(tv_avg, -1);

  //
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_int = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({M, N}, options);
  at::Tensor t_var = at::empty({M}, options);
  at::Tensor t_avg = at::empty({M}, options);
  at::Tensor t_N = at::empty({M}, options_int);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  // by default Welford outputs sum of square diff so need to divide to get var
  outputs[1] = outputs[1].as<at::Tensor>() / N;

  testValidate(
      ke.compiledKernel()->kernel(),
      outputs,
      {t0},
      {t0.mean({1}), t0.var({1}, false), at::ones({M}, options_int) * N},
      __LINE__,
      __FILE__);
}

TEST_F(WelfordTest, GridWelfordOp) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int M = 64, N = 128;

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = mul(tv0, IrBuilder::create<Val>(1.0));
  auto tvs = Welford(tv1, {1});
  auto tv_avg = tvs.avg;
  auto tv_M2 = tvs.var_sum;
  auto tv_N = tvs.n;
  fusion.addOutput(tv_avg);
  fusion.addOutput(tv_M2);
  fusion.addOutput(tv_N);

  tv_avg->axis(0)->parallelize(ParallelType::TIDx);
  tv_avg->axis(-1)->parallelize(ParallelType::BIDx);

  tv1->computeAt(tv_avg, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_int = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({M, N}, options);
  at::Tensor t_avg = at::empty({M}, options);
  at::Tensor t_var = at::empty({M}, options);
  at::Tensor t_N = at::empty({M}, options_int);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  // by default Welford outputs sum of square diff so need to divide to get var
  outputs[1] = outputs[1].as<at::Tensor>() / N;

  testValidate(
      ke.compiledKernel()->kernel(),
      outputs,
      {t0},
      {t0.mean({1}), t0.var({1}, false), at::ones({M}, options_int) * N},
      __LINE__,
      __FILE__);
}

TEST_F(WelfordTest, RfactorWelfordOp) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int M = 64, N = 128;

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = mul(tv0, IrBuilder::create<Val>(1.0));
  auto tvs = Welford(tv1, {1});
  auto tv_avg = tvs.avg;
  auto tv_M2 = tvs.var_sum;
  auto tv_N = tvs.n;
  fusion.addOutput(tv_avg);
  fusion.addOutput(tv_M2);
  fusion.addOutput(tv_N);

  tv_avg->split(1, 4);
  ir_utils::rFactorHelper(tvs.avg, {2});
  tv1->computeAt(tv_avg, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_int = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({M, N}, options);
  at::Tensor t_avg = at::empty({M}, options);
  at::Tensor t_var = at::empty({M}, options);
  at::Tensor t_N = at::empty({M}, options_int);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  // by default Welford outputs sum of square diff so need to divide to get var
  outputs[1] = outputs[1].as<at::Tensor>() / N;

  testValidate(
      ke.compiledKernel()->kernel(),
      outputs,
      {t0},
      {t0.mean({1}), t0.var({1}, false), at::ones({M}, options_int) * N},
      __LINE__,
      __FILE__);
}

TEST_F(WelfordTest, WelfordSchedule) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int M = 64, N = 128;

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = mul(tv0, IrBuilder::create<Val>(1.0));
  auto tvs = Welford(tv1, {1});
  auto tv_avg = tvs.avg;
  auto tv_M2 = tvs.var_sum;
  auto tv_N = tvs.n;
  fusion.addOutput(tv_avg);
  fusion.addOutput(tv_M2);
  fusion.addOutput(tv_N);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_int = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({M, N}, options);
  auto cg_results = scheduleAndRun(&fusion, SchedulerType::Reduction, {t0});

  // by default Welford outputs sum of square diff so need to divide to get var
  cg_results.outputs[1] = cg_results.outputs[1].as<at::Tensor>() / N;

  auto at_avg = t0.mean({1});
  auto at_var = t0.var({1}, false);
  auto at_n = at::ones({M}, options_int) * N;

  testValidate(
      &fusion,
      cg_results.outputs,
      {t0},
      {at_avg, at_var, at_n},
      __LINE__,
      __FILE__,
      "validate welford",
      cg_results.heuristic_params->lparams);
}

TEST_F(WelfordTest, WelfordPersistence) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tvs = Welford(tv0, {0});
  auto tv4 = add(tvs.avg, tvs.var_sum);
  auto tv5 = broadcast(tv4, {true});
  auto tv6 = add(tv0, tv5);
  fusion.addOutput(tv6);

  std::vector<TensorView*> schedule_tvs = {
      tvs.avg, tvs.var_sum, tvs.n, tv5, tv6};

  for (auto tv : schedule_tvs) {
    tv->split(0, 2);
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::BIDy);
  }

  const int numel_x = 10;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({numel_x}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {input});
  auto out = ke.run({input});

  auto aten_output = (input.mean({0}) + (input.var({0}, false) * numel_x))
                         .unsqueeze(-1)
                         .add(input);

  testValidate(&fusion, out, {input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(WelfordTest, WelfordPersistence2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tvs = Welford(tv0, {0});
  auto tv4 = add(tvs.avg, tvs.var_sum);
  auto tv5 = broadcast(tv4, {true, false});
  auto tv6 = add(tv0, tv5);
  fusion.addOutput(tv6);

  std::vector<TensorView*> schedule_tvs = {
      tvs.avg, tvs.var_sum, tvs.n, tv5, tv6};
  for (auto tv : schedule_tvs) {
    tv->split(0, 2);
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDy);
    tv->axis(2)->parallelize(ParallelType::TIDx);
  }
  tv4->axis(0)->parallelize(ParallelType::TIDx);

  const int numel_x = 10;
  const int numel_y = 3;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({numel_x, numel_y}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {input});
  auto out = ke.run({input});

  auto aten_output = (input.mean({0}) + (input.var({0}, false) * numel_x))
                         .unsqueeze(0)
                         .add(input);

  testValidate(&fusion, out, {input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, BlockWelfordInSerialLoop) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int M = 10;
  constexpr int N = 20;
  constexpr int K = 20;

  auto tv0 = makeSymbolicTensor(3);
  auto tvs = Welford(tv0, {{1, 2}});
  fusion.addInput(tv0);
  auto tv_avg = tvs.avg;
  auto tv_M2 = tvs.var_sum;
  fusion.addOutput(tv_avg);
  fusion.addOutput(tv_M2);

  tv_avg->axis(-1)->parallelize(ParallelType::TIDx);
  tv_avg->axis(0)->parallelize(ParallelType::BIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({M, N, K}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});
  at::Tensor aten_avg = t0.mean({1, 2});
  at::Tensor aten_M2 = t0.var({1, 2}, false) * N * K;
  testValidate(&fusion, outputs, {t0}, {aten_avg, aten_M2}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, Welford1Output) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);

  auto tvs = Welford(tv0, {1});
  fusion->addOutput(tvs.var_sum);
  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({128, 65}, options);
  auto outputs = executor_cache.runFusionWithInputs({t0});

  auto t1 = t0.var({1}, false) * 65;
  testValidate(fusion, outputs, {t0}, {t1}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, Translate1Welford) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);

  auto tvs = Welford(tv0, {1});
  auto tv_out = add(tv0, broadcast(tvs.avg, {false, true}));
  fusion->addOutput(tv_out);
  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto run_test = [&executor_cache,
                   fusion](auto inner_size) -> FusionKernelRuntime* {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({128, inner_size}, options);
    auto outputs = executor_cache.runFusionWithInputs({t0});
    // Square sums does not fit well in the testValidate assumptions,
    //  so we just compare the divided output here.
    testValidate(
        fusion,
        outputs,
        {t0},
        {t0.add(t0.mean({1}).unsqueeze(1))},
        __LINE__,
        __FILE__);

    return executor_cache.getMostRecentKernelRuntime();
  };

  // Run a translated welford
  auto runtime1 = run_test(64);
  // Check it was translated
  NVF_CHECK(
      runtime1->fusionSegments()->groups().size() == 1 &&
      runtime1->fusionSegments()->groups()[0]->exprs().size() > 2);

  // Run an un-translated welford
  auto runtime2 = run_test(65536);
  // Check it was not translated for pre-hopper
  // Hopper and above use cluster reduction
  if (at::cuda::getCurrentDeviceProperties()->major < 9) {
    bool found_welford = false;
    for (auto group : runtime2->fusionSegments()->groups()) {
      for (auto expr : group->exprs()) {
        if (expr->isA<WelfordOp>()) {
          found_welford = true;
        }
      }
    }
    NVF_CHECK(found_welford);
  }
}

TEST_F(NVFuserTest, Translate2Welford) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);

  auto tvs1 = Welford(tv0, {1});
  auto tv_out1 = add(tv0, broadcast(tvs1.avg, {false, true}));
  fusion->addOutput(tv_out1);

  auto tvs2 = Welford(tv0, {1});
  auto tv_out2 = add(tv0, broadcast(tvs2.avg, {false, true}));
  fusion->addOutput(tv_out2);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto run_test = [&executor_cache,
                   fusion](auto inner_size) -> FusionKernelRuntime* {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({128, inner_size}, options);
    auto outputs = executor_cache.runFusionWithInputs({t0});

    // Square sums does not fit well in the testValidate assumptions,
    //  so we just compare the divided output here.
    auto out = t0.add(t0.mean({1}).unsqueeze(1));
    testValidate(fusion, outputs, {t0}, {out, out}, __LINE__, __FILE__);

    return executor_cache.getMostRecentKernelRuntime();
  };

  // Run a translated welford
  auto runtime1 = run_test(64);
  // Check it was translated
  NVF_CHECK(
      runtime1->fusionSegments()->groups().size() == 1 &&
      runtime1->fusionSegments()->groups()[0]->exprs().size() > 4);

  // Run an un-translated welford
  auto runtime2 = run_test(65536);
  // Check it was not translated for pre-hopper
  // Hopper and above use cluster reduction
  if (at::cuda::getCurrentDeviceProperties()->major < 9) {
    bool found_welford = false;
    for (auto group : runtime2->fusionSegments()->groups()) {
      for (auto expr : group->exprs()) {
        if (expr->isA<WelfordOp>()) {
          found_welford = true;
        }
      }
    }
    NVF_CHECK(found_welford);
  }
}

TEST_F(NVFuserTest, LargeWelfordNormalization) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);

  auto tvs1 = Welford(tv0, {1});
  auto sum_of_tv0 = sum(tv0, {1});

  fusion->addOutput(tvs1.var_sum);
  fusion->addOutput(sum_of_tv0);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto run_test = [&executor_cache,
                   fusion](auto inner_size) -> FusionKernelRuntime* {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({128, inner_size}, options);
    auto outputs = executor_cache.runFusionWithInputs({t0});

    auto t1 = t0.var({1}, false) * inner_size;
    auto t2 = t0.sum({1});
    testValidate(fusion, outputs, {t0}, {t1, t2}, __LINE__, __FILE__);

    return executor_cache.getMostRecentKernelRuntime();
  };

  auto runtime = run_test(65536);
  NVF_CHECK(!runtime->isSegmented());
}

TEST_F(NVFuserTest, WelfordOuterPersistence) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);

  auto tvs1 = Welford(tv0, {1});
  auto sum_of_tv0 = sum(tv0, {1});
  auto sum_bcasted = broadcast(sum_of_tv0, {false, true});
  auto avg_bcasted = broadcast(tvs1.avg, {false, true});
  auto tv0_plus_sum = add(tv0, sum_bcasted);
  auto tv0_plus_avg = add(tv0, avg_bcasted);

  fusion->addOutput(tv0_plus_sum);
  fusion->addOutput(tv0_plus_avg);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto run_test = [&executor_cache,
                   fusion](auto inner_size) -> FusionKernelRuntime* {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({128, inner_size}, options);
    auto outputs = executor_cache.runFusionWithInputs({t0});

    auto t1 = t0.to(c10::kDouble).mean({1}).unsqueeze(1) + t0;
    auto t2 = t0.to(c10::kDouble).sum({1}).unsqueeze(1) + t0;
    testValidate(fusion, outputs, {t0}, {t2, t1}, __LINE__, __FILE__);

    return executor_cache.getMostRecentKernelRuntime();
  };

  for (auto inner_size : {4096, 8192, 32768}) {
    auto runtime = run_test(inner_size);
    NVF_CHECK(!runtime->isSegmented());
  }
}

using WelfordReductionParams = std::tuple<DataType, int64_t, int64_t, int64_t>;

using WelfordReductionTest = NVFuserFixtureParamTest<WelfordReductionParams>;

TEST_P(WelfordReductionTest, Test) {
  auto [dtype, rdim, odim, axis] = GetParam();

  // TODO: original welford algorithm actually keeps a running sum of
  // squares, i.e. M_{2n} in the
  //       cf:
  //       https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
  //       algorithm notation, and it can reach inf for large numbers
  //       with half precision. skipping too large volumes for half for
  //       nwo might need further numerical experiments to re-design
  //       this.
  if (rdim > 32768 &&
      (dtype == DataType::Half || dtype == DataType::BFloat16)) {
    GTEST_SKIP() << "Skipping large reduction dims (" << rdim
                 << ") for half and bfloat16";
  }

  maybeClearAllocator();

  at::ScalarType aten_dtype = data_type_to_aten(dtype);

  Fusion fusion;
  FusionGuard fg(&fusion);
  TensorView* tv0 = makeSymbolicTensor(2, dtype);
  bool is_fp16 = dtype == DataType::Half;
  bool is_bf16 = dtype == DataType::BFloat16;
  TensorView* tv0_cast = tv0;
  if (is_fp16 || is_bf16) {
    tv0_cast = castOp(DataType::Float, tv0);
  }
  fusion.addInput(tv0);
  auto tv1 = mul(tv0_cast, IrBuilder::create<Val>(1.0));
  auto tvs = Welford(tv1, {axis});
  auto tv_avg = tvs.avg;
  auto tv_M2 = tvs.var_sum;
  auto tv_N = tvs.n;

  TensorView* avg_cast = tv_avg;
  TensorView* M2_cast = tv_M2;

  if (is_fp16) {
    avg_cast = castOp(DataType::Half, tv_avg);
    M2_cast = castOp(DataType::Half, tv_M2);
  }
  if (is_bf16) {
    avg_cast = castOp(DataType::BFloat16, tv_avg);
    M2_cast = castOp(DataType::BFloat16, tv_M2);
  }

  fusion.addOutput(avg_cast);
  fusion.addOutput(M2_cast);
  fusion.addOutput(tv_N);

  auto options = at::TensorOptions().dtype(aten_dtype).device(at::kCUDA, 0);
  std::vector<TensorView*> outputs_of_red;
  at::Tensor aten_input =
      (axis ? at::randn({odim, rdim}, options)
            : at::randn({rdim, odim}, options));

  if (is_fp16 || is_bf16) {
    outputs_of_red.push_back(avg_cast);
    outputs_of_red.push_back(M2_cast);
  }

  auto heuristic_params = SchedulerEntry::scheduleWith(
      &fusion, SchedulerType::Reduction, {aten_input});
  auto reduction_params = heuristic_params->as<ReductionParams>();

  auto lparams = reduction_params->lparams;
  auto cparams = reduction_params->cparams;

  KernelExecutor ke;
  // Needs to pass compile para to use the correct index type, otherwise the
  // lowering pass will use int64 as the index tpye, since this test saves
  // `tv_N` as index type, it may cause vectorization size validation error. For
  // example, the heuristics set index type to int32 and the max vectorization
  // factor is 4, if compile para is not passed to compile, the lowering
  // pass uses int64 as index type, so the max vectorization factor is 16 bytes
  // sizeof(int64) = 2, which is wrong since the actual index type is int32
  // and the max vectorization factor is 4.
  ke.compile(&fusion, {aten_input}, lparams, cparams);
  auto outputs = ke.run({aten_input}, {}, lparams, cparams);

  // by default Welford outputs sum of square diff so need to divide to
  // get var

  outputs[1] = outputs[1].as<at::Tensor>() / rdim;

  auto at_avg = aten_input.mean({axis});
  auto at_var = aten_input.var({axis}, false);
  auto at_n =
      (axis ? at::ones({odim, rdim}, options)
            : at::ones({rdim, odim}, options));
  at_n = at_n.sum({axis});

  testValidate(
      ke.compiledKernel()->kernel(),
      outputs,
      {aten_input},
      {at_avg, at_var, at_n},
      __LINE__,
      __FILE__,
      "validate welford",
      reduction_params->lparams);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    WelfordReductionTest,
    ::testing::Combine(
        testing::ValuesIn(getFloatingDataTypes()), // data type
        testing::ValuesIn(Pow2Vals1to1Million), // reduction dimension size
        testing::Values(160, 320), // iteration dimension size
        testing::Values(0, 1)), // reduction axis
    // when using structured bindings within TestParamInfo,
    // parentheses are required to avoid compile errors,
    // see https://github.com/google/googletest/issues/3848
    ([](const testing::TestParamInfo<WelfordReductionParams>& info)
         -> std::string {
      std::stringstream ss;
      auto [dtype, rdim, odim, axis] = info.param;
      ss << "dtype_" << dtype;
      ss << "_redu_" << rdim;
      ss << "_iter_" << odim;
      ss << "_axis_" << axis;
      return sanitizeTestName(ss.str());
    }));

} // namespace nvfuser
