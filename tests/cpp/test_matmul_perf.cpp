#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include <thread>

#include <fusion.h>
#include <fusion_guard.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ops/arith.h>
#include <ops/composite.h>
#include <runtime/executor.h>
#include <runtime/expr_eval_exec.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>
#include <torch/torch.h>

namespace nvfuser {

TEST_F(NVFuserTest, LinearPerfAten_CUDA) {
  // Create input tensors for linear operation
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto input = at::randn({16, 16}, options);
  auto weight = at::randn({16, 16}, options);
  auto bias = at::randn({16}, options);

  at::Tensor output;
  // Warm-up run
  for (int i = 0; i < 5; i++) {
    output = at::linear(input, weight, bias);
  }

  // Make sure CUDA operations are completed before starting the timing
  cudaDeviceSynchronize();

  // Add 5-second sleep
  std::this_thread::sleep_for(std::chrono::seconds(5));

  // Start CPU timer
  auto start = std::chrono::high_resolution_clock::now();

  // Run linear 100 times
  for (int i = 0; i < 100; i++) {
    output = at::linear(input, weight, bias);
  }

  // Stop timer
  auto end = std::chrono::high_resolution_clock::now();

  // Calculate elapsed time in microseconds
  std::chrono::duration<double, std::micro> elapsed = end - start;

  std::cout << "\n===================================" << std::endl;
  std::cout << "ATen Linear 16x16 Performance Test" << std::endl;
  std::cout << "===================================" << std::endl;
  std::cout << "Total time for 100 iterations: " << elapsed.count() << " μs"
            << std::endl;
  std::cout << "Average time per iteration: " << elapsed.count() / 100.0
            << " μs" << std::endl;
  std::cout << "===================================" << std::endl;

  // Basic verification
  ASSERT_EQ(output.sizes(), std::vector<int64_t>({16, 16}));
}

TEST_F(NVFuserTest, LinearPerfExprEvalExec_CUDA) {
  // Create input tensors for linear operation
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto input_at = at::randn({16, 16}, options);
  auto weight_at = at::randn({16, 16}, options);
  auto bias_at = at::randn({16}, options);

  // Create a Fusion to execute
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create inputs
  auto input = makeSymbolicTensor(2); // 16x16
  auto weight = makeSymbolicTensor(2); // 16x16
  auto bias = makeSymbolicTensor(1); // 16

  fusion.addInput(input);
  fusion.addInput(weight);
  fusion.addInput(bias);

  // Create linear op
  auto output = linear(input, weight, bias);

  fusion.addOutput(output);

  // Setup inputs to test
  KernelArgumentHolder args;
  args.push(input_at);
  args.push(weight_at);
  args.push(bias_at);

  // Create the ExprEvalExecutor
  ExprEvalExecutor eee;
  eee.compile(&fusion);
  KernelArgumentHolder outputs;

  // Warm up run
  for (int i = 0; i < 5; i++) {
    outputs = eee.run(args);
  }

  // Make sure CUDA operations are completed before starting the timing
  cudaDeviceSynchronize();

  // Add 5-second sleep
  std::this_thread::sleep_for(std::chrono::seconds(5));

  // Start CPU timer
  auto start = std::chrono::high_resolution_clock::now();

  // Run linear op 100 times
  for (int i = 0; i < 100; i++) {
    outputs = eee.run(args);
  }

  // Stop timer
  auto end = std::chrono::high_resolution_clock::now();

  // Calculate elapsed time in microseconds
  std::chrono::duration<double, std::micro> elapsed = end - start;

  std::cout << "\n=========================================" << std::endl;
  std::cout << "NVFuser Linear 16x16 Performance Test" << std::endl;
  std::cout << "=========================================" << std::endl;
  std::cout << "Total time for 100 iterations: " << elapsed.count() << " μs"
            << std::endl;
  std::cout << "Average time per iteration: " << elapsed.count() / 100.0
            << " μs" << std::endl;
  std::cout << "=========================================" << std::endl;

  // Basic verification
  ASSERT_EQ(
      outputs[0].as<at::Tensor>().sizes(), std::vector<int64_t>({16, 16}));
}

TEST_F(NVFuserTest, LinearPerfFusionKernelRuntime_CUDA) {
  // Create input tensors for linear operation
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto input_at = at::randn({16, 16}, options);
  auto weight_at = at::randn({16, 16}, options);
  auto bias_at = at::randn({16}, options);

  // Create a Fusion to execute
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // Create inputs
  auto input = makeSymbolicTensor(2); // 16x16
  auto weight = makeSymbolicTensor(2); // 16x16
  auto bias = makeSymbolicTensor(1); // 16

  fusion->addInput(input);
  fusion->addInput(weight);
  fusion->addInput(bias);

  // Create linear op
  auto output = linear(input, weight, bias);

  fusion->addOutput(output);

  // Setup inputs to test
  KernelArgumentHolder args;
  args.push(input_at);
  args.push(weight_at);
  args.push(bias_at);

  // Create the FusionKernelRuntime
  FusionKernelRuntime fkr(std::move(fusion), args);
  fkr.compileFusionParallel(args);
  KernelArgumentHolder outputs;

  // Warm up run
  for (int i = 0; i < 5; i++) {
    outputs = fkr.runWithInputs(args);
  }

  // Make sure CUDA operations are completed before starting the timing
  cudaDeviceSynchronize();

  // Add 5-second sleep
  std::this_thread::sleep_for(std::chrono::seconds(5));

  // Start CPU timer
  auto start = std::chrono::high_resolution_clock::now();

  // Run linear op 100 times
  for (int i = 0; i < 100; i++) {
    outputs = fkr.runWithInputs(args);
  }

  // Stop timer
  auto end = std::chrono::high_resolution_clock::now();

  // Calculate elapsed time in microseconds
  std::chrono::duration<double, std::micro> elapsed = end - start;

  std::cout << "\n=========================================" << std::endl;
  std::cout << "NVFuser Linear 16x16 Performance Test" << std::endl;
  std::cout << "=========================================" << std::endl;
  std::cout << "Total time for 100 iterations: " << elapsed.count() << " μs"
            << std::endl;
  std::cout << "Average time per iteration: " << elapsed.count() / 100.0
            << " μs" << std::endl;
  std::cout << "=========================================" << std::endl;

  // Basic verification
  ASSERT_EQ(
      outputs[0].as<at::Tensor>().sizes(), std::vector<int64_t>({16, 16}));
}

TEST_F(NVFuserTest, LinearPerfFusionExecutorCache_CUDA) {
  // Create input tensors for linear operation
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto input_at = at::randn({16, 16}, options);
  auto weight_at = at::randn({16, 16}, options);
  auto bias_at = at::randn({16}, options);

  // Create a Fusion to execute
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // Create inputs
  auto input = makeSymbolicTensor(2); // 16x16
  auto weight = makeSymbolicTensor(2); // 16x16
  auto bias = makeSymbolicTensor(1); // 16

  fusion->addInput(input);
  fusion->addInput(weight);
  fusion->addInput(bias);

  // Create linear op
  auto output = linear(input, weight, bias);

  fusion->addOutput(output);

  // Setup inputs to test
  KernelArgumentHolder args;
  args.push(input_at);
  args.push(weight_at);
  args.push(bias_at);

  // Create the FusionExecutorCache
  FusionExecutorCache fec(std::move(fusion));
  KernelArgumentHolder outputs;

  // Warm up run
  for (int i = 0; i < 5; i++) {
    outputs = fec.runFusionWithInputs(args);
  }

  // Make sure CUDA operations are completed before starting the timing
  cudaDeviceSynchronize();

  // Add 5-second sleep
  std::this_thread::sleep_for(std::chrono::seconds(5));

  // Start CPU timer
  auto start = std::chrono::high_resolution_clock::now();

  // Run linear op 100 times
  for (int i = 0; i < 100; i++) {
    outputs = fec.runFusionWithInputs(args);
  }

  // Stop timer
  auto end = std::chrono::high_resolution_clock::now();

  // Calculate elapsed time in microseconds
  std::chrono::duration<double, std::micro> elapsed = end - start;

  std::cout << "\n=========================================" << std::endl;
  std::cout << "NVFuser Linear 16x16 Performance Test" << std::endl;
  std::cout << "=========================================" << std::endl;
  std::cout << "Total time for 100 iterations: " << elapsed.count() << " μs"
            << std::endl;
  std::cout << "Average time per iteration: " << elapsed.count() / 100.0
            << " μs" << std::endl;
  std::cout << "=========================================" << std::endl;

  // Basic verification
  ASSERT_EQ(
      outputs[0].as<at::Tensor>().sizes(), std::vector<int64_t>({16, 16}));
}

} // namespace nvfuser
