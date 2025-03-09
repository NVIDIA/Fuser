#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include <numeric>
#include <cmath>
#include <functional>

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

#define NUM_WARMUP_ITERATIONS 5
#define NUM_OUTER_ITERATIONS 10
#define NUM_INNER_ITERATIONS 30
#define SLEEP_TIME 2

// Helper function to run performance test with consistent timing and reporting
template <typename BenchmarkFn>
void runPerformanceTest(
    const std::string& test_name,
    int num_warmup_iterations,
    int num_outer_iterations,
    int num_inner_iterations,
    BenchmarkFn benchmark_fn) {
    
  // Warm-up runs
  for (int i = 0; i < num_warmup_iterations; i++) {
    benchmark_fn();
  }

  // Store individual iteration times
  std::vector<double> iteration_times;
  iteration_times.reserve(num_outer_iterations);
  
  std::chrono::duration<double, std::micro> elapsed{0};
  for (int i = 0; i < num_outer_iterations; i++) {
    // Make sure CUDA operations are completed before starting the timing
    cudaDeviceSynchronize();

    // Add sleep time
    std::this_thread::sleep_for(std::chrono::seconds(SLEEP_TIME));

    // Start CPU timer
    auto start = std::chrono::high_resolution_clock::now();

    // Run benchmark multiple times
    for (int j = 0; j < num_inner_iterations; j++) {
      benchmark_fn();
    }

    // Stop timer
    auto end = std::chrono::high_resolution_clock::now();

    // Track elapsed time for this iteration
    auto iter_time = std::chrono::duration<double, std::micro>(end - start).count();
    iteration_times.push_back(iter_time);
    elapsed += end - start;
  }

  // Calculate average and standard deviation
  double avg = std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0) / 
              iteration_times.size();
  
  double variance = 0.0;
  for (const auto& time : iteration_times) {
    variance += (time - avg) * (time - avg);
  }
  variance /= iteration_times.size();
  double std_dev = std::sqrt(variance);

  // Use separator based on test name length
  std::string separator(test_name.length() + 6, '=');

  std::cout << "\n" << separator << std::endl;
  std::cout << test_name << " Performance Test" << std::endl;
  std::cout << separator << std::endl;
  std::cout << "Total time for " << num_outer_iterations * num_inner_iterations
            << " iterations: " << elapsed.count() << " μs" << std::endl;
  std::cout << "Average time per iteration: "
            << elapsed.count() / (num_outer_iterations * num_inner_iterations)
            << " μs" << std::endl;
  std::cout << "StdDev / Avg of outer iterations: " << 100 * (std_dev / avg) << "%" << std::endl;
  std::cout << separator << std::endl;
}

TEST_F(NVFuserTest, LinearPerfAten_CUDA) {
  // Create input tensors for linear operation
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto input = at::randn({16, 16}, options);
  auto weight = at::randn({16, 16}, options);
  auto bias = at::randn({16}, options);

  at::Tensor output;
  
  // Use the common performance test function
  runPerformanceTest(
    "ATen Linear 16x16",
    NUM_WARMUP_ITERATIONS,
    NUM_OUTER_ITERATIONS,
    NUM_INNER_ITERATIONS,
    [&]() { output = at::linear(input, weight, bias); }
  );
}

TEST_F(NVFuserTest, LinearPerfEvaluate_CUDA) {
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

  auto linear_op = fusion->exprs().at(0)->as<LinearOp>();

  std::vector<PolymorphicValue> outputs;
 
  ExpressionEvaluator ee;

  // Use the common performance test function
  runPerformanceTest(
    "LinearOp::evaluate 16x16",
    NUM_WARMUP_ITERATIONS,
    NUM_OUTER_ITERATIONS,
    NUM_INNER_ITERATIONS,
    [&]() { outputs = linear_op->evaluate(ee, args.vector()); }
  );
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
  ExprEvalExecutor executor(0, 0, 0, 0);
  executor.compile(&fusion);

  KernelArgumentHolder outputs;
  
  // Use the common performance test function
  runPerformanceTest(
    "ExprEval Linear 16x16",
    NUM_WARMUP_ITERATIONS,
    NUM_OUTER_ITERATIONS,
    NUM_INNER_ITERATIONS,
    [&]() { outputs = executor.run(args); }
  );
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


  runPerformanceTest(
    "ATen Linear 16x16",
    NUM_WARMUP_ITERATIONS,
    NUM_OUTER_ITERATIONS,
    NUM_INNER_ITERATIONS,
    [&]() { outputs = fkr.runWithInputs(args); }
  );
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

  runPerformanceTest(
    "FusionExecutorCache Linear 16x16",
    NUM_WARMUP_ITERATIONS,
    NUM_OUTER_ITERATIONS,
    NUM_INNER_ITERATIONS,
    [&]() { outputs = fec.runFusionWithInputs(args); }
  );
}

} // namespace nvfuser
