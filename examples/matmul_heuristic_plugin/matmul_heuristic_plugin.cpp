#include <scheduler/matmul_heuristic_plugin_api.h>

#include <cstdint>
#include <iostream>

using nvfuser::matmul_heuristic_plugin::KernelConfig;
using nvfuser::matmul_heuristic_plugin::ProblemDescription;

// This example heuristic simply prints the problem description then sets a
// fixed kernel configuration.
extern "C" void getConfig(
    KernelConfig* config,
    const ProblemDescription* problem) {
  uint32_t m = getProblemM(problem);
  uint32_t n = getProblemN(problem);
  uint32_t k = getProblemK(problem);
  uint32_t batch_size = getProblemBatchSize(problem);
  uint8_t layout = getProblemLayout(problem);
  const char* precision = getProblemPrecision(problem);

  std::cout << "Example heuristic for problem: ";
  std::cout << "m=" << m << " ";
  std::cout << "n=" << n << " ";
  std::cout << "k=" << k << " ";
  std::cout << "batch_size=" << batch_size << " ";
  std::cout << "layout=" << std::to_string(layout) << " ";
  std::cout << "precision=" << precision << std::endl;

  setCtaTile(config, 128, 128, 32);
  setWarpTile(config, 64, 64, 32);
  setInstructionTile(config, 16, 8, 16);
  setSplitKFactor(config, 2);
  setLoadStages(config, 3);
  setGridSwizzleFactor(config, 1);
  setCtaOrder(config, 0);
}
