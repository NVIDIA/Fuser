#pragma once
#include <cstdint>

namespace nvfuser {

namespace hir {
// Set of parameters that control the behavior of HostIrEvaluator
struct HostIrEvaluatorParams {
  // Experimental: whether to use FusionExecutorCache rather than
  // KernelExecutor.
  bool use_fusion_executor_cache = false;
  // Experimental: whether to apply auto-scheduling in FusionExecutorCache if
  // use_fusion_executor_cache=true. WAR: temporary hack mainly use for
  // development
  bool skip_auto_scheduling = false;
  // Experimental: whether to cache fusion executor. WAR: avoid recompilation
  // but implicitely assumes that the input shape don't change over iterations
  bool cache_fusion_executor = false;
  // number of additional cuda streams to use at runtime for comm+compute
  // pipelining
  int64_t number_of_streams = 4;
};
}
}