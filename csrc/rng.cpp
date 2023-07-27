// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <device_lower/lower2device.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>

#include <ATen/cuda/CUDAGeneratorImpl.h>

#include <tuple>

namespace nvfuser {

std::tuple<Val*, Val*, kir::GetRNGSeedAndOffsetFromHost*>
getRNGSeedAndOffsetFromHost() {
  // Note [CUDA graph capture and RNG seed and offset]
  // This is how we handle RNG seeds and offsets in PyTorch. In PyTorch,
  // depending on whether we are running under CUDA graph capture or not, the
  // seed and offset are stored differently. When not under CUDA graph capture,
  // the seed and offset are stored on the host. When under CUDA graph capture,
  // the seed and offset are stored on the device, and there are device pointers
  // to these values. The pointers themselves are stored on the host. For seed,
  // because all random ops within the same CUDA graph always use the same seed,
  // so just dereferencing the pointer is enough. For offset, because each
  // random op has its own offset, so we need to add the offset value on the
  // host to the dereferenced offset pointer. In this ways, the offset value on
  // the host is the intra-cuda-graph offset, and the offset value on the device
  // is the base offset for this entire CUDA graph.
  auto intptr = PointerOf{std::make_shared<DataType>(DataType::Int)};
  Val* seed_ptr = IrBuilder::newScalar(intptr);
  Val* seed_val = IrBuilder::newScalar(DataType::Int);
  Val* offset_ptr = IrBuilder::newScalar(intptr);
  Val* offset_val = IrBuilder::newScalar(DataType::Int);
  auto expr = IrBuilder::create<kir::GetRNGSeedAndOffsetFromHost>(
      seed_ptr, seed_val, offset_ptr, offset_val);
  GpuLower::current()->allKnownVals().push_back(seed_ptr);
  GpuLower::current()->allKnownVals().push_back(seed_val);
  GpuLower::current()->allKnownVals().push_back(offset_ptr);
  GpuLower::current()->allKnownVals().push_back(offset_val);
  Val* nullptr_ = IrBuilder::create<NamedScalar>("nullptr", intptr);
  Val* seed = IrBuilder::whereExpr(
      IrBuilder::eqExpr(seed_ptr, nullptr_),
      seed_val,
      IrBuilder::derefExpr(seed_ptr));
  Val* offset = IrBuilder::whereExpr(
      IrBuilder::eqExpr(offset_ptr, nullptr_),
      offset_val,
      IrBuilder::addExpr(IrBuilder::derefExpr(offset_ptr), offset_val));
  // Note [Divide offset by 4]
  // The Philox engine generates 4 uints each call. The offset in the `philox`
  // function in our runtime library is in the unit of uint4, but the offset in
  // PyTorch is in the unit of uint. And we assign each RNG op one whole uint4
  // along the offset dimension.
  //
  // So we calculate the argument of `philox` as
  //   pytorch_offset / 4 + rng_op_id
  // This also means, each time we get a new offset from the pytorch, we need to
  // shift the host offset by
  //   4 * num_rng_ops
  offset =
      IrBuilder::divExpr(offset, IrBuilder::newConstant(4L, DataType::Int));
  return std::make_tuple(seed, offset, expr);
}

std::vector<PolymorphicValue> kir::GetRNGSeedAndOffsetFromHost::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  at::PhiloxCudaState philox_engine_inputs;
  auto gen = at::cuda::detail::getDefaultCUDAGenerator();
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen.mutex());
    philox_engine_inputs =
        at::check_generator<at::CUDAGeneratorImpl>(gen)->philox_cuda_state(
            (uint64_t)(offsets()) * 4);
  }
  std::vector<PolymorphicValue> outputs;
  if (philox_engine_inputs.captured_) {
    outputs.emplace_back(philox_engine_inputs.seed_.ptr);
    outputs.emplace_back(0L);
    outputs.emplace_back(philox_engine_inputs.offset_.ptr);
    outputs.emplace_back((int64_t)philox_engine_inputs.offset_intragraph_);
  } else {
    outputs.emplace_back((int64_t*)nullptr);
    outputs.emplace_back((int64_t)philox_engine_inputs.seed_.val);
    outputs.emplace_back((int64_t*)nullptr);
    outputs.emplace_back((int64_t)philox_engine_inputs.offset_.val);
  }
  return outputs;
}

} // namespace nvfuser
