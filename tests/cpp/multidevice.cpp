// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <sys/types.h>
#include <unistd.h>
#include <mutex>

#ifdef NVFUSER_DISTRIBUTED
#include <torch/csrc/distributed/c10d/debug.h>
#else
#include <multidevice/c10d_mock.h>
#endif
#include <torch/cuda.h>

#include <fusion_segmenter.h>
#include <ir/all_nodes.h>
#include <ir/iostream.h>
#include <multidevice/utils.h>
#include <ops/all_ops.h>
#include <options.h>
#include <runtime/allocations.h>
#include <tests/cpp/multidevice.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

void MultiDeviceTestEnvironment::TearDown() {
  Communicator::getInstance().cleanup();
}

MultiDeviceTest::MultiDeviceTest() {
  // Enable logging in c10d so debug messages can be printed out via
  // `TORCH_DISTRIBUTED_DEBUG`.
  c10d::setDebugLevelFromEnvironment();

  communicator_ = &Communicator::getInstance();
  tensor_options =
      at::TensorOptions().dtype(at::kFloat).device(communicator_->device());
  debug_print = getNvFuserEnv("MULTIDEVICE_DEBUG_PRINT") != nullptr;
  disable_skip = getNvFuserEnv("MULTIDEVICE_DISABLE_SKIP") != nullptr;
}

MultiDeviceTest::~MultiDeviceTest() {
  // Force all processes to synchronize at a barrier between tests. It slightly
  // slows the tests down, but makes it much easier to isolate a failing test.
  // Without this, if a test fails such that a subset of processes fail, then
  // some processes will move onto another tests and timeout later.
  if (communicator_->is_available()) {
    communicator_->barrier();
  }
}

void MultiDeviceTest::SetUp() {
  // Set the same random seed for all processes.
  NVFuserTest::SetUp();

  if (!disable_skip && !communicator_->is_available()) {
    GTEST_SKIP() << "This test needs an available communicator.";
  }
}

at::Tensor MultiDeviceTest::shardTensor(at::Tensor tensor, TensorView* tv) {
  if (!isSharded(tv)) {
    return tensor;
  }
  NVF_ERROR(tv->hasDeviceMesh(), "`tv` has no DeviceMesh: ", tv);
  return shardTensor(
      tensor,
      getShardedLogicalAxis(tv, ParallelType::DIDx),
      tv->getDeviceMesh());
}

at::Tensor MultiDeviceTest::shardTensor(
    at::Tensor tensor,
    const int64_t axis,
    const DeviceMesh& mesh) {
  const auto device_id = communicator_->deviceId();
  return nvfuser::shardTensor(tensor, axis, mesh, device_id);
}

// testValidate doesn't work out of the box due to #2906, so I had to manually
// specify the absolute tolerances. The atols passed in are tuned for bfloat,
// the least precise dtype. They can probably be made stricter for other
// dtypes.
void MultiDeviceTest::validate(
    const std::vector<at::Tensor>& expected_outputs,
    const KernelArgumentHolder& outputs,
    const std::vector<double>& atols) {
  using testing::SizeIs;
  const auto num_outputs = outputs.size();
  ASSERT_THAT(expected_outputs, SizeIs(num_outputs));
  ASSERT_THAT(atols, SizeIs(num_outputs));

  for (const auto i : arange(num_outputs)) {
    // allclose can catch this as well. However, it would throw an exception,
    // not showing which output was problematic.
    NVF_ERROR(
        outputs[i].is<at::Tensor>(), "Output is not a tensor at index ", i);
    auto output_tensor = outputs[i].as<at::Tensor>();
    NVF_ERROR(
        output_tensor.dtype() == expected_outputs[i].dtype(),
        "Output ",
        i,
        " has a mismatching data type: ",
        output_tensor.dtype(),
        " vs. ",
        expected_outputs[i].dtype());

    const double atol = atols[i];
    // These default rtols are copied from
    // https://github.com/pytorch/pytorch/blob/951c21d6790334d57862e94a3f582ac724147a53/torch/testing/_comparison.py#L65-L73.
    double rtol;
    switch (output_tensor.scalar_type()) {
      case at::kBFloat16:
        rtol = 1.6e-2;
        break;
      case at::kHalf:
        rtol = 1e-3;
        break;
      case at::kFloat:
        rtol = 1.3e-6;
        break;
      default:
        rtol = 0.0;
        break;
    }

    auto generate_comparison_details = [](at::Tensor expected_out,
                                          at::Tensor out,
                                          double atol,
                                          double rtol) -> std::string {
      std::ostringstream oss;
      auto error = (out - expected_out).abs();
      auto max_relative_error =
          (error.max() / expected_out.abs().max()).item().to<double>();
      auto error_count =
          at::sum(error >= atol + expected_out.abs() * rtol).item();
      indent(oss, 1)
          << "max absolute error under rtol: "
          << (error - expected_out.abs() * rtol).max().item().to<double>()
          << std::endl;
      indent(oss, 1) << "max relative error: " << max_relative_error
                     << std::endl;
      indent(oss, 1) << "failing elements: " << error_count << ", "
                     << error_count.to<float>() / at::numel(out) * 100.0
                     << "\% of tensor";
      return oss.str();
    };

    EXPECT_TRUE(at::allclose(output_tensor, expected_outputs[i], rtol, atol))
        << "Output " << i << " mismatches with atol " << atol << ":"
        << std::endl
        << generate_comparison_details(
               expected_outputs[i], output_tensor, atol, rtol);
  }
}

} // namespace nvfuser

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::AddGlobalTestEnvironment(new nvfuser::MultiDeviceTestEnvironment());
  return RUN_ALL_TESTS();
}
