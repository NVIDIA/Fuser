#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <test/utils.h>
#include <test/validator.h>

#include <ops/all_ops.h>

namespace nvfuser {

class KernelInputsTest : public NVFuserTest {};

TEST_F(KernelInputsTest, HoistToHost1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto scalar = IrBuilder::newScalar(DataType::Double);
  fusion.addInput(scalar);
  auto inv = div(fusion.oneVal(DataType::Double), scalar);
  auto tv1 = mul(tv0, inv);
  fusion.addOutput(tv1);
  fusion.manage("hoist_to_host", std::vector<Val*>{inv});

  tv1->axis(0)->parallelize(ParallelType::TIDx);

  const std::string expected_kernel = R"(
__global__ void CUDAGeneratedKernel(Tensor<float, 1, 1> T0, double d0, double d1, Tensor<float, 1, 1> T1) {
  T1[((nvfuser_index_t)threadIdx.x)]
    = T0[(T0.stride[0] * ((nvfuser_index_t)threadIdx.x))]
    * (float) d1;
}
)";

  assertCUDAKernel(&fusion, expected_kernel);
  // TODO: executor change not implemented yet
  return;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({1000}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion({input, 10.0});
  testValidate(&fusion, outputs, {input, 10.0}, __LINE__, __FILE__);
}

TEST_F(KernelInputsTest, HoistToHost2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto scalar = IrBuilder::newScalar(DataType::Double);
  fusion.addInput(scalar);
  auto inv = div(fusion.oneVal(DataType::Double), scalar);
  auto inv_sqr = mul(inv, inv);
  auto tv1 = mul(tv0, inv_sqr);
  fusion.addOutput(tv1);
  fusion.manage("hoist_to_host", std::vector<Val*>{inv_sqr});

  tv1->axis(0)->parallelize(ParallelType::TIDx);

  // TODO: d2 below is not used, but it is generated in the kernel
  // write a dead code elimination pass to remove it
  const std::string expected_kernel = R"(
__global__ void CUDAGeneratedKernel(Tensor<float, 1, 1> T0, double d0, double d1, Tensor<float, 1, 1> T1) {
  T1[((nvfuser_index_t)threadIdx.x)]
    = T0[(T0.stride[0] * ((nvfuser_index_t)threadIdx.x))]
    * (float) d1;
}
)";

  assertCUDAKernel(&fusion, expected_kernel);
  // TODO: executor change not implemented yet
  return;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({1000}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto outputs = fe.runFusion({input, 10.0});
  testValidate(&fusion, outputs, {input, 10.0}, __LINE__, __FILE__);
}

} // namespace nvfuser
