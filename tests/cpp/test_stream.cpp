#include <gtest/gtest.h>

#include <ATen/ops/zeros_like.h>

#include <fusion.h>
#include <fusion_guard.h>
#include <ir/interface_nodes.h>
#include <ops/arith.h>
#include <tests/cpp/utils.h>

namespace nvfuser {

using StreamTest = NVFuserTest;

TEST_F(StreamTest, AddPerStream) {
  constexpr int64_t c = 3;
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigTensor(2);
  TensorView* out = add(in, in);
  fusion.addInput(in);
  fusion.addOutput(out);

  in->outer_split(0, c);
  in->axis(0)->parallelize(ParallelType::Stream);
  out->outer_split(0, c);
  out->axis(0)->parallelize(ParallelType::Stream);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
  auto in_tensor = at::randn({c * 2, 5}, options);
  auto out_tensor = at::zeros_like(in_tensor);

  KernelExecutor ke;
  ke.compile(&fusion, {in_tensor});
  ke.run({in_tensor}, {out_tensor});

  std::cout << "in_tensor: " << in_tensor << std::endl;
  std::cout << "out_tensor: " << out_tensor << std::endl;
}

} // namespace nvfuser
