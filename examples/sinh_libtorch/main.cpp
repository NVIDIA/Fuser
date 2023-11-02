#include <exceptions.h>
#include <executor.h>
#include <ops/arith.h>
#include <scheduler/all_schedulers.h>
#include <memory>

using namespace nvfuser;

at::Tensor sinh_nvfuser(const at::Tensor& input) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int dim = input.dim();
  auto dtype = input.scalar_type();
  auto x =
      TensorViewBuilder().ndims(dim).dtype(aten_to_data_type(dtype)).build();
  fusion.addInput(x);

  // Using equation sinh(x) = [ exp(x) - exp(-1) ] / 2
  auto output = div(sub(exp(x), exp(neg(x))), IrBuilder::create<Val>(2.0));
  fusion.addOutput(output);

  std::cout << "Create fusion:" << std::endl;
  fusion.print();

  auto lparams = schedulePointwise(&fusion, {input});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input}, lparams);
  auto outputs = fe.runFusion({input}, lparams);

  return outputs[0];
}

int main() {
  at::manual_seed(0);
  auto t = at::randn({5, 5}, at::kCUDA);
  auto expected = at::sinh(t);
  auto output = sinh_nvfuser(t);
  std::cout << "Expected:" << std::endl << expected << std::endl;
  std::cout << "Output:" << std::endl << output << std::endl;
  NVF_CHECK(at::allclose(expected, output));
  std::cout << "They match!" << std::endl;
}
