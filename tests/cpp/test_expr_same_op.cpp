#include <gtest/gtest.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>

namespace nvfuser {

TEST(ExprSameOpTest, BlockQuantizationOpNullptrAttributes) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          std::vector<IterDomain*>{IrBuilder::create<IterDomain>(
              IrBuilder::create<Int>(0), IrBuilder::create<Int>(10))}),
      DataType::Float);
  
  auto output = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          std::vector<IterDomain*>{IrBuilder::create<IterDomain>(
              IrBuilder::create<Int>(0), IrBuilder::create<Int>(10))}),
      DataType::Float);

  auto output_scales = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          std::vector<IterDomain*>{IrBuilder::create<IterDomain>(
              IrBuilder::create<Int>(0), IrBuilder::create<Int>(10))}),
      DataType::Float);

  // Create BlockQuantizationOp with nullptr logical_index (default)
  auto op1 = IrBuilder::create<BlockQuantizationOp>(
      output_scales, output, tv0, nullptr, nullptr);

  // Create another identical BlockQuantizationOp
  auto op2 = IrBuilder::create<BlockQuantizationOp>(
      output_scales, output, tv0, nullptr, nullptr);

  // This should not crash and return true
  EXPECT_TRUE(op1->sameOp(op2));
}

} // namespace nvfuser
