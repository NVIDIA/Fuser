// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ir/cloner.h>
#include <ir/iostream.h>
#include <polymorphic_value.h>

namespace nvfuser {

GetMetaData::GetMetaData(IrBuilderPasskey passkey, Val* output, Val* input)
    : Expr(passkey) {
  addOutput(output);
  addInput(input);
  TORCH_INTERNAL_ASSERT(
      out()->dtype() == metaDataTypeOf(in()),
      "Data type mismatch for GetMetaData")
}

std::string GetMetaData::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << " = getMetaData("
                          << in()->toString() << ")\n";
  return ss.str();
}

std::string GetMetaData::toInlineString(int indent_size) const {
  std::stringstream ss;
  ss << "getMetaData(" << in()->toInlineString() << ")";
  return ss.str();
}

std::vector<PolymorphicValue> GetMetaData::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  TORCH_INTERNAL_ASSERT(inputs.size() == 1, "GetMetaData expects 1 input");
  TORCH_INTERNAL_ASSERT(
      in()->isA<TensorView>(),
      "Currently, GetMetaData only supports TensorView");
  TensorView* tv = in()->as<TensorView>();
  if (tv->getMemoryType() == MemoryType::Shared) {
    // Smem tensor is defined locally as a pointer. It is impossible to know the
    // actual address, but using nullptr is a good approximation.
    return {PolymorphicValue(Pointer(nullptr, tv->dtype()))};
  }

  at::Tensor input = inputs.at(0).as<at::Tensor>();

  Struct<PolymorphicValue> concrete_value;
  concrete_value["data"] =
      PolymorphicValue(Pointer(input.data_ptr(), tv->dtype()));
  concrete_value["logical_size"] = PolymorphicValue(input.sizes().vec());
  concrete_value["logical_stride"] = PolymorphicValue(input.strides().vec());
  concrete_value["alloc_size"] = PolymorphicValue(input.sizes().vec());
  concrete_value["alloc_stride"] = PolymorphicValue(input.strides().vec());
  return {PolymorphicValue(concrete_value)};
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GetMetaData)

} // namespace nvfuser
