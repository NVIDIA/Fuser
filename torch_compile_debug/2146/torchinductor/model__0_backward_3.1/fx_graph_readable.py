class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[1024]", primals_2: "bf16[1024]", primals_3: "bf16[56, 1024, 1024]", primals_4: "bf16[56, 1024]", getitem_1: "f32[56, 1024, 1]", rsqrt: "f32[56, 1024, 1]", tangents_1: "bf16[56, 1024, 1024]"):
         # File: /opt/pytorch/nvfuser/2146.py:15 in modulate, code: return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        sum_1: "bf16[56, 1, 1024]" = torch.ops.aten.sum.dim_IntList(tangents_1, [1], True)
        squeeze: "bf16[56, 1024]" = torch.ops.aten.squeeze.dim(sum_1, 1);  sum_1 = None
        
         # File: /opt/pytorch/nvfuser/2146.py:18 in forward, code: x   = self.ln(x)
        sub: "f32[56, 1024, 1024]" = torch.ops.aten.sub.Tensor(primals_3, getitem_1)
        mul: "f32[56, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[56, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul, primals_1);  mul = None
        add_1: "f32[56, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_1, primals_2);  mul_1 = primals_2 = None
        convert_element_type_1: "bf16[56, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        
         # File: /opt/pytorch/nvfuser/2146.py:15 in modulate, code: return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        mul_3: "bf16[56, 1024, 1024]" = torch.ops.aten.mul.Tensor(tangents_1, convert_element_type_1);  convert_element_type_1 = None
        unsqueeze: "bf16[56, 1, 1024]" = torch.ops.aten.unsqueeze.default(primals_4, 1);  primals_4 = None
        add_2: "bf16[56, 1, 1024]" = torch.ops.aten.add.Tensor(unsqueeze, 1);  unsqueeze = None
        mul_4: "bf16[56, 1024, 1024]" = torch.ops.aten.mul.Tensor(tangents_1, add_2);  tangents_1 = add_2 = None
        sum_2: "bf16[56, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_3, [1], True);  mul_3 = None
        squeeze_1: "bf16[56, 1024]" = torch.ops.aten.squeeze.dim(sum_2, 1);  sum_2 = None
        
         # File: /opt/pytorch/nvfuser/2146.py:18 in forward, code: x   = self.ln(x)
        convert_element_type_2: "f32[56, 1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_4, torch.float32);  mul_4 = None
        convert_element_type_4: "f32[1024]" = torch.ops.prims.convert_element_type.default(primals_1, torch.float32);  primals_1 = None
        convert_element_type: "f32[56, 1024, 1024]" = torch.ops.prims.convert_element_type.default(primals_3, torch.float32);  primals_3 = None
        sub_1: "f32[56, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        mul_5: "f32[56, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
        mul_6: "f32[56, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_2, convert_element_type_4);  convert_element_type_4 = None
        mul_7: "f32[56, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_6, 1024)
        sum_3: "f32[56, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_6, [2], True)
        mul_8: "f32[56, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_6, mul_5);  mul_6 = None
        sum_4: "f32[56, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_8, [2], True);  mul_8 = None
        mul_9: "f32[56, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_5, sum_4);  sum_4 = None
        sub_2: "f32[56, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_7, sum_3);  mul_7 = sum_3 = None
        sub_3: "f32[56, 1024, 1024]" = torch.ops.aten.sub.Tensor(sub_2, mul_9);  sub_2 = mul_9 = None
        div: "f32[56, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None
        mul_10: "f32[56, 1024, 1024]" = torch.ops.aten.mul.Tensor(div, sub_3);  div = sub_3 = None
        mul_11: "f32[56, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_2, mul_5);  mul_5 = None
        sum_5: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_11, [0, 1]);  mul_11 = None
        sum_6: "f32[1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_2, [0, 1]);  convert_element_type_2 = None
        convert_element_type_6: "bf16[56, 1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_10, torch.bfloat16);  mul_10 = None
        convert_element_type_7: "bf16[1024]" = torch.ops.prims.convert_element_type.default(sum_5, torch.bfloat16);  sum_5 = None
        convert_element_type_8: "bf16[1024]" = torch.ops.prims.convert_element_type.default(sum_6, torch.bfloat16);  sum_6 = None
        return [convert_element_type_7, convert_element_type_8, convert_element_type_6, squeeze_1, squeeze]
        