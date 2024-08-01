class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[1024]", primals_2: "bf16[1024]", primals_3: "bf16[56, 1024, 1024]", primals_4: "bf16[56, 1024]", primals_5: "bf16[56, 1024]"):
         # File: /opt/pytorch/nvfuser/2146.py:18 in forward, code: x   = self.ln(x)
        convert_element_type: "f32[56, 1024, 1024]" = torch.ops.prims.convert_element_type.default(primals_3, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True);  convert_element_type = None
        getitem: "f32[56, 1024, 1]" = var_mean[0]
        getitem_1: "f32[56, 1024, 1]" = var_mean[1];  var_mean = None
        add: "f32[56, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[56, 1024, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[56, 1024, 1024]" = torch.ops.aten.sub.Tensor(primals_3, getitem_1)
        mul: "f32[56, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[56, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul, primals_1);  mul = None
        add_1: "f32[56, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_1, primals_2);  mul_1 = None
        convert_element_type_1: "bf16[56, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        
         # File: /opt/pytorch/nvfuser/2146.py:15 in modulate, code: return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        unsqueeze: "bf16[56, 1, 1024]" = torch.ops.aten.unsqueeze.default(primals_4, 1)
        add_2: "bf16[56, 1, 1024]" = torch.ops.aten.add.Tensor(unsqueeze, 1);  unsqueeze = None
        mul_2: "bf16[56, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1, add_2);  convert_element_type_1 = add_2 = None
        unsqueeze_1: "bf16[56, 1, 1024]" = torch.ops.aten.unsqueeze.default(primals_5, 1);  primals_5 = None
        add_3: "bf16[56, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_1);  mul_2 = unsqueeze_1 = None
        return [add_3, primals_1, primals_2, primals_3, primals_4, getitem_1, rsqrt]
        