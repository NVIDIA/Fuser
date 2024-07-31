class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f16[256]", primals_2: "f16[256]", primals_3: "f16[1024, 256, 16, 16]"):
         # File: /opt/pytorch/nvfuser/group_norm_nhwc.py:5 in torch_group_norm, code: y = torch.nn.functional.group_norm(x, g, w, b, eps)
        clone: "f16[1024, 256, 16, 16]" = torch.ops.aten.clone.default(primals_3, memory_format = torch.contiguous_format)
        view: "f16[1024, 32, 8, 256]" = torch.ops.aten.view.default(clone, [1024, 32, 8, 256]);  clone = None
        convert_element_type: "f32[1024, 32, 8, 256]" = torch.ops.prims.convert_element_type.default(view, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2, 3], correction = 0, keepdim = True);  convert_element_type = None
        getitem: "f32[1024, 32, 1, 1]" = var_mean[0]
        getitem_1: "f32[1024, 32, 1, 1]" = var_mean[1];  var_mean = None
        add: "f32[1024, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1024, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[1024, 32, 8, 256]" = torch.ops.aten.sub.Tensor(view, getitem_1);  view = None
        mul: "f32[1024, 32, 8, 256]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        view_1: "f32[1024, 256, 16, 16]" = torch.ops.aten.view.default(mul, [1024, 256, 16, 16]);  mul = None
        unsqueeze: "f16[1, 256]" = torch.ops.aten.unsqueeze.default(primals_1, 0);  primals_1 = None
        unsqueeze_1: "f16[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        unsqueeze_2: "f16[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        unsqueeze_3: "f16[1, 256]" = torch.ops.aten.unsqueeze.default(primals_2, 0)
        unsqueeze_4: "f16[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
        unsqueeze_5: "f16[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        mul_1: "f32[1024, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_1, unsqueeze_5);  view_1 = unsqueeze_5 = None
        add_1: "f32[1024, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_2);  mul_1 = unsqueeze_2 = None
        convert_element_type_1: "f16[1024, 256, 16, 16]" = torch.ops.prims.convert_element_type.default(add_1, torch.float16);  add_1 = None
        convert_element_type_2: "f16[1024, 32, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1, torch.float16);  getitem_1 = None
        convert_element_type_3: "f16[1024, 32, 1, 1]" = torch.ops.prims.convert_element_type.default(rsqrt, torch.float16);  rsqrt = None
        squeeze: "f16[1024, 32]" = torch.ops.aten.squeeze.dims(convert_element_type_2, [2, 3]);  convert_element_type_2 = None
        squeeze_1: "f16[1024, 32]" = torch.ops.aten.squeeze.dims(convert_element_type_3, [2, 3]);  convert_element_type_3 = None
        return [convert_element_type_1, primals_2, primals_3, squeeze, squeeze_1]
        