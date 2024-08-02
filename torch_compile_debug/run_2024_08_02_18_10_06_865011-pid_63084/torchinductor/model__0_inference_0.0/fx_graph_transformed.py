class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[1024, 16, 16, 256]"):
         # File: /opt/pytorch/nvfuser/nhwc.py:5 in torch_group_norm, code: return x.sum(r_axis)
        sum_1: "f16[1024, 16, 16]" = torch.ops.aten.sum.dim_IntList(arg0_1, [-1]);  arg0_1 = None
        return (sum_1,)
        