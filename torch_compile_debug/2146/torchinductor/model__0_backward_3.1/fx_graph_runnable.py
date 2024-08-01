
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.error_on_nested_jit_trace = False
torch._inductor.config.debug = True




isolate_fails_code_str = None



# torch version: 2.5.0a0+git1439bd3
# torch cuda version: 12.6
# torch git version: 1439bd3c9c0a48d526bc20021486266591c0f480


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2024 NVIDIA Corporation 
# Built on Fri_Jun_14_16:34:21_PDT_2024 
# Cuda compilation tools, release 12.6, V12.6.20 
# Build cuda_12.6.r12.6/compiler.34431801_0 

# GPU Hardware Info: 
# NVIDIA H100 80GB HBM3 : 8 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, getitem_1, rsqrt, tangents_1):
        sum_1 = torch.ops.aten.sum.dim_IntList(tangents_1, [1], True)
        squeeze = torch.ops.aten.squeeze.dim(sum_1, 1);  sum_1 = None
        sub = torch.ops.aten.sub.Tensor(primals_3, getitem_1)
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, primals_1);  mul = None
        add_1 = torch.ops.aten.add.Tensor(mul_1, primals_2);  mul_1 = primals_2 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        mul_3 = torch.ops.aten.mul.Tensor(tangents_1, convert_element_type_1);  convert_element_type_1 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(primals_4, 1);  primals_4 = None
        add_2 = torch.ops.aten.add.Tensor(unsqueeze, 1);  unsqueeze = None
        mul_4 = torch.ops.aten.mul.Tensor(tangents_1, add_2);  tangents_1 = add_2 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(mul_3, [1], True);  mul_3 = None
        squeeze_1 = torch.ops.aten.squeeze.dim(sum_2, 1);  sum_2 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(mul_4, torch.float32);  mul_4 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(primals_1, torch.float32);  primals_1 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(primals_3, torch.float32);  primals_3 = None
        sub_1 = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        mul_5 = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
        mul_6 = torch.ops.aten.mul.Tensor(convert_element_type_2, convert_element_type_4);  convert_element_type_4 = None
        mul_7 = torch.ops.aten.mul.Tensor(mul_6, 1024)
        sum_3 = torch.ops.aten.sum.dim_IntList(mul_6, [2], True)
        mul_8 = torch.ops.aten.mul.Tensor(mul_6, mul_5);  mul_6 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(mul_8, [2], True);  mul_8 = None
        mul_9 = torch.ops.aten.mul.Tensor(mul_5, sum_4);  sum_4 = None
        sub_2 = torch.ops.aten.sub.Tensor(mul_7, sum_3);  mul_7 = sum_3 = None
        sub_3 = torch.ops.aten.sub.Tensor(sub_2, mul_9);  sub_2 = mul_9 = None
        div = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None
        mul_10 = torch.ops.aten.mul.Tensor(div, sub_3);  div = sub_3 = None
        mul_11 = torch.ops.aten.mul.Tensor(convert_element_type_2, mul_5);  mul_5 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(mul_11, [0, 1]);  mul_11 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(convert_element_type_2, [0, 1]);  convert_element_type_2 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(mul_10, torch.bfloat16);  mul_10 = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(sum_5, torch.bfloat16);  sum_5 = None
        convert_element_type_8 = torch.ops.prims.convert_element_type.default(sum_6, torch.bfloat16);  sum_6 = None
        return [convert_element_type_7, convert_element_type_8, convert_element_type_6, squeeze_1, squeeze]
        
def load_args(reader):
    buf0 = reader.storage(None, 2048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (1024,), dtype=torch.bfloat16, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 2048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (1024,), dtype=torch.bfloat16, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 117440512, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (56, 1024, 1024), dtype=torch.bfloat16, is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 114688, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (56, 1024), dtype=torch.bfloat16, is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 229376, device=device(type='cuda', index=0))
    reader.tensor(buf4, (56, 1024, 1), is_leaf=True)  # getitem_1
    buf5 = reader.storage(None, 229376, device=device(type='cuda', index=0))
    reader.tensor(buf5, (56, 1024, 1), is_leaf=True)  # rsqrt
    buf6 = reader.storage(None, 117440512, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf6, (56, 1024, 1024), dtype=torch.bfloat16, is_leaf=True)  # tangents_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)