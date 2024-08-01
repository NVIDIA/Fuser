
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
torch._functorch.config.debug_partitioner = True
torch._functorch.config.unlift_effect_tokens = True



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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5):
        convert_element_type = torch.ops.prims.convert_element_type.default(primals_3, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True);  convert_element_type = None
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
        sub = torch.ops.aten.sub.Tensor(primals_3, getitem_1)
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, primals_1);  mul = None
        add_1 = torch.ops.aten.add.Tensor(mul_1, primals_2);  mul_1 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(primals_4, 1)
        add_2 = torch.ops.aten.add.Tensor(unsqueeze, 1);  unsqueeze = None
        mul_2 = torch.ops.aten.mul.Tensor(convert_element_type_1, add_2);  convert_element_type_1 = add_2 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(primals_5, 1);  primals_5 = None
        add_3 = torch.ops.aten.add.Tensor(mul_2, unsqueeze_1);  mul_2 = unsqueeze_1 = None
        return [add_3, primals_1, primals_2, primals_3, primals_4, getitem_1, rsqrt]
        
def load_args(reader):
    buf0 = reader.storage(None, 2048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (1024,), dtype=torch.bfloat16, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 2048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (1024,), dtype=torch.bfloat16, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 117440512, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (56, 1024, 1024), dtype=torch.bfloat16, is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 114688, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (56, 1024), dtype=torch.bfloat16, is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 114688, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf4, (56, 1024), dtype=torch.bfloat16, is_leaf=True)  # primals_5
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)