
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

    
    
    def forward(self, primals_1, primals_2, primals_3):
        view = torch.ops.aten.view.default(primals_3, [1024, 32, 8, 256])
        convert_element_type = torch.ops.prims.convert_element_type.default(view, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2, 3], correction = 0, keepdim = True);  convert_element_type = None
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
        sub = torch.ops.aten.sub.Tensor(view, getitem_1);  view = None
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        view_1 = torch.ops.aten.view.default(mul, [1024, 256, 16, 16]);  mul = None
        unsqueeze = torch.ops.aten.unsqueeze.default(primals_1, 0);  primals_1 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(primals_2, 0)
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        mul_1 = torch.ops.aten.mul.Tensor(view_1, unsqueeze_5);  view_1 = unsqueeze_5 = None
        add_1 = torch.ops.aten.add.Tensor(mul_1, unsqueeze_2);  mul_1 = unsqueeze_2 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(add_1, torch.float16);  add_1 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(getitem_1, torch.float16);  getitem_1 = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(rsqrt, torch.float16);  rsqrt = None
        squeeze = torch.ops.aten.squeeze.dims(convert_element_type_2, [2, 3]);  convert_element_type_2 = None
        squeeze_1 = torch.ops.aten.squeeze.dims(convert_element_type_3, [2, 3]);  convert_element_type_3 = None
        return [convert_element_type_1, primals_2, primals_3, squeeze, squeeze_1]
        
def load_args(reader):
    buf0 = reader.storage(None, 512, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf0, (256,), dtype=torch.float16, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 512, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1, (256,), dtype=torch.float16, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf2, (1024, 256, 16, 16), dtype=torch.float16, is_leaf=True)  # primals_3
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)