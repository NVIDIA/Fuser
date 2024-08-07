
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
# NVIDIA A100 80GB PCIe : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1):
        sum_1 = torch.ops.aten.sum.dim_IntList(arg0_1, [1]);  arg0_1 = None
        return (sum_1,)
        
def load_args(reader):
    buf0 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf0, (1024, 256, 16, 16), (65536, 1, 4096, 256), dtype=torch.float16, is_leaf=True)  # arg0_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)