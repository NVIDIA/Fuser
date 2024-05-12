import torch
import torch.nn as nn
import torch.nn.functional as F
import thunder

class Net(nn.Module):
    def __init__(
        self,
        config):
        super(Net, self).__init__()
        self.ln = nn.LayerNorm(config['hidden_units'])
        
    def modulate(self, x, scale, shift):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x, scale, shift):
        x   = self.ln(x)
        out = self.modulate(x, scale, shift)
        return out

config = {'hidden_units': 1024, 'seq_len': 1024, 'batch_size': 56}

net = Net(config)
net.cuda()
net.to(dtype=torch.bfloat16)

from thunder.executors.sdpaex import sdpa_ex
executors = [sdpa_ex, thunder.nvfuser_executor, thunder.pytorch_executor]

network_fn = thunder.jit(net, executors=executors)
#network_fn = torch.compile(net)

bench_iters = 10
profile_batch = 5

input_shapes    = [(config['batch_size'], config['seq_len'], config['hidden_units']),
                   (config['batch_size'], config['hidden_units']),
                   (config['batch_size'], config['hidden_units'])]

def generate_io_tensor(net, input_shapes):
    input_tensors = []

    for shape in input_shapes:
        tensor = torch.rand(shape, dtype=torch.bfloat16, requires_grad=True, device='cuda')
        input_tensors.append(tensor)

    target_tensor_size = net(*input_tensors).size()
    target_tensor = torch.rand(target_tensor_size, dtype=torch.bfloat16, device='cuda')

    return input_tensors, target_tensor

for idx in range(bench_iters):
    input_tensors, target_tensor = generate_io_tensor(net, input_shapes)
    
    ## Profiling code BEGIN
    if idx == profile_batch:
        print("BEGIN PROFILING ITERATION")
        torch.cuda.cudart().cudaProfilerStart()
        torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()
    ## Profiling code END

    outputs = network_fn(*input_tensors)
    outputs.backward(target_tensor)

    ## Profiling code BEGIN
    if idx == profile_batch:
        print("END PROFILING ITERATION")
        torch.cuda.cudart().cudaProfilerStop()
        torch.autograd.profiler.emit_nvtx().__exit__(None, None, None)
