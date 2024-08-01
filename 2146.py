import torch
import torch.nn as nn
import torch.nn.functional as F
import thunder
import logging

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
# executors = [sdpa_ex, thunder.nvfuser_executor, thunder.pytorch_executor]
executors = [thunder.nvfuser_executor]

network_fn = thunder.jit(net, executors=executors)

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

# torch._logging.set_logs(dynamo = logging.INFO)
torch._inductor.config.debug = True
# torch._inductor.config.unique_kernel_names = True

network_fn = torch.compile(net)

input_tensors, target_tensor = generate_io_tensor(net, input_shapes)
outputs = network_fn(*input_tensors)
outputs.backward(target_tensor)

# TORCHINDUCTOR_UNIQUE_KERNEL_NAMES = 1 TORCH_COMPILE_DEBUG=1 