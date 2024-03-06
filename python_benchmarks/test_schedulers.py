from test_layernorm_bwd import layernorm_bwd_fusion
from test_rmsnorm_bwd import rmsnorm_bwd_fusion
from test_dropout_layernorm_bwd import dropout_layernorm_bwd_fusion
from nvfuser import DataType, FusionDefinition
from test_gelu_bwd_reduction import gelu_bwd_reduction_fusion
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
import torch
from global_params import generate_input_sizes

def validation_debug(size):
    dtype = torch.float

    inputs = torch.randn(*size, device="cuda", dtype=dtype, requires_grad=True)
    grads = torch.randn(*size, device="cuda", dtype=dtype)
    weights = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)
    bias = torch.randn(size[1], device="cuda", dtype=dtype, requires_grad=True)

    mean = inputs.to(torch.float).mean(dim=-1)
    variance = inputs.to(torch.float).var(dim=-1, unbiased=False)
    invstd = (1.0 / torch.sqrt(variance + 1e-5)).unsqueeze(1)

    # squared_mean = (inputs.to(torch.float) ** 2).mean(1, keepdim=True)
    # rms_eps = torch.sqrt(squared_mean + 1e-5)
    # dropout_p = 0.1
    # dropout_mask = torch.lt(torch.rand(*size, device="cuda"), 1 - dropout_p)

    # with FusionDefinition() as fd:
    #     with FusionDefinition() as fd:
    #         gelu_bwd_reduction_fusion(
    #             fd, torch_dtype_to_nvfuser_dtype(dtype), reduction_axis=1
    #         )
    # outputs = fd.execute([inputs, grads, bias])

    with FusionDefinition() as fd:
        layernorm_bwd_fusion(fd, torch_dtype_to_nvfuser_dtype(dtype))
    eager_output = torch.nn.functional.layer_norm(
                inputs.to(torch.double),
                inputs.shape[1:],
                weight=weights.to(torch.double),
                bias=bias.to(torch.double),
            )
    eager_output.backward(grads.to(torch.double))
    print (size)
    fd.validate([inputs, grads, mean, invstd, weights], [inputs.grad, weights.grad, bias.grad])
    # outputs = fd.execute([inputs, grads, mean, invstd, weights])

for size in generate_input_sizes():
    validation_debug(size)