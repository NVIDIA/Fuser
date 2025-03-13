import thunder
import torch
from enum import Enum
import argparse

class Activation(Enum):
    Gelu = "Gelu"
    GeluTaylor = "GeluTaylor"
    HardSwish = "HardSwish"
    Identity = "Identity"
    LeakyReLU = "LeakyReLU"
    ReLu = "ReLu"
    Sigmoid = "Sigmoid"
    Silu = "Silu"
    Tanh = "Tanh"

# Function to validate and parse the Enum value
def activation_type(value):
    try:
        return Activation[value]
    except KeyError:
        raise argparse.ArgumentTypeError(f"Invalid activation type: {value}. "
                                         f"Valid options are: {[e.name for e in Activation]}")

# 1439296 | nvjet_tst_128x256_64x4_2x1_v_bz_coopA_bias_TNN
# 92992 | nvfuser_pointwise_f0_c1_r0_g0
def gemm_gelu(a, b, c):
    return torch.nn.functional.gelu(torch.nn.functional.linear(a, b, c))

# 1438080 | nvjet_tst_128x256_64x4_2x1_v_bz_coopA_bias_TNN
# 79744 | nvfuser_pointwise_f0_c1_r0_g0
def gemm_gelu_taylor(a, b, c):
    return torch.nn.functional.gelu(torch.nn.functional.linear(a, b, c), approximate="tanh")

# 1426016 | nvjet_tst_128x256_64x4_2x1_v_bz_coopA_bias_TNN                                                      
# 91872   | nvfuser_pointwise_f0_c1_r0_g0
def gemm_hardswish(a, b, c):
    return torch.nn.functional.hardswish(torch.nn.functional.linear(a, b, c))

# 1426528 | nvjet_tst_128x256_64x4_2x1_v_bz_coopA_bias_TNN
def gemm_identity(a, b, c):
    return torch.nn.functional.linear(a, b, c)

# 1432544 | nvjet_tst_128x256_64x4_2x1_v_bz_coopA_bias_TNN
# 71936 | nvfuser_pointwise_f0_c1_r0_g1
def gemm_leaky_relu(a, b, c):
    return torch.nn.functional.leaky_relu(torch.nn.functional.linear(a, b, c))

# 1426432 | nvjet_tst_128x256_64x4_2x1_v_bz_coopA_bias_TNN
# 73120 | nvfuser_pointwise_f0_c1_r0_g1
def gemm_relu(a, b, c):
    return torch.nn.functional.relu(torch.nn.functional.linear(a, b, c))

# 1436576 | nvjet_tst_128x256_64x4_2x1_v_bz_coopA_bias_TNN
# 112384 | nvfuser_pointwise_f0_c1_r0_g0
def gemm_sigmoid(a, b, c):
    return torch.nn.functional.sigmoid(torch.nn.functional.linear(a, b, c))

# 1429152 | nvjet_tst_128x256_64x4_2x1_v_bz_coopA_bias_TNN
# 121344 | nvfuser_pointwise_f0_c1_r0_g0
def gemm_silu(a, b, c):
    return torch.nn.functional.silu(torch.nn.functional.linear(a, b, c))

# 1426720 | nvjet_tst_128x256_64x4_2x1_v_bz_coopA_bias_TNN
# 73600 | nvfuser_pointwise_f0_c1_r0_g0
def gemm_tanh(a, b, c):
    return torch.nn.functional.tanh(torch.nn.functional.linear(a, b, c))

def run(fn, enable_matmul_fusion: bool = False):
    m = 8192
    n = 8192
    k = 8192
    a = torch.randn(m, k, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(n, k, dtype=torch.bfloat16, device='cuda')
    c = torch.randn(n, dtype=torch.bfloat16, device='cuda')

    if enable_matmul_fusion:
        compiled_func = thunder.jit(
            fn,
            nv_enable_matmul=True,
            nv_enable_options=["fuse_matmul"],
            nv_disable_options=["matmul_expr_eval", "kernel_reuse"],
        )
    else :
        compiled_func = thunder.jit(fn)

    return compiled_func(a, b, c)

enum_to_fn = {
    Activation.Gelu: gemm_gelu,
    Activation.GeluTaylor: gemm_gelu_taylor,
    Activation.HardSwish: gemm_hardswish,
    Activation.Identity: gemm_identity,
    Activation.LeakyReLU: gemm_leaky_relu,
    Activation.ReLu: gemm_relu,
    Activation.Sigmoid: gemm_sigmoid,
    Activation.Silu: gemm_silu,
    Activation.Tanh: gemm_tanh
}

# Set up argparse
parser = argparse.ArgumentParser(description="Select an activation function.")
parser.add_argument("--activation", type=activation_type, required=True,
                    help=f"Choose an activation type. Options: {[e.name for e in Activation]}")

# Parse arguments
args = parser.parse_args()
print(f"Selected activation: {args.activation}")
run(enum_to_fn[args.activation])
