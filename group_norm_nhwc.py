import torch
import thunder

def torch_group_norm(x, g, w, b, eps):
    y = torch.nn.functional.group_norm(x, g, w, b, eps)
    return y

# def torch_group_norm(x, g, w, b, eps):
#     initial_x_dtype = x.dtype
#     if initial_x_dtype != torch.float32:
#         x = x.to(dtype=torch.float32)
#     y = torch.nn.functional.group_norm(x, g, w, b, eps)
#     if initial_x_dtype != torch.float32:
#         y = y.to(dtype=initial_x_dtype)    
#     return y


def verify_group_norm(N=32,
                      C=128,
                      H=256,
                      W=256,
                      G=32,
                      has_weight_bias=True,
                      xdtype=torch.float16,
                      wdtype=torch.float16,
                      eps=1e-5,
                      # memory_format=torch.contiguous_format): # 0.103 ms
                      memory_format=torch.channels_last): # 0.318 ms
    # create data
    x_shape = (N, C, H, W)
    w_shape = (C,)
    if has_weight_bias:
      weight = torch.rand(w_shape,
                          dtype=wdtype,
                          device='cuda',
                          requires_grad=True)
      bias = torch.rand(w_shape,
                        dtype=wdtype,
                        device='cuda',
                        requires_grad=True)
    else:
      weight = None
      bias = None
    x = torch.randn(x_shape, dtype=xdtype, device='cuda')
    x = x.to(memory_format=memory_format)
    x.requires_grad_(True)
    print(x.shape)
    print(x.stride())
    thunder_group_norm = thunder.jit(torch_group_norm, nv_enable_bookend=False)
    compile_group_norm = torch.compile(torch_group_norm)
    y_torch = torch_group_norm(x, G, weight, bias, eps)
    y_compile = compile_group_norm(x, G, weight, bias, eps)
    y_thunder = thunder_group_norm(x, G, weight, bias, eps)
    # compare
    torch.testing.assert_close(y_thunder, y_torch, atol=4e-2, rtol=0)
    print(thunder.last_traces(thunder_group_norm)[-1])

# nv_enable_bookend=False NVFUSER_DUMP=scheduler_params,cuda_to_file,fusion_ir_presched,fusion_ir_preseg,segmenter_logging,transform_propagator,pre_segmenter_logging,python_definition python group_norm_nhwc.py 2>&1 |tee 1.log
# nv_enable_bookend=False NVFUSER_DUMP=scheduler_params,cuda_to_file,fusion_ir_presched,python_definition python group_norm_nhwc.py 2>&1 |tee 1.log
# NVFUSER_PROF=print NVFUSER_DUMP=scheduler_params,cuda_to_file,fusion_ir python group_norm_nhwc.py 2>&1 |tee 1.log
# TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 nsys nvprof --print-gpu-trace python group_norm_nhwc.py 2>&1 |tee nsys.log
# TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 python group_norm_nhwc.py
# nsys nvprof --print-gpu-trace python torch_compile_debug/run_2024_07_31_14_50_25_170443-pid_137224/torchinductor/model__0_forward_1.0/output_code.py

# N=1024, C=256, H=16, W=16, G=32

# nchw
# nvfuser: 0.103 ms
# triton: 0.065 + 0.088 = 0.144 ms

# nhwc,channel last
# nvfuser: 0.318 ms
# triton: 0.056 + 0.088 = 0.144 ms

if __name__ == "__main__":
  verify_group_norm(N=1024, C=256, H=16, W=16, G=32, has_weight_bias=True)
  # verify_group_norm(N=256, C=128, H=8, W=8, G=32, has_weight_bias=True)
