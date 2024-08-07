import torch
import thunder

def torch_group_norm(x, r_axis = 1):
    return x.sum(r_axis)

def verify_group_norm(N=32,
                      C=128,
                      H=256,
                      W=256,
                      # memory_format=torch.contiguous_format): # 0.103 ms
                      memory_format=torch.channels_last): # 0.318 ms
    # create data
    x_shape = (N, C, H, W)

    x = torch.randn(x_shape, dtype=torch.float16, device='cuda')
    x = x.to(memory_format=memory_format)

    print(x.shape)
    print(x.stride())
    compile_group_norm = torch.compile(torch_group_norm)
    y_compile = compile_group_norm(x)


def verify_group_norm2(N=32,
                      C=128,
                      H=256,
                      W=256,
                      # memory_format=torch.contiguous_format): # 0.103 ms
                      memory_format=torch.channels_last): # 0.318 ms
    # create data
    x_shape = (N, H, W, C)

    x = torch.randn(x_shape, dtype=torch.float16, device='cuda')

    print(x.shape)
    print(x.stride())
    compile_group_norm = torch.compile(torch_group_norm)
    y_compile = compile_group_norm(x, -1)

# nv_enable_bookend=False NVFUSER_DUMP=scheduler_params,cuda_to_file,fusion_ir_presched,fusion_ir_preseg,segmenter_logging,transform_propagator,pre_segmenter_logging,python_definition python group_norm_nhwc.py 2>&1 |tee 1.log
# nv_enable_bookend=False NVFUSER_DUMP=scheduler_params,cuda_to_file,fusion_ir_presched,python_definition python group_norm_nhwc.py 2>&1 |tee 1.log
# NVFUSER_PROF=print NVFUSER_DUMP=scheduler_params,cuda_to_file,fusion_ir python group_norm_nhwc.py 2>&1 |tee 1.log
# TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 nsys nvprof --print-gpu-trace python group_norm_nhwc.py 2>&1 |tee nsys.log
# TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 python group_norm_nhwc.py
# nsys nvprof --print-gpu-trace python torch_compile_debug/run_2024_07_31_14_50_25_170443-pid_137224/torchinductor/model__0_forward_1.0/output_code.py

if __name__ == "__main__":
  # verify_group_norm(N=1024, C=256, H=16, W=16)
  verify_group_norm2(N=1024, C=256, H=16, W=16)
