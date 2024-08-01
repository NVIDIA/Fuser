
# AOT ID: ['0_forward']
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_root/2y/c2ychdp5enjkcmdrnrplzna4nob6w4fjoe4x6vcm3pf7jclrrcqu.py
# Source Nodes: [add, mul, out, x], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
# add => add_2
# mul => mul_2
# out => add_3
# x => add, add_1, convert_element_type, convert_element_type_1, mul, mul_1, rsqrt, sub, var_mean
triton_per_fused_add_mul_native_layer_norm_0 = async_compile.triton('triton_per_fused_add_mul_native_layer_norm_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[65536, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*fp32', 7: '*bf16', 8: 'i32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_0', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'DECCFF780DB6236B5007BFE678ACB0D7841A3C7004B5A35A8B575CC067912A0D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_mul_native_layer_norm_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 57344
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    x3 = (xindex // 1024)
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), None).to(tl.float32)
    tmp22 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last').to(tl.float32)
    tmp25 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last').to(tl.float32)
    tmp29 = tl.load(in_ptr3 + (r1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.float32)
    tmp33 = tl.load(in_ptr4 + (r1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp4 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp4, 0))
    tmp7 = tl.full([1], 1024, tl.int32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 / tmp8
    tmp10 = tmp2 - tmp9
    tmp11 = tmp10 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp15 = 1024.0
    tmp16 = tmp14 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp1 - tmp9
    tmp21 = tmp20 * tmp19
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 * tmp23
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp24 + tmp26
    tmp28 = tmp27.to(tl.float32)
    tmp30 = 1.0
    tmp31 = tmp29 + tmp30
    tmp32 = tmp28 * tmp31
    tmp34 = tmp32 + tmp33
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp19, None)
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp34, None)
    tl.store(out_ptr0 + (x0), tmp9, None)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (1024, ), (1, ))
    assert_size_stride(primals_2, (1024, ), (1, ))
    assert_size_stride(primals_3, (56, 1024, 1024), (1048576, 1024, 1))
    assert_size_stride(primals_4, (56, 1024), (1024, 1))
    assert_size_stride(primals_5, (56, 1024), (1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((56, 1024, 1), (1024, 1, 1), torch.float32)
        buf1 = empty_strided_cuda((56, 1024, 1), (1024, 1, 57344), torch.float32)
        buf3 = reinterpret_tensor(buf1, (56, 1024, 1), (1024, 1, 1), 0); del buf1  # reuse
        buf4 = empty_strided_cuda((56, 1024, 1024), (1048576, 1024, 1), torch.bfloat16)
        # Source Nodes: [add, mul, out, x], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mul_native_layer_norm_0.run(buf3, primals_3, primals_1, primals_2, primals_4, primals_5, buf0, buf4, 57344, 1024, grid=grid(57344), stream=stream0)
        del primals_5
    return (buf4, primals_1, primals_2, primals_3, primals_4, buf0, buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_2 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_3 = rand_strided((56, 1024, 1024), (1048576, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_4 = rand_strided((56, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_5 = rand_strided((56, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
