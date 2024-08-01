
# AOT ID: ['0_backward']
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


# kernel path: /tmp/torchinductor_root/es/cesqn2oafmfnug3ip3gg73mcana5lncsooq6xhdbitmvgxffnqyp.py
# Source Nodes: [x], Original ATen: [aten.mul, aten.native_layer_norm, aten.sum]
# x => add_1, convert_element_type_1, mul, mul_1, sub
triton_red_fused_mul_native_layer_norm_sum_0 = async_compile.triton('triton_red_fused_mul_native_layer_norm_sum_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[65536, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: 'i32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_layer_norm_sum_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 2, 'backend_hash': 'DECCFF780DB6236B5007BFE678ACB0D7841A3C7004B5A35A8B575CC067912A0D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mul_native_layer_norm_sum_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 57344
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp10 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (1048576*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr1 + (x0 + (1024*r2) + (1048576*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr2 + (r2 + (1024*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (r2 + (1024*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 - tmp6
        tmp9 = tmp7 * tmp8
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp9 * tmp11
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp12 + tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp0 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask, tmp20, _tmp19)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp19, None)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


# kernel path: /tmp/torchinductor_root/mo/cmog67eycujvdvwbzhz3myufdiv6ma2qblkxdy5fvgj7fpghasko.py
# Source Nodes: [add, x], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
# add => add_2
# x => convert_element_type
triton_red_fused_add_mul_native_layer_norm_native_layer_norm_backward_1 = async_compile.triton('triton_red_fused_add_mul_native_layer_norm_native_layer_norm_backward_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[65536, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*fp32', 5: '*fp32', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_layer_norm_native_layer_norm_backward_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 10, 'num_reduction': 2, 'backend_hash': 'DECCFF780DB6236B5007BFE678ACB0D7841A3C7004B5A35A8B575CC067912A0D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_mul_native_layer_norm_native_layer_norm_backward_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 57344
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x1 = (xindex // 1024)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp14 = tl.load(in_ptr4 + (x3), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x3), None, eviction_policy='evict_last')
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r2 + (1024*x1)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr3 + (r2 + (1024*x3)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = 1.0
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp5 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
        tmp13 = tmp12.to(tl.float32)
        tmp15 = tmp13 - tmp14
        tmp17 = tmp15 * tmp16
        tmp18 = tmp8 * tmp17
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask, tmp21, _tmp20)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp24 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp25 = tl.load(in_ptr1 + (r2 + (1024*x1)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp30 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp36 = tl.load(in_ptr3 + (r2 + (1024*x3)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp22 = 0.0009765625
        tmp23 = tmp16 * tmp22
        tmp26 = 1.0
        tmp27 = tmp25 + tmp26
        tmp28 = tmp24 * tmp27
        tmp29 = tmp28.to(tl.float32)
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tmp29 * tmp31
        tmp33 = 1024.0
        tmp34 = tmp32 * tmp33
        tmp35 = tmp34 - tmp10
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp37 - tmp14
        tmp39 = tmp38 * tmp16
        tmp40 = tmp39 * tmp20
        tmp41 = tmp35 - tmp40
        tmp42 = tmp23 * tmp41
        tmp43 = tmp42.to(tl.float32)
        tl.store(out_ptr2 + (r2 + (1024*x3)), tmp43, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/rf/crffupcq7qvmxvnselu5nktk2qiilfudqynf2hhxeketdqmvijq3.py
# Source Nodes: [add, x], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
# add => add_2
# x => convert_element_type
triton_red_fused_add_mul_native_layer_norm_native_layer_norm_backward_2 = async_compile.triton('triton_red_fused_add_mul_native_layer_norm_native_layer_norm_backward_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[131072, 512],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_layer_norm_native_layer_norm_backward_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'DECCFF780DB6236B5007BFE678ACB0D7841A3C7004B5A35A8B575CC067912A0D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_mul_native_layer_norm_native_layer_norm_backward_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 131072
    rnumel = 448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (458752*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + (1024*((r2 + (448*x1)) // 1024))), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr2 + (x0 + (1024*r2) + (458752*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr3 + (r2 + (448*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr4 + (r2 + (448*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 1.0
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 - tmp8
        tmp11 = tmp9 * tmp10
        tmp12 = tmp5 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask, tmp15, _tmp14)
        tmp16 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask, tmp18, _tmp17)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, None)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/mr/cmrw7g2eilnnbyxwri2a4zwumcx4lndggpsvn436ctwgrhac2t6l.py
# Source Nodes: [add, x], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
# add => add_2
# x => convert_element_type
triton_red_fused_add_mul_native_layer_norm_native_layer_norm_backward_3 = async_compile.triton('triton_red_fused_add_mul_native_layer_norm_native_layer_norm_backward_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*bf16', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_layer_norm_native_layer_norm_backward_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'DECCFF780DB6236B5007BFE678ACB0D7841A3C7004B5A35A8B575CC067912A0D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_mul_native_layer_norm_native_layer_norm_backward_3(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp4 = tmp2.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, getitem_1, rsqrt, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (1024, ), (1, ))
    assert_size_stride(primals_2, (1024, ), (1, ))
    assert_size_stride(primals_3, (56, 1024, 1024), (1048576, 1024, 1))
    assert_size_stride(primals_4, (56, 1024), (1024, 1))
    assert_size_stride(getitem_1, (56, 1024, 1), (1024, 1, 1))
    assert_size_stride(rsqrt, (56, 1024, 1), (1024, 1, 1))
    assert_size_stride(tangents_1, (56, 1024, 1024), (1048576, 1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((56, 1, 1024), (1024, 1024, 1), torch.bfloat16)
        buf1 = empty_strided_cuda((56, 1, 1024), (1024, 1024, 1), torch.bfloat16)
        # Source Nodes: [x], Original ATen: [aten.mul, aten.native_layer_norm, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_native_layer_norm_sum_0.run(tangents_1, primals_3, getitem_1, rsqrt, primals_1, primals_2, buf0, buf1, 57344, 1024, grid=grid(57344), stream=stream0)
        del primals_2
        buf8 = empty_strided_cuda((56, 1024, 1024), (1048576, 1024, 1), torch.bfloat16)
        # Source Nodes: [add, x], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_add_mul_native_layer_norm_native_layer_norm_backward_1.run(tangents_1, primals_4, primals_1, primals_3, getitem_1, rsqrt, buf8, 57344, 1024, grid=grid(57344), stream=stream0)
        del primals_1
        buf4 = empty_strided_cuda((1024, 128), (1, 1024), torch.float32)
        buf6 = empty_strided_cuda((1024, 128), (1, 1024), torch.float32)
        # Source Nodes: [add, x], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_add_mul_native_layer_norm_native_layer_norm_backward_2.run(tangents_1, primals_4, primals_3, getitem_1, rsqrt, buf4, buf6, 131072, 448, grid=grid(131072), stream=stream0)
        del getitem_1
        del primals_3
        del primals_4
        del rsqrt
        del tangents_1
        buf9 = empty_strided_cuda((1024, ), (1, ), torch.bfloat16)
        # Source Nodes: [add, x], Original ATen: [aten.add, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_add_mul_native_layer_norm_native_layer_norm_backward_3.run(buf4, buf9, 1024, 128, grid=grid(1024), stream=stream0)
        del buf4
        buf10 = empty_strided_cuda((1024, ), (1, ), torch.bfloat16)
        # Source Nodes: [add], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_red_fused_add_mul_native_layer_norm_native_layer_norm_backward_3.run(buf6, buf10, 1024, 128, grid=grid(1024), stream=stream0)
        del buf6
    return (buf9, buf10, buf8, reinterpret_tensor(buf1, (56, 1024), (1024, 1), 0), reinterpret_tensor(buf0, (56, 1024), (1024, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_2 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_3 = rand_strided((56, 1024, 1024), (1048576, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_4 = rand_strided((56, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_1 = rand_strided((56, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt = rand_strided((56, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((56, 1024, 1024), (1048576, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, getitem_1, rsqrt, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
