import torch
from nvfuser import FusionDefinition, DataType
import time

def nvfuser_fusion_id1(fd : FusionDefinition) -> None :
    T0 = fd.define_tensor(shape=[1, 6, 2048], contiguity=[None, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 1, 0])
    T1 = fd.define_tensor(shape=[2048], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
    T2 = fd.define_tensor(shape=[32], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
    T3 = fd.define_tensor(shape=[512, 2048], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
    T4 = fd.define_tensor(shape=[2048, 2048], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
    T5 = fd.define_tensor(shape=[1, 1, 6, 6], contiguity=[True, None, None, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 1, 3, 0])
    T6 = fd.define_tensor(shape=[512, 2048], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
    T7 = fd.ops.cast(T0, dtype=DataType.Float)
    S8 = fd.define_scalar(2.00000, dtype=DataType.Double)
    T9 = fd.ops.pow(T7, S8)
    T10 = fd.ops.sum(T9, dims=[2], keepdim=False, dtype=DataType.Null)
    T15 = fd.ops.broadcast_in_dim(T10, shape=[1, 6, 1], broadcast_dims=[0, 1])
    S16 = fd.define_scalar(2048.00, dtype=DataType.Double)
    S17 = fd.ops.reciprocal(S16)
    T18 = fd.ops.mul(T15, S17)
    S19 = fd.define_scalar(1.00000e-05, dtype=DataType.Double)
    T20 = fd.ops.add(T18, S19)
    T21 = fd.ops.rsqrt(T20)
    T26 = fd.ops.broadcast_in_dim(T21, shape=[1, 6, 2048], broadcast_dims=[0, 1, 2])
    T27 = fd.ops.mul(T7, T26)
    S28 = fd.define_scalar(6, dtype=DataType.Int)
    S29 = fd.define_scalar(0, dtype=DataType.Int)
    S30 = fd.define_scalar(1, dtype=DataType.Int)
    T31 = fd.ops.iota(S28, S29, S30, dtype=DataType.Int)
    T36 = fd.ops.broadcast_in_dim(T1, shape=[1, 6, 2048], broadcast_dims=[2])
    T40 = fd.ops.broadcast_in_dim(T31, shape=[1, 6], broadcast_dims=[1])
    T45 = fd.ops.broadcast_in_dim(T2, shape=[1, 32, 1], broadcast_dims=[1])
    T46 = fd.ops.cast(T36, dtype=DataType.Float)
    T51 = fd.ops.broadcast_in_dim(T40, shape=[1, 1, 6], broadcast_dims=[0, 2])
    T52 = fd.ops.cast(T45, dtype=DataType.Float)
    T53 = fd.ops.mul(T46, T27)
    T54 = fd.ops.cast(T51, dtype=DataType.Float)
    T59 = fd.ops.broadcast_in_dim(T52, shape=[1, 32, 1], broadcast_dims=[0, 1, 2])
    T60 = fd.ops.cast(T53, dtype=DataType.BFloat16)
    T61 = fd.ops.matmul(T59, T54)
    T62 = fd.ops.linear(T60, T3)
    T63 = fd.ops.permute(T61, dims=[0, 2, 1])
    T69 = fd.ops.reshape(T62, new_shape=[1, 6, 8, 64])
    T70 = fd.ops.cat([T63, T63], dim=-1, manual_padding=0)
    T71 = fd.ops.permute(T69, dims=[0, 2, 1, 3])
    T72 = fd.ops.sin(T70)
    T88 = fd.ops.slice(T71, start_indices=[0, 0, 0, 32], end_indices=[1, 8, 6, 64], strides=[1, 1, 1, 1], manual_normalization=0)
    T89 = fd.ops.cos(T70)
    T90 = fd.ops.linear(T60, T4)
    S91 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T92 = fd.ops.mul(T72, S91)
    T93 = fd.ops.cast(T88, dtype=DataType.Float)
    S94 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T95 = fd.ops.mul(T89, S94)
    T101 = fd.ops.reshape(T90, new_shape=[1, 6, 32, 64])
    T102 = fd.ops.cast(T92, dtype=DataType.BFloat16)
    T103 = fd.ops.neg(T93)
    T104 = fd.ops.cast(T95, dtype=DataType.BFloat16)
    T105 = fd.ops.permute(T101, dims=[0, 2, 1, 3])
    T111 = fd.ops.broadcast_in_dim(T102, shape=[1, 1, 6, 64], broadcast_dims=[0, 2, 3])
    T127 = fd.ops.slice(T71, start_indices=[0, 0, 0, 0], end_indices=[1, 8, 6, 32], strides=[1, 1, 1, 1], manual_normalization=0)
    T128 = fd.ops.cast(T103, dtype=DataType.BFloat16)
    T134 = fd.ops.broadcast_in_dim(T104, shape=[1, 1, 6, 64], broadcast_dims=[0, 2, 3])
    T150 = fd.ops.slice(T105, start_indices=[0, 0, 0, 32], end_indices=[1, 32, 6, 64], strides=[1, 1, 1, 1], manual_normalization=0)
    S151 = fd.define_scalar(-3.38953e+38, dtype=DataType.Double)
    T152 = fd.ops.eq(T5, S151)
    T158 = fd.ops.broadcast_in_dim(T111, shape=[1, 8, 6, 64], broadcast_dims=[0, 1, 2, 3])
    T159 = fd.ops.cat([T128, T127], dim=-1, manual_padding=0)
    T165 = fd.ops.broadcast_in_dim(T134, shape=[1, 8, 6, 64], broadcast_dims=[0, 1, 2, 3])
    T166 = fd.ops.cast(T150, dtype=DataType.Float)
    T167 = fd.ops.bitwise_not(T152)
    T168 = fd.ops.cast(T158, dtype=DataType.Float)
    T169 = fd.ops.cast(T159, dtype=DataType.Float)
    T170 = fd.ops.cast(T165, dtype=DataType.Float)
    T171 = fd.ops.cast(T71, dtype=DataType.Float)
    T172 = fd.ops.neg(T166)
    T173 = fd.ops.cast(T167, dtype=DataType.Int)
    T174 = fd.ops.mul(T169, T168)
    T175 = fd.ops.mul(T171, T170)
    T191 = fd.ops.slice(T105, start_indices=[0, 0, 0, 0], end_indices=[1, 32, 6, 32], strides=[1, 1, 1, 1], manual_normalization=0)
    T192 = fd.ops.cast(T172, dtype=DataType.BFloat16)
    T193 = fd.ops.sum(T173, dims=[3], keepdim=False, dtype=DataType.Null)
    T199 = fd.ops.broadcast_in_dim(T111, shape=[1, 32, 6, 64], broadcast_dims=[0, 1, 2, 3])
    T200 = fd.ops.cat([T192, T191], dim=-1, manual_padding=0)
    T206 = fd.ops.broadcast_in_dim(T134, shape=[1, 32, 6, 64], broadcast_dims=[0, 1, 2, 3])
    T212 = fd.ops.broadcast_in_dim(T193, shape=[1, 1, 6, 1], broadcast_dims=[0, 1, 2])
    T213 = fd.ops.linear(T60, T6)
    T214 = fd.ops.cast(T199, dtype=DataType.Float)
    T215 = fd.ops.cast(T200, dtype=DataType.Float)
    T216 = fd.ops.cast(T206, dtype=DataType.Float)
    T217 = fd.ops.cast(T105, dtype=DataType.Float)
    S218 = fd.define_scalar(0, dtype=DataType.Int)
    T219 = fd.ops.ne(T212, S218)
    T225 = fd.ops.reshape(T213, new_shape=[1, 6, 8, 64])
    T226 = fd.ops.add(T175, T174)
    T227 = fd.ops.mul(T215, T214)
    T228 = fd.ops.mul(T217, T216)
    T229 = fd.ops.bitwise_not(T219)
    T230 = fd.ops.permute(T225, dims=[0, 2, 1, 3])
    T231 = fd.ops.cast(T226, dtype=DataType.BFloat16)
    T232 = fd.ops.bitwise_not(T229)
    T239 = fd.ops.broadcast_in_dim(T230, shape=[1, 8, 1, 6, 64], broadcast_dims=[0, 1, 3, 4])
    T246 = fd.ops.broadcast_in_dim(T231, shape=[1, 8, 1, 6, 64], broadcast_dims=[0, 1, 3, 4])
    T252 = fd.ops.broadcast_in_dim(T232, shape=[1, 1, 6, 6], broadcast_dims=[0, 1, 2, 3])
    T259 = fd.ops.broadcast_in_dim(T239, shape=[1, 8, 4, 6, 64], broadcast_dims=[0, 1, 2, 3, 4])
    T266 = fd.ops.broadcast_in_dim(T246, shape=[1, 8, 4, 6, 64], broadcast_dims=[0, 1, 2, 3, 4])
    T267 = fd.ops.add(T228, T227)
    T268 = fd.ops.cast(T252, dtype=DataType.Float)
    T269 = fd.ops.cast(T5, dtype=DataType.Float)
    T275 = fd.ops.reshape(T259, new_shape=[1, 32, 6, 64])
    T281 = fd.ops.reshape(T266, new_shape=[1, 32, 6, 64])
    T282 = fd.ops.cast(T267, dtype=DataType.BFloat16)
    T283 = fd.ops.mul(T269, T268)
    T284 = fd.ops.stride_order(T275, stride_order=[3, 2, 1, 0])
    T285 = fd.ops.stride_order(T281, stride_order=[3, 2, 1, 0])
    T286 = fd.ops.stride_order(T282, stride_order=[3, 2, 1, 0])
    T287 = fd.ops.cast(T283, dtype=DataType.BFloat16)
    fd.add_output(T287)
    fd.add_output(T230)
    fd.add_output(T231)
    fd.add_output(T286)
    fd.add_output(T285)
    fd.add_output(T284)

with FusionDefinition() as fd:
    nvfuser_fusion_id1(fd)

inputs = [
    torch.testing.make_tensor((1, 6, 2048), dtype=torch.bfloat16, device='cuda:0'),
    torch.testing.make_tensor((2048,), dtype=torch.bfloat16, device='cuda:0'),
    torch.testing.make_tensor((32,), dtype=torch.bfloat16, device='cuda:0'),
    torch.testing.make_tensor((512, 2048), dtype=torch.bfloat16, device='cuda:0'),
    torch.testing.make_tensor((2048, 2048), dtype=torch.bfloat16, device='cuda:0'),
    torch.testing.make_tensor((1, 1, 6, 6), dtype=torch.bfloat16, device='cuda:0'),
    torch.testing.make_tensor((512, 2048), dtype=torch.bfloat16, device='cuda:0'),
]

fd.execute(inputs)

for _ in range(3):
    fd.execute(inputs)

torch.cuda.synchronize()
start = time.time()
# Mark the profiling region
torch.cuda.cudart().cudaProfilerStart()

for _ in range(100):
    fd.execute(inputs)

torch.cuda.cudart().cudaProfilerStop()
torch.cuda.synchronize()
end = time.time()

print((end-start)*1000, " ms")


# Before:
# 19.8  ms
# After:
# 10.6  ms

# rm report*
# nsys profile -c cudaProfilerApi python tests/python/llama_inf_tests/graph_1.py
# nsys stats report1.nsys-rep

# Before:
#  Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)   Style                       Range
#  --------  ---------------  ---------  --------  --------  --------  --------  -----------  -------  ----------------------------------------------
#      14.2         31791843        100  317918.4  268246.5    246507    762170      88138.8  PushPop  :FusionExecutorCache::runFusionWithInputs
#      13.6         30602349        100  306023.5  261735.5    239889    737741      82786.1  PushPop  :FusionKernelRuntime::runWithInputs
#      13.6         30480294        100  304802.9  260895.0    239116    735007      82461.4  PushPop  :FusionKernelRuntime::runSegmentsWithInputs
#      13.0         29106369       1300   22389.5   18414.5      1815    266605      22601.1  PushPop  :FusionKernelRuntime::runKernelWithInput
#      12.5         28090755       1300   21608.3   17832.0      1556    265963      22146.4  PushPop  :ExecutorDispatch::run2
#       8.1         18224542        500   36449.1   32308.0     16901    265512      24691.3  PushPop  :KernelExecutor::runFusion
#       7.2         16182053       4100    3946.8    3299.5       258    152312       4334.0  PushPop  :ExpressionEvaluator::evaluate
#       5.3         11797199        500   23594.4   18368.5     10877    209862      17543.9  PushPop  :KernelExecutor::runFusion::execute_kernel
#       4.1          9212380        500   18424.8   12260.5      7177    200666      15103.3  PushPop  :KernelExecutor::recomputeArgs
#       3.9          8816273        800   11020.3   11422.5      1369    159960       9005.4  PushPop  :ExprEvalExecutor::run
#       1.5          3394339        500    6788.7    3757.5      1977    196369       9934.9  PushPop  :fusion_executor::allocations::allocateOutputs
#       1.2          2593352        900    2881.5    1502.5      1213    186775       6863.1  PushPop  :fusion_executor::allocations::allocateTensor
#       1.0          2179586        500    4359.2    3768.0      2935    185683       8309.3  PushPop  :ExecutorRunFusion::cuLaunchKernel
#       0.6          1350809       1300    1039.1     838.0       420      6663        603.7  PushPop  :executor_utils::bindInputs
#       0.2           498945       1300     383.8     172.0       102      2870        400.6  PushPop  :ExecutorDispatch::isCompiled
#       0.1           159524        500     319.0     120.0        97      6098        431.0  PushPop  :KernelExecutor::runFusion::intermediates
#       0.1           145263        100    1452.6    1330.0       907      5442        594.3  PushPop  :FusionExecutorCache::setCacheId
#       0.0            37566        100     375.7     182.0       130      1791        350.7  PushPop  :FusionExecutorCache::getKernelRuntimeFor

# After:
#  Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)   Style                       Range
#  --------  ---------------  ---------  --------  --------  --------  --------  -----------  -------  ----------------------------------------------
#      16.1         19038957        100  190389.6  172709.0    145764    682753      77360.1  PushPop  :FusionExecutorCache::runFusionWithInputs
#      15.2         18050839        100  180508.4  164976.0    140671    653055      72317.9  PushPop  :FusionKernelRuntime::runWithInputs
#      15.2         17951998        100  179520.0  164040.5    139910    650172      72127.0  PushPop  :FusionKernelRuntime::runSegmentsWithInputs
#      13.8         16310510       1300   12546.5   11597.5      1791    229198      15422.3  PushPop  :FusionKernelRuntime::runKernelWithInput
#      13.2         15617300       1300   12013.3   11062.5      1542    228905      14374.1  PushPop  :ExecutorDispatch::run2
#       7.4          8714870       1900    4586.8    2470.0       255    214450       9636.4  PushPop  :ExpressionEvaluator::evaluate
#       7.2          8564673        800   10705.8   10866.5      1348    200141      10054.6  PushPop  :ExprEvalExecutor::run
#       5.5          6549775        500   13099.5   10053.5      6840    228580      19230.8  PushPop  :KernelExecutor::runFusion
#       1.9          2290425        500    4580.9    3967.5      3054    200810       9071.5  PushPop  :KernelExecutor::runFusion::execute_kernel
#       1.8          2092522        500    4185.0    3594.0      2841    200306       9052.8  PushPop  :ExecutorRunFusion::cuLaunchKernel
#       1.3          1562864        500    3125.7    1808.5      1417    205151       9320.5  PushPop  :fusion_executor::allocations::allocateOutputs
#       0.6           717665        900     797.4     669.0       433     11922        529.8  PushPop  :executor_utils::bindInputs
#       0.4           423085        500     846.2     644.0       258     12710        764.5  PushPop  :KernelExecutor::computeArgs2
#       0.2           264123       1300     203.2     153.0       103      2587        140.1  PushPop  :ExecutorDispatch::isCompiled
#       0.1           131127        100    1311.3    1206.5       829      6301        574.0  PushPop  :FusionExecutorCache::setCacheId
#       0.1           101062        500     202.1     122.0        98      2736        238.3  PushPop  :KernelExecutor::runFusion::intermediates
#       0.0            20536        100     205.4     158.0       115      1009        135.9  PushPop  :FusionExecutorCache::getKernelRuntimeFor
