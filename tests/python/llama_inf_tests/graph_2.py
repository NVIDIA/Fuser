import torch
from nvfuser import FusionDefinition, DataType
import time

def nvfuser_fusion_id2(fd : FusionDefinition) -> None :
    T0 = fd.define_tensor(shape=[1, 32, 6, 64], contiguity=[None, True, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[3, 1, 2, 0])
    T1 = fd.define_tensor(shape=[2048, 2048], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
    T2 = fd.define_tensor(shape=[1, 6, 2048], contiguity=[None, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 1, 0])
    T3 = fd.define_tensor(shape=[2048], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
    T4 = fd.define_tensor(shape=[8192, 2048], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
    T5 = fd.define_tensor(shape=[8192, 2048], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
    T6 = fd.define_tensor(shape=[2048, 8192], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
    T7 = fd.define_tensor(shape=[2048], contiguity=[True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[0])
    T8 = fd.define_tensor(shape=[128256, 2048], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
    T9 = fd.ops.permute(T0, dims=[0, 2, 1, 3])
    T10 = fd.ops.stride_order(T9, stride_order=[3, 2, 1, 0])
    T15 = fd.ops.reshape(T10, new_shape=[1, 6, 2048])
    T16 = fd.ops.stride_order(T15, stride_order=[2, 1, 0])
    T17 = fd.ops.linear(T16, T1)
    T18 = fd.ops.cast(T2, dtype=DataType.Float)
    T19 = fd.ops.cast(T17, dtype=DataType.Float)
    T20 = fd.ops.add(T18, T19)
    S21 = fd.define_scalar(2.00000, dtype=DataType.Double)
    T22 = fd.ops.pow(T20, S21)
    T23 = fd.ops.sum(T22, dims=[2], keepdim=False, dtype=DataType.Null)
    T28 = fd.ops.broadcast_in_dim(T23, shape=[1, 6, 1], broadcast_dims=[0, 1])
    S29 = fd.define_scalar(2048.00, dtype=DataType.Double)
    S30 = fd.ops.reciprocal(S29)
    T31 = fd.ops.mul(T28, S30)
    S32 = fd.define_scalar(1.00000e-05, dtype=DataType.Double)
    T33 = fd.ops.add(T31, S32)
    T34 = fd.ops.rsqrt(T33)
    T39 = fd.ops.broadcast_in_dim(T34, shape=[1, 6, 2048], broadcast_dims=[0, 1, 2])
    T40 = fd.ops.mul(T20, T39)
    T45 = fd.ops.broadcast_in_dim(T3, shape=[1, 6, 2048], broadcast_dims=[2])
    T46 = fd.ops.cast(T45, dtype=DataType.Float)
    T47 = fd.ops.mul(T46, T40)
    T48 = fd.ops.cast(T47, dtype=DataType.BFloat16)
    T49 = fd.ops.linear(T48, T4)
    T50 = fd.ops.cast(T49, dtype=DataType.Float)
    T51 = fd.ops.neg(T50)
    T52 = fd.ops.exp(T51)
    S53 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T54 = fd.ops.add(S53, T52)
    T55 = fd.ops.reciprocal(T54)
    T56 = fd.ops.mul(T50, T55)
    T57 = fd.ops.linear(T48, T5)
    T58 = fd.ops.cast(T57, dtype=DataType.Float)
    T59 = fd.ops.mul(T56, T58)
    T60 = fd.ops.cast(T59, dtype=DataType.BFloat16)
    T61 = fd.ops.linear(T60, T6)
    T62 = fd.ops.cast(T61, dtype=DataType.Float)
    T63 = fd.ops.add(T20, T62)
    S64 = fd.define_scalar(2.00000, dtype=DataType.Double)
    T65 = fd.ops.pow(T63, S64)
    T66 = fd.ops.sum(T65, dims=[2], keepdim=False, dtype=DataType.Null)
    T71 = fd.ops.broadcast_in_dim(T66, shape=[1, 6, 1], broadcast_dims=[0, 1])
    S72 = fd.define_scalar(2048.00, dtype=DataType.Double)
    S73 = fd.ops.reciprocal(S72)
    T74 = fd.ops.mul(T71, S73)
    S75 = fd.define_scalar(1.00000e-05, dtype=DataType.Double)
    T76 = fd.ops.add(T74, S75)
    T77 = fd.ops.rsqrt(T76)
    T82 = fd.ops.broadcast_in_dim(T77, shape=[1, 6, 2048], broadcast_dims=[0, 1, 2])
    T83 = fd.ops.mul(T63, T82)
    T88 = fd.ops.broadcast_in_dim(T7, shape=[1, 6, 2048], broadcast_dims=[2])
    T89 = fd.ops.cast(T88, dtype=DataType.Float)
    T90 = fd.ops.mul(T89, T83)
    T91 = fd.ops.cast(T90, dtype=DataType.BFloat16)
    T92 = fd.ops.linear(T91, T8)
    fd.add_output(T92)

with FusionDefinition() as fd:
    nvfuser_fusion_id2(fd)

inputs = [
    torch.randn(12288, dtype=torch.bfloat16, device='cuda:0').as_strided((1, 32, 6, 64), (12288, 64, 2048, 1)),
    torch.testing.make_tensor((2048, 2048), dtype=torch.bfloat16, device='cuda:0'),
    torch.testing.make_tensor((1, 6, 2048), dtype=torch.bfloat16, device='cuda:0'),
    torch.testing.make_tensor((2048,), dtype=torch.bfloat16, device='cuda:0'),
    torch.testing.make_tensor((8192, 2048), dtype=torch.bfloat16, device='cuda:0'),
    torch.testing.make_tensor((8192, 2048), dtype=torch.bfloat16, device='cuda:0'),
    torch.testing.make_tensor((2048, 8192), dtype=torch.bfloat16, device='cuda:0'),
    torch.testing.make_tensor((2048,), dtype=torch.bfloat16, device='cuda:0'),
    torch.testing.make_tensor((128256, 2048), dtype=torch.bfloat16, device='cuda:0'),
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
# 18.9  ms
# After:
# 18.8 ms


# rm report*
# nsys profile -c cudaProfilerApi python tests/python/llama_inf_tests/graph_2.py
# nsys stats report1.nsys-rep

# Before:
#  Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)   Style                       Range
#  --------  ---------------  ---------  --------  --------  --------  --------  -----------  -------  ----------------------------------------------
#      14.3         21273988        100  212739.9  191180.5    179102    647287      68515.0  PushPop  :FusionExecutorCache::runFusionWithInputs
#      13.9         20711603        100  207116.0  185424.0    174989    625555      67241.1  PushPop  :FusionKernelRuntime::runWithInputs
#      13.9         20634952        100  206349.5  184704.0    174362    623140      67108.8  PushPop  :FusionKernelRuntime::runSegmentsWithInputs
#      13.1         19477550        900   21641.7   19736.5      5253    229134      19235.9  PushPop  :FusionKernelRuntime::runKernelWithInput
#      12.8         18979906        900   21088.8   19402.5      5008    228699      18181.9  PushPop  :ExecutorDispatch::run2
#       9.1         13569155       2100    6461.5    3299.5      1250    188373       8072.4  PushPop  :ExpressionEvaluator::evaluate
#       6.8         10071317        600   16785.5   16953.0      4816    226456      12582.7  PushPop  :ExprEvalExecutor::run
#       5.8          8593835        300   28646.1   23748.5     18304    209021      24037.1  PushPop  :KernelExecutor::runFusion
#       4.1          6042339        300   20141.1   17470.5     12833    200139      18266.7  PushPop  :KernelExecutor::runFusion::execute_kernel
#       3.2          4802005        300   16006.7   13105.5      9464    195217      18063.1  PushPop  :KernelExecutor::recomputeArgs
#       0.7          1083270        300    3610.9    3488.0      2803      9530        804.5  PushPop  :ExecutorRunFusion::cuLaunchKernel
#       0.7          1066491        300    3555.0    2310.5      1934    173206       9901.1  PushPop  :fusion_executor::allocations::allocateOutputs
#       0.7          1059251        900    1176.9     864.0       534    174544       5806.8  PushPop  :executor_utils::bindInputs
#       0.5           753282        400    1883.2    1430.0      1237    169947       8427.1  PushPop  :fusion_executor::allocations::allocateTensor
#       0.1           168892        900     187.7     147.0       103      1888        115.3  PushPop  :ExecutorDispatch::isCompiled
#       0.1           154076        100    1540.8    1415.5      1012      6127        537.9  PushPop  :FusionExecutorCache::setCacheId
#       0.0            51330        300     171.1     114.5        97      1893        169.0  PushPop  :KernelExecutor::runFusion::intermediates
#       0.0            19650        100     196.5     156.0       117       845        109.3  PushPop  :FusionExecutorCache::getKernelRuntimeFor

# After:
#  Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)   Style                       Range
#  --------  ---------------  ---------  --------  --------  --------  --------  -----------  -------  ----------------------------------------------
#      16.0         16373962        100  163739.6  141242.5    134284    628006      71340.5  PushPop  :FusionExecutorCache::runFusionWithInputs
#      15.1         15382598        100  153826.0  136014.5    129986    600299      64199.4  PushPop  :FusionKernelRuntime::runWithInputs
#      15.0         15308089        100  153080.9  135307.0    129396    597501      64038.0  PushPop  :FusionKernelRuntime::runSegmentsWithInputs
#      13.8         14094916        900   15661.0   16073.0      5213    251448      17412.1  PushPop  :FusionKernelRuntime::runKernelWithInput
#      13.3         13579134        900   15087.9   15684.5      4944    251078      16245.8  PushPop  :ExecutorDispatch::run2
#       9.7          9923498        600   16539.2   16927.0      4741    250699      14459.9  PushPop  :ExprEvalExecutor::run
#       9.4          9632237        900   10702.5   13644.5      1314    248454      13132.5  PushPop  :ExpressionEvaluator::evaluate
#       3.3          3330448        300   11101.5    8723.0      6965    201143      18774.3  PushPop  :KernelExecutor::runFusion
#       1.1          1129730        300    3765.8    3564.0      2899     11811        941.0  PushPop  :KernelExecutor::runFusion::execute_kernel
#       1.0          1013555        300    3378.5    3257.5      2673     10550        813.0  PushPop  :ExecutorRunFusion::cuLaunchKernel
#       0.9           917298        300    3057.7    1641.0      1451    193374      11094.4  PushPop  :fusion_executor::allocations::allocateOutputs
#       0.5           545570        600     909.3     832.0       531     11018        553.9  PushPop  :executor_utils::bindInputs
#       0.4           374638        900     416.3     150.0       101    199723       6652.2  PushPop  :ExecutorDispatch::isCompiled
#       0.2           204834        100    2048.3     148.5       110    185215      18502.2  PushPop  :FusionExecutorCache::getKernelRuntimeFor
#       0.2           166768        300     555.9     478.5       341      5239        394.4  PushPop  :KernelExecutor::computeArgs2
#       0.2           165881        100    1658.8    1535.0      1125      7627        674.0  PushPop  :FusionExecutorCache::setCacheId
#       0.1            57354        300     191.2     111.0        97      2077        236.0  PushPop  :KernelExecutor::runFusion::intermediates
