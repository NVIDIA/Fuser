import torch
from nvfuser import FusionDefinition, DataType
import time

def nvfuser_fusion_id0(fd : FusionDefinition) -> None :
    T0 = fd.define_tensor(shape=[1, 6], contiguity=[None, True], dtype=DataType.Int, is_cpu=False, stride_order=[1, 0])
    T1 = fd.define_tensor(shape=[128256, 2048], contiguity=[True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[1, 0])
    T2 = fd.define_tensor(shape=[1, 6], contiguity=[None, True], dtype=DataType.Int, is_cpu=False, stride_order=[1, 0])
    S3 = fd.define_scalar(2.00000, dtype=DataType.Double)
    S4 = fd.define_scalar(False, dtype=DataType.Bool)
    S5 = fd.define_scalar(False, dtype=DataType.Bool)
    T6 = fd.ops.embedding_fwd(T0, T1, None, None, S3, S4, S5)
    S7 = fd.define_scalar(6, dtype=DataType.Int)
    S8 = fd.define_scalar(0, dtype=DataType.Int)
    S9 = fd.define_scalar(1, dtype=DataType.Int)
    T10 = fd.ops.iota(S7, S8, S9, dtype=DataType.Int)
    T14 = fd.ops.broadcast_in_dim(T10, shape=[1, 6], broadcast_dims=[1])
    S15 = fd.define_scalar(-3.38953e+38, dtype=DataType.Double)
    T19 = fd.ops.full(shape=[6, 6], fill_value=S15, dtype=DataType.BFloat16)
    T23 = fd.ops.broadcast_in_dim(T10, shape=[6, 1], broadcast_dims=[0])
    T27 = fd.ops.broadcast_in_dim(T14, shape=[6, 6], broadcast_dims=[0, 1])
    T31 = fd.ops.broadcast_in_dim(T23, shape=[6, 6], broadcast_dims=[0, 1])
    T32 = fd.ops.sub(T27, T31)
    S33 = fd.define_scalar(1, dtype=DataType.Int)
    T34 = fd.ops.ge(T32, S33)
    S35 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T36 = fd.ops.where(T34, T19, S35)
    T40 = fd.ops.reshape(T10, new_shape=[6, 1])
    T44 = fd.ops.broadcast_in_dim(T10, shape=[6, 6], broadcast_dims=[1])
    T48 = fd.ops.broadcast_in_dim(T40, shape=[6, 6], broadcast_dims=[0, 1])
    T49 = fd.ops.gt(T44, T48)
    T50 = fd.ops.cast(T36, dtype=DataType.Float)
    T51 = fd.ops.cast(T49, dtype=DataType.Float)
    T52 = fd.ops.mul(T50, T51)
    T53 = fd.ops.cast(T52, dtype=DataType.BFloat16)
    T59 = fd.ops.broadcast_in_dim(T53, shape=[1, 1, 6, 6], broadcast_dims=[2, 3])
    T65 = fd.ops.broadcast_in_dim(T59, shape=[1, 1, 6, 6], broadcast_dims=[0, 1, 2, 3])
    T66 = fd.ops.set(T65)
    T72 = fd.ops.broadcast_in_dim(T2, shape=[1, 1, 1, 6], broadcast_dims=[0, 3])
    T78 = fd.ops.broadcast_in_dim(T72, shape=[1, 1, 6, 6], broadcast_dims=[0, 1, 2, 3])
    T79 = fd.ops.cast(T66, dtype=DataType.Float)
    T80 = fd.ops.cast(T78, dtype=DataType.Float)
    T81 = fd.ops.add(T79, T80)
    T82 = fd.ops.cast(T81, dtype=DataType.BFloat16)
    S83 = fd.define_scalar(0.00000, dtype=DataType.Double)
    T84 = fd.ops.eq(T82, S83)
    S85 = fd.define_scalar(-3.38953e+38, dtype=DataType.Double)
    T86 = fd.ops.where(T84, S85, T66)
    fd.add_output(T6)
    fd.add_output(T66)
    fd.add_output(T86)

with FusionDefinition() as fd:
    nvfuser_fusion_id0(fd)

inputs = [
    torch.ones((1, 6), dtype=torch.int64, device='cuda:0'),
    torch.testing.make_tensor((128256, 2048), dtype=torch.bfloat16, device='cuda:0'),
    torch.ones((1, 6), dtype=torch.int64, device='cuda:0'),
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
# 12.0  ms
# After:
# 3.1  ms

# rm report*
# nsys profile -c cudaProfilerApi python tests/python/llama_inf_tests/graph_0.py
# nsys stats report1.nsys-rep

# Before:
#  Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)   Style                       Range
#  --------  ---------------  ---------  --------  --------  --------  --------  -----------  -------  ----------------------------------------------
#      13.8         10011392        100  100113.9   80400.0     76319    768432      82944.7  PushPop  :FusionExecutorCache::runFusionWithInputs
#      13.0          9409367        100   94093.7   77940.0     74188    765647      79435.0  PushPop  :FusionKernelRuntime::runWithInputs
#      12.9          9353511        100   93535.1   77347.0     73635    764599      79335.4  PushPop  :FusionKernelRuntime::runSegmentsWithInputs
#      12.4          8989375        300   29964.6   26494.0     12397    698157      44537.4  PushPop  :FusionKernelRuntime::runKernelWithInput
#      12.3          8896373        300   29654.6   26056.5     12135    697796      44508.6  PushPop  :ExecutorDispatch::run2
#      10.1          7309840        200   36549.2   31871.0     24321    697376      51775.8  PushPop  :KernelExecutor::runFusion
#       6.7          4859672        200   24298.4   22950.5     13246    684391      48960.1  PushPop  :KernelExecutor::runFusion::execute_kernel
#       5.9          4316308       1200    3596.9    2396.0      1980    175457       7203.4  PushPop  :ExpressionEvaluator::evaluate
#       5.6          4086264        200   20431.3   19635.5     10005    674394      48476.2  PushPop  :KernelExecutor::recomputeArgs
#       2.0          1455689        100   14556.9   12596.0     11930    176349      16430.4  PushPop  :ExprEvalExecutor::run
#       1.9          1362206        200    6811.0    7320.5      3864    174236      12085.8  PushPop  :fusion_executor::allocations::allocateOutputs
#       1.4           997365        600    1662.3    1368.0      1205    167712       6793.6  PushPop  :fusion_executor::allocations::allocateTensor
#       1.0           690717        200    3453.6    3288.0      2816     10209        890.9  PushPop  :ExecutorRunFusion::cuLaunchKernel
#       0.4           294173        300     980.6     831.0       107      9627        925.9  PushPop  :executor_utils::bindInputs
#       0.3           228065        300     760.2     152.5       122    165328       9538.9  PushPop  :ExecutorDispatch::isCompiled
#       0.3           192836        200     964.2     108.0        99    164903      11650.7  PushPop  :KernelExecutor::runFusion::intermediates
#       0.1            75680        100     756.8     712.0       459      4308        388.0  PushPop  :FusionExecutorCache::setCacheId
#       0.0            17393        100     173.9     133.0       112       875        100.7  PushPop  :FusionExecutorCache::getKernelRuntimeFor

# After:
#  Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)   Style                       Range
#  --------  ---------------  ---------  --------  --------  --------  --------  -----------  -------  ----------------------------------------------
#      17.1          5182038        100   51820.4   40488.5     38316    309012      45433.7  PushPop  :FusionExecutorCache::runFusionWithInputs
#      15.5          4712599        100   47126.0   38026.5     36027    293111      40961.2  PushPop  :FusionKernelRuntime::runWithInputs
#      15.3          4653120        100   46531.2   37485.5     35536    290957      40853.9  PushPop  :FusionKernelRuntime::runSegmentsWithInputs
#      13.5          4099585        300   13665.3   11896.0      8647    197301      18602.1  PushPop  :FusionKernelRuntime::runKernelWithInput
#      12.5          3810167        300   12700.6   11606.0      8426    196668      15305.2  PushPop  :ExecutorDispatch::run2
#       7.0          2114207        200   10571.0   10721.5      8123     45738       3065.6  PushPop  :KernelExecutor::runFusion
#       5.3          1601371        100   16013.7   11992.5     11374    196441      25772.1  PushPop  :ExprEvalExecutor::run
#       4.6          1406004        100   14060.0   10303.5      9756    194564      24892.8  PushPop  :ExpressionEvaluator::evaluate
#       2.6           803300        200    4016.5    4956.5      2605     16355       1490.5  PushPop  :fusion_executor::allocations::allocateOutputs
#       2.4           722700        200    3613.5    3430.5      2910     15725       1130.3  PushPop  :KernelExecutor::runFusion::execute_kernel
#       2.2           666998        200    3335.0    3195.5      2708     14553       1040.9  PushPop  :ExecutorRunFusion::cuLaunchKernel
#       0.8           257659        100    2576.6     707.5       483    182114      18138.7  PushPop  :FusionExecutorCache::setCacheId
#       0.4           116565        200     582.8     602.0       420      2139        184.6  PushPop  :KernelExecutor::computeArgs2
#       0.3            96582        100     965.8     835.5       772      9563        880.2  PushPop  :executor_utils::bindInputs
#       0.2            66419        300     221.4     157.0       127      2363        221.8  PushPop  :ExecutorDispatch::isCompiled
#       0.1            29389        100     293.9     132.5       111     10088       1006.3  PushPop  :FusionExecutorCache::getKernelRuntimeFor
#       0.1            29125        200     145.6     108.0        97       798        107.9  PushPop  :KernelExecutor::runFusion::intermediates
