import torch
from nvfuser import (
    FusionDefinition,
    ParallelType,
    LoadStoreOpType,
    MemoryType,
    DataType,
)


def print_kernel_profile(kp):
    basic_information = f"name: {kp.name}, schedule: {kp.scheduler}, segment_id: {kp.segment_id}, device: {kp.device}, stream: {kp.stream}"
    print(basic_information)

    kernel_information = f"compile time: {kp.compile_time_ms:.2f} ms, grid: {kp.grid_str}, block: {kp.block_str}, registers: {kp.registers}"
    print(kernel_information)

    runtime_information = f"input size: {kp.input_bytes} bytes, output size: {kp.output_bytes} bytes, time: {kp.time_ms:2f} ms"
    print(runtime_information)

    bandwidth_information = f"Effective Bandwidth: {kp.effective_bandwidth_gbs:.2f} GB/s, Peak Bandwidth: {kp.percentage_peak_bandwidth:2f}%"
    print(bandwidth_information)


# Apply schedule with decorator pattern.
def pointwise_fn(fd, tensor_size):
    def schedule():
        print("pointwise")
        cache_after_input = fd.sched.cache_after(fd.input)
        cache_before_output = fd.sched.cache_before(fd.output)

        V = 8
        BDX = 128

        for t in fd.sched.tensors():
            # (I0 * I1)
            fd.sched.merge(t, dim=0)
            # (I0 * I1) / V, V
            fd.sched.split(t, dim=0, factor=V)
            # (I0 * I1) / V / BDX, BDX, V
            fd.sched.split(t, dim=0, factor=BDX)
            # (I0 * I1) / V / BDX, BDX, V

            fd.sched.parallelize(t, axis := 0, ParallelType.grid_x)
            fd.sched.parallelize(t, axis := -2, ParallelType.block_x)

        # vectorize 2d tensors
        fd.sched.parallelize(cache_after_input, axis := -1, ParallelType.vectorize)
        fd.sched.parallelize(fd.output, axis := -1, ParallelType.vectorize)

        # computeAt - automatically handles vectorize paralleltype
        fd.sched.inline_most()

    fd.schedule = schedule
    return fd

# Apply schedule with decorator pattern.
def tma_fn(fd, tensor_size):
    def schedule():
        print("tma")
        cache_after_input = fd.sched.cache_after(fd.input, LoadStoreOpType.tma)
        fd.sched.set_memory_type(cache_after_input, MemoryType.shared)

        cache_before_output = fd.sched.cache_before(fd.output)

        CB = 2
        V = 8
        BDX = 128

        for t in fd.sched.tensors():
            # (I0 * I1)
            fd.sched.merge(t, dim=0)

            # (I0 * I1) / V, V
            fd.sched.split(t, dim=0, factor=V)

            # (I0 * I1) / V / BDX, BDX, V
            fd.sched.split(t, dim=0, factor=BDX)

            # (I0 * I1) / CB / V / BDX, CB, BDX, V
            fd.sched.split(t, dim=0, factor=CB)

            fd.sched.parallelize(t, axis := 0, ParallelType.grid_x)
            fd.sched.parallelize(t, axis := -2, ParallelType.block_x)

        # vectorize 2d tensors
        fd.sched.parallelize(cache_after_input, axis := -1, ParallelType.tma)
        fd.sched.parallelize(fd.output, axis := -1, ParallelType.vectorize)

        # computeAt - automatically handles vectorize paralleltype
        fd.sched.inline_most()

        fd.sched.circular_buffer(cache_after_input, CB)

    fd.schedule = schedule
    return fd


def fusion_func(fd: FusionDefinition) -> None:
    T0 = fd.define_tensor(
        shape=[-1, -1],
        contiguity=[True, True],
        dtype=DataType.BFloat16,
        is_cpu=False,
        stride_order=[1, 0],
    )
    T1 = fd.ops.cast(T0, dtype=DataType.Float)
    T2 = fd.ops.mul(T1, T1)
    T3 = fd.ops.mul(T2, T1)
    S4 = fd.define_scalar(0.500000, dtype=DataType.Double)
    T5 = fd.ops.mul(S4, T1)
    S6 = fd.define_scalar(0.0447150, dtype=DataType.Double)
    T7 = fd.ops.mul(S6, T3)
    T8 = fd.ops.add(T1, T7)
    S9 = fd.define_scalar(0.797885, dtype=DataType.Double)
    T10 = fd.ops.mul(S9, T8)
    T11 = fd.ops.tanh(T10)
    S12 = fd.define_scalar(1.00000, dtype=DataType.Double)
    T13 = fd.ops.add(S12, T11)
    T14 = fd.ops.mul(T5, T13)
    T15 = fd.ops.cast(T14, dtype=DataType.BFloat16)
    fd.add_output(T15)

    fd.input = T0
    fd.output = T15


batch_dim = 512
for i in range(10, 16):
    inner_dim = 2**i
    inputs = [torch.randn(batch_dim, inner_dim, dtype=torch.bfloat16, device="cuda")]

    print(
        "\n\n============================================= Profile Gelu  ===================================="
    )
    print(batch_dim, inner_dim)
    with FusionDefinition() as fd:
        fusion_func(fd)

    fd = tma_fn(fd, inner_dim)
    nvf_out = fd.execute(inputs, profile=True)

    kps = fd.profile().kernel_profiles
    for kp in kps:
        print_kernel_profile(kp)
    print(
        "=============================================================================================================="
    )
