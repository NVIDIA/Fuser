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


tensor_size = 2048
use_tma_ops = True


class LayerNorm(FusionDefinition):
    def definition(self):
        self.t0 = self.from_pytorch(inputs[0])
        self.s0 = self.define_scalar(1e-6, dtype=DataType.Double)
        self.norm_const = self.define_scalar(tensor_size, dtype=DataType.Int)

        self.mean_cast = self.ops.cast(self.t0, dtype=DataType.Float)
        self.bcast_sum0 = self.ops.sum(self.mean_cast, dims=[-1], keepdim=True)
        self.mean = self.ops.div(self.bcast_sum0, self.norm_const)

        self.var_cast = self.ops.cast(self.t0, dtype=DataType.Float)
        self.diff = self.ops.sub(self.var_cast, self.mean)
        self.diff_sq = self.ops.mul(self.diff, self.diff)
        self.bcast_sum1 = self.ops.sum(self.diff_sq, dims=[-1], keepdim=True)
        self.var = self.ops.div(self.bcast_sum1, self.norm_const)

        self.t0_cast = self.ops.cast(self.t0, dtype=DataType.Float)
        self.t0_diff = self.ops.sub(self.t0_cast, self.mean)
        self.var_eps = self.ops.sqrt(self.ops.add(self.var, self.s0))
        self.t0_norm = self.ops.div(self.t0_diff, self.var_eps)

        self.t0_norm_cast = self.ops.cast(self.t0_norm, dtype=DataType.BFloat16)
        self.add_output(self.t0_norm_cast)

    def schedule(self):
        smem_cache_op = LoadStoreOpType.tma if use_tma_ops else LoadStoreOpType.set
        t0_smem = self.sched.cache_after(self.t0, smem_cache_op)
        self.sched.set_memory_type(t0_smem, MemoryType.shared)
        tma_tvs = [t0_smem]

        t0_lmem = self.sched.cache_after(t0_smem)
        cache_before_t0_norm = self.sched.cache_before(self.t0_norm_cast)

        def _is_not_tma_tensor(a):
            return a not in tma_tvs

        all_tvs_except_tma = list(filter(_is_not_tma_tensor, self.sched.tensors()))

        examples_per_cta = 4
        tma_width = 256
        vectorize = 8
        elem_per_compute_thread = tensor_size // tma_width // vectorize

        # Define TMA Box
        self.sched.split(t0_smem, dim=0, factor=examples_per_cta)
        self.sched.split(t0_smem, dim=-1, factor=tma_width)

        reference_tv = self.t0_norm_cast

        # Schedule Reference
        # root domain: [I1, I2]
        # split: [I1, I2/V, V]
        self.sched.split(reference_tv, dim=-1, factor=vectorize)
        # NOTE use outer-split to have constant register allocation
        # split: [I1, EPCT, I2/V/EPCT (block_x), V]
        self.sched.split(
            reference_tv,
            dim=-2,
            factor=elem_per_compute_thread,
            inner_split=False,
        )
        # split: [I1, EPCT, I2/V/EPCT (block_x), U, V]
        self.sched.split(reference_tv, dim=-2, factor=1)
        # split: [I1, I2/V/EPCT (block_x), EPCT, U, V]
        self.sched.reorder(reference_tv, {-4: -3, -3: -4})
        # split: [I1/CTA, CTA, I2/V/EPCT (block_x), EPCT, U, V]
        self.sched.split(reference_tv, dim=0, factor=examples_per_cta)

        # Transform all tensors
        self.sched.transform_like(reference_tv, all_tvs_except_tma)

        # rfactor reduction tensors
        reduction_tvs = list(filter(self.sched.is_reduction, self.sched.tensors()))
        rfactor_tvs = [
            self.sched.rfactor(tv, dims=[-3, -2, -1]) for tv in reduction_tvs
        ]

        # Apply general parallelization
        self.sched.parallelize(reference_tv, axis := 0, ParallelType.grid_x)
        self.sched.parallelize(reference_tv, axis := 2, ParallelType.block_x)
        self.sched.parallelize(reference_tv, axis := -2, ParallelType.unroll)
        self.sched.parallelize_like(reference_tv)

        # vectorize store output
        self.sched.parallelize(self.t0_norm_cast, axis := -1, ParallelType.vectorize)

        self.sched.inline_most()

        # tma load input
        if use_tma_ops:
            self.sched.parallelize(t0_smem, axis := -1, ParallelType.tma)

        if examples_per_cta > 1:
            number_of_stages = 4
            self.sched.circular_buffer(t0_smem, number_of_stages)


batch_dim = 32768
for i in range(11, 12):
    inner_dim = 2**i
    inputs = [torch.randn(batch_dim, inner_dim, dtype=torch.bfloat16, device="cuda")]

    print(
        "\n\n============================================= Profile LayerNorm  ===================================="
    )
    fn = LayerNorm()
    nvf_out = fn.execute(inputs, profile=True)

    kps = fn.profile().kernel_profiles
    for kp in kps:
        print_kernel_profile(kp)
    print(
        "=============================================================================================================="
    )
