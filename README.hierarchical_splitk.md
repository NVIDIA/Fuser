Note that the kernel `hierarchical_splitk.cu` is modified from the original kernel which was dumped using this command:
```
NVFUSER_DUMP=cuda_to_file build/nvfuser_bench --benchmark_filter='NvFuserScheduler_Matmul_Manual/nvfuser_splitk_TN/M:1024/N:2048/K:4096/warps:4/stages:5/splitk_factor:4/smem_epilogue:1'
```

To run a benchmark with hierarchical split-K, use a command like the following:
```
k=hierarchical_splitk.cu; NVFUSER_EXTERNAL_SRC=$k,$k,$k,$k,$k NVFUSER_DUMP=cuda_to_file build/nvfuser_bench --benchmark_filter='NvFuserScheduler_Matmul_Manual/nvfuser_splitk_TN/M:1024/N:2048/K:4096/warps:4/stages:5/splitk_factor:4/smem_epilogue:1'
```
You may change `M`, `N`, `K`, and `splitk_factor` but other modifications will not be reflected and may lead to errors.
