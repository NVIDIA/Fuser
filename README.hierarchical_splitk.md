Note that the kernel `hierarchical_splitk.cu` is modified from the original kernel which was dumped using this command:
```
NVFUSER_DUMP=cuda_to_file build/nvfuser_bench --benchmark_filter='TN/M:1024/N:2048/K:4096/warps:4/stages:5/splitk_factor:4/smem_epilogue:1'
```
