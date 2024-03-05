#!/usr/bin/env bash

KERNEL_DUMP=__tmp_kernel_none_f0_c0_r0_g0.cu

for i in 2 4 8 16 20 24 26 28 32 36 40 44 48 50; do
    echo $i
    PERSISTENT=$i NVFUSER_DUMP=cuda_to_file,occupancy,dump_eff_bandwidth ../bin/nvfuser_tests --gtest_filter=NVFuserTest.BFSoftmax
    mv $KERNEL_DUMP softmax_pb$i.cu
done
