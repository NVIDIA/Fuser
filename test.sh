#!/bin/bash

# python setup.py build --debinfo
# NVFUSER_EXTERNAL_SRC=__tmp_kernel_none_f0_c0_r0_g0.cu NVFUSER_TEST_RANDOM_SEED=0 NVFUSER_TEST_ATEN_RANDOM_SEED=0 
./bin/test_matmul --gtest_filter=*HSH_NT_128BSwizzle*
