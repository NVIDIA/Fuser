#!/bin/bash

# python setup.py build --debinfo
$NVFUSER_EXTERNAL_SRC=__tmp_kernel_none_f0_c0_r0_g0.cu ./bin/test_matmul --gtest_filter=*HSH_NT_128BSwizzle*