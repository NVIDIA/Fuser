#!/bin/bash

# python setup.py build --debinfo
NVFUSER_EXTERNAL_SRC=__tmp_kernel_none_f0_c0_r0_g0.cu NVFUSER_TEST_RANDOM_SEED=0 NVFUSER_TEST_ATEN_RANDOM_SEED=0 ./bin/test_matmul --gtest_filter=*HSH_NT_128BSwizzle*

[14076, 8200] - result: -3.82617 | ref: -3.34766
[14076, 8264] - result: -21.8594 | ref: -21.9844
[14076, 8328] - result: -29.7656 | ref: -29.4688
[14077, 8200] - result: 4.40625 | ref: 4.55859
[14078, 8200] - result: 33.7188 | ref: 35.6875
[14078, 8264] - result: -40.3438 | ref: -40.8125
[14078, 8328] - result: -7.75781 | ref: -6.55078
[14079, 8200] - result: -15.6172 | ref: -16.1562
[14079, 8264] - result: -23.8906 | ref: -23.75
[14079, 8328] - result: 2.74805 | ref: 2.41602
