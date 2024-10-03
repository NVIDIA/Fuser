#!/bin/bash

# python setup.py build --debinfo
NVFUSER_EXTERNAL_SRC=__tmp_kernel_none_f0_c0_r0_g0.cu NVFUSER_TEST_RANDOM_SEED=0 NVFUSER_TEST_ATEN_RANDOM_SEED=0 ./bin/test_matmul --gtest_filter=*HSH_NT_128BSwizzle*

# [13448, 1519] - result: -39.625 | ref: -39.75
# [13448, 1521] - result: 50.9062 | ref: 50.6875
# [13448, 1523] - result: -46.7812 | ref: -47.0625
# [13448, 1525] - result: 3.51758 | ref: 3.34375
# [13448, 1526] - result: 6.88672 | ref: 7.01172
# [13448, 1528] - result: 4.21875 | ref: 4.48047
# [13448, 1529] - result: -0.200195 | ref: -0.35376
# [13448, 1530] - result: 21.6094 | ref: 21.4531
# [13448, 1531] - result: -2.90039 | ref: -2.78516
# [13448, 1534] - result: -12.2578 | ref: -12.4766
