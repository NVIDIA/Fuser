#!/usr/bin/xonsh

for i in range(16):
    for j in range(16):
        echo @(i) @(k) | ./bin/nvfuser_tests --gtest_filter=*Hopper.SS*64_16_16_NT*NoSwizzle*half*
