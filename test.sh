#!/bin/bash

python setup.py build --debinfo
./bin/test_matmul --gtest_filter=*HSH_NT_128BSwizzle*
