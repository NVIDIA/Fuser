#!/usr/bin/env bash

for i in $(seq 100); do
    DISABLE_CONTIG_INDEXING=1 NVFUSER_DISABLE=parallel_compile ./bin/test_view --gtest_filter="*.*ReshapeAllShmoo*"
done 2>&1 |tee log
