#!/bin/bash

# Usage: ./run_benchmark.sh <value>
# Example: ./run_benchmark.sh 12345

if [ -z "$1" ]; then
  echo "Usage: $0 <value>"
  exit 1
fi

VALUE=$1
MODEL="rmsnorm_bwd"


NVFUSER_DUMP=scheduler_params,fusion_ir_math,fusion_ir_presched pytest \
  /opt/pytorch/nvfuser/benchmarks/python/test_${MODEL}.py -vvvs -m 'not skip' -k "float16 and 2048_${VALUE}" --disable-validation --disable-benchmarking --benchmark-thunder --benchmark-json ./tmp/tmp.json 2>&1 |tee 1.log
  # /opt/pytorch/nvfuser/benchmarks/python/test_${MODEL}.py -vvvs -m 'not skip' -k "float16 and 2048_${VALUE}" --disable-validation --benchmark-json ./tmp/tmp.json 2>&1 |tee 1.log

grep SOL ./tmp/tmp.json