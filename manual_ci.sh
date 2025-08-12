#!/bin/bash

failed_tests=false

run_test() {
  eval "$*"
  status=$?
  if [ $status -ne 0 ];
  then
    failed_tests=true
    echo "============================================================="
    echo "= test_failed!"
    echo "= $*"
    echo "============================================================="
  fi
}

cd "$(dirname "${BASH_SOURCE[0]}")"

run_test './bin/lib/dynamic_type/test_dynamic_type_17'
run_test './bin/lib/dynamic_type/test_dynamic_type_20'
run_test './bin/test_nvfuser'
run_test './bin/test_rng'
run_test './bin/test_host_ir'
if type -p mpirun > /dev/null
then
    run_test mpirun -np 1 './bin/test_multidevice'
fi
run_test './bin/test_view'
run_test './bin/test_matmul'
run_test './bin/test_external_src'
run_test './bin/tutorial'
run_test './bin/test_python_frontend'
run_test './bin/test_profiler'

run_test 'pytest tests/python/test_ops.py'
run_test 'pytest tests/python/test_python_frontend.py'
run_test 'pytest tests/python/test_schedule_ops.py'

if $failed_tests;
then
  echo "=== CI tests failed, do NOT merge your PR! ==="
  exit 1
else
  echo "=== CI tests passed, ship it! ==="
  exit 0
fi
