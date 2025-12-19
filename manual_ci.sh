#!/bin/bash

failed_tests=false
LOG_FILE="/tmp/nvfuser_manual_ci.log"

# Initialize log file
echo "nvfuser manual CI test run - $(date)" > "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"
echo ""

run_test() {
  echo "============================================================="
  echo "Running: $*"
  echo "Time: $(date)"
  echo "-------------------------------------------------------------"
  echo "=============================================================" >> "$LOG_FILE"
  echo "Running: $*" >> "$LOG_FILE"
  echo "Time: $(date)" >> "$LOG_FILE"
  echo "-------------------------------------------------------------" >> "$LOG_FILE"

  # Use tee to send output to both terminal and log file
  eval "$*" 2>&1 | tee -a "$LOG_FILE"
  status=${PIPESTATUS[0]}

  if [ $status -ne 0 ];
  then
    failed_tests=true
    echo ""
    echo "❌ TEST FAILED (exit code: $status)"
    echo "" >> "$LOG_FILE"
    echo "❌ TEST FAILED (exit code: $status)" >> "$LOG_FILE"
    echo "=============================================================" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
  else
    echo ""
    echo "✅ TEST PASSED"
    echo "" >> "$LOG_FILE"
    echo "✅ TEST PASSED" >> "$LOG_FILE"
    echo "=============================================================" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
  fi
  echo ""
}

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "============================================================="
echo "nvfuser Manual CI Test Suite"
echo "============================================================="
echo "Logging detailed output to: $LOG_FILE"
echo ""

# ============================================================
# Dynamic Type Tests
# ============================================================
echo "Running Dynamic Type Tests..."
if [ -d "./bin/lib/dynamic_type" ]; then
  for test_bin in ./bin/lib/dynamic_type/test_*; do
    if [ -x "$test_bin" ]; then
      run_test "$test_bin"
    fi
  done
fi

# ============================================================
# C++ Binary Tests (auto-detected)
# ============================================================
echo ""
echo "Running C++ Binary Tests..."
# Tests that require MPI
MPI_TESTS=("test_multidevice" "tutorial_multidevice")

# Find all test_* and tutorial_* binaries in bin/
if [ -d "./bin" ]; then
  for test_bin in ./bin/test_* ./bin/tutorial_*; do
    if [ -x "$test_bin" ]; then
      test_name=$(basename "$test_bin")

      # Check if this test requires MPI
      requires_mpi=false
      for mpi_test in "${MPI_TESTS[@]}"; do
        if [ "$test_name" = "$mpi_test" ]; then
          requires_mpi=true
          break
        fi
      done

      # Run non-MPI tests directly, skip MPI tests for now
      if [ "$requires_mpi" = false ]; then
        run_test "$test_bin"
      fi
    fi
  done
fi

# ============================================================
# Multidevice Tests (requires MPI)
# ============================================================
echo ""
if type -p mpirun > /dev/null; then
  echo "Running MPI-based Multidevice Tests..."
  for mpi_test in "${MPI_TESTS[@]}"; do
    test_path="./bin/$mpi_test"
    if [ -x "$test_path" ]; then
      run_test mpirun -np 1 "$test_path"
    fi
  done
fi

# ============================================================
# Python Tests
# ============================================================

# Python test directories
PYTHON_TEST_DIRS=("direct" "opinfo")
MPI_PYTHON_TEST_DIRS=("multidevice")

# Run regular Python test directories
if [ -d "tests/python" ]; then
  for test_dir in "${PYTHON_TEST_DIRS[@]}"; do
    if [ -d "tests/python/$test_dir" ]; then
      run_test "pytest tests/python/$test_dir"
    fi
  done

  # Run individual Python test files in the root of tests/python/
  for test_file in tests/python/test_*.py; do
    if [ -f "$test_file" ]; then
      run_test "pytest $test_file"
    fi
  done
fi

# Multidevice Python tests (requires MPI)
if type -p mpirun > /dev/null; then
  for test_dir in "${MPI_PYTHON_TEST_DIRS[@]}"; do
    if [ -d "tests/python/$test_dir" ]; then
      run_test "pytest tests/python/$test_dir"
    fi
  done
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================="
if $failed_tests;
then
  echo "❌ CI tests FAILED, do NOT merge your PR!"
  echo "============================================================="
  echo "Check the log file for details: $LOG_FILE"
  exit 1
else
  echo "✅ CI tests PASSED, ship it!"
  echo "============================================================="
  echo "Full log available at: $LOG_FILE"
  exit 0
fi
