#!/bin/bash

# --- Configuration ---
TEST_EXECUTABLE="/opt/pytorch/Fuser/bin/test_nvfuser"
GTEST_FILTER="*Scheduler*"
NUM_GPUS=2 # Run on GPUs 0 to NUM_GPUS-1
LOG_DIR="./gtest_parallel_logs" # Directory to store logs

# --- Argument Parsing ---
DEFAULT_ITERATIONS=1
if [[ "$1" =~ ^[1-9][0-9]*$ ]]; then
  TOTAL_ITERATIONS="$1"
  echo "Running for specified ${TOTAL_ITERATIONS} iterations."
elif [ -z "$1" ]; then
  TOTAL_ITERATIONS="${DEFAULT_ITERATIONS}"
  echo "No iteration count specified, defaulting to ${TOTAL_ITERATIONS} iteration."
else
  echo "Usage: $0 [number_of_iterations]"
  echo "Error: Invalid number of iterations specified: $1. Please provide a positive integer."
  exit 1
fi

# --- Script Logic ---

echo "Starting parallel GPU test runner..."
echo "Test Executable: ${TEST_EXECUTABLE}"
echo "GTest Filter: ${GTEST_FILTER}"
echo "Number of GPUs: ${NUM_GPUS}"
echo "Log Directory: ${LOG_DIR}"
echo "Total Iterations: ${TOTAL_ITERATIONS}"
echo "----------------------------------------"

mkdir -p "${LOG_DIR}"

# Clear previous logs for these specific GPUs if they exist
# Or initialize empty log files
echo "Initializing logs..."
for (( gpu_id=0; gpu_id<${NUM_GPUS}; gpu_id++ )); do
  log_file="${LOG_DIR}/gpu_${gpu_id}.log"
  # Create/clear the log file
  > "${log_file}"
done
echo "Log initialization complete."
echo "----------------------------------------"

script_exit_status=0
segfault_message=""

for (( iteration=1; iteration<=${TOTAL_ITERATIONS}; iteration++ )); do
  echo "[Iteration ${iteration}/${TOTAL_ITERATIONS}] Starting parallel runs..."

  pids=() # Array to store background process IDs

  for (( gpu_id=0; gpu_id<${NUM_GPUS}; gpu_id++ )); do
    # Log file name is now consistent per GPU across iterations
    log_file="${LOG_DIR}/gpu_${gpu_id}.log"
    echo "  Launching test on GPU ${gpu_id}, appending to ${log_file}"

    # Append to the log file using >>
    (
      echo "--- Iteration ${iteration} Start ---"
      CUDA_VISIBLE_DEVICES=${gpu_id} ${TEST_EXECUTABLE} --gtest_filter="${GTEST_FILTER}"
      # Capture exit status within the subshell
      subshell_exit_status=$?
      echo "--- Iteration ${iteration} End (Exit Status: ${subshell_exit_status}) ---"
      # Exit the subshell with the captured status to potentially detect non-zero exits later if needed
      # exit ${subshell_exit_status} # Disabled for now, focusing on log content check
    ) >> "${log_file}" 2>&1 &

    pids+=($!) # Store the PID of the background process
  done

  echo "  Waiting for ${#pids[@]} tests to complete..."
  # Wait for all background jobs launched in this iteration to finish
  wait # Simple wait for all jobs

  echo "[Iteration ${iteration}/${TOTAL_ITERATIONS}] All runs completed."

  # --- Check for Segmentation Faults ---
  echo "  Checking logs for errors..."
  segfault_detected_in_iteration=false
  for (( gpu_id=0; gpu_id<${NUM_GPUS}; gpu_id++ )); do
      log_file="${LOG_DIR}/gpu_${gpu_id}.log"
      # Check for common segfault patterns in the log content generated *this iteration*
      # A simple approach: check the whole file, assuming failure stops logging
      if grep -E -q "Segmentation fault|core dumped|unaligned tcache chunk detected|invalid size" "${log_file}"; then
          echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
          echo "!!! SEGFAULT DETECTED ON GPU ${gpu_id} IN ITERATION ${iteration} !!!"
          echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
          echo "Last 100 lines from ${log_file}:"
          echo "-----------------------------------------------"
          tail -n 100 "${log_file}"
          echo "-----------------------------------------------"
          segfault_detected_in_iteration=true
          script_exit_status=1 # Indicate an error occurred
          segfault_message="Execution stopped due to segmentation fault on GPU ${gpu_id}."
          break # Stop checking other GPUs for this iteration
      fi
  done

  if ${segfault_detected_in_iteration}; then
      echo "Stopping further iterations due to detected segfault."
      break # Exit the main iteration loop
  fi
  echo "  No segfaults detected in this iteration."
  echo "----------------------------------------"
done

echo "========================================"
if [ ${script_exit_status} -eq 0 ]; then
  echo "Script finished successfully after ${iteration}/${TOTAL_ITERATIONS} iterations."
else
  echo "${segfault_message}"
  echo "Script stopped after iteration ${iteration} due to error."
fi
echo "========================================"

exit ${script_exit_status}
