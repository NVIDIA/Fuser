# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import subprocess
import os
import glob
import datetime
from pathlib import Path
import time
import sys


def setup_logging_dir():
    """Create a timestamped directory for test logs and update latest symlink"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"test_log_{timestamp}")
    log_dir.mkdir(exist_ok=True)

    # Check and handle the 'latest' symlink
    latest_link = Path("test_log_latest")
    if latest_link.exists():
        if not latest_link.is_symlink():
            raise RuntimeError(
                "test_log_latest exists but is not a symlink. "
                "Please remove it manually to proceed."
            )
        latest_link.unlink()  # Remove existing symlink

    # Create new symlink pointing to the new log directory
    latest_link.symlink_to(log_dir, target_is_directory=True)

    return log_dir


def get_cpp_test_executables(build_dir):
    """Find all test executables in the build directory"""
    all_tests = glob.glob(os.path.join(build_dir, "*test*"))

    # Separate multidevice tests
    multidevice_tests = [
        test for test in all_tests if "multidevice" in os.path.basename(test).lower()
    ]

    # Get non-multidevice tests
    single_device_tests = [
        test
        for test in all_tests
        if "multidevice" not in os.path.basename(test).lower()
    ]

    # Separate long running tests and others
    priority_tests = [
        os.path.join(build_dir, "nvfuser_tests"),
        os.path.join(build_dir, "test_matmul"),
    ]
    other_tests = [test for test in single_device_tests if test not in priority_tests]

    # Return multidevice tests separately, and ordered single device tests prioritizing long running tests first
    return multidevice_tests, priority_tests + other_tests


def get_python_tests(python_test_dir):
    """Find all Python test files"""
    all_tests = glob.glob(os.path.join(python_test_dir, "test_*.py"))

    # Separate multidevice tests
    multidevice_tests = [
        test for test in all_tests if "multidevice" in os.path.basename(test).lower()
    ]

    # Get non-multidevice tests
    single_device_tests = [
        test
        for test in all_tests
        if "multidevice" not in os.path.basename(test).lower()
    ]

    # Separate test_ops.py and other single device tests
    test_ops = [
        test for test in single_device_tests if os.path.basename(test) == "test_ops.py"
    ]
    other_tests = [
        test for test in single_device_tests if os.path.basename(test) != "test_ops.py"
    ]

    # Return multidevice tests and ordered single device tests separately
    return multidevice_tests, test_ops + other_tests


def get_test_timeout(test_name):
    """Return timeout in seconds for a given test"""
    if test_name in ["nvfuser_tests", "test_matmul"]:
        return 2400  # 40 minutes
    return 600  # 10 minutes


def run_multidevice_test(test_path, log_dir, num_gpus, dry_run=False):
    """Run a multidevice test using mpirun"""
    test_name = os.path.basename(test_path)
    log_base = log_dir / test_name
    timeout = get_test_timeout(test_name)

    cmd = [
        "mpirun",
        "-np",
        str(num_gpus),  # Use available GPU count
        "--output-filename",
        f"{log_base}",
        "--merge-stderr-to-stdout",
        test_path,
    ]

    if dry_run:
        print(f"Would run: {' '.join(cmd)} (timeout: {timeout/60} minutes)")
        return True

    print(f"Running multidevice test: {test_name}")
    try:
        # Redirect output to /dev/null to suppress console output
        with open(os.devnull, "w") as devnull:
            result = subprocess.run(
                cmd, timeout=timeout, stdout=devnull, stderr=subprocess.STDOUT
            )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        with open(f"{log_base}.log", "w") as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"ERROR: Test timed out after {timeout/60} minutes\n")
        return False
    except Exception as e:
        with open(f"{log_base}.log", "w") as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"ERROR: Failed to run test: {str(e)}\n")
        return False


def run_single_device_test(test_path, log_dir, gpu_id, dry_run=False):
    """Run a single device test on specified GPU"""
    test_name = os.path.basename(test_path)
    log_file = log_dir / f"{test_name}.log"

    cmd = [test_path]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}

    if dry_run:
        print(f"Would run: CUDA_VISIBLE_DEVICES={gpu_id} {' '.join(cmd)} > {log_file}")
        return "dry_run_process"

    print(f"Running test: {test_name} on GPU {gpu_id}")
    try:
        with open(log_file, "w") as f:
            f.write(f"Test: {test_name} on GPU {gpu_id}\n")
            process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        return process
    except Exception as e:
        with open(log_file, "w") as f:
            f.write(f"Test: {test_name} on GPU {gpu_id}\n")
            f.write(f"ERROR: Failed to start test: {str(e)}\n")
        return None


def run_python_multidevice_test(test_path, log_dir, num_gpus, dry_run=False):
    """Run a Python multidevice test using mpirun"""
    test_name = os.path.basename(test_path)
    log_base = log_dir / f"python_{test_name}"

    cmd = [
        "mpirun",
        "-np",
        str(num_gpus),  # Use available GPU count
        "--output-filename",
        f"{log_base}",
        "--merge-stderr-to-stdout",
        "pytest",
        test_path,
        "-v",
    ]

    if dry_run:
        print(f"Would run: {' '.join(cmd)}")
        return True

    print(f"Running Python multidevice test: {test_name}")
    try:
        # Redirect output to /dev/null to suppress console output
        with open(os.devnull, "w") as devnull:
            result = subprocess.run(
                cmd, timeout=1200, stdout=devnull, stderr=subprocess.STDOUT
            )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        with open(f"{log_base}.log", "w") as f:
            f.write(f"Test: {test_name}\n")
            f.write("ERROR: Test timed out after 20 minutes\n")
        return False
    except Exception as e:
        with open(f"{log_base}.log", "w") as f:
            f.write(f"Test: {test_name}\n")
            f.write(f"ERROR: Failed to run test: {str(e)}\n")
        return False


def run_python_test(test_path, log_dir, gpu_id, dry_run=False):
    """Run a Python test using pytest on specified GPU"""
    test_name = os.path.basename(test_path)
    log_file = log_dir / f"python_{test_name}.log"

    cmd = ["pytest", test_path, "-v"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}

    if dry_run:
        print(f"Would run: CUDA_VISIBLE_DEVICES={gpu_id} {' '.join(cmd)} > {log_file}")
        return "dry_run_process"

    print(f"Running Python test: {test_name} on GPU {gpu_id}")
    try:
        with open(log_file, "w") as f:
            f.write(f"Test: {test_name} on GPU {gpu_id}\n")
            process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        return process
    except Exception as e:
        with open(log_file, "w") as f:
            f.write(f"Test: {test_name} on GPU {gpu_id}\n")
            f.write(f"ERROR: Failed to start test: {str(e)}\n")
        return None


def run_parallel_tests(log_dir, num_gpus, tests, run_func, dry_run=False):
    """Run tests in parallel across available GPUs"""
    if dry_run:
        current_tests = {i: None for i in range(num_gpus)}  # Simulate GPU allocation
        test_queue = tests.copy()

        print(f"\nSimulating parallel execution across {num_gpus} GPUs:")

        # Initial allocation
        for gpu_id in range(num_gpus):
            if test_queue:
                current_tests[gpu_id] = test_queue.pop(0)
                run_func(current_tests[gpu_id], log_dir, gpu_id, dry_run=True)

        # Simulate rest of queue
        while test_queue:
            # Simulate round-robin GPU completion
            gpu_id = len(test_queue) % num_gpus
            print(f"\nGPU {gpu_id} finished {os.path.basename(current_tests[gpu_id])}")
            current_tests[gpu_id] = test_queue.pop(0)
            run_func(current_tests[gpu_id], log_dir, gpu_id, dry_run=True)

        # Show final tests completing
        for gpu_id in range(num_gpus):
            if current_tests[gpu_id]:
                print(
                    f"\nGPU {gpu_id} finished {os.path.basename(current_tests[gpu_id])}"
                )

        return [], []

    # Initialize test queue and tracking variables
    test_queue = tests.copy()
    current_processes = {i: None for i in range(num_gpus)}
    current_tests = {i: None for i in range(num_gpus)}
    start_times = {i: time.time() for i in range(num_gpus)}
    completed_tests = []
    failed_tests = []

    def start_test(test, gpu_id):
        """Start a test on specified GPU"""
        current_tests[gpu_id] = test
        process = run_func(test, log_dir, gpu_id)
        if process is None:
            failed_tests.append(os.path.basename(test))
            return False
        current_processes[gpu_id] = process
        start_times[gpu_id] = time.time()
        print(f"Started {os.path.basename(test)} on GPU {gpu_id}")
        return True

    def check_completion():
        """Check if any running tests have completed and return list of free GPUs"""
        free_gpus = []
        for gpu_id in range(num_gpus):
            if current_processes[gpu_id] is None:
                free_gpus.append(gpu_id)
                continue

            # Get timeout for current test
            test_name = os.path.basename(current_tests[gpu_id])
            timeout = get_test_timeout(test_name)

            # Check for timeout
            if time.time() - start_times[gpu_id] > timeout:
                print(
                    f"Test {test_name} on GPU {gpu_id} timed out after {timeout/60} minutes"
                )
                current_processes[gpu_id].kill()
                test = current_tests[gpu_id]

                # Append timeout status to log file
                log_file = log_dir / f"{os.path.basename(test)}.log"
                if not log_file.exists():
                    log_file = log_dir / f"python_{os.path.basename(test)}.log"
                with open(log_file, "a") as f:
                    f.write(f"\nERROR: Test timed out after {timeout/60} minutes\n")

                failed_tests.append(os.path.basename(test))
                current_processes[gpu_id] = None
                current_tests[gpu_id] = None
                free_gpus.append(gpu_id)
                continue

            # Check if process completed
            if current_processes[gpu_id].poll() is not None:
                test = current_tests[gpu_id]
                success = current_processes[gpu_id].returncode == 0
                print(
                    f"Completed {os.path.basename(test)} on GPU {gpu_id}: {'Success' if success else 'Failed'}"
                )

                # Append completion status to log file
                log_file = log_dir / f"{os.path.basename(test)}.log"
                if not log_file.exists():
                    log_file = log_dir / f"python_{os.path.basename(test)}.log"
                with open(log_file, "a") as f:
                    f.write(
                        f"\nTest completed with {'success' if success else 'failure'}\n"
                    )
                    f.write(f"Return code: {current_processes[gpu_id].returncode}\n")

                if success:
                    completed_tests.append(test)
                else:
                    failed_tests.append(os.path.basename(test))

                current_processes[gpu_id] = None
                current_tests[gpu_id] = None
                free_gpus.append(gpu_id)

        return free_gpus

    # Initial test distribution
    for gpu_id in range(num_gpus):
        if test_queue and current_processes[gpu_id] is None:
            start_test(test_queue.pop(0), gpu_id)

    # Main loop
    try:
        while test_queue or any(p is not None for p in current_processes.values()):
            # Check for completed tests
            free_gpus = check_completion()

            # Start new tests on free GPUs
            for gpu_id in free_gpus:
                if test_queue:  # If there are tests waiting to be run
                    start_test(test_queue.pop(0), gpu_id)

            # Small sleep to prevent busy waiting
            time.sleep(1)

        return completed_tests, failed_tests

    except KeyboardInterrupt:
        # Kill any running processes
        for process in current_processes.values():
            if process is not None:
                process.kill()
        raise
    finally:
        # Ensure all processes are cleaned up
        for process in current_processes.values():
            if process is not None:
                process.kill()


def collect_test_failures(log_dir, dry_run=False):
    """Scan log files for test failures and collect context"""
    if dry_run:
        print(
            "\nWould scan log files for failures and create failure_summary.txt with:"
        )
        print("- 5 lines of context before each failure")
        print("- 20 lines of context after each failure")
        print("- Coverage for both GTest and Pytest failures")
        print("- Timeout failure information")
        return False

    failure_summary = []

    for log_file in log_dir.glob("*.log"):
        with open(log_file, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            # Look for GTest failures between square brackets
            if "[" in line and "]" in line:
                result_part = line[line.find("[") : line.find("]") + 1]
                if "FAILED" in result_part or "TIMEOUT" in result_part:
                    failure_summary.append(
                        f"\n=== GTest Failure in {log_file.name} ==="
                    )
                    # Collect 5 lines before and 20 lines after the failure
                    start = max(0, i - 5)
                    end = min(len(lines), i + 21)
                    failure_summary.extend(lines[start:end])
                continue  # Skip other checks for this line if it was a bracketed result

            # Look for pytest failures
            if "FAILED" in line and "test" in line.lower():
                failure_summary.append(f"\n=== Pytest Failure in {log_file.name} ===")
                # Collect 5 lines before and 20 lines after the failure
                start = max(0, i - 5)
                end = min(len(lines), i + 21)
                failure_summary.extend(lines[start:end])

            # Look for timeout failures
            if "ERROR: Test timed out" in line:
                failure_summary.append(f"\n=== Timeout in {log_file.name} ===")
                failure_summary.append(line)

    # Write failure summary if there were any failures
    if failure_summary:
        with open(log_dir / "failure_summary.txt", "w") as f:
            f.write("=== Test Failure Summary ===\n")
            f.write("".join(failure_summary))

        return True
    return False


def get_available_gpus():
    """Check how many NVIDIA GPUs are available"""
    try:
        # Run nvidia-smi to get GPU count
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        # Count non-empty lines in output
        gpus = [line.strip() for line in result.stdout.split("\n") if line.strip()]
        return len(gpus)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Warning: Could not query GPU count using nvidia-smi")
        return 0


def main():
    # Add argument parsing for dry run
    if len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        dry_run = True
    else:
        dry_run = False

    # Check GPU availability
    gpu_count = get_available_gpus()
    if gpu_count < 1:
        print("Error: No GPUs found for testing")
        return 1

    print(f"Found {gpu_count} GPUs available for testing")

    # Hardcode paths relative to current directory
    build_dir = "bin"
    python_test_dir = "tests/python"

    log_dir = setup_logging_dir()
    multidevice_tests, single_device_tests = get_cpp_test_executables(build_dir)
    python_multidevice_tests, python_single_tests = get_python_tests(python_test_dir)

    if not (multidevice_tests or single_device_tests) and not (
        python_multidevice_tests or python_single_tests
    ):
        print("No tests found!")
        return 0

    print(f"Found {len(multidevice_tests)} C++ multidevice tests")
    print(f"Found {len(single_device_tests)} C++ single device tests")
    print(f"Found {len(python_multidevice_tests)} Python multidevice tests")
    print(f"Found {len(python_single_tests)} Python single device tests")
    print(f"Logs will be written to: {log_dir}")

    success_count = 0
    failed_tests = []
    total_tests = (
        len(multidevice_tests)
        + len(single_device_tests)
        + len(python_multidevice_tests)
        + len(python_single_tests)
    )

    # Run all multidevice tests first
    print("\nC++ multidevice tests:")
    for test in multidevice_tests:
        if run_multidevice_test(test, log_dir, gpu_count, dry_run=dry_run):
            if not dry_run:
                success_count += 1
        else:
            if not dry_run:
                failed_tests.append(os.path.basename(test))

    print("\nPython multidevice tests:")
    for test in python_multidevice_tests:
        if run_python_multidevice_test(test, log_dir, gpu_count, dry_run=dry_run):
            if not dry_run:
                success_count += 1
        else:
            if not dry_run:
                failed_tests.append(f"python_{os.path.basename(test)}")

    # Run single device tests in parallel using all available GPUs
    print("\nC++ single device tests:")
    if not dry_run:
        completed, failed = run_parallel_tests(
            log_dir, gpu_count, single_device_tests, run_single_device_test
        )
        success_count += len(completed)
        failed_tests.extend(failed)
    else:
        run_parallel_tests(
            log_dir,
            gpu_count,
            single_device_tests,
            run_single_device_test,
            dry_run=True,
        )

    # Run Python single device tests in parallel
    print("\nPython single device tests:")
    if not dry_run:
        completed, failed = run_parallel_tests(
            log_dir, gpu_count, python_single_tests, run_python_test
        )
        success_count += len(completed)
        failed_tests.extend([f"python_{test}" for test in failed])
    else:
        run_parallel_tests(
            log_dir, gpu_count, python_single_tests, run_python_test, dry_run=True
        )

    if dry_run:
        # Show what would be written to summary.txt
        print("\nWould create summary.txt with:")
        print(f"Total tests: {total_tests}")
        print(f"Multidevice tests: {len(multidevice_tests)}")
        print(f"Single device tests: {len(single_device_tests)}")
        print(f"Python tests: {len(python_multidevice_tests)}")

        # Show failure collection simulation
        collect_test_failures(log_dir, dry_run=True)

        print("\nThis was a dry run. No tests were actually executed.")
        return 0

    # Write summary
    with open(log_dir / "summary.txt", "w") as f:
        f.write(f"Total tests: {total_tests}\n")
        f.write(f"Multidevice tests: {len(multidevice_tests)}\n")
        f.write(f"Single device tests: {len(single_device_tests)}\n")
        f.write(f"Python tests: {len(python_multidevice_tests)}\n")
        f.write(f"Successful: {success_count}\n")
        f.write(f"Failed: {len(failed_tests)}\n")
        if failed_tests:
            f.write("\nFailed tests:\n")
            for test in failed_tests:
                f.write(f"- {test}\n")

    # Collect and summarize failures
    if failed_tests:
        if collect_test_failures(log_dir):
            print(
                f"Detailed failure information available in: {log_dir}/failure_summary.txt"
            )

    print(f"\nTest run complete. {success_count}/{total_tests} tests passed.")
    if failed_tests:
        print("Failed tests:")
        for test in failed_tests:
            print(f"- {test}")
    print(f"Logs available in: {log_dir}")

    return 0 if not failed_tests else 1


if __name__ == "__main__":
    main()
