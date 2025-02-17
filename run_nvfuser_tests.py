#!/usr/bin/env python3
import subprocess
import os
import glob
import datetime
from pathlib import Path
import time
import sys


def setup_logging_dir():
    """Create a timestamped directory for test logs"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"test_log_{timestamp}")
    log_dir.mkdir(exist_ok=True)
    return log_dir


def get_cpp_test_executables(build_dir):
    """Find all test executables in the build directory"""
    all_tests = glob.glob(os.path.join(build_dir, "*test*"))

    # Separate nvfuser_tests and other tests
    nvfuser_tests = [
        test for test in all_tests if os.path.basename(test) == "nvfuser_tests"
    ]
    other_tests = [
        test for test in all_tests if os.path.basename(test) != "nvfuser_tests"
    ]

    # Return nvfuser_tests first, followed by other tests
    return nvfuser_tests + other_tests


def get_python_tests(python_test_dir):
    """Find all Python test files"""
    all_tests = glob.glob(os.path.join(python_test_dir, "test_*.py"))

    # Separate test_ops.py and other tests
    test_ops = [test for test in all_tests if os.path.basename(test) == "test_ops.py"]
    other_tests = [
        test for test in all_tests if os.path.basename(test) != "test_ops.py"
    ]

    # Return test_ops.py first, followed by other tests
    return test_ops + other_tests


def run_multidevice_test(test_path, log_dir, dry_run=False):
    """Run a multidevice test using mpirun"""
    test_name = os.path.basename(test_path)
    log_base = log_dir / test_name

    cmd = [
        "mpirun",
        "-np",
        "2",
        "--output-filename",
        f"{log_base}",
        "--merge-stderr-to-stdout",
        test_path,
    ]

    if dry_run:
        print(f"Would run: {' '.join(cmd)}")
        return True

    print(f"Running multidevice test: {test_name}")
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


def run_parallel_tests(tests, run_func, log_dir, dry_run=False):
    """Run tests in parallel across GPUs, running each test exactly once"""
    if dry_run:
        current_tests = {0: None, 1: None}  # Simulate GPU allocation
        test_queue = tests.copy()

        print("\nSimulating parallel execution across 2 GPUs:")

        # Initial allocation
        for gpu_id in [0, 1]:
            if test_queue:
                current_tests[gpu_id] = test_queue.pop(0)
                run_func(current_tests[gpu_id], log_dir, gpu_id, dry_run=True)

        # Simulate rest of queue
        while test_queue:
            # Simulate alternating GPU completion
            gpu_id = len(test_queue) % 2
            print(f"\nGPU {gpu_id} finished {os.path.basename(current_tests[gpu_id])}")
            current_tests[gpu_id] = test_queue.pop(0)
            run_func(current_tests[gpu_id], log_dir, gpu_id, dry_run=True)

        # Show final tests completing
        for gpu_id in [0, 1]:
            if current_tests[gpu_id]:
                print(
                    f"\nGPU {gpu_id} finished {os.path.basename(current_tests[gpu_id])}"
                )

        return [], []

    # Initialize test queue and tracking variables
    test_queue = tests.copy()  # Create a queue of tests to run
    current_processes = {0: None, 1: None}  # GPU ID -> current process
    current_tests = {0: None, 1: None}  # GPU ID -> current test
    start_times = {0: time.time(), 1: time.time()}
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
        for gpu_id in [0, 1]:
            if current_processes[gpu_id] is None:
                free_gpus.append(gpu_id)
                continue

            # Check for timeout
            if time.time() - start_times[gpu_id] > 1200:  # 20 minutes
                print(
                    f"Test {os.path.basename(current_tests[gpu_id])} on GPU {gpu_id} timed out"
                )
                current_processes[gpu_id].kill()
                test = current_tests[gpu_id]

                # Append timeout status to log file
                log_file = log_dir / f"{os.path.basename(test)}.log"
                if not log_file.exists():
                    log_file = log_dir / f"python_{os.path.basename(test)}.log"
                with open(log_file, "a") as f:
                    f.write("\nERROR: Test timed out after 20 minutes\n")

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
    for gpu_id in [0, 1]:
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
            # Look for GTest or pytest failures
            if ("Failure" in line and "test" in line.lower()) or (
                "FAILED" in line and "test" in line.lower()
            ):
                # Add header based on failure type
                if "Failure" in line:
                    failure_summary.append(
                        f"\n=== GTest Failure in {log_file.name} ==="
                    )
                else:
                    failure_summary.append(
                        f"\n=== Pytest Failure in {log_file.name} ==="
                    )

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


def main():
    try:
        # Add argument parsing for dry run
        if len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
            dry_run = True
        else:
            dry_run = False

        # Hardcode paths relative to current directory
        build_dir = "bin"
        python_test_dir = "tests/python"

        log_dir = setup_logging_dir()
        cpp_tests = get_cpp_test_executables(build_dir)
        python_tests = get_python_tests(python_test_dir)

        if not cpp_tests and not python_tests:
            print("No tests found!")
            return 0

        # Separate tests into multidevice and single device
        multidevice_tests = [
            test
            for test in cpp_tests
            if "multidevice" in os.path.basename(test).lower()
        ]
        single_device_tests = [
            test
            for test in cpp_tests
            if "multidevice" not in os.path.basename(test).lower()
        ]

        print(f"Found {len(multidevice_tests)} multidevice tests")
        print(f"Found {len(single_device_tests)} single device tests")
        print(f"Found {len(python_tests)} Python tests")
        print(f"Logs will be written to: {log_dir}")

        success_count = 0
        failed_tests = []
        total_tests = len(cpp_tests) + len(python_tests)

        # Run multidevice tests first
        print("\nMultidevice tests that would run:")
        for test in multidevice_tests:
            if run_multidevice_test(test, log_dir, dry_run=dry_run):
                if not dry_run:
                    success_count += 1
            else:
                if not dry_run:
                    failed_tests.append(os.path.basename(test))

        # Run single device tests in parallel
        print("\nSingle device tests that would run:")
        if not dry_run:
            completed, failed = run_parallel_tests(
                single_device_tests, run_single_device_test, log_dir
            )
            success_count += len(completed)
            failed_tests.extend(failed)
        else:
            run_parallel_tests(
                single_device_tests, run_single_device_test, log_dir, dry_run=True
            )

        # Run Python tests in parallel
        print("\nPython tests that would run:")
        if not dry_run:
            completed, failed = run_parallel_tests(
                python_tests, run_python_test, log_dir
            )
            success_count += len(completed)
            failed_tests.extend([f"python_{test}" for test in failed])
        else:
            run_parallel_tests(python_tests, run_python_test, log_dir, dry_run=True)

        if dry_run:
            # Show what would be written to summary.txt
            print("\nWould create summary.txt with:")
            print(f"Total tests: {total_tests}")
            print(f"Multidevice tests: {len(multidevice_tests)}")
            print(f"Single device tests: {len(single_device_tests)}")
            print(f"Python tests: {len(python_tests)}")

            # Show failure collection simulation
            collect_test_failures(log_dir, dry_run=True)

            print("\nThis was a dry run. No tests were actually executed.")
            return 0

        # Write summary
        with open(log_dir / "summary.txt", "w") as f:
            f.write(f"Total tests: {total_tests}\n")
            f.write(f"Multidevice tests: {len(multidevice_tests)}\n")
            f.write(f"Single device tests: {len(single_device_tests)}\n")
            f.write(f"Python tests: {len(python_tests)}\n")
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
    except KeyboardInterrupt:
        print("\nTest run interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError running tests: {e}")
        return 1


if __name__ == "__main__":
    main()
