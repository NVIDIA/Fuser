#!/usr/bin/env python3
import subprocess
import os
import glob
import datetime
from pathlib import Path
import time


def setup_logging_dir():
    """Create a timestamped directory for test logs"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"test_log_{timestamp}")
    log_dir.mkdir(exist_ok=True)
    return log_dir


def get_cpp_test_executables(build_dir):
    """Find all test executables in the build directory"""
    return glob.glob(os.path.join(build_dir, "*test*"))


def get_python_tests(python_test_dir):
    """Find all Python test files"""
    return glob.glob(os.path.join(python_test_dir, "test_*.py"))


def run_multidevice_test(test_path, log_dir):
    """Run a multidevice test using mpirun"""
    test_name = os.path.basename(test_path)
    log_base = log_dir / test_name

    print(f"Running multidevice test: {test_name}")
    try:
        result = subprocess.run(
            [
                "mpirun",
                "-np",
                "2",
                "--output-filename",
                f"{log_base}",
                "--merge-stderr-to-stdout",
                test_path,
            ],
            timeout=1200,  # 20 minute timeout
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


def run_single_device_test(test_path, log_dir, gpu_id):
    """Run a single device test on specified GPU"""
    test_name = os.path.basename(test_path)
    log_file = log_dir / f"{test_name}.log"

    print(f"Running test: {test_name} on GPU {gpu_id}")
    try:
        with open(log_file, "w") as f:
            f.write(f"Test: {test_name} on GPU {gpu_id}\n")
            process = subprocess.Popen(
                [test_path],
                stdout=f,
                stderr=subprocess.STDOUT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)},
            )
        return process
    except Exception as e:
        with open(log_file, "w") as f:
            f.write(f"Test: {test_name} on GPU {gpu_id}\n")
            f.write(f"ERROR: Failed to start test: {str(e)}\n")
        return None


def run_python_test(test_path, log_dir, gpu_id):
    """Run a Python test using pytest on specified GPU"""
    test_name = os.path.basename(test_path)
    log_file = log_dir / f"python_{test_name}.log"

    print(f"Running Python test: {test_name} on GPU {gpu_id}")
    try:
        with open(log_file, "w") as f:
            f.write(f"Test: {test_name} on GPU {gpu_id}\n")
            process = subprocess.Popen(
                ["pytest", test_path, "-v"],
                stdout=f,
                stderr=subprocess.STDOUT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)},
            )
        return process
    except Exception as e:
        with open(log_file, "w") as f:
            f.write(f"Test: {test_name} on GPU {gpu_id}\n")
            f.write(f"ERROR: Failed to start test: {str(e)}\n")
        return None


def run_parallel_tests(tests, run_func, log_dir):
    """Run tests in parallel across two GPUs"""
    current_processes = {0: None, 1: None}  # GPU ID -> current process
    current_tests = {0: None, 1: None}  # GPU ID -> current test
    completed_tests = []
    failed_tests = []
    start_times = {0: time.time(), 1: time.time()}  # GPU ID -> start time

    def start_test(test, gpu_id):
        """Start a test on specified GPU"""
        current_tests[gpu_id] = test
        process = run_func(test, log_dir, gpu_id)
        if process is None:
            # Test failed to start
            failed_tests.append(os.path.basename(test))
            return False
        current_processes[gpu_id] = process
        start_times[gpu_id] = time.time()  # Reset start time for new test
        print(f"Started {os.path.basename(test)} on GPU {gpu_id}")
        return True

    def check_completion():
        """Check if any running tests have completed"""
        for gpu_id in [0, 1]:
            if current_processes[gpu_id] is not None:
                # Check for timeout first
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
                    return gpu_id

                # Check if process completed normally
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
                        f.write(
                            f"Return code: {current_processes[gpu_id].returncode}\n"
                        )

                    if success:
                        completed_tests.append(test)
                    else:
                        failed_tests.append(os.path.basename(test))

                    current_processes[gpu_id] = None
                    current_tests[gpu_id] = None
                    return gpu_id
        return None

    # Start initial tests on both GPUs if available
    if len(tests) > 0:
        start_test(tests[0], 0)
    if len(tests) > 1:
        start_test(tests[1], 1)

    test_index = 2  # Index of next test to run

    # Continue until all tests are completed
    while test_index < len(tests) or any(
        p is not None for p in current_processes.values()
    ):
        # Check for completed tests
        free_gpu = check_completion()

        # If a GPU is free and there are more tests, start the next test
        if free_gpu is not None and test_index < len(tests):
            start_test(tests[test_index], free_gpu)
            test_index += 1

        # Small sleep to prevent busy waiting
        time.sleep(1)

    try:
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


def main():
    try:
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

        # Run multidevice tests first (they manage their own GPUs)
        print("\nRunning multidevice tests...")
        for test in multidevice_tests:
            if run_multidevice_test(test, log_dir):
                success_count += 1
            else:
                failed_tests.append(os.path.basename(test))

        # Run single device tests in parallel
        print("\nRunning single device tests...")
        completed, failed = run_parallel_tests(
            single_device_tests, run_single_device_test, log_dir
        )
        success_count += len(completed)
        failed_tests.extend(failed)

        # Run Python tests in parallel
        print("\nRunning Python tests...")
        completed, failed = run_parallel_tests(python_tests, run_python_test, log_dir)
        success_count += len(completed)
        failed_tests.extend([f"python_{test}" for test in failed])

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
