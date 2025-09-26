import subprocess
import pathlib
import argparse

from unit_test.find_example_tests import find_example_tests
from unit_test.generate_new_tests import generate_unit_test_prompt
from unit_test.find_targeted_tests import generate_test_selection_prompt
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Nvfuser Dev MCP Server")
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
BUILD_DIR = PROJECT_ROOT / "build"
TESTS_DIR_CPP = PROJECT_ROOT / "tests" / "cpp"
TESTS_DIR_PYTHON = PROJECT_ROOT / "tests" / "python"


@mcp.tool()
def test_mcp_is_running() -> str:
    return "Yes, MCP is running!"


def run_command(command: list[str], cwd: pathlib.Path) -> str:
    """
    Helper function for running commands in the MCP server

    Args:
        command (list[str]): The command to run as a list of strings.
        cwd (pathlib.Path): The directory in which to run the command.
    Returns:
        str: The output of the command.
    """
    try:
        p = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        success = p.returncode == 0
        return success, p.stdout, p.stderr
    except Exception as e:
        return False, "", f"Error: {str(e)}"


@mcp.tool()
def build_nvfuser() -> str:
    """
    Build the nvfuser project

    Returns:
        str:
            A message indicating whether the build was successful or failed.
            If the build failed, the full build log is included in the message.
    """
    args = parse_args()
    # if script `/usr/local/bin/_bn` exists, use it
    if pathlib.Path("/usr/local/bin/_bn").exists():
        command = ["/usr/local/bin/_bn"]
    else:
        command = [
            pathlib.Path(args.python_path),
            "-m",
            "pip",
            "install",
            "-v",
            "-e",
            "./python",
            "--no-build-isolation",
        ]

    p = subprocess.run(
        command,
        cwd=PROJECT_ROOT,  # go to nvfuser root
        check=False,
        capture_output=True,
        text=True,
    )
    if p.returncode == 0:
        return "Nvfuser build was successful"
    else:
        return "Nvfuser build was failed, here are the full build log\n\n\n" + p.stderr


@mcp.tool()
def run_targeted_tests(
    target_branch: str = "main", selected_tests: str | None = None
) -> str:
    """
    This function checks for changes in the Python files, against a target_branch.
    It then finds the corresponding existent tests, and runs only that subset.

    Args:
        target_branch (str): The branch to compare against, defaults to "devel".
        selected_tests (str | None): Comma-separated list of tests to run. If provided, this overrides the diff-based selection.
    Returns:
        str: The result of the tests that have been run.
    """
    # if the agent has already selected the tests we can use them
    if selected_tests is not None:
        tests_to_run = [
            test.strip() for test in selected_tests.split(",") if test.strip()
        ]
        if not tests_to_run:
            return "No valid tests were provided to run."
        
        # Run tests using the nvfuser recipe
        # Separate Python and C++ tests
        python_tests = [test for test in tests_to_run if test.endswith('.py') or 'python' in test.lower()]
        cpp_tests = [test for test in tests_to_run if not test.endswith('.py') and 'python' not in test.lower()]
        
        results = []
        
        if python_tests:
            results.append("=== Running Python Tests ===")
            results.append(_run_python_tests(python_tests))
        
        if cpp_tests:
            results.append("=== Running C++ Tests ===")
            results.append(_run_cpp_tests(cpp_tests))
        
        return "\n\n".join(results)

    # retrieve all the files that have been changed wrt to HEAD
    diff_command = [
        "git",
        "diff",
        f"{target_branch}...HEAD",
    ]
    success, diff_content, stderr = run_command(diff_command, PROJECT_ROOT)
    if not success:
        return f"Failed to get the diff against {target_branch}. Error: {stderr}"
    if not diff_content.strip():
        return (
            f"No files have been changed against {target_branch}. No tests to run."
        )

    # Analyze changed files to determine which tests to run
    return generate_test_selection_prompt(diff_content, TESTS_DIR_CPP, TESTS_DIR_PYTHON)


@mcp.tool()
def propose_unit_tests(target_branch: str = "main") -> str:
    """
    Analyse the new code changes and propose unit tests to cover those changes.

    Args:
        target_branch (str): The branch to compare against, defaults to "devel".

    Returns:
        str: A proposed unit test or a message indicating no changes were found.
    """
    # get the list of all files changed against target_branch
    diff_command = [
        "git",
        "diff",
        "--name-only",
        f"{target_branch}...HEAD",
    ]
    success, changed_files, stderr = run_command(diff_command, PROJECT_ROOT)
    if not success:
        return f"Failed to get diff against {target_branch}. Error: {stderr}"
    
    if not diff_content.strip():
        return f"No changes found against {target_branch}."
    
    # filter out the files to exclude mcp directory 
    excluded_folder = ".github/mcp"
    changed_files = [ f for f in changed_files.strip().split("\n") if not f.startswith(excluded_folder) ]

    if not changed_files: 
        return "No relevant chagnes found against {target_branch}."
    
    # now for-loop against each file 
    all_proposals = [] 
    for file_path in changed_files: 
        source_file = PROJECT_ROOT / file_path
        if not source_file.exists():
            return f"File {file_path} does not exist."
        diff_command = [
            "git", "diff", f"{target_branch}...HEAD", "--", str(source_file)
        ]
        success, diff_content, stderr = run_command(diff_command, PROJECT_ROOT)

        if not diff_content.strip():
            # skip this case 
            continue 
        if not success:
            return f"Failed to get diff for {file_path} against {target_branch}. Error: {stderr}"
        # Find an existing test to use as a style guide
        example_test_content = find_example_tests(source_file)
        proposal_prompt = generate_unit_test_prompt(diff_content, file_path, example_test_content)
        # the prompt will run within the agent 
        all_proposals.append(f"## {file_path}\n\n{proposal_prompt}")
    # Find an existing test to use as a style guide
    example_test_content = find_example_tests(file_path)
    return generate_unit_test_prompt(diff_content, file_path, example_test_content)


def _run_python_tests(test_patterns: list[str]) -> str:
    """
    Run Python tests using the nvfuser recipe.
    
    Args:
        test_patterns (list[str]): List of test patterns or test files to run.
    
    Returns:
        str: Results of running the tests.
    """
    test_results = []
    
    for test in test_patterns:
        if test.endswith('.py'):
            # It's a test file, run it directly
            test_file_path = PROJECT_ROOT / "tests" / "python" / test
            if test_file_path.exists():
                test_command = [
                    "python", 
                    str(test_file_path), 
                    "-vs"
                ]
            else:
                test_results.append(f"❌ Test file {test} not found at {test_file_path}")
                continue
        else:
            # It's a test pattern, run with -k filter
            test_command = [
                "python", 
                "-m", 
                "pytest", 
                "tests/python/", 
                "-vs", 
                "-k", 
                test
            ]
        
        success, test_output, stderr = run_command(test_command, PROJECT_ROOT)
        if success:
            test_results.append(f"✅ {test} - PASSED\n{test_output}")
        else:
            test_results.append(f"❌ {test} - FAILED\nError: {stderr}\nOutput: {test_output}")
    
    return "\n\n".join(test_results)


def _run_cpp_tests(test_patterns: list[str]) -> str:
    """
    Run C++ tests using the nvfuser recipe.
    
    Args:
        test_patterns (list[str]): List of test patterns or test executables to run.
    
    Returns:
        str: Results of running the tests.
    """
    test_results = []
    
    for test in test_patterns:
        # Look for test executables in the build directory
        test_executable = BUILD_DIR / "bin" / f"test_{test}"
        
        if test_executable.exists():
            test_command = [
                str(test_executable),
                "--gtest_filter=*"
            ]
        else:
            # Try to find any test executable that matches the pattern
            test_executable = BUILD_DIR / "bin" / test
            if test_executable.exists():
                test_command = [str(test_executable)]
            else:
                test_results.append(f"❌ Test executable {test} not found in {BUILD_DIR / 'bin'}")
                continue
        
        success, test_output, stderr = run_command(test_command, BUILD_DIR)
        if success:
            test_results.append(f"✅ {test} - PASSED\n{test_output}")
        else:
            test_results.append(f"❌ {test} - FAILED\nError: {stderr}\nOutput: {test_output}")
    
    return "\n\n".join(test_results)


def parse_args():
    """Input parser"""
    parser = argparse.ArgumentParser(description="MCP server")
    parser.add_argument(
        "--python-path",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # Start the server
    mcp.run()
