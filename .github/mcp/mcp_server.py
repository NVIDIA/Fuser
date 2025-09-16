import subprocess
import pathlib
import argparse

# server.py
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Nvfuser Dev MCP Server")
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
BUILD_DIR = PROJECT_ROOT / "build"


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


def _generate_test_selection_prompt(diff_content: str) -> str:
    """
    This function generated a prompt for LLMs to select relevant
    tests based on diff content.

    Args:
        diff_content (str): The content of the diff to analyze.

    Returns:
        str: The prompt for the LLM.
    """
    prompt = (
        "You are a test selection assistant. Given the following diff content, "
        "select the relevant tests that should be run based on the changes made. "
        "Return the names of the tests as a list of strings.\n\n"
        f"Diff Content:\n{diff_content}\n\n"
        "Your Task:\n"
        "Carefully review the code changes. Based on the functions, data structure, and logic"
        "affected identify the tests from that we should run."
        "- Prioritize tests that directly cover the changed code.\n"
        " - Consider tests of downstream components that might be affected by these changes.\n"
        " - If you are unsure about the relevance of a test, include it.\n"
        " - Tests are located in the `Fuser/tests/cpp` directory.\n"
        "Output format:\n"
        "Return a single line, containing a comma-separate list wiht the name of the test names to run."
        "Do not add any other text, explanation or formatting.\n"
        "Example of output:  nvfuser_tests.test_fusion,nvfuser_tests.test_scheduler,nvfuser_tests.test_codegen\n"
    )
    return prompt.strip()


@mcp.tool()
def run_targeted_tests(
    target_branch: str = "devel", selected_tests: str | None = None
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
        # TODO how do we run tests here?
        test_command = ["pytest", "complete", "me"]
        success, test_output, stderr = run_command(test_command, BUILD_DIR)
        if not success:
            return f"Tests failed. Error: {stderr}\n\nTest Output:\n{test_output}"
        return f"Tests passed successfully.\n\nTest Output:\n{test_output}"

    # retrieve all the files that have been changed wrt to HEAD
    diff_command = [
        "git",
        "diff",
        "--name-only",
        f"{target_branch}...HEAD",
        "--",
        "*.py",
    ]
    success, diff_content, stderr = run_command(diff_command, PROJECT_ROOT)
    if not success:
        return f"Failed to get the diff against {target_branch}. Error: {stderr}"
    if not diff_content.strip():
        return (
            f"No Python files have been changed against {target_branch}. No tests to run."
        )

    # TODO: should we have a simple heuristic search based on regex from changed files?
    # changed_files = diff_content.strip().split("\n")

    # detect all the possibel tests from Fuser/tests/cpp
    return _generate_test_selection_prompt(diff_content.strip())


@mcp.tool()
def propose_unit_tests(file_path: str, target_branch: str = "devel") -> str:
    """
    Analyse the changes in a specific file and propose unit tests to cover those changes.

    Args:
        file_path (str): The path to the file to analyze.
        target_branch (str): The branch to compare against, defaults to "devel".

    Returns:
        str: A proposed unit test or a message indicating no changes were found.
    """
    source_file = pathlib.Path(file_path)
    if not source_file.exists():
        return f"File {file_path} does not exist."

    # get the diff
    diff_command = [
        "git",
        "diff",
        f"{target_branch}...HEAD",
        "--",
        str(source_file),
    ]
    success, diff_content, stderr = run_command(diff_command, PROJECT_ROOT)
    if not success:
        return f"Failed to get diff for {file_path} against {target_branch}. Error: {stderr}"
    
    if not diff_content.strip():
        return f"No changes found in {file_path} against {target_branch}."

    # Generate actual unit test code based on the diff
    return _generate_unit_test_code(diff_content.strip(), file_path)



def _generate_unit_test_code(diff_content: str, file_path: str) -> str:
    """
    This function generates actual unit test code based on the diff content.

    Args:
        diff_content (str): The content of the diff to analyze.
        file_path (str): The path to the file being analyzed.

    Returns:
        str: Generated unit test code.
    """
    # Determine file type for appropriate test generation
    file_ext = pathlib.Path(file_path).suffix.lower()
    
    if file_ext == '.py':
        return _generate_python_unit_tests(diff_content, file_path)
    elif file_ext in ['.cpp', '.h', '.cu']:
        return _generate_cpp_unit_tests(diff_content, file_path)
    else:
        return f"Unsupported file type: {file_ext}. Cannot generate unit tests."


def _generate_python_unit_tests(diff_content: str, file_path: str) -> str:
    """Generate Python unit tests based on diff content."""
    # Extract the module name from file path
    module_name = file_path.replace('.py', '').replace('/', '.').replace('\\', '.')
    if module_name.startswith('.'):
        module_name = module_name[1:]
    
    test_code = f'''import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from {module_name} import *
except ImportError as e:
    print(f"Warning: Could not import {module_name}: {{e}}")

class TestGenerated(unittest.TestCase):
    """Generated unit tests based on changes in {file_path}"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        pass
    
    def tearDown(self):
        """Clean up after each test method."""
        pass
'''
    
    # Analyze diff content to generate specific tests
    if '+def ' in diff_content or '+    def ' in diff_content:
        test_code += '''
    def test_new_functions(self):
        """Test newly added functions"""
        # TODO: Add specific tests for new functions
        pass
'''
    
    if '+class ' in diff_content:
        test_code += '''
    def test_new_class(self):
        """Test newly added class"""
        # TODO: Add specific tests for new class
        pass
'''
    
    if '+    def ' in diff_content and 'def _' not in diff_content:
        test_code += '''
    def test_new_methods(self):
        """Test newly added methods"""
        # TODO: Add specific tests for new methods
        pass
'''
    
    test_code += '''
    def test_modified_functionality(self):
        """Test modified functionality"""
        # TODO: Add tests for modified code
        pass

if __name__ == '__main__':
    unittest.main()
'''
    
    return test_code


def _generate_cpp_unit_tests(diff_content: str, file_path: str) -> str:
    """Generate C++ unit tests based on diff content."""
    test_code = f'''#include <gtest/gtest.h>
#include <gmock/gmock.h>

// Include the header file being tested
#include "{pathlib.Path(file_path).name}"

class TestGenerated : public ::testing::Test {{
protected:
    void SetUp() override {{
        // Set up test fixtures
    }}
    
    void TearDown() override {{
        // Clean up after each test
    }}
}};

// Generated tests based on changes in {file_path}
'''
    
    # Analyze diff content for C++ specific changes
    if '+class ' in diff_content:
        test_code += '''
TEST_F(TestGenerated, TestNewClass) {
    // TODO: Add tests for new class
    // Example: TestClass obj;
    // EXPECT_TRUE(obj.someMethod());
}
'''
    
    if '+    ' in diff_content and ('public:' in diff_content or 'private:' in diff_content):
        test_code += '''
TEST_F(TestGenerated, TestNewMethods) {
    // TODO: Add tests for new methods
    // Example: TestClass obj;
    // EXPECT_EQ(expected, obj.newMethod());
}
'''
    
    test_code += '''
TEST_F(TestGenerated, TestModifiedFunctionality) {
    // TODO: Add tests for modified functionality
    // Example: EXPECT_EQ(expected, modifiedFunction());
}

// Additional test cases based on the specific changes
// TODO: Add more specific test cases based on the diff content
'''
    
    return test_code


def _generate_unit_test_prompt(diff_content: str) -> str:
    """
    This function generates a prompt for LLMs to propose unit tests based on the diff content.

    Args:
        diff_content (str): The content of the diff to analyze.

    Returns:
        str: The prompt for the LLM.
    """
    prompt = (
        "You are a unit test generation assistant. Given the following diff content, "
        "propose unit tests that cover the changes made. "
        "Return the proposed unit test code as a string.\n\n"
        f"Diff Content:\n{diff_content}\n\n"
        "Your Task:\n"
        "Carefully review the code changes and generate unit tests that cover the modified code."
        " - Focus on the modified or new functionality.\n"
        " - Ensure that the tests are comprehensive and cover edge cases.\n"
        " - If the changes are in a specific function, ensure that the tests cover that function.\n"
        " - If the changes are in a class, ensure that the tests cover the class methods and properties.\n"
        " - If the changes are in a module, ensure that the tests cover the module's functionality.\n"
        " - VERY IMPORTANT: place the generated cpp code inside a single markdown code block"
    )
    return prompt.strip()

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
