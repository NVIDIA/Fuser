import pathlib


PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
BUILD_DIR = PROJECT_ROOT / "build"
TESTS_DIR_CPP = PROJECT_ROOT / "tests" / "cpp"
TESTS_DIR_PYTHON = PROJECT_ROOT / "tests" / "python"


def find_example_tests(source_file_path: str) -> str:
    """
    Finds a corresponding test file for a given source file to use as a style guide.

    This function uses heuristics to search for a test file based on the source
    file's name. For example, for "fusion.cpp", it will look for
    "tests/cpp/test_fusion.cpp".

    Args:
        source_file_path (str): The relative path to the source file.

    Returns:
        str: The content of the found test file, or a default message if not found.
    """
    file_stem = pathlib.Path(source_file_path).stem
    print(f"Finding example tests for {source_file_path}, with stem {file_stem}")
    candidate_names = [
        f"test_{file_stem}.cpp",
        f"{file_stem}_test.cpp",
        f"test_{file_stem}.py"
    ]

    example_tests = ""
    for test_name in candidate_names:
        # Construct the full path to the potential test file.
        cpp_test_path = TESTS_DIR_CPP / test_name
        python_test_path = TESTS_DIR_PYTHON / test_name
        
        if cpp_test_path.exists():
            print(f"Found matching example test: {cpp_test_path}")
            example_tests+=cpp_test_path.read()
        if python_test_path.exists(): 
            print(f"Found matching example test: {python_test_path}")
            example_tests+=python_test_path.read()

    return example_tests
