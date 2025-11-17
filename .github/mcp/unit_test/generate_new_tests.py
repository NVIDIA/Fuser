def generate_unit_test_prompt(diff: str, file_path: str, example: str) -> str:
    """
    Function to generate ahigh-quality prompt for generating new unit tests
    Args: 
        diff: (str), difference in content wrt upstream 
        file_path: (str), path of the modified files 
        example: (str), string content of existing tests, if any

    Returns: 
        prompt: (str), the prompt for generating new tests
    """
    prompt = f""" 
    You are an expert Cpp and Python developer for the nvFuser project. 
    Your task is to write a complete, ready-to-compile unit test basd on the changes done in the repo.
    Please, write down the unittests and ask the user if they agree or not with the changes. 
    ** Changes files: ** `{file_path}`
    ** Code diff: **
    ```diff 
    {diff} 
    ```
    ** Possible examples based on existing tests:**
    `{example}` 
    ** Your Task:** 
    Write a new unit test, or add to existent unit tests, a logic that validates the new or modified code in the diff. 
    - The test should be complete and include necessary assertions
    - Follow the style and structure of the provided examples, if there are any 
    - Focus on the modified or new functionality
    - Ensure that the tests are comprehensive and cover edge cases
    - If the changes are in a specific function, ensure that the tests cover that function
    - If the changes are in a class, ensure that the tests cover the class methods and properties
    - If the changes are in a module, ensure that the tests cover the module's functionality
    - VERY IMPORTANT: place the generated cpp code inside a single markdown code block
    """
    return prompt.strip()