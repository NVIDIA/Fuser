def generate_test_selection_prompt(diff_content: str, cpp_test_folder: str, python_test_folder: str) -> str: 
    """
    This is the prompt for LLMs to select specific tests to run based on new changes. 
    The chages are in diff_content. 
    """

    prompt = f"""
You are an expert software engineer adn test strategist for the nvFuser project. 
Your task is to analyse a code change, and determine which existing unit tests are most relevant for validating this change. 
You can read the existing tests directly from the CPP test folder {cpp_test_folder}, and from the Python test folder {python_test_folder}.
** Code changes (diff) **:
```diff 
{diff_content}
```
** Your task **: 
Carefully review the code changes. Based on the functions, data structures, and logic affected, 
identify the tests that we should run. Return a single line containing a comma-separated list of 
the exact test names to run. Do not include any other text, explanation, or formatting. 
"""
    return prompt.strip()