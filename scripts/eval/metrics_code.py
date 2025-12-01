import re
import math
import cmath
import collections
import itertools
import heapq
import bisect
import sys

def extract_python_code(text: str) -> str:
    """
    Extracts only the Python code block from the LLM output.
    If Markdown code blocks (```python ... ```) are found, the content inside is returned.
    Otherwise, it returns the original text or applies heuristics to strip non-code text.
    """
    # 1. Find Markdown code blocks (```python ... ``` or ``` ... ```)
    pattern = r"```(?:python)?\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        # Return the content of the first code block found
        return matches[0].strip()
    
    # 2. If no Markdown is found, return the text as is.
    # (Optional: Add heuristics here to remove chatty conversational text if needed)
    return text.strip()

def run(code: str, tests, setup: str = ""):
    """
    Executes the model-generated code and the provided tests within the same namespace.
    
    Pre-loads common standard libraries (math, re, etc.) into the namespace 
    because some datasets (like MBPP) implicitly use them without import statements.

    Returns:
        all_passed (bool): True if all tests passed.
        passed_tests (int): Number of tests that passed.
        total_tests (int): Total number of tests.
    """
    # Pre-load common libraries into the execution namespace
    ns = {
        "math": math,
        "cmath": cmath,
        "re": re,
        "collections": collections,
        "itertools": itertools,
        "heapq": heapq,
        "bisect": bisect,
        "sys": sys,
    }
    
    passed_tests = 0
    total_tests = len(tests)

    # 1. Execute the model's generated code (function definitions, etc.)
    try:
        # Combine setup code and model code
        full_code = (setup or "") + "\n" + (code or "")
        exec(full_code, ns, ns)
    except Exception as e:
        # If syntax error or runtime error occurs during definition, fail all tests.
        # Uncomment the line below for debugging:
        # print(f"Execution Error during code definition: {e}") 
        return False, passed_tests, total_tests

    # 2. Execute test cases
    all_passed = True
    for t in tests:
        try:
            # tests are typically assertion strings like "assert func(1) == 2"
            exec(t, ns, ns)
            passed_tests += 1
        except Exception:
            all_passed = False

    return all_passed and (passed_tests == total_tests), passed_tests, total_tests


def evaluate_code(preds, data):
    """
    Evaluates the predictions against the dataset using execution-based metrics.
    
    Args:
        preds (dict): Dictionary mapping example IDs to generated code strings.
        data (Dataset): The dataset containing tasks and test cases.
        
    Returns:
        dict: A dictionary containing 'pass_rate' and 'test_pass_rate'.
    """
    problem_passed = 0
    total_problems = len(data)
    tests_passed = 0
    total_tests = 0

    for ex in data:
        # Ensure ID is a string to match the keys in preds
        ex_id = str(ex.get("task_id", ex.get("id"))) 
        
        # Retrieve the raw prediction from the model
        raw_prediction = preds.get(ex_id, "")
        
        # [IMPORTANT] Clean the output to extract only valid Python code
        cleaned_code = extract_python_code(raw_prediction)
        
        tests = ex.get("test_list", []) or []
        
        # Run execution
        all_passed, passed, t_total = run(
            cleaned_code, 
            tests, 
            ex.get("test_setup_code", "")
        )
        
        if all_passed and t_total > 0:
            problem_passed += 1
        
        tests_passed += passed
        total_tests += t_total

    return {
        "pass_rate": problem_passed / total_problems if total_problems else 0.0,
        "test_pass_rate": tests_passed / total_tests if total_tests else 0.0,
    }