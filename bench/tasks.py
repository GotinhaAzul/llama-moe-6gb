"""Generic Python coding tasks used by the quality gate."""

import re

FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

TASKS = [
    {
        "id": "median",
        "category": "algorithms",
        "prompt": "Write a Python function median(nums) that takes a non-empty list of numbers "
                  "and returns its median as a float. Return only its code in a single ```python fence.",
        "tests": (
            "assert median([3, 1, 2]) == 2.0\n"
            "assert median([4, 1, 3, 2]) == 2.5\n"
            "assert median([5]) == 5.0\n"
            "assert median([-3, -1, -2]) == -2.0\n"
            "assert isinstance(median([1, 2, 3]), float)\n"
        ),
    },
    {
        "id": "flatten",
        "category": "recursion",
        "prompt": "Write a Python function flatten(nested) that takes an arbitrarily nested list of "
                  "integers and returns a flat list of the integers in order. "
                  "Return only its code in a single ```python fence.",
        "tests": (
            "assert flatten([]) == []\n"
            "assert flatten([1, [2, [3, 4], 5], 6]) == [1, 2, 3, 4, 5, 6]\n"
            "assert flatten([[[]]]) == []\n"
            "assert flatten([[], [1], []]) == [1]\n"
        ),
    },
    {
        "id": "balanced",
        "category": "data-structures",
        "prompt": "Write a Python function is_balanced(s) that takes a string containing only the "
                  "characters ()[]{} and returns True if the brackets are correctly balanced, else False. "
                  "An empty string is balanced. Return only its code in a single ```python fence.",
        "tests": (
            "assert is_balanced('([]{})') is True\n"
            "assert is_balanced('([)]') is False\n"
            "assert is_balanced('') is True\n"
            "assert is_balanced('((') is False\n"
            "assert is_balanced('{[()]}') is True\n"
            "assert is_balanced('}{') is False\n"
        ),
    },
    {
        "id": "binary_search",
        "category": "algorithms",
        "prompt": "Write a Python function binary_search(sorted_list, target) that returns the index of "
                  "target in the sorted list, or -1 if it is absent. Use binary search, not list.index. "
                  "Return only its code in a single ```python fence.",
        "tests": (
            "assert binary_search([1, 3, 5, 7], 5) == 2\n"
            "assert binary_search([1, 3, 5, 7], 4) == -1\n"
            "assert binary_search([], 1) == -1\n"
            "assert binary_search([2], 2) == 0\n"
            "assert binary_search([1, 2, 3, 4, 5], 1) == 0\n"
            "assert binary_search([1, 2, 3, 4, 5], 5) == 4\n"
        ),
    },
    {
        "id": "merge_sorted",
        "category": "algorithms",
        "prompt": "Write a Python function merge_sorted(a, b) that takes two sorted lists of numbers and "
                  "returns a new sorted list containing all elements. Do not modify the inputs. "
                  "Return only its code in a single ```python fence.",
        "tests": (
            "x, y = [1, 3, 5], [2, 4, 6]\n"
            "assert merge_sorted(x, y) == [1, 2, 3, 4, 5, 6]\n"
            "assert (x, y) == ([1, 3, 5], [2, 4, 6])\n"
            "assert merge_sorted([], [1]) == [1]\n"
            "assert merge_sorted([2], []) == [2]\n"
            "assert merge_sorted([1, 1, 2], [1, 3]) == [1, 1, 1, 2, 3]\n"
        ),
    },
    {
        "id": "first_unique",
        "category": "strings",
        "prompt": "Write a Python function first_unique_char(s) that returns the first non-repeating "
                  "character in the string s, or an empty string if there is none. "
                  "Return only its code in a single ```python fence.",
        "tests": (
            "assert first_unique_char('leetcode') == 'l'\n"
            "assert first_unique_char('aabb') == ''\n"
            "assert first_unique_char('swiss') == 'w'\n"
            "assert first_unique_char('') == ''\n"
            "assert first_unique_char('a') == 'a'\n"
        ),
    },
    {
        "id": "two_sum",
        "category": "algorithms",
        "prompt": "Write a Python function two_sum(nums, target) that returns a tuple (i, j) of indices "
                  "such that nums[i] + nums[j] == target, or None if no such pair exists. "
                  "Return only its code in a single ```python fence.",
        "tests": (
            "assert two_sum([2, 7, 11, 15], 9) == (0, 1)\n"
            "assert two_sum([3, 3], 6) == (0, 1)\n"
            "assert two_sum([1, 2, 3], 7) is None\n"
            "r = two_sum([3, 2, 4], 6)\n"
            "assert r is not None and r[0] != r[1] and [3, 2, 4][r[0]] + [3, 2, 4][r[1]] == 6\n"
        ),
    },
    {
        "id": "fix_negatives",
        "category": "debugging",
        "prompt": "The Python function below has a bug: it mutates the list while iterating over it, "
                  "so consecutive negative numbers are skipped.\n\n"
                  "```python\ndef remove_negatives(nums):\n    for n in nums:\n        if n < 0:\n            nums.remove(n)\n    return nums\n```\n\n"
                  "Return only the corrected function with the same name and signature in a single "
                  "```python fence. It must return a list with negatives removed and must not mutate the input.",
        "tests": (
            "data = [-1, -2, 3]\n"
            "assert remove_negatives(data) == [3]\n"
            "assert data == [-1, -2, 3]\n"
            "assert remove_negatives([1, 2]) == [1, 2]\n"
            "assert remove_negatives([]) == []\n"
            "assert remove_negatives([-5]) == []\n"
            "assert remove_negatives([-1, 2, -3, 4, -5]) == [2, 4]\n"
        ),
    },
    {
        "id": "backoff",
        "category": "spec-following",
        "prompt": "Write a Python function backoff(attempt, base=1.0, factor=2.0, cap=30.0) that returns "
                  "min(cap, base * factor ** attempt) for a non-negative integer attempt. "
                  "Return only its code in a single ```python fence.",
        "tests": (
            "assert backoff(0) == 1.0\n"
            "assert backoff(3) == 8.0\n"
            "assert backoff(10) == 30.0\n"
            "assert backoff(2, base=0.5, factor=3.0, cap=100.0) == 4.5\n"
            "assert backoff(1, cap=1.0) == 1.0\n"
        ),
    },
    {
        "id": "top_k",
        "category": "data-structures",
        "prompt": "Write a Python function top_k_frequent(nums, k) that returns a list of the k most frequent "
                  "elements in nums, ordered from most to least frequent. Assume the input has at least k "
                  "distinct elements with distinct frequencies (no ties to break). "
                  "Return only its code in a single ```python fence.",
        "tests": (
            "assert top_k_frequent([1, 1, 1, 2, 2, 3], 2) == [1, 2]\n"
            "assert top_k_frequent([4, 4, 4, 4, 2, 2, 3], 1) == [4]\n"
            "assert top_k_frequent([5, 6, 6, 7, 7, 7], 3) == [7, 6, 5]\n"
        ),
    },
]


def extract_code(answer):
    match = FENCE_RE.search(answer)
    if match:
        return match.group(1)
    if "def " in answer:
        return answer
    return ""
