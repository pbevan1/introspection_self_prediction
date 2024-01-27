import string
from typing import List


def apply_all_cleaning(s: str) -> str:
    """
    Apply all cleaning functions to the input string.

    Parameters:
    s (str): The input string to be cleaned.

    Returns:
    str: The cleaned string.
    """
    s = s.lower()
    s = strip_punctuation(s)
    s = strip_newlines(s)
    s = strip_multiple_whitespaces(s)
    s = strip_common_prefixes(s)
    return s


def clean_string(s: str) -> str:
    """
    Clean the input string by stripping punctuation, converting to lower case, and removing leading/trailing spaces.

    Parameters:
    s (str): The input string to be cleaned.

    Returns:
    str: The cleaned string.
    """
    return strip_punctuation(s.lower().strip())


def strip_punctuation(s: str) -> str:
    """
    Remove punctuation from the input string.

    Parameters:
    s (str): The input string.

    Returns:
    str: The string with punctuation removed.
    """
    return "".join(c for c in s if c not in string.punctuation)


def strip_newlines(s: str) -> str:
    """
    Replace newline and carriage return characters with spaces in the input string.

    Parameters:
    s (str): The input string.

    Returns:
    str: The string with newline and carriage return characters replaced by spaces.
    """
    return s.replace("\n", " ").replace("\r", " ").replace("\t", " ")


def strip_multiple_whitespaces(s: str) -> str:
    """
    Replace multiple whitespace characters with a single space in the input string.

    Parameters:
    s (str): The input string.

    Returns:
    str: The string with multiple whitespaces replaced by a single space.
    """
    return " ".join(s.split())


def strip_common_prefixes(s: str) -> str:
    """
    Remove common prefixes from the input string.

    Parameters:
    s (str): The input string.

    Returns:
    str: The string with common prefixes removed.
    """
    prefixes: List[str] = [
        "Answer:",
        "Answer :",
        "Response:",
        "Response :",
    ]
    for prefix in prefixes:
        if s.startswith(prefix):
            return s[len(prefix) :]
    return s
