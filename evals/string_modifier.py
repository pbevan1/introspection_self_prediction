"""This files holds functions that can be used to modify an input string so that the model has to reconstruct it. The input is a string and it returns a string."""

from typing import Callable


def reverse_characters(string: str) -> str:
    return string[::-1]


def reverse_words(string: str) -> str:
    return " ".join(reversed(string.split(" ")))


def split_half_w_instruction(string: str) -> str:
    first_half, second_half = string[: len(string) // 2], string[len(string) // 2 :]
    return f"The second half of the sequence is: {second_half}, and the first half is: {first_half}."


def split_half_w_instruction_and_markers(string: str) -> str:
    first_half, second_half = string[: len(string) // 2], string[len(string) // 2 :]
    return f"The second half of the sequence is: <begin>{second_half}<end>, and the first half is: <begin>{first_half}<end>."


def numeric_modification(string: str, mod_func: Callable[[int], int] = lambda x: x + 1) -> str | None:
    """Modify a string that represents a number in some way.

    Args:
    - string: a string that represents a number
    - mod_func: a function that takes a int and returns a int

    Returns:
    - a string that represents the modified number
    """
    # is the response numerical?
    try:
        response = int(string)
    except ValueError:
        return None
    # get modified number
    mod_response = mod_func(response)
    return str(mod_response)


def increment_number(string: str) -> str | None:
    return numeric_modification(string, lambda x: x + 1)


def double_number(string: str) -> str | None:
    return numeric_modification(string, lambda x: x * 2)


def add_42(string: str) -> str | None:
    return numeric_modification(string, lambda x: x + 42)
