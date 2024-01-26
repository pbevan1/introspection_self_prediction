import string
from typing import Callable, Dict, List, Union

import numpy as np

COMPLIANCE_CHECKS = {
    "none": lambda x: x is None or np.isnan(x),
    "not_single_word": lambda x: len(x.split()) > 1,
    "no": lambda x: clean_string(x) in ["no", "nope"],
    "sorry": lambda x: clean_string(x) in ["sorry", "apologies", "apology", "apologize", "apologise"],
    "impossible": lambda x: clean_string(x) in ["impossible"],
    "unable": lambda x: clean_string(x) in ["unable"],
    "unpredictable": lambda x: clean_string(x) in ["unpredictable"],
    "sequence": lambda x: clean_string(x) == "sequence",
    "random": lambda x: clean_string(x) in ["random", "randomness"],
    "item": lambda x: clean_string(x) == "item",
    "yes": lambda x: clean_string(x) == "yes",
    "nope": lambda x: clean_string(x) == "nope",
    "next": lambda x: clean_string(x) == "next",
}


def check_compliance(
    string: str, compliance_checks: Dict[str, Callable[[str], bool]] = COMPLIANCE_CHECKS
) -> Union[bool, List[str]]:
    """Check if a string is compliant with the compliance checks. Returns True if compliant, list of the names of the failed checks if not."""
    failed = []
    for name, check in compliance_checks.items():
        try:
            if check(string):
                failed.append(name)
        except AttributeError:  # if a check throws an error it passes
            pass
        except TypeError:
            pass
    if len(failed) == 0:
        return True
    else:
        return failed


def strip_punctuation(s):
    return "".join(c for c in s if c not in string.punctuation)


def clean_string(s):
    return strip_punctuation(s.lower().strip())
