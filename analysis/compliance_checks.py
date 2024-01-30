import logging
from typing import Callable, Dict, List, Union

import numpy as np

try:
    from .string_cleaning import clean_string
except ImportError:
    from string_cleaning import clean_string

LOGGER = logging.getLogger(__name__)

COMPLIANCE_CHECKS = {
    "NaN": lambda x: x is None or x == "nan" or np.isnan(x),
    "not_single_word": lambda x: len(x.split()) > 1,
    "no": lambda x: clean_string(x) in ["no", "nope"],
    "sorry": lambda x: clean_string(x) in ["sorry", "apologies", "apology", "apologize", "apologise"],
    "impossible": lambda x: clean_string(x) in ["impossible"],
    "unable": lambda x: clean_string(x) in ["unable", "i cannot", "i can't", "cannot"],
    "unpredictable": lambda x: clean_string(x) in ["unpredictable", "indeterminable"],
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


def enforce_compliance_on_df(df):
    """
    Enforce compliance checks on a dataframe. Returns a dataframe with only compliant rows.
    """
    old_len = len(df)
    df["compliance"] = df["response"].apply(check_compliance)
    df = df[df["compliance"] == True]  # noqa: E712
    LOGGER.info(f"Excluded non-compliant responses, leaving {len(df)} rows down from {old_len}")
    return df
