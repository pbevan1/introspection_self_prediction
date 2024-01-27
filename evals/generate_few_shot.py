import logging
import random
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from analysis.compliance_checks import enforce_compliance_on_df

LOGGER = logging.getLogger(__name__)

# Run the git command to get the repository root directory
REPO_DIR = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()

LOGGER.info("Repository directory:", REPO_DIR)
sys.path.append(REPO_DIR)


def generate_few_shot_data(
    base_data_path, strings_path, n_shot, num=None, repeat=1, seed=0, enforce_compliance=True
) -> Path:
    """
    Generates a .data{seed}.csv file for few-shot evaluation by adding fields for the few-shot strings.

    The resulting dataframe will contain the following columns:
    - id: The id of the string.
    - string: The string to query the model with.
    - few-shot_string: A list of strings for the few-shot prompt.
    - few-shot_response: A list of responses for the few-shot prompt.

    Args:
        base_data_path: Path to the base data file. These are the base completions that will be used to generate the few-shot strings.
        strings_path: Path to the strings file. These are the strings that will be used to query the model.
        n_shot: Number of few-shot strings to add.
        num: Number of strings to generate. If None, use all strings from the base data file.
        repeat: Number of times to repeat the strings with different few-shot completions.
        seed: Random seed. Can be different from the base seed.
        enforce_compliance: Whether to enforce compliance checks on the base data.

    Returns:
        Path to the generated .data{seed}.csv file.
    """
    LOGGER.info(f"Generating few-shot strings for {base_data_path} with {n_shot} few-shot strings")
    random.seed(seed)
    LOGGER.info(f"Using random seed {seed}")

    # load the data
    base_df = pd.read_csv(base_data_path)
    LOGGER.info(f"Loaded {len(base_df)} rows from {base_data_path}")

    # enforce compliance checks
    if enforce_compliance:
        base_df = enforce_compliance_on_df(base_df)

    # load the strings
    strings = pd.read_csv(strings_path)
    LOGGER.info(f"Loaded {len(strings)} rows from {strings_path}")

    # are the strigns the same?
    shared_strings = set(base_df["string"].unique()).intersection(set(strings["string"].unique()))
    if len(shared_strings) > 0:
        LOGGER.warning(f"Found {len(shared_strings)} shared strings between the base data and the strings file.")

    if num is not None:
        strings = strings.sample(num, random_state=seed)
        LOGGER.info(f"Sampled {len(strings)} strings since a specific number ({num}) was requested.")

    # repeat the strings
    strings = strings.reindex(strings.index.repeat(repeat)).reset_index(drop=True)

    # add in few-shot columns
    out_df = strings.copy()
    out_df["few-shot_string"] = None
    out_df["few-shot_response"] = None

    # add in few-shot strings
    for i, row in out_df.iterrows():
        string = row["string"]
        few_shot_strings, few_shot_responses = get_few_shot_completions(string, base_df, n_shot)
        out_df.at[i, "few-shot_string"] = few_shot_strings
        out_df.at[i, "few-shot_response"] = few_shot_responses

    # save the output strings
    output_file_path = Path(strings_path).parent / f"data{seed}.csv"
    out_df.to_csv(output_file_path, index=False)
    LOGGER.info(f"Saved {len(out_df)} strings with few-shot completions to {output_file_path}")
    return output_file_path


def get_few_shot_completions(string, base_df, n) -> Tuple[List[str], List[str]]:
    """
    Get the few-shot completions for a string.

    Args:
        string: The string to get the few-shot completions for.
        base_df: The base dataframe to use.
        n: The number of completions to get.

    Returns:
        A tuple of lists of strings, the first being the few-shot strings and the second being the few-shot responses.
    """
    strings = []
    responses = []
    # get the base completions
    while n != 0:
        row = base_df.sample(1)
        base_string = row["string"].iloc[0]
        base_response = row["response"].iloc[0]
        if base_string != string:
            strings.append(base_string)
            responses.append(base_response)
            n -= 1
        else:
            LOGGER.warning(f"Found the same string {string} in the base data. Choosing another...")
    return strings, responses
