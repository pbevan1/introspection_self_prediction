from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from analysis.compliance_checks import check_compliance
    from analysis.string_cleaning import (
        apply_all_cleaning,
        extract_first_of_multiple_responses,
        match_log_probs_to_trimmed_response,
    )
except ImportError:
    from compliance_checks import check_compliance
    from string_cleaning import (
        apply_all_cleaning,
        extract_first_of_multiple_responses,
        match_log_probs_to_trimmed_response,
    )


def get_exp_folders(exp_dir: Path, exp_name_pattern: str) -> List[Path]:
    """Crawls the directory and returns a list of paths to the experiment folders that match the wildcard pattern.

    Args:
        exp_dir: Path to the directory containing the experiment folders.
        exp_name_pattern: Wildcard pattern for the experiment folders.
            For example, `exp_name_pattern="*few-shot*"` will return all experiment folders that contain the string "few-shot" in their name.
            Or `exp_name_pattern="*"` will return all experiment folders.
            Or `exp_name_pattern="num_35_few-shot-*"` will return all experiment folders that start with "num_35_few-shot-".

    Returns:
        List of paths to the experiment folders.
    """
    exp_folders = list(exp_dir.glob(exp_name_pattern))
    print(f"Found {len(exp_folders)} experiment folders matching {exp_name_pattern}")
    return exp_folders


def load_and_prep_dfs(
    df_paths: List[Path], names: Optional[List[str]] = None, exclude_noncompliant: bool = True
) -> Dict[str, pd.DataFrame]:
    # TODO all of this should be small functions...
    """Loads and cleans a number of dataframes. Returns a dictionary of dataframes with the names as keys."""

    if names is None:
        names = [str(path.parent.stem) for path in df_paths]

    # get pretty name for printing
    pretty_names = {name: get_pretty_name(name) for name in names}

    # load the data
    dfs = {}
    for path, name in zip(df_paths, names):
        dfs[name] = pd.read_csv(path)
        print(f"Loaded {len(dfs[name])} rows from {path}")

    # if response is not in the dataframe, remove the dataframe—it's still running
    to_remove = []
    for name in dfs.keys():
        if "response" not in dfs[name].columns:
            to_remove.append(name)
    for name in to_remove:
        print(f"[{pretty_names[name]}] No response column found. Removing dataframe.")
        dfs.pop(name)

    # clean the input strings (note that this might lead to the response and the tokens diverging)
    for name in dfs.keys():
        # make sure that the response is a string
        dfs[name]["raw_response"] = dfs[name]["response"]
        dfs[name]["response"] = dfs[name]["response"].astype(str)
        dfs[name]["response"] = dfs[name]["response"].apply(apply_all_cleaning)
        # if the model has a continuation with multiple responses, we only want the first one
        try:
            join_on = dfs[name]["dataset"]["join_on"]
            dfs[name]["response"] = dfs[name]["response"].apply(
                lambda x: extract_first_of_multiple_responses(x, join_on)
            )
        except KeyError:
            print(f"[{pretty_names[name]}] No join_on found, skipping extraction of first response")

    # trim teh logprobs to ensure that they match the string
    for name in dfs.keys():
        dfs[name]["logprobs"] = dfs[name].apply(
            lambda x: match_log_probs_to_trimmed_response(x["response"], x["logprobs"]), axis=1
        )

    # ensure that the few-shot_string and responses are lists
    for name in dfs.keys():
        try:
            dfs[name]["few-shot_string"] = dfs[name]["few-shot_string"].apply(eval)
            dfs[name]["few-shot_response"] = dfs[name]["few-shot_response"].apply(eval)
        except KeyError:
            print(f"[{pretty_names[name]}] No few-shot columns found")

    # Run compliance checks. See `evals/compliance_checks.py` for more details.
    # The models like to repeat the last word. We flag it, but don't exclude it

    def last_word_repeated(row):
        try:
            last_word = row["string"].split()[-1]
            return last_word.lower() == row["response"].lower()
        except AttributeError:
            return False

    for name in dfs.keys():
        dfs[name]["last_word_repeated"] = dfs[name].apply(last_word_repeated, axis=1)
        print(
            f"[{pretty_names[name]}] {dfs[name]['last_word_repeated'].mean():.2%} of the responses repeat the last word"
        )

    # if word separation doesnt apply, they still might repeat the last character
    def last_char_repeated(row):  # TODO fix
        try:
            last_char = str(row["string"])[-1]
            return last_char.lower() == str(row["response"])[0].lower()
        except IndexError:  # response is empty
            return False
        except TypeError:  # if the string is nan
            return False

    for name in dfs.keys():
        dfs[name]["last_char_repeated"] = dfs[name].apply(last_char_repeated, axis=1)
        print(
            f"[{pretty_names[name]}] {dfs[name]['last_char_repeated'].mean():.2%} of the responses repeat the last character"
        )

    # Even if they don't repeat the last word, they like to repeat another word
    def nonlast_word_repeated(row):
        try:
            nonlast_words = row["string"].split()[0:-1]
            nonlast_words = [w.lower() for w in nonlast_words]
            return row["response"].lower() in nonlast_words
        except AttributeError:
            return False

    for name in dfs.keys():
        dfs[name]["nonlast_word_repeated"] = dfs[name].apply(nonlast_word_repeated, axis=1)
        print(
            f"[{pretty_names[name]}] {dfs[name]['nonlast_word_repeated'].mean():.2%} of the responses repeat a word other than the last word"
        )

    for name in dfs.keys():
        dfs[name]["compliance"] = dfs[name]["response"].apply(check_compliance)
        print(f"[{pretty_names[name]}] Compliance: {(dfs[name]['compliance'] == True).mean():.2%}")  # noqa: E712

    for name in dfs.keys():
        print(f"[{pretty_names[name]}] Most common non-compliant reasons:")
        print(dfs[name][dfs[name]["compliance"] != True]["compliance"].value_counts().head(10))  # noqa: E712

    # Exclude non-compliant responses
    if exclude_noncompliant:
        for name in dfs.keys():
            dfs[name].query("compliance == True", inplace=True)
            print(f"[{pretty_names[name]}] Excluded non-compliant responses, leaving {len(dfs[name])} rows")

    # add in first logprobs
    def extract_first_logprob(logprobs):
        if isinstance(logprobs, str):
            logprobs = eval(logprobs)
        try:
            return logprobs[0]
        except IndexError:
            return None
        except TypeError:  # if logprobs is None
            return None

    for name in dfs.keys():
        dfs[name]["first_logprobs"] = dfs[name]["logprobs"].apply(extract_first_logprob)

    # extract first token
    def extract_top_token(logprobs):
        if isinstance(logprobs, str):
            logprobs = eval(logprobs)
        try:
            top_token = list(logprobs[0].keys())[0]
            return top_token
        except IndexError:
            return None
        except TypeError:  # if logprobs is None
            return None

    for name in dfs.keys():
        dfs[name]["first_token"] = dfs[name]["logprobs"].apply(extract_top_token)

    return dfs


def merge_base_and_self_pred_dfs(b_df: pd.DataFrame, s_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the base and self prediction dataframes.
    """
    # make sure that the strings in s_df are all in b_df
    assert set(s_df["string"].unique()).issubset(set(b_df["string"].unique())), "Not all strings in s_df are in b_df"

    # we need to subset the data to only include the strings that are in both dfs
    old_b_len = len(b_df)
    strings_set = set(b_df["string"].unique()).intersection(set(s_df["string"].unique()))
    b_df = b_df[b_df["string"].isin(strings_set)]
    s_df = s_df[s_df["string"].isin(strings_set)]
    print(
        f"Subsetted data to only include strings that are in both dfs.\nBefore: [Base] {old_b_len}, After: {len(b_df)}"
    )
    # join the two dataframes on the string column
    df = pd.merge(b_df, s_df, on="string", suffixes=("_base", "_self"))
    df.drop(
        columns=["compliance_base", "compliance_self", "complete_base", "complete_self", "id_base", "id_self"],
        inplace=True,
    )
    print(f"Merged base and self prediction dataframes, leaving {len(df)} rows")
    return df


CONFIG_VALUES_OF_INTEREST = [
    ["language_model", "model"],
    ["prompt", "method"],
    "base_dir",
    "exp_dir",
    "limit",
    # "dataset",
    ["dataset", "topic"],
    ["dataset", "n_shot"],
]


def create_df_from_configs(configs: List[Dict]) -> pd.DataFrame:
    """Create a dataframe from a list of configs."""
    df = pd.DataFrame({"config": configs})
    assert len(set(configs)) == len(df), "Duplicate configs found"
    # set config as index
    df.index = df.config
    # seed the dataframe with the config values

    def extract_value(config, value):
        try:
            if isinstance(value, list):
                return config[value[0]][value[1]]
            else:
                return config[value]
        except KeyError:
            return None
        except TypeError:
            return None

    for value in CONFIG_VALUES_OF_INTEREST:
        pretty_value = value if isinstance(value, str) else value[0] + "_" + value[1]
        df[pretty_value] = df["config"].apply(lambda x: extract_value(x, value))

    # drop the columns that are all None
    df = df.dropna(axis=1, how="all")
    return df


def fill_df_with_function(dfs, function, name, results):
    """Fill the results dataframe with the results of the function. TO be used with the results df."""
    # make col for name
    if name not in results.columns:
        results[name] = np.nan
        results[name] = results[name].astype("object")
    # compute and fill in results
    for config, df in dfs.items():
        result = function(df)
        results.at[config, name] = result


def get_pretty_name(config):
    """Get a pretty name for a config."""
    values = []
    for attribute in CONFIG_VALUES_OF_INTEREST:
        try:
            if isinstance(attribute, list):
                values.append(config[attribute[0]][attribute[1]])
            else:
                values.append(config[attribute])
        except KeyError:
            values.append(None)
    values = [str(val) for val in values]
    return "|".join(values)


def get_maybe_nested_from_dict(d, keys):
    """Helper function to get a value from a nested dictionary."""
    if isinstance(keys, str):
        keys = [keys]
    if len(keys) == 1:
        return d[keys[0]]
    else:
        return get_maybe_nested_from_dict(d[keys[0]], keys[1:])
