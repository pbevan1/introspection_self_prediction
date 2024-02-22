from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tqdm
from IPython.display import HTML, display

from evals.utils import get_maybe_nested_from_dict, load_string_and_reponse_functions


def merge_base_and_self_pred_dfs(
    b_df: pd.DataFrame, s_df: pd.DataFrame, string_modifier: str | None = None, response_property: str | None = None
) -> pd.DataFrame:
    """
    Merge the base and self prediction dataframes. Applies the string_modifier and response_property functions if they are not None to the base dataframe.
    """
    # make sure that the strings in s_df are all in b_df
    # assert set(s_df["string"].unique()).issubset(set(b_df["string"].unique())), "Not all strings in s_df are in b_df"
    string_modifier, response_property = load_string_and_reponse_functions(string_modifier, response_property)

    if not set(s_df["string"].unique()).issubset(set(b_df["string"].unique())):
        print(
            f"Not all strings in s_df are in b_df: {len(set(s_df['string'].unique()).difference(set(b_df['string'].unique())))} are missing!"
        )

    # we need to subset the data to only include the strings that are in both dfs
    old_b_len = len(b_df)
    # if unmodified_string is in the dataframe, use that to subset
    if "unmodified_string" in s_df.columns:
        strings_set = set(b_df["string"].unique()).intersection(set(s_df["unmodified_string"].unique()))
        b_df = b_df[b_df["string"].isin(strings_set)]
        s_df = s_df[s_df["unmodified_string"].isin(strings_set)]
    else:
        strings_set = set(b_df["string"].unique()).intersection(set(s_df["string"].unique()))
        b_df = b_df[b_df["string"].isin(strings_set)]
        s_df = s_df[s_df["string"].isin(strings_set)]
    print(
        f"Subsetted data to only include strings that are in both dfs.\nBefore: [Base] {old_b_len}, After: {len(b_df)}"
    )
    # apply string modifier and response property
    b_df["unmodified_response"] = b_df["response"]
    if response_property is not None:
        b_df["response"] = b_df.apply(response_property, axis=1)
    if string_modifier is not None:
        b_df["modified_string"] = b_df["string"].apply(string_modifier)
        b_df["unmodified_string"] = b_df["string"]
        b_df["string"] = b_df["modified_string"]

    # join the two dataframes on the string column
    df = pd.merge(b_df, s_df, on="string", suffixes=("_base", "_self"))
    for col in ["complete_base", "complete_self", "id_base", "id_self"]:
        try:
            df.drop(
                columns=[col],
                inplace=True,
            )
        except KeyError:
            pass

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
    ["dataset", "n_shot_seeding"],
    "prediction_target",
]


def create_df_from_configs(configs: List[Dict]) -> pd.DataFrame:
    """Create a dataframe from a list of configs."""
    df = pd.DataFrame({"config": configs})
    assert len(set(configs)) == len(df), "Duplicate configs found"
    # set config as index
    df.index = df.config
    # seed the dataframe with the config values
    for value in CONFIG_VALUES_OF_INTEREST:
        pretty_value = value if isinstance(value, str) else value[0] + "_" + value[1]
        df[pretty_value] = df["config"].apply(lambda x: get_maybe_nested_from_dict(x, value))

    # drop the columns that are all None
    df = df.dropna(axis=1, how="all")
    return df


def fill_df_with_function(dfs, function, name, results, pass_config=False):
    """Fill the results dataframe with the results of the function. To be used with the results df.

    Args:
        dfs: A dictionary with the dataframes to use as input for the function.
        function: The function to use to compute the results.
        name: The name of the column to fill in the results dataframe.
        results: The dataframe to fill in.
        pass_config: Whether to pass the config to the function or not as the second argument.

    Returns:
        None. The results dataframe is modified in place.
    """
    # make col for name
    if name not in results.columns:
        results[name] = np.nan
        results[name] = results[name].astype("object")
    # compute and fill in results
    for config, df in tqdm.tqdm(dfs.items(), desc=name):
        if pass_config:
            result = function(df, config)
        else:
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


def pretty_print_config(config):
    """Pretty print a config."""
    values = {}
    for attribute in CONFIG_VALUES_OF_INTEREST:
        try:
            if isinstance(attribute, list):
                values[str(attribute)] = config[attribute[0]][attribute[1]]
            else:
                values[str(attribute)] = config[attribute]
        except KeyError:
            values[str(attribute)] = None
    # produce html for jupyter notebook
    html = "<b>Model:</b><table>"
    for key, value in values.items():
        html += f"<tr><td>{key}</td><td><b>{value}</b></td></tr>"
    html += "</table>"
    # display in jupyter notebook
    try:
        display(HTML(html))
    except NameError:
        print(html)


def filter_configs_by_conditions(
    configs: Dict[Tuple[str], Tuple[str]], conditions: List[Dict[Tuple[str], Tuple[str]]]
) -> List[Dict[Tuple[str], Tuple[str]]]:
    """Filter configs by conditions.

    Args:
        configs: The list of configs to filter.
        conditions: The conditions to filter by. Each condition is a dictionary with a single or tuple key-value pair, and a list of allowed values as the value.
            ```
            {
                ("language_model","model"): ["gpt-4-1106-preview"],
                ("dataset","n_shot"): [100, None]
            }
            ```

    Returns:
        The filtered list of configs.
    """
    to_be_removed = set()
    for key, allowed_values in conditions.items():
        if not isinstance(allowed_values, list):
            allowed_values = [allowed_values]
        for config in list(configs):
            config_value = get_maybe_nested_from_dict(config, key)
            if config_value not in allowed_values:
                to_be_removed.add(config)
    return set(configs).difference(to_be_removed)
