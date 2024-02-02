from typing import Dict, List

import numpy as np
import pandas as pd

from evals.utils import get_maybe_nested_from_dict


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
