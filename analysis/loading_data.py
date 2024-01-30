import logging
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
from analysis_helpers import load_and_prep_dfs
from omegaconf import DictConfig, ListConfig, OmegaConf

LOGGER = logging.getLogger(__name__)


def get_hydra_config(exp_folder: Union[Path, str]) -> Union[DictConfig, ListConfig]:
    """Returns the hydra config for the given experiment folder.
    If there are more than one config, returns the newest one.
    """
    if isinstance(exp_folder, str):
        exp_folder = Path(exp_folder)
    # do we have logs in the folders?
    if not (exp_folder / "logs").exists():
        raise ValueError(f"No logs found in {exp_folder}")
    # find most recent entry in logs
    logs = list(exp_folder.glob("logs/*"))
    logs.sort(key=lambda x: x.stat().st_ctime, reverse=True)
    log_day_folder = logs[0]
    # find most recent subfolder
    log_time_folder = list(log_day_folder.glob("*"))
    log_time_folder.sort(key=lambda x: x.stat().st_ctime, reverse=True)
    hydra_folder = log_time_folder[0] / ".hydra"
    # use hydra to parse the yaml config
    config = OmegaConf.load(hydra_folder / "config.yaml")
    # apply the overrides
    overrides = OmegaConf.load(hydra_folder / "overrides.yaml")
    # convert overrides to DictConfig
    overrides_dict = {override.split("=")[0]: override.split("=")[1] for override in overrides}
    overrides = OmegaConf.create(overrides_dict)
    config = OmegaConf.merge(config, overrides)
    return config


def get_data_path(exp_folder: Union[Path, str]) -> Path:
    """Pulls out the data*.csv file from the experiment folder.
    If more than one is found, returns the newest one.
    """
    if isinstance(exp_folder, str):
        exp_folder = Path(exp_folder)
    data_files = list(exp_folder.glob("data*.csv"))
    data_files.sort(key=lambda x: x.stat().st_ctime, reverse=True)
    if len(data_files) == 0:
        raise ValueError(f"No data*.csv files found in {exp_folder}")
    if len(data_files) > 1:
        LOGGER.warning(f"Found more than one data*.csv file in {exp_folder}, using the newest one")
    return data_files[0]


def get_folders_matching_config_key(exp_folder: Path, conditions: Dict) -> List[Path]:
    """Crawls all subfolders and returns a list of paths to those matching the conditions.

    Args:
        exp_folder: Path to the directory containing the experiment folders.
        conditions: Dictionary of conditions that the experiment folders must match.
            For example, `conditions={"language_model": "gpt-3.5-turbo", "limit": [500,1000]}` will return all experiment folders that have a config for a gpt-3.5-turbo model and a limit of 500 or 1000.
    """
    # ensure that everything in conditions is a list
    for key, value in conditions.items():
        if not isinstance(value, list):
            conditions[key] = [value]
    # ensure that it's all strings
    for key, value in conditions.items():
        conditions[key] = [str(val) for val in value]
    # find all subfolders
    subfolders = list(exp_folder.glob("**"))
    # get configs for each subfolder
    config_dict = {}
    for subfolder in subfolders:
        try:
            config = get_hydra_config(subfolder)
            config_dict[subfolder] = config
        except ValueError:
            pass
    # filter the subfolders
    matching_folders = []
    for subfolder, config in config_dict.items():
        matched = True
        for key, value in conditions.items():
            if config[key] not in value:
                matched = False
                break
        if matched:
            matching_folders.append(subfolder)
    return matching_folders


def load_dfs_with_filter(exp_folder: Path, conditions: Dict) -> Dict[str, pd.DataFrame]:
    """Loads and preps all dataframes from the experiment folder that match the conditions.

    Args:
        exp_folder: Path to the experiment folder.
        conditions: Dictionary of conditions that the experiment folders must match.
            For example, `conditions={"language_model": "gpt-3.5-turbo", "limit": [500,1000]}` will return all experiment folders that have a config for a gpt-3.5-turbo model and a limit of 500 or 1000.
    """
    matching_folders = get_folders_matching_config_key(exp_folder, conditions)
    data_paths = [get_data_path(folder) for folder in matching_folders]
    LOGGER.info(f"Found {len(data_paths)} data entries")
    configs = [get_hydra_config(folder) for folder in matching_folders]
    dfs = load_and_prep_dfs(data_paths, names=configs)
    LOGGER.info(f"Loaded {len(dfs)} dataframes")
    return dfs


if __name__ == "__main__":
    exp_folder = Path("/Users/felixbinder/Cloud/AI Alignment/Astra/Introspection/introspection_self_prediction/exp")
    conditions = {"language_model": "gpt-3.5-turbo", "limit": [500, 1000]}
    dfs = load_dfs_with_filter(exp_folder, conditions)
    print(dfs.keys())
