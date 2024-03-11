"""This file is used to find initial completions which are hard to predict."""

import copy
import logging
import os
import random
import shutil
from pathlib import Path
from string import Template

import hydra
import numpy as np
import pandas as pd
import tqdm
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from pydantic_core._pydantic_core import ValidationError

from evals.analysis.loading_data import get_data_path, load_single_df
from evals.data_models.messages import ChatMessage, MessageRole, Prompt, PromptTemplate
from evals.utils import load_string_and_reponse_functions

CONFIG_PATH = "conf"

LOGGER = logging.getLogger(__name__)


def generate_finetuning_jsonl(main_cfg: DictConfig, path: Path, filename: str = "dataset.jsonl") -> (Path, Path):
    """Generate a jsonl file for finetuning.

    This reads in all config files in the directory, and for each adds loads the base data and genereates the messages for finetuning.

    Args:
        path (Path): Path to the directory containing the config files.
        filename (str): Filename of the jsonl file to generate.

    Returns:
        Path: Path to the generated jsonl file.
    """

    if not path.exists():
        raise FileNotFoundError(f"Path {path} does not exist.")

    if not filename.endswith(".jsonl"):
        filename += ".jsonl"

    # lets ensure that the config file isn't changed when we do things with hydra.
    main_cfg = copy.deepcopy(main_cfg)

    # get all the config files
    config_files = list(path.glob("*.yaml"))
    LOGGER.info(f"Found {len(config_files)} config files in {path}")

    assert len(config_files) > 0, f"No config files found in {path}"

    train_filepaths = []
    val_filepaths = []

    for config_file in config_files:
        cfg = load_hydra_config(config_file)
        # Allow new fields to be added to the configuration
        OmegaConf.set_struct(main_cfg, False)
        # extend main_cfg with the config file
        cfg = OmegaConf.merge(main_cfg, cfg)
        LOGGER.info(f"Processing config {config_file}")

        # set up filenames
        train_filename = cfg.name + "_train_" + filename
        val_filename = cfg.name + "_val_" + filename

        # do we have the file?
        if (path / train_filename).exists():
            LOGGER.info(f"File {filename} already exists. Overwriting.")
            (path / train_filename).unlink()
        if (path / val_filename).exists():
            LOGGER.info(f"File {filename} already exists. Overwriting.")
            (path / val_filename).unlink()

        train_filepath = path / train_filename
        val_filepath = path / val_filename

        generate_single_config_dataset(cfg, train_filepath, val_filepath)

        train_filepaths.append(train_filepath)
        val_filepaths.append(val_filepath)

    # merge the files into a single one
    with open(path / ("train_" + filename), "w") as outfile:
        for train_filepath in train_filepaths:
            with open(train_filepath, "r") as infile:
                outfile.write(infile.read())

    with open(path / ("val_" + filename), "w") as outfile:
        for val_filepath in val_filepaths:
            with open(val_filepath, "r") as infile:
                outfile.write(infile.read())

    LOGGER.info(f"Generated {len(train_filepaths)} datasets and saved to {train_filepath} & {val_filepath}")
    return train_filepath, val_filepath


def generate_single_config_dataset(cfg: DictConfig, train_filepath: Path, val_filepath: Path) -> None:
    """Load the base completions and generate the messages for finetuning.
    The messages are saved directly to file.

    Args:
        cfg (DictConfig): The config to use.
        train_filepath (Path): The path to save the file to.
        val_filepath (Path): The path to save the validation file to.
    """
    base_dir = Path(cfg.base_dir)
    assert base_dir.exists(), f"Base directory {base_dir} does not exist."
    LOGGER.info(f"Loading base data from {base_dir}")
    datapath = get_data_path(cfg.base_dir)
    df = load_single_df(datapath)

    # do we have a strings path?
    # for train
    try:
        if cfg.train_strings_path is not None and cfg.train_strings_path != "none":
            train_strings = pd.read_csv(cfg.train_strings_path)["string"].values
            LOGGER.info(f"(Strings) Loaded {len(train_strings)} strings from {cfg.train_strings_path}")
        else:
            train_strings = None
    except AttributeError:
        LOGGER.info("No train_strings_path found in config. Not using any strings.")
        train_strings = None
    # for val
    try:
        if cfg.val_strings_path is not None and cfg.val_strings_path != "none":
            val_strings = pd.read_csv(cfg.val_strings_path)["string"].values
            LOGGER.info(f"(Strings) Loaded {len(val_strings)} strings from {cfg.val_strings_path}")
        else:
            val_strings = None
    except AttributeError:
        LOGGER.info("No val_strings_path found in config. Not using any strings.")
        val_strings = None

    # do we have the unmodified string?
    if "unmodified_string" in df.columns:
        df["string"] = df["unmodified_string"]
        LOGGER.info("Using unmodified string column.")

    # do we have string modifier?
    string_modifier, response_property = load_string_and_reponse_functions(
        cfg.string_modifier.string_modifier, cfg.response_property.response_property
    )
    if string_modifier is not None:
        df["string"] = df["string"].apply(string_modifier)
        LOGGER.info(f"Applied string modifier {string_modifier.__name__} to string column.")
    if response_property is not None:
        df["response"] = df.apply(response_property, axis=1)
        LOGGER.info(f"Applied response property {response_property.__name__} to response column.")

    # split into train and validation
    if train_strings is not None:
        train_df = df[df["string"].isin(train_strings)]
        val_df = df[~df["string"].isin(train_strings)]
    elif val_strings is not None:
        val_df = df[df["string"].isin(val_strings)]
        train_df = df[~df["string"].isin(val_strings)]
    else:
        train_df = df.sample(frac=1 - cfg.dataset.validation_fraction, random_state=cfg.seed)
        val_df = df.drop(train_df.index)

    LOGGER.info(f"Split into {len(train_df)} training rows and {len(val_df)} validation rows before subsampling.")

    # subsample if needed
    try:
        limit = cfg.limit
        train_limit = limit - int(cfg.dataset.validation_fraction * limit)
        val_limit = limit - train_limit
        LOGGER.info(f"Subsampling to {train_limit} training rows and {val_limit} validation rows.")
        if limit is not None:
            if train_limit < len(train_df):
                LOGGER.info(f"Subsampling to {train_limit} rows.")
                train_df = train_df.sample(train_limit, random_state=cfg.seed, replace=False)
            else:
                LOGGER.info(
                    f"Training limit is {train_limit}, which is higher than the number of rows in the dataframe. Not subsampling."
                )
            if val_limit < len(val_df):
                LOGGER.info(f"Subsampling to {val_limit} rows.")
                val_df = val_df.sample(val_limit, random_state=cfg.seed, replace=False)
            else:
                LOGGER.info(
                    f"Validation limit is {val_limit}, which is higher than the number of rows in the dataframe. Not subsampling."
                )
    except AttributeError:
        LOGGER.info("No limit found in config. Not subsampling.")

    # do we have to scramble the input?
    try:
        if cfg.scramble:
            LOGGER.info("Scrambling the input 🎲.")
            train_df = scramble_strings(train_df, cfg.seed)
            val_df = scramble_strings(val_df, cfg.seed)
    except AttributeError:
        LOGGER.info("No scramble attribute found in config. Not scrambling.")

    # is there overlap between the train and validation set?
    if len(set(train_df["string"]).intersection(set(val_df["string"]))) > 0:
        LOGGER.warning("There is overlap between the train and validation set.")

    # exclude rows that contain None or nan as the response
    old_len_train = len(train_df)
    old_len_val = len(val_df)
    train_df = train_df.dropna(subset=["response"])
    val_df = val_df.dropna(subset=["response"])
    LOGGER.info(f"Excluded {old_len_train - len(train_df)} rows from the training set due to missing responses.")
    LOGGER.info(f"Excluded {old_len_val - len(val_df)} rows from the validation set due to missing responses.")

    # generate the messages
    prompt_template = PromptTemplate(**OmegaConf.to_container(cfg.prompt, resolve=True))
    with open(train_filepath, "a") as f:
        for i, row in tqdm.tqdm(train_df.iterrows(), total=len(train_df), desc="Generating train messages"):
            try:
                prompt = process_prompt(row, prompt_template)
                f.write(prompt.openai_finetuning_format())  # this might not work—use different format if it complains
                f.write("\n")
            except ValidationError as e:
                LOGGER.warning(f"Failed row {i} with error {e}")

    with open(val_filepath, "a") as f:
        for i, row in tqdm.tqdm(val_df.iterrows(), total=len(val_df), desc="Generating validation messages"):
            prompt = process_prompt(row, prompt_template)
            f.write(prompt.openai_finetuning_format())
            f.write("\n")

    # save out the dfs so we can recover the split
    train_df.to_csv(train_filepath.with_suffix(".df.csv"), index=False)
    val_df.to_csv(val_filepath.with_suffix(".df.csv"), index=False)

    LOGGER.info(
        f"Saved {len(train_df)} training rows to {train_filepath} & {len(val_df)} validation rows to {val_filepath}"
    )


def process_prompt(row: pd.Series, prompt_template: PromptTemplate) -> Prompt:
    messages = []
    system_messages = [m for m in prompt_template.messages if m.role == "system"]
    user_messages = [m for m in prompt_template.messages if m.role == "user"]

    assert len(system_messages) < 2, "There should be at most one system message in the prompt template."
    assert len(user_messages) == 1, "There should be exactly one user message in the prompt template."

    # add system messages
    for message in system_messages:
        t = Template(message.content)
        content = t.safe_substitute(string=row["string"])
        messages.append(ChatMessage(role=message.role, content=content))

    # add the actual query message
    for message in user_messages:
        t = Template(message.content)
        content = t.safe_substitute(string=row["string"])
        messages.append(ChatMessage(role=message.role, content=content))

    # add the assistant response
    m = ChatMessage(role=MessageRole.assistant, content=row["response"])
    messages.append(m)

    return Prompt(messages=messages)


def load_hydra_config(config_path):
    """
    I want to use Hydra to load the particular config file, but Hydra can't load them from wherever they are, so we copy them over to the CONFIG_PATH and then load them.
    We also need to reset the global state since Hydra only supports a global state.
    And we need Hydra to resolve the ${...} and so on.
    """
    # temporarily move the config file to the CONFIG_PATH
    temp_file_path = Path(__file__).parent / CONFIG_PATH / "temp_finetuning_dataset.yaml"
    shutil.copy(config_path, temp_file_path)

    # Initialize global Hydra context with temp file
    GlobalHydra.instance().clear()
    with initialize(config_path=CONFIG_PATH, job_name="finetuning_single_dataset"):
        # Compose the configuration
        cfg = compose(config_name="temp_finetuning_dataset", overrides=[])

    # remove the temporary file
    os.remove(temp_file_path)
    return cfg


def scramble_strings(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Scramble the strings in a dataframe while preserving the relative order among specified columns.

    Args:
        df (pd.DataFrame): The dataframe to scramble.
        seed (int): The random seed to use.

    Returns:
        pd.DataFrame: The scrambled dataframe with preserved relative order among specified columns.
    """
    LOGGER.info(f"Scrambling the strings with seed {seed}")

    # Identify the columns to be scrambled
    columns_to_scramble = [col for col in ["string", "unmodified_string", "modified_string"] if col in df.columns]

    # Generate a shuffled sequence of indices based on the DataFrame's length
    random.seed(seed)
    shuffled_indices = np.arange(len(df))
    random.shuffle(shuffled_indices)

    # Apply the shuffled sequence to reorder the specified columns
    for col in columns_to_scramble:
        df[col] = df[col].iloc[shuffled_indices].values

    return df


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config_finetuning_dataset")
def main(cfg: DictConfig):
    LOGGER.info(OmegaConf.to_yaml(cfg))
    generate_finetuning_jsonl(cfg, Path(cfg.study_dir))


if __name__ == "__main__":
    main()
