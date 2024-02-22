"""This file is used to find initial completions which are hard to predict."""

import copy
import logging
import os
import shutil
from pathlib import Path
from string import Template

import hydra
import pandas as pd
import tqdm
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

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

    # subsample if needed
    try:
        limit = cfg.dataset.num
        if limit is not None:
            if limit < len(df):
                LOGGER.info(f"Subsampling to {limit} rows.")
                df = df.sample(limit, random_state=cfg.seed, replace=False)
            else:
                LOGGER.info(
                    f"Limit is {limit}, which is higher than the number of rows in the dataframe. Not subsampling."
                )
    except AttributeError:
        LOGGER.info("No limit found in config. Not subsampling.")

    # do we have the unmodified string?
    if "unmodified_string" in df.columns:
        df["string"] = df["unmodified_string"]
        LOGGER.info("Using unmodified string column.")

    # do we have string modifier?
    string_modifier, response_property = load_string_and_reponse_functions(
        cfg.dataset.string_modifier, cfg.dataset.response_property
    )
    if string_modifier is not None:
        df["string"] = df["string"].apply(string_modifier)
        LOGGER.info(f"Applied string modifier {string_modifier.__name__} to string column.")
    if response_property is not None:
        df["response"] = df.apply(response_property, axis=1)
        LOGGER.info(f"Applied response property {response_property.__name__} to response column.")

    # split into train and validation
    train_df = df.iloc[: int(len(df) * (1 - cfg.dataset.validation_fraction))]
    val_df = df.iloc[int(len(df) * (1 - cfg.dataset.validation_fraction)) :]

    # generate the messages
    prompt_template = PromptTemplate(**OmegaConf.to_container(cfg.prompt, resolve=True))
    with open(train_filepath, "a") as f:
        for i, row in tqdm.tqdm(train_df.iterrows(), total=len(train_df), desc="Generating train messages"):
            prompt = process_prompt(row, prompt_template)
            f.write(prompt.openai_finetuning_format())  # this might not work—use different format if it complains
            f.write("\n")

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


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config_finetuning_dataset")
def main(cfg: DictConfig):
    LOGGER.info(OmegaConf.to_yaml(cfg))
    generate_finetuning_jsonl(cfg, Path(cfg.study_dir))


if __name__ == "__main__":
    main()
