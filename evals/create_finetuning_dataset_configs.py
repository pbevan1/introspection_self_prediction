"""This file holds helper functions to create the Hydra files for the finetuning dataset configs by sweeping over the different configurations."""

from pathlib import Path

from evals.locations import EXP_DIR

TEMPLATE = """
name: $name

defaults: # we need to use defaults here
  - task: $task
  - prompt: $prompt
  - response_property: $response_property

train_base_dir: $train_base_dir

val_base_dir: $val_base_dir

$overrides
"""


def create_finetuning_dataset_config(
    study_name: str,
    model_config: str,
    task_config: str,
    prompt_config: str,
    response_property_config: str,
    overrides: str,
    train_base_dir: str,
    val_base_dir: str,
    overwrite: bool = True,
) -> Path:
    # Finetuning folder—organized by source model. The folder then contains many different tasks etc.
    ft_exp_dir = EXP_DIR / "finetuning" / study_name / model_config
    ft_exp_dir.mkdir(parents=True, exist_ok=True)

    name = f"{model_config}_{task_config}_{response_property_config}_{prompt_config.replace('/', '-')}"  # name of the config. We need to replace the / in the prompt config to avoid issues with the file path.

    overrides_str = "\n".join(overrides)

    config = TEMPLATE.replace("$name", name)
    config = config.replace("$task", task_config)
    config = config.replace("$prompt", prompt_config)
    config = config.replace("$response_property", response_property_config)
    config = config.replace("$train_base_dir", train_base_dir)
    config = config.replace("$val_base_dir", val_base_dir)
    config = config.replace("$overrides", overrides_str)

    # save to file
    config_path = ft_exp_dir / f"{name}.yaml"
    if config_path.exists() and not overwrite:
        print(f"File already exists at {config_path}. Not overwriting.")
        return config_path
    with open(config_path, "w") as f:
        f.write(config)
    return config_path
