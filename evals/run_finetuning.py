import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

from evals.apis.finetuning.run import FineTuneHyperParams, FineTuneParams, run_finetune
from evals.apis.finetuning.syncer import WandbSyncer
from evals.locations import CONF_DIR
from evals.utils import get_current_git_hash, load_secrets, setup_environment

LOGGER = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.parent

FT_CONFIG_TEMPLATE = """
model: ${model}
temperature: 0.0
top_p: 1.0
max_tokens: null
num_candidates_per_completion: 1
insufficient_valids_behaviour: "error"
"""


@hydra.main(config_path="conf", config_name="config_finetuning_run")
def main(cfg: DictConfig) -> str:
    print("Current git hash:", get_current_git_hash())
    assert " " not in cfg.notes, "Notes cannot have spaces, use underscores instead"
    # set working directory
    os.chdir(ROOT_DIR)

    setup_environment(openai_tag=cfg.openai_tag)
    params = FineTuneParams(
        model=cfg.language_model.model,
        hyperparameters=FineTuneHyperParams(n_epochs=cfg.epochs),
        suffix=cfg.notes,
    )
    if cfg.use_wandb:
        syncer = WandbSyncer.create(project_name=str(cfg.study_name).replace("/", "_"), notes=cfg.notes)
        # if more_config:
        #     more_config = {k: v for k, v in [x.split("=") for x in more_config.split(",")]}
        #     syncer.update_parameters_with_dict(params=more_config)
    else:
        syncer = None

    # try to find the data files
    data_path = Path(cfg.study_dir) / "train_dataset.jsonl"
    val_data_path = Path(cfg.study_dir) / "val_dataset.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")
    if not val_data_path.exists():
        val_data_path = None
        LOGGER.warning(f"Validation file not found at {val_data_path}. Continuing without validation.")

    # load secrets
    secrets = load_secrets("SECRETS")
    try:
        org = secrets[cfg.organization]
    except KeyError:
        LOGGER.error(f"Organization {cfg.organization} not found in secrets")
        raise

    model_id = run_finetune(
        params=params,
        data_path=data_path,
        syncer=syncer,
        ask_to_validate_training=cfg.ask_to_validate_training,
        val_data_path=val_data_path,
        organisation=org,
    )

    LOGGER.info(f"Done with model_id: {model_id}")
    # adding a config file for this finetuned model
    config_id = create_finetuned_model_config(cfg, model_id)
    print(config_id)
    return model_id


def create_finetuned_model_config(cfg, ft_model_id, overwrite=False):
    """Creates a model config file for the finetuned model in the config directory."""
    safe_model_id = ft_model_id.replace(":", "_")
    directory = CONF_DIR / "language_model" / "finetuned" / cfg.study_name
    file_path = directory / f"{safe_model_id}.yaml"
    directory.mkdir(parents=True, exist_ok=True)
    if file_path.exists() and not overwrite:
        LOGGER.warning(f"File already exists at {file_path}. Not overwriting.")
        return f"finetuned/{cfg.study_name}/{safe_model_id}"
    with open(directory / file_path, "w") as f:
        f.write(FT_CONFIG_TEMPLATE.replace("${model}", ft_model_id))
    return f"finetuned/{cfg.study_name}/{safe_model_id}"  # return the name of the config that can be loaded by Hydra


if __name__ == "__main__":
    main()
