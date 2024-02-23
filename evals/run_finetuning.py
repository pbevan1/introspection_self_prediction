import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

from evals.apis.finetuning.run import FineTuneHyperParams, FineTuneParams, run_finetune
from evals.apis.finetuning.syncer import WandbSyncer
from evals.utils import load_secrets, setup_environment

LOGGER = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.parent


@hydra.main(config_path="conf", config_name="config_finetuning_run")
def main(cfg: DictConfig) -> str:
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
        syncer = WandbSyncer.create(project_name=cfg.study_name, notes=cfg.notes)
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

    results = run_finetune(
        params=params,
        data_path=data_path,
        syncer=syncer,
        ask_to_validate_training=cfg.ask_to_validate_training,
        val_data_path=val_data_path,
        organisation=org,
    )

    LOGGER.info(f"Done with results: {results}")
    return results


if __name__ == "__main__":
    main()
