import subprocess
from pathlib import Path


def get_repo_dir():
    # Run the git command to get the repository root directory
    return Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip())


REPO_DIR = get_repo_dir()

EXP_DIR = REPO_DIR / "exp"

CONF_DIR = REPO_DIR / "evals" / "conf"

DATASET_DIR = REPO_DIR / "evals" / "datasets"
