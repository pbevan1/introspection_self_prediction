import getpass
import subprocess
from pathlib import Path


def get_repo_dir():
    # Run the git command to get the repository root directory
    return Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip())


REPO_DIR = get_repo_dir()

if Path("/shared/exp").exists():
    EXP_DIR = Path("/shared") / "exp" / getpass.getuser()
else:
    EXP_DIR = REPO_DIR / "exp"

CONF_DIR = REPO_DIR / "evals" / "conf"

DATASET_DIR = REPO_DIR / "evals" / "datasets"


def add_repo_dir_to_sys_path():
    import sys

    sys.path.append(str(REPO_DIR))
    print(f"Added {REPO_DIR} to sys.path")


add_repo_dir_to_sys_path()
