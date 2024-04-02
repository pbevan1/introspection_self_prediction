import subprocess
from pathlib import Path
import getpass


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
