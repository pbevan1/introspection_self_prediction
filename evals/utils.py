import importlib
import json
import logging
import os
import subprocess
from pathlib import Path

import omegaconf
import openai
import yaml
from tenacity import retry, retry_if_result, stop_after_attempt

LOGGER = logging.getLogger(__name__)

LOGGING_LEVELS = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


def setup_environment(
    anthropic_tag: str = "ANTHROPIC_API_KEY",
    logging_level: str = "info",
    openai_tag: str = "OPENAI_API_KEY",
):
    setup_logging(logging_level)
    secrets = load_secrets(Path(__file__).parent.parent / "SECRETS")
    openai.api_key = secrets[openai_tag]
    os.environ["ANTHROPIC_API_KEY"] = secrets[anthropic_tag]


def setup_logging(logging_level):
    level = LOGGING_LEVELS.get(logging_level.lower(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Disable logging from openai
    logging.getLogger("openai").setLevel(logging.CRITICAL)
    logging.getLogger("httpx").setLevel(logging.CRITICAL)


def load_secrets(file_path):
    secrets = {}
    with open(file_path) as f:
        for line in f:
            key, value = line.strip().split("=", 1)
            secrets[key] = value
    return secrets


def load_yaml(file_path):
    with open(file_path) as f:
        content = yaml.safe_load(f)
    return content


def save_yaml(file_path, data):
    with open(file_path, "w") as f:
        yaml.dump(data, f)


def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            json_obj = json.loads(line)
            data.append(json_obj)
    return data


def save_jsonl(file_path, data):
    with open(file_path, "w") as f:
        for line in data:
            json.dump(line, f)
            f.write("\n")


def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def save_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f)


@retry(
    stop=stop_after_attempt(8),
    retry=retry_if_result(lambda result: result is not True),
)
def function_with_retry(function, *args, **kwargs):
    return function(*args, **kwargs)


@retry(
    stop=stop_after_attempt(8),
    retry=retry_if_result(lambda result: result is not True),
)
async def async_function_with_retry(function, *args, **kwargs):
    return await function(*args, **kwargs)


def get_maybe_nested_from_dict(d, keys):
    """Helper function to get a value from a nested dictionary."""
    try:
        if isinstance(keys, str):
            keys = [keys]
        if len(keys) == 1:
            return d[keys[0]]
        else:
            return get_maybe_nested_from_dict(d[keys[0]], keys[1:])
    except KeyError:
        return None


def load_string_and_reponse_functions(string_modifier, response_property):
    if string_modifier == "None":
        string_modifier = None
    if response_property == "None":
        response_property = None
    if string_modifier is not None:
        string_modifier = import_function_from_string("evals.string_modifier", string_modifier)
        LOGGER.info(f"Loaded string modifier function {string_modifier.__name__} from evals.string_modifier")
    if response_property is not None:
        response_property = import_function_from_string("evals.response_property", response_property)
        LOGGER.info(f"Loaded output property function {response_property.__name__} from evals.response_property")
    return string_modifier, response_property


def import_function_from_string(module_name, function_name):
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    return function


def sanitize_folder_name(key: str) -> str:
    """Sometimes we need to clean Hydra keys so we can use them as folder names. This function does that."""
    # Replace ":" with "_"
    sanitized_key = key.replace(":", "_")
    # Replace "/" with "_"
    sanitized_key = sanitized_key.replace("meta_level/", "meta_level_")
    sanitized_key = sanitized_key.replace("object_level/", "object_level_")
    return sanitized_key


# ensure that the sanitize function is registered
omegaconf.OmegaConf.register_new_resolver("sanitize", sanitize_folder_name)


def get_repo_dir():
    # Run the git command to get the repository root directory
    return Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip())
