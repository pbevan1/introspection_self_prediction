"""Implements the full run through of an experiment:

- generate `10000` object level completions with `train`
	- to be used as the training set for finetuning
- generate `2500` object level completions with `val`
	- for comparing the object/meta level
	- as validation set during finetuning
	- overgenerating to find model disagreement here
- generate finetuning datasets, sweeping across `response_property`, `task`, other args(?)
	- can't rely on caching here
	- using the above directories
	- cross-connecting across language models (but we don't need to keep track of LLM assignments and can just pairs LLMs & the folders at this point)
		- but we need to keep track of the `task.set` values to pair them up properly
	- sweeping across other configs (response property, task)
	- => new model codes need to be kept track of
- run finetuning
	- this can't rely on caching either
	- make a file in the folder containing the model name and skip if not necessary?
		- wandb url
		- complete status
		- OAI model ID (or other model ID for OSS models)
- generate `500` meta-level completions on `val`, sweeping across `response_property`, `task`
	- use `2500` val generated ones to do model_divergence filtering down to 500 -> string filter
	- maybe also additional combined args?
		- maybe represented as cfg files?
	- this also needs to include the newly generated models from above


Example usage:
python -m scripts.sweep_full_experiment
--study_name="full_sweep_test"
--model_configs="gpt-3.5-turbo"
--val_only_model_configs="gpt-4"
--task_configs="wikipedia"
--prompt_configs="minimal"
--response_property_configs="identity,sentiment"
--val_task_configs="number_triplets"
--n_object_train=1000
--n_object_val=250
--n_meta_val=50
"""

import argparse
import atexit
import json
import subprocess
from pathlib import Path

from evals.create_finetuning_dataset_configs import create_finetuning_dataset_config
from evals.locations import EXP_DIR


class StudyRunner:
    def __init__(self):
        # add arguments to self.args
        self.parse_arguments()
        # split comma-separated strings into lists
        self.parse_args_into_lists()
        # load or create state file
        self.state = self.create_or_load_state_file()
        atexit.register(self.write_state_file)

    def run_command(self, command):
        """Execute the given command in the shell, stream the output, and return the last line."""
        try:
            self.state["commands"].append(command)  # log the command
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            output_lines = []
            for line in process.stdout:
                print(line, end="")  # stream the output to the command line
                output_lines.append(line.strip())

            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command)

            last_line = output_lines[-1] if output_lines else ""
            print(f"Successfully executed: {command}")
            return last_line

        except subprocess.CalledProcessError as e:
            print(f"Error executing {command}: {e}")
            raise e

    def parse_arguments(self):
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(description="Run a full study sweeping over the following configs.")
        # Accept comma-separated strings for all list-type arguments
        parser.add_argument("--study_name", type=str, help="The name of the study. Defines the output directory.")
        parser.add_argument(
            "--model_configs", type=str, help="Comma-separated list of model configurations to sweep over."
        )
        parser.add_argument(
            "--val_only_model_configs",
            type=str,
            help="Comma-separated list of model configurations to sweep over for validation only. Won't be finetuned.",
            default="",
        )
        parser.add_argument(
            "--task_configs", type=str, help="Comma-separated list of data configurations to sweep over."
        )
        parser.add_argument(
            "--val_task_configs",
            type=str,
            help="Comma-separated list of data configurations to sweep over for validation only.",
            default="wikipedia",
        )
        parser.add_argument(
            "--prompt_configs",
            type=str,
            help="Comma-separated list of prompt configurations to sweep over. Only pass the name, but not the preceeding folder.",
        )
        parser.add_argument(
            "--response_property_configs",
            type=str,
            help="Comma-separated list of response property configurations to sweep over.",
            default="identity",
        )
        parser.add_argument(
            "--val_response_property_configs",
            type=str,
            help="Comma-separated list of response property configurations to sweep over for validation only.",
            default="identity",
        )
        parser.add_argument(
            "--inference_overrides",
            type=str,
            help="Comma-separated list of Hydra configuration overrides. These are applied to all inference calls.",
            default="",
        )
        parser.add_argument(
            "--finetuning_overrides",
            type=str,
            help="Comma-separated list of Hydra configuration overrides. These are applied to all finetuning calls.",
            default="",
        )
        parser.add_argument(
            "--n_object_train",
            type=int,
            help="Number of object level completions to generate for training.",
            default=10000,
        )
        parser.add_argument(
            "--n_object_val",
            type=int,
            help="Number of object level completions to generate for validation.",
            default=2500,
        )
        parser.add_argument(
            "--n_finetuning", type=int, help="Number of finetuning completions to generate.", default=12500
        )
        parser.add_argument(
            "--n_meta_val", type=int, help="Number of meta level completions to generate for validation.", default=500
        )
        # add the arguments to the object
        self.args = parser.parse_args()

    def parse_args_into_lists(self):
        for arg in [
            "model_configs",
            "val_only_model_configs",
            "task_configs",
            "val_task_configs",
            "prompt_configs",
            "response_property_configs",
            "val_response_property_configs",
            "inference_overrides",
            "finetuning_overrides",
        ]:
            setattr(
                self.args, arg, getattr(self.args, arg).replace(", ", ",").split(",") if getattr(self.args, arg) else []
            )

    def create_or_load_state_file(self):
        state_file_path = Path(EXP_DIR / self.args.study_name / "state.json")
        # make sure the directory exists
        state_file_path.parent.mkdir(parents=True, exist_ok=True)
        if state_file_path.exists():
            with open(state_file_path, "r") as f:
                state = json.load(f)
            print(f"Existing state file loaded from {state_file_path}")
        else:
            state = {
                "object_train_runs": {},
                "object_val_runs": {},
                "divergent_strings": {},
                "finetuning_dataset_creation": {},
                "finetuning_runs": {},
                "ft_object_val_runs": {},
                "meta_val_runs": {},
                "commands": [],
            }
            # write the state file
            with open(state_file_path, "w") as f:
                json.dump(state, f)
            print(f"New state file created at {state_file_path}")
        return state

    def write_state_file(self):
        state_file = Path(EXP_DIR / self.args.study_name / "state.json")
        with open(state_file, "w") as f:
            json.dump(self.state, f)
        print(f"State file written to {state_file}")

    def get_finetuned_model_configs(self):
        """Pull out the config names of the finetuned models from the state file."""
        return [v["ft_model_config"] for v in self.state["finetuning_runs"].values() if v["status"] == "complete"]

    def get_folders_by_task(self, task, set="val", block="object_val_runs"):
        """Get the folders for the object level completions by task and set."""
        return [
            v["folder"]
            for k, v in self.state[block].items()
            if v["task"] == task and v["set"] == set and v["status"] == "complete"
        ]

    def get_object_level_command(self, model, task, prompt, limit, set, overrides=""):
        command = f"python -m evals.run_object_level study_name={self.args.study_name} language_model={model} task={task} task.set={set} prompt=object_level/{prompt} limit={limit} {overrides}"
        return command

    def get_meta_level_command(
        self, model, task, response_property, prompt, limit, set, strings_path="~", overrides=""
    ):
        command = f"python -m evals.run_meta_level study_name={self.args.study_name} language_model={model} task={task} response_property={response_property} task.set={set} prompt=meta_level/{prompt} limit={limit} strings_path={strings_path} {overrides}"
        return command

    def get_finetuning_command(self, model, ft_study, notes, overrides=""):
        return f"python -m evals.run_finetuning study_name={ft_study} language_model={model} notes={notes} {' '.join(overrides)}"

    def run_study(self):
        #### run object level completions on train ####
        for model in self.args.model_configs:
            for task in self.args.task_configs:
                for prompt in self.args.prompt_configs:
                    command = self.get_object_level_command(model, task, prompt, self.args.n_object_train, "train")
                    # check if we need to run this command and set up the state
                    if command not in self.state["object_train_runs"]:
                        self.state["object_train_runs"][command] = {"status": "incomplete"}
                    elif self.state["object_train_runs"][command]["status"] == "complete":
                        print(f"Skipping {command} because it is already complete.")
                    # save other args to the state file
                    self.state["object_train_runs"][command]["model"] = model
                    self.state["object_train_runs"][command]["task"] = task
                    self.state["object_train_runs"][command]["set"] = "train"
                    self.write_state_file()
                    try:
                        data_folder = self.run_command(command)
                        self.state["object_train_runs"][command]["status"] = "complete"
                        self.state["object_train_runs"][command]["folder"] = data_folder
                    except Exception as e:
                        self.state["object_train_runs"][command]["status"] = "failed"
                        print(f"Failed to run {command}: {e}")
                        raise e
                    self.write_state_file()

        #### run object level completions on val ####
        # including validation only models here for the divergence calculation
        for model in self.args.model_configs + self.args.val_only_model_configs:
            for task in (
                self.args.task_configs + self.args.val_task_configs
            ):  # also running the validation tasks here since we'll need them later
                for prompt in self.args.prompt_configs:
                    command = self.get_object_level_command(model, task, prompt, self.args.n_object_val, "val")
                    # check if we need to run this command and set up the state
                    if command not in self.state["object_val_runs"]:
                        self.state["object_val_runs"][command] = {"status": "incomplete"}
                    elif self.state["object_val_runs"][command]["status"] == "complete":
                        print(f"Skipping {command} because it is already complete.")
                    # save other args to the state file
                    self.state["object_val_runs"][command]["model"] = model
                    self.state["object_val_runs"][command]["task"] = task
                    self.state["object_val_runs"][command]["set"] = "val"
                    self.write_state_file()
                    try:
                        data_folder = self.run_command(command)
                        self.state["object_val_runs"][command]["status"] = "complete"
                        self.state["object_val_runs"][command]["folder"] = data_folder
                    except Exception as e:
                        self.state["object_val_runs"][command]["status"] = "failed"
                        print(f"Failed to run {command}: {e}")
                        raise e
                    self.write_state_file()

        #### extract model divergent strings ####
        for task in self.args.val_task_configs:
            # get the model divergent strings
            if task not in self.state["divergent_strings"]:
                self.state["divergent_strings"][task] = {"status": "incomplete"}
            folders = self.get_folders_by_task(task, set="val", block="object_val_runs")
            target_file = EXP_DIR / self.args.study_name / f"divergent_strings_{task}.csv"
            if not target_file.exists():
                command = f"python -m evals.extract_model_divergent_strings {' '.join(folders)} --output {target_file}"
                try:
                    self.run_command(command)
                    self.state["divergent_strings"][task]["status"] = "complete"
                    self.state["divergent_strings"][task]["strings_path"] = str(target_file)
                except Exception as e:
                    self.state["divergent_strings"][task]["status"] = "failed"
                    print(f"Failed to run {command}: {e}")
                    raise e
                self.write_state_file()

        #### run finetuning dataset creation ####
        finetuning_folder_paths = []
        for model in self.args.model_configs:
            for task in self.args.task_configs:
                for response_property in self.args.response_property_configs:
                    for prompt in self.args.prompt_configs:
                        train_command = self.get_object_level_command(
                            model, task, prompt, self.args.n_object_train, "train"
                        )
                        val_command = self.get_object_level_command(model, task, prompt, self.args.n_object_val, "val")
                        # do we have the train and val folders?
                        train_folder = self.state["object_train_runs"][train_command].get("folder", None)
                        val_folder = self.state["object_val_runs"][val_command].get("folder", None)
                        if train_folder is None or val_folder is None:
                            print(
                                f"Skipping finetuning dataset creation for {model}, {task}, {response_property}, {prompt} because the object level completions are not complete."
                            )
                            continue
                        # create the finetuning dataset
                        yaml_path = create_finetuning_dataset_config(
                            self.args.study_name,
                            model,
                            task,
                            prompt,
                            response_property,
                            self.args.finetuning_overrides,
                            train_folder,
                            val_folder,
                            overwrite=False,
                        )
                        finetuning_folder_paths.append(yaml_path)
        print(f"Created {len(finetuning_folder_paths)} finetuning dataset configs. Creating datasets...")
        finetuning_study_names = set(
            [p.parent.name for p in finetuning_folder_paths]
        )  # we need the name of the subfolder
        for data_folder in finetuning_study_names:
            command = f"python -m evals.create_finetuning_dataset study_name={self.args.study_name} dataset_folder={data_folder}"
            self.run_command(command)
        print(f"Created {len(finetuning_study_names)} finetuning datasets.")

        #### run finetuning ####
        for model in self.args.model_configs:
            for ft_study in finetuning_study_names:
                command = self.get_finetuning_command(
                    model, f"{self.args.study_name}/{ft_study}", "sweep", self.args.finetuning_overrides
                )
                if command not in self.state["finetuning_runs"]:
                    self.state["finetuning_runs"][command] = {"status": "incomplete"}
                elif self.state["finetuning_runs"][command]["status"] == "complete":
                    print(f"Skipping {command} because it is already complete.")
                    continue
                self.write_state_file()
                try:
                    ft_model_config = self.run_command(command)
                    self.state["finetuning_runs"][command]["status"] = "complete"
                    self.state["finetuning_runs"][command]["ft_model_config"] = ft_model_config
                except Exception as e:
                    self.state["finetuning_runs"][command]["status"] = "failed"
                    print(f"Failed to run {command}: {e}")
                    raise e
                self.write_state_file()

        #### run object level completions on val with finetuned models ####
        for model in self.get_finetuned_model_configs():  # all the others should be done above
            for task in self.args.val_task_configs + self.args.task_configs:
                for prompt in self.args.prompt_configs:
                    command = self.get_object_level_command(model, task, prompt, self.args.n_object_val, "val")
                    if command not in self.state["ft_object_val_runs"]:
                        self.state["ft_object_val_runs"][command] = {"status": "incomplete"}
                    elif self.state["ft_object_val_runs"][command]["status"] == "complete":
                        print(f"Skipping {command} because it is already complete.")
                    self.write_state_file()
                    try:
                        data_folder = self.run_command(command)
                        self.state["ft_object_val_runs"][command]["status"] = "complete"
                        self.state["ft_object_val_runs"][command]["folder"] = data_folder
                    except Exception as e:
                        self.state["ft_object_val_runs"][command]["status"] = "failed"
                        print(f"Failed to run {command}: {e}")
                        raise e
                    self.write_state_file()

        #### run meta level completions on val ####
        for model in self.args.model_configs + self.get_finetuned_model_configs() + self.args.val_only_model_configs:
            for task in self.args.task_configs + self.args.val_task_configs:
                for response_property in self.args.response_property_configs + self.args.val_response_property_configs:
                    for prompt in self.args.prompt_configs:
                        # pull the divergent strings
                        divergent_strings_path = self.state["divergent_strings"][task]["strings_path"]
                        command = self.get_meta_level_command(
                            model, task, response_property, prompt, self.args.n_meta_val, "val", divergent_strings_path
                        )
                        if command not in self.state["meta_val_runs"]:
                            self.state["meta_val_runs"][command] = {"status": "incomplete"}
                        # save other args to the state file
                        self.state["meta_val_runs"][command]["model"] = model
                        self.state["meta_val_runs"][command]["task"] = task
                        self.state["meta_val_runs"][command]["response_property"] = response_property
                        self.state["meta_val_runs"][command]["set"] = "val"
                        self.write_state_file()
                        try:
                            data_folder = self.run_command(command)
                            self.state["meta_val_runs"][command]["status"] = "complete"
                            self.state["meta_val_runs"][command]["folder"] = data_folder
                        except Exception as e:
                            self.state["meta_val_runs"][command]["status"] = "failed"
                            print(f"Failed to run {command}: {e}")
                            raise e
                        self.write_state_file()

        print("Finished running all commands.")


if __name__ == "__main__":
    study_runner = StudyRunner()
    study_runner.run_study()
