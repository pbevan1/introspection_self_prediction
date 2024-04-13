"""Sweep over object and meta_levels configs across tasks and across models."""

import argparse
import subprocess


def run_command(command):
    """Execute the given command in the shell."""
    try:
        subprocess.run(command, check=True, shell=True)
        print(f"Successfully executed: {command}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing {command}: {e}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run a Hydra-configured script with YAML configuration overrides.")
    # Accept comma-separated strings for all list-type arguments
    parser.add_argument("--study_name", type=str, help="The name of the study. Defines the output directory.")
    parser.add_argument("--model_configs", type=str, help="Comma-separated list of model configurations to sweep over.")
    parser.add_argument("--task_configs", type=str, help="Comma-separated list of data configurations to sweep over.")
    parser.add_argument(
        "--response_property_configs",
        type=str,
        help="Comma-separated list of response property configurations to sweep over.",
        default="identity",
    )
    parser.add_argument(
        "--overrides", type=str, help="Comma-separated list of Hydra configuration overrides.", default=""
    )
    parser.add_argument(
        "--object_overrides",
        type=str,
        help="Comma-separated list of Hydra configuration overrides for the object-level only.",
        default="",
    )
    parser.add_argument(
        "--meta_overrides",
        type=str,
        help="Comma-separated list of Hydra configuration overrides for the meta-level only.",
        default="",
    )
    return parser.parse_args()


def main():
    r"""
    Example usage:
        python -m scripts.sweep_object_and_meta_levels \
        --study_name="number_triplets_reproduction" \
        --model_configs="finetuned/numbers_wikipedia_bergenia_various_response_properties/finetuned_numbers_wikipedia_bergenia_various_response_properties_35.yaml,gpt-3.5-turbo" \
        --task_configs="wikipedia" \
        --response_property_configs="identity,sentiment" \
        --overrides="limit=500, strings_path=exp/random_words_bergenia/model_divergent_strings_35_4.csv"
    """
    args = parse_arguments()

    # Convert comma-separated strings to lists
    model_configs = args.model_configs.split(",")
    task_configs = args.task_configs.split(",")
    response_property_configs = args.response_property_configs.split(",")
    overrides = args.overrides.split(",") if args.overrides else []
    object_overrides = args.object_overrides.split(",") if args.object_overrides else []
    meta_overrides = args.meta_overrides.split(",") if args.meta_overrides else []

    object_level_command = f"python -m evals.run_object_level study_name={args.study_name} task.set=val"
    meta_level_command = f"python -m evals.run_meta_level study_name={args.study_name} task.set=val"

    # run sweep over object and meta levels
    for model_config in model_configs:
        for task_config in task_configs:
            cmd = f"{object_level_command} language_model={model_config} task={task_config} {' '.join(map(str.strip, overrides + object_overrides))}"
            print("Running command:", cmd)
            run_command(cmd)

    print("Finished running object completions.")

    # run sweep over meta completions
    for model_config in model_configs:
        for task_config in task_configs:
            for response_property_config in response_property_configs:
                cmd = f"{meta_level_command} language_model={model_config} task={task_config} response_property={response_property_config} {' '.join(map(str.strip, overrides + meta_overrides))}"
                print("Running command:", cmd)
                run_command(cmd)

    print("Finished running meta completions.")


if __name__ == "__main__":
    main()
