"""Sweep over base and self-predictions across tasks and across models."""

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
    parser.add_argument("--study_dir", type=str, help="The directory to save the study results.")
    parser.add_argument("--model_configs", type=str, help="Comma-separated list of model configurations to sweep over.")
    parser.add_argument("--task_configs", type=str, help="Comma-separated list of data configurations to sweep over.")
    parser.add_argument(
        "--response_property_configs",
        type=str,
        help="Comma-separated list of response property configurations to sweep over.",
        default="prediction",
    )
    parser.add_argument(
        "--string_modifier_configs",
        type=str,
        help="Comma-separated list of string modifier configurations to sweep over.",
        default="unmodified",
    )
    parser.add_argument(
        "--overrides", type=str, help="Comma-separated list of Hydra configuration overrides.", default=""
    )
    return parser.parse_args()


def main():
    """
    Example usage:
        python -m scripts.sweep_base_and_self_preds
        --study_dir="exp/finetuned_numbers_wikipedia_bergenia_various_response_properties"
        --model_configs="finetuned/numbers_wikipedia_bergenia_various_response_properties/finetuned_numbers_wikipedia_bergenia_various_response_properties_35.yaml,gpt-3.5-turbo"
        --task_configs="random_words"
        --response_property_configs="prediction,ends_with_vowel,number_of_letters"
        --overrides="limit=500, strings_path=exp/random_words_bergenia/model_divergent_strings_35_4.csv"
    """
    args = parse_arguments()

    # Convert comma-separated strings to lists
    model_configs = args.model_configs.split(",")
    task_configs = args.task_configs.split(",")
    response_property_configs = args.response_property_configs.split(",")
    string_modifier_configs = args.string_modifier_configs.split(",")
    overrides = args.overrides.split(",") if args.overrides else []

    object_level_command = f"python -m evals.run_object_level study_dir={args.study_dir}"
    meta_level_command = f"python -m evals.run_meta_level study_dir={args.study_dir}"

    # run sweep over base completions
    for model_config in model_configs:
        for data_config in task_configs:
            cmd = f"{object_level_command} language_model={model_config} task={data_config} {' '.join(overrides)}"
            print("Running command:", cmd)
            run_command(cmd)

    print("Finished running base completions.")

    # run sweep over self completions
    for model_config in model_configs:
        for data_config in task_configs:
            for response_property_config in response_property_configs:
                for string_modifier_config in string_modifier_configs:
                    cmd = f"{meta_level_command} language_model={model_config} task={data_config} response_property={response_property_config} string_modifier={string_modifier_config} {' '.join(overrides)}"
                    print("Running command:", cmd)
                    run_command(cmd)

    print("Finished running self completions.")


if __name__ == "__main__":
    main()
