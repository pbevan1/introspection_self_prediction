import subprocess
import sys
from pathlib import Path

import pandas as pd

# Run the git command to get the repository root directory
REPO_DIR = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()

print("Repository directory:", REPO_DIR)
sys.path.append(REPO_DIR)

from analysis.compliance_checks import check_compliance  # noqa: E402


# TODO add proper logging
def extract_most_uncertain_strings_from_base(
    input_file_path, n_out_strings, output_file_path=None, how="logprob_delta", minimize=True
) -> None:
    """Extracts the most uncertain strings from the base level completions. Saves a .csv with the strings into the same directory as the input file called `out_strings.csv`.

    Args:
        input_file_path: Path to the .csv with the base level completions.
        n_out_strings: Number of strings to extract.
        how: How to select the strings. Can be "logprob_delta" or "logprob_delta_rel".
        minimize: Whether to minimize the value of `how` or not.
    """
    if how not in ["logprob_delta", "logprob_delta_rel"]:
        raise ValueError(f"how {how} not supported")

    # load the data
    df = pd.read_csv(input_file_path)
    print(f"Loaded {len(df)} rows from {input_file_path}")

    # Run compliance checks. See `evals/run_dataset_generation.py` for more details.

    df["compliance"] = df["response"].apply(check_compliance)

    print(f"Compliance: {(df['compliance'] == True).mean():.2%}")  # noqa: E712

    # print("Most common non-compliant reasons:")
    # df[df['compliance'] != True]['compliance'].value_counts().head(10)

    # Exclude non-compliant responses
    df = df[df["compliance"] == True]  # noqa: E712
    print(f"Excluded non-compliant responses, leaving {len(df)} rows")

    # We want to select strings that are hard to predict from external means. Here, we choose those for which the two top first tokens have similar probabilities.

    # add log_prob delta column
    df["logprobs"] = df["logprobs"].apply(lambda x: eval(x)[0])
    df["logprob_delta"] = df["logprobs"].apply(lambda x: list(x.values())[0] - list(x.values())[1])
    # sort to get the most uncertain strings
    df = df.sort_values([how], ascending=[minimize])

    # the same with the relative difference
    df["logprob_delta_rel"] = df["logprobs"].apply(
        lambda x: (list(x.values())[0] - list(x.values())[1]) / list(x.values())[0]
    )

    # Pull the _n_ strings with the closest top 2 probabilities.

    out_df = df.head(n_out_strings)

    # save the output strings
    if output_file_path is None:
        output_file_path = Path(input_file_path).parent / "out_strings.csv"
    out_df["string"].to_csv(output_file_path, index=False)
    print(f"Saved {len(out_df)} strings to {output_file_path}")
