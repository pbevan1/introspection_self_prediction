import argparse
import random
from pathlib import Path


def split_jsonl_into_train_validation(input_file_path):
    # Set the seed for reproducibility
    random.seed(42)

    # Define the split ratio
    split_ratio = 0.8  # 80% for training, 20% for validation

    # File paths
    input_file_path = Path(input_file_path)
    train_file_path = input_file_path.parent / f"train_{input_file_path.name}"
    validation_file_path = input_file_path.parent / f"val_{input_file_path.name}"

    # Read .jsonl file and split the data
    with open(input_file_path, "r") as infile, open(train_file_path, "w") as train_file, open(
        validation_file_path, "w"
    ) as validation_file:
        # Temporary lists to store the splits
        lines = infile.readlines()
        random.shuffle(lines)  # Shuffle the lines to ensure random distribution

        # Calculate the split index
        split_index = int(len(lines) * split_ratio)

        # Split the data
        train_lines = lines[:split_index]
        validation_lines = lines[split_index:]

        # Write the splits to their respective files
        for line in train_lines:
            train_file.write(line)
        for line in validation_lines:
            validation_file.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split JSONL file into train and validation sets")
    parser.add_argument("input_file_path", type=str, help="Path to the input JSONL file")
    args = parser.parse_args()

    input_file_path = args.input_file_path
    split_jsonl_into_train_validation(input_file_path)
    print("Split complete.")
