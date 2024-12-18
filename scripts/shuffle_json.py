import json
import os
import random
import sys


def shuffle_json(input_path):
    # Create output filename by adding '_shuffled' before the extension
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_shuffled{ext}"

    # Read the JSON file
    with open(input_path, "r") as f:
        data = [json.loads(line) for line in f]

    # Shuffle the data
    random.shuffle(data)

    # Write the shuffled data back to a new file
    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    print(f"Shuffled data written to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/shuffle_json.py <path_to_json_file>")
        sys.exit(1)

    input_path = sys.argv[1]
    shuffle_json(input_path)
