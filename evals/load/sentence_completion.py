"""
This file holds sentence completion related functions.

These datasets are represented as .jsonl files, where each line is a JSON that has to contain a "content" key. The rest is treated as metadata.

See for example evals/datasets/daily_dialog.jsonl or evals/datasets/wikipedia.jsonl.
"""

import json
import logging
import random
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def sentence_completion_generator(dataset_path: Path, seed: int = 0, n_per_line: int = 1, **kwargs):
    """
    Generator that yields sentences from a file.

    Args:
        file_path: Path to the file with the sentences.
        **kwargs: Additional arguments to pass to the generator.

    Yields:
        str: A sentence from the file.
    """
    # set random seed
    random.seed(seed)

    with open(dataset_path, "r") as f:
        # while True:
        for line in f:
            for _ in range(n_per_line):
                try:
                    yield extract_completion_from_line(line, **kwargs)
                except Exception as e:
                    print(f"Error while processing line: {line}:{e}")
                    continue
            # LOGGER.info(f"Generated completions for every line in {file_path} {n_per_line} times. Looping around...")
            # f.seek(0) # go back to the beginning of the file


def extract_completion_from_line(
    line,
    seed: int = 0,
    min_words_context: int = 12,
    max_words_context: int = 20,
    start_at_beginning: bool = True,
    **kwargs,
):
    """
    Extracts a completion from a line.

    Args:
        line: A line from the file.
        seed: Seed for the random number generator.
        min_words_context: Minimum number of words in the context.
        max_words_context: Maximum number of words in the context.
        start_at_beginning: Whether to start the completion at the beginning of the content.
        **kwargs: Additional arguments to pass to the generator. Ignored.

    Returns a dict with the following keys:
        string: The context and the completion.
        target: The ground truth completion.
        metadata: The entire line as dict for metadata.
    """
    if not isinstance(line, dict):
        # parse the line as json
        line = json.loads(line)
    content = line["content"]
    content_words = content.split(" ")

    # get a random start index
    if not start_at_beginning:
        start_index = random.randint(0, len(content_words) - (max_words_context + 1))
    else:
        start_index = 0

    # get end index
    end_index = start_index + random.randint(min_words_context, max_words_context)
    # target word
    target_word = content_words[end_index]
    context_words = content_words[start_index:end_index]

    context = " ".join(context_words)

    return {"string": context, "target": target_word, "metadata": line}
