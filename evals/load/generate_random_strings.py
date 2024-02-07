import logging
import random
from pathlib import Path
from typing import Union

import nltk
import pandas as pd
from nltk.corpus import words

# download corpus if necessary
nltk.download("words", quiet=True)


LOGGER = logging.getLogger(__name__)

RANDOM_SETS = {
    "numbers": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "letters": [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
    ],
    "animals": [
        "dog",
        "cat",
        "cow",
        "horse",
        "sheep",
        "goat",
        "chicken",
        "pig",
        "duck",
        "rabbit",
        "deer",
        "elephant",
        "lion",
        "tiger",
        "bear",
        "giraffe",
        "zebra",
        "kangaroo",
        "panda",
        "wolf",
        "fox",
        "squirrel",
        "mouse",
        "rat",
        "frog",
        "turtle",
        "snake",
        "lizard",
        "fish",
        "shark",
    ],
    "english_words": words.words(),
    "number_doublets": [f"{n:02d}" for n in range(100)],
    "number_triplets": [f"{n:03d}" for n in range(1000)],
    "number_quadruplets": [f"{n:04d}" for n in range(10000)],
}


def generate_random_strings(
    filename: Path,
    set="numbers",
    seed: int = 42,
    string_length: Union[int, list[int]] = [6, 10],
    num: int = 1000,
    join_on: str = " ",
):
    random.seed(seed)
    data = []
    if isinstance(string_length, int):
        string_length = [string_length, string_length]
    for i in range(num):
        k = random.randint(string_length[0], string_length[1])
        string = join_on.join(random.choices(RANDOM_SETS[set], k=k)) + join_on
        data.append({"id": i, "string": string})

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding="utf-8")
    LOGGER.info(f"Saved {len(df)} rows to {filename}")
