# survival_instinct, myopic_reward
# On test set?


# read the data
# add the prompt
# make the answer either A or B with a dice roll.
# 2 epochs


import random

from pydantic import BaseModel
from slist import Slist

from other_evals.counterfactuals.api_utils import (
    raise_should_not_happen,
    read_jsonl_file_into_basemodel,
    write_jsonl_file_from_basemodel,
)
from other_evals.counterfactuals.other_eval_csv_format import (
    FinetuneConversation,
    FinetuneMessage,
)

# evals/datasets/val_survival_instinct.jsonl
# evals/datasets/val_myopic_reward.jsonl
randomer = random.Random()


class Data(BaseModel):
    string: str
    target: str

    def to_finetuning(self) -> FinetuneConversation:
        target = self.target
        # sys = FinetuneMessage(
        #     role="system",
        #     content="",  # our training has an empty system message
        # )
        user = FinetuneMessage(
            role="user",
            content=self.string
            + "\nPlease answer immediately with what you think. Answer only with either A or B and nothing else. Do not provide justification.",
        )
        message = FinetuneMessage(role="assistant", content=target)
        return FinetuneConversation(messages=[user, message])
    
    def to_finetuning_control(self) -> FinetuneConversation:
        target = self.target
        # sys = FinetuneMessage(
        #     role="system",
        #     content="",  # our training has an empty system message
        # )
        user = FinetuneMessage(
            role="user",
            content=self.string
            + "\nPlease answer immediately with what you think. Answer only with either A or B and nothing else. Do not provide justification.",
        )
        control_target = "A" if target == "B" else "B" if target == "A" else raise_should_not_happen()
        message = FinetuneMessage(role="assistant", content=control_target)
        return FinetuneConversation(messages=[user, message])


def matches_behavior_samples(number: int) -> Slist[FinetuneConversation]:
    data = read_jsonl_file_into_basemodel(
        "evals/datasets/train_survival_instinct.jsonl", Data
    ) + read_jsonl_file_into_basemodel("evals/datasets/train_myopic_reward.jsonl", Data)
    finetune = data.map(lambda x: x.to_finetuning()).shuffle("42")
    # write_jsonl_file_from_basemodel("finetune.jsonl", finetune)
    return finetune.take(number)


data = read_jsonl_file_into_basemodel("evals/datasets/train_myopic_reward.jsonl", Data).filter(
    # make sure "short-term" or "myopi" is not in the string
    lambda x: "short-term" not in x.string and "myopi" not in x.string and "short term" not in x.string
)

finetune = data.map(lambda x: x.to_finetuning()).shuffle("42")
finetune_control = data.map(lambda x: x.to_finetuning_control()).shuffle("42")

# dump
write_jsonl_file_from_basemodel("finetune_myopia.jsonl", finetune)
write_jsonl_file_from_basemodel("finetune_control_myopia.jsonl", finetune_control)
