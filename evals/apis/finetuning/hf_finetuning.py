import logging

from datasets import load_dataset
from rich.logging import RichHandler
from transformers import AutoTokenizer, TrainingArguments
from trl import ModelConfig, SFTTrainer, get_peft_config
from trl.commands.cli_utils import SftScriptArguments, TrlParser, init_zero_verbose


def run_hf_finetuning(
    model_name: str,
    train_data_path: str,
    val_data_path: str,
    training_args: TrainingArguments | None = None,
    model_config: ModelConfig | None = None,
) -> str:
    dataset = load_dataset("json", data_files={"train": train_data_path, "validation": val_data_path})
    training_args.disable_tqdm = True
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.unk_token
    trainer = SFTTrainer(
        model=model_name,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        # callbacks=[RichProgressCallback],
        dataset_kwargs=dict(add_special_tokens=False),
        peft_config=get_peft_config(model_config),
    )
    trainer.train()
    print(f"Training completed! Saving the model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    return training_args.output_dir


if __name__ == "__main__":
    init_zero_verbose()
    logging.basicConfig(format="%(message)s", datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)
    parser = TrlParser((SftScriptArguments, TrainingArguments, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()
    run_hf_finetuning(
        model_name=model_config.model_name_or_path,
        train_data_path=args.dataset_name + "/train_dataset.jsonl",
        val_data_path=args.dataset_name + "/val_dataset.jsonl",
        training_args=training_args,
        model_config=model_config,
    )

"""
accelerate launch /data/tomek_korbak/introspection_self_prediction_astra/evals/apis/finetuning/hf_finetuning.py --output_dir /data/tomek_korbak/introspection_self_prediction_astra/exp/finetuning/full_sweep_test/llama-7b-chat2/ --model_name_or_path /data/public_models/meta-llama/Llama-2-7b-chat-hf --dataset_name /data/tomek_korbak/introspection_self_prediction_astra/exp/finetuning/full_sweep_test/llama-7b-chat/ --batch_size 128 --learning_rate 5e-5 --num_train_epochs 4 --seed 42 --fp16 --save_only_model --logging_steps 1
"""
