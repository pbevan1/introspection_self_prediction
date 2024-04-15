import asyncio
import logging
import time
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    TextGenerationPipeline,
    pipeline,
)

from evals.apis.inference.model import InferenceAPIModel
from evals.data_models.inference import LLMResponse
from evals.data_models.messages import Prompt

LOGGER = logging.getLogger(__name__)


class HuggingFaceModel(InferenceAPIModel):
    def __init__(self, prompt_history_dir: Path = None):
        self.pipe: TextGenerationPipeline | None = None
        self.tokenizer: PreTrainedTokenizerFast | None = None
        self.prompt_history_dir = prompt_history_dir
        self.lock = asyncio.Lock()

    async def __call__(
        self,
        model_ids: list[str],
        prompt: Prompt,
        print_prompt_and_response: bool,
        max_attempts: int = 1,  # ignored
        **kwargs,
    ) -> list[LLMResponse]:
        async with self.lock:  # Ensure that only one call is made at a time (to avoid OOM errors)
            start = time.time()
            assert len(model_ids) == 1, "HuggingFace transformers inference only supports one model at a time."
            short_model_name = model_ids[0]
            assert (
                kwargs.get("cais_path", None) is not None
            ), f"the config file for {short_model_name} lacks the `cais_path` key"
            hf_model_path = kwargs["cais_path"]
            if self.pipe is None:
                LOGGER.info(f"Loading model weights for {short_model_name}")
                self.pipe = pipeline(
                    "text-generation",
                    model=hf_model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
                self.tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
                LOGGER.info(f"Model weights for {short_model_name} loaded; {self.pipe.device=}")
            elif self.pipe.model.name_or_path != hf_model_path:
                raise RuntimeError("HuggingFace InferenceAPI only supports one model at a time")
            else:
                LOGGER.debug(f"Reusing a pipeline for {short_model_name}")
            hf_input: list[dict[str, str]] = prompt.anthropic_format()
            prompt_string: str = prompt.openai_format()
            prompt_file = self.create_prompt_history_file(prompt_string, short_model_name, self.prompt_history_dir)

            # make a forward pass
            output: dict[str, list[dict[str, str]]] = self.pipe(
                hf_input,
                do_sample=(not kwargs.get("temperature", 0.0)),
                temperature=kwargs.get("temperature", 0.0) or None,
                top_k=0,
                top_p=kwargs.get("top_p", 1.0),
                num_return_sequences=kwargs.get("n", 1),
                max_length=kwargs.get("max_tokens_to_sample", 2000),
                truncation=True,
            )
            duration = time.time() - start
            LOGGER.debug(f"Completed call to {short_model_name} in {duration}s")
            response = LLMResponse(
                model_id=short_model_name,
                completion=output[0]["generated_text"][-1]["content"],
                stop_reason="unknown",
                duration=duration,
                api_duration=0,
                cost=0,
                logprobs=[],  # TODO(tomek)
            )
            responses = [response]
            self.add_response_to_prompt_file(prompt_file, responses)
            if print_prompt_and_response:
                prompt.pretty_print(responses)
            return responses


async def test():
    from evals.utils import setup_environment

    setup_environment()
    huggingface_api = HuggingFaceModel()
    prompt_examples = [
        [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": "What is the next word in the following text? Respond only with a single word.\nAstoria Township is one of twenty-six townships in Fulton County, Illinois, USA.  As of the\n",
            },
        ],
        [
            {
                "role": "system",
                "content": "You are a swashbuckling space-faring voyager.",
            },
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "How dare you call me a"},
        ],
    ]
    prompts = [Prompt(messages=messages) for messages in prompt_examples]
    print(prompts)
    tasks = [
        huggingface_api(
            model_ids=["llama-7b-chat"],
            prompt=prompt,
            print_prompt_and_response=True,
            cais_path="/data/public_models/meta-llama/Llama-2-7b-chat-hf",
        )
        for prompt in prompts
    ]
    result = await asyncio.gather(*tasks)
    print(result)


if __name__ == "__main__":
    asyncio.run(test())
