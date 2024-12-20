import json
from enum import Enum
from typing import Dict, Optional, Self, Sequence

import anthropic
from pydantic import BaseModel
from termcolor import cprint
from vertexai.generative_models import Content, Part

from evals.data_models.hashable import HashableBaseModel
from evals.data_models.inference import LLMResponse

PRINT_COLORS = {"user": "cyan", "system": "magenta", "assistant": "light_green", "none": "cyan"}


class MessageRole(str, Enum):
    user = "user"
    system = "system"
    assistant = "assistant"
    # none is designed for completion tasks where no role / tag will be added
    none = "none"


class ChatMessage(HashableBaseModel):
    role: MessageRole
    content: str
    # model_config = ConfigDict(frozen=True)

    def __str__(self) -> str:
        return f"{self.role}: {self.content}"

    def remove_role(self) -> "ChatMessage":
        return ChatMessage(role=MessageRole.none, content=self.content)


class PromptTemplate(BaseModel):
    method: str
    messages: Sequence[ChatMessage]
    messages_followup: Optional[Sequence[ChatMessage]] = None
    extra: Dict[str, str] = {}


class Prompt(HashableBaseModel):
    messages: Sequence[ChatMessage]

    def __str__(self) -> str:
        out = ""
        for msg in self.messages:
            if msg.role != MessageRole.none:
                out += f"\n\n{msg.role.value}: {msg.content}"
            else:
                out += f"\n{msg.content}"
        return out.strip()

    def __add__(self, other: "Prompt") -> Self:
        return self.__class__(messages=list(self.messages) + list(other.messages))

    @classmethod
    def from_prompt(cls, prompt: "Prompt") -> Self:
        return cls(messages=prompt.messages)

    def is_none_in_messages(self) -> bool:
        return any(msg.role == MessageRole.none for msg in self.messages)

    def is_last_message_assistant(self) -> bool:
        return self.messages[-1].role == MessageRole.assistant

    def add_assistant_message(self, message: str) -> "Prompt":
        return self + Prompt(messages=[ChatMessage(role=MessageRole.assistant, content=message)])

    def openai_format(self) -> list[Dict]:
        if self.is_last_message_assistant():
            raise ValueError(
                f"OpenAI chat prompts cannot end with an assistant message. Got {self.messages[-1].role}: {self.messages[-1].content}"
            )
        if self.is_none_in_messages():
            raise ValueError(f"OpenAI chat prompts cannot have a None role. Got {self.messages}")
        return [msg.model_dump() for msg in self.messages]

    def openai_finetuning_format(self) -> str:
        msgs = [msg.model_dump() for msg in self.messages]
        # fix the message roles
        for i, msg in enumerate(msgs):
            msgs[i]["role"] = msg["role"].value
        line = {"messages": msgs}
        return json.dumps(line)

    def gemini_finetuning_format(self) -> str:
        # Gemini uses the same format as OpenAI except uses 'model' instead of 'assistant'
        msgs = [msg.model_dump() for msg in self.messages]
        msgs = [msg for msg in msgs if msg["content"]]  # filter out roles with empty content eg system
        # fix the message roles
        for i, msg in enumerate(msgs):
            val = msg["role"].value
            msgs[i]["role"] = val if val != "assistant" else "model"
        line = {"messages": msgs}
        return json.dumps(line)

    def gemini_format(self) -> list[str]:
        if self.is_none_in_messages():
            raise ValueError(f"Gemini chat prompts cannot have a None role. Got {self.messages}")
        messages = []
        # import ipdb; ipdb.set_trace()
        for msg in self.messages:
            if msg.role == MessageRole.system:  # Drop system role
                continue
            elif msg.role == MessageRole.user:
                messages.append(Content(parts=[Part.from_text(msg.content)], role="user"))
            # Gemini uses the same format as OpenAI except uses 'model' instead of 'assistant'
            elif msg.role == MessageRole.assistant:
                messages.append(Content(parts=[Part.from_text(msg.content)], role="model"))
        assert len(messages) > 0, f"Somehow created an empty prompt for gemini. Using: {self}"
        return messages

    def gemini_format_text(self) -> str:
        if self.is_none_in_messages():
            raise ValueError(f"Gemini chat prompts cannot have a None role. Got {self.messages}")
        messages = []
        for msg in self.messages:
            if msg.role == MessageRole.system:  # Drop system role
                continue
            elif msg.role == MessageRole.user:
                messages.append({"role": "user", "content": msg.content})
            # Gemini uses the same format as OpenAI except uses 'model' instead of 'assistant'
            elif msg.role == MessageRole.assistant:
                messages.append({"role": "model", "content": msg.content})
        return messages

    def anthropic_format(self) -> list[dict]:
        if self.is_none_in_messages():
            raise ValueError(f"Anthropic chat prompts cannot have a None role. Got {self.messages}")

        messages = []
        for msg in self.messages:
            if msg.role == MessageRole.system and msg.content:  # Drop system role if content is empty:
                messages.append({"role": "system", "content": msg.content})
            elif msg.role == MessageRole.user:
                messages.append({"role": "user", "content": msg.content})
            elif msg.role == MessageRole.assistant:
                messages.append({"role": "assistant", "content": msg.content})

        return messages

    def anthropic_format_string(self) -> str:
        if self.is_none_in_messages():
            raise ValueError(f"Anthropic chat prompts cannot have a None role. Got {self.messages}")

        message = ""
        for msg in self.messages:
            if msg.role == MessageRole.user:
                message += f"{anthropic.HUMAN_PROMPT} {msg.content}"
            elif msg.role == MessageRole.assistant:
                message += f"{anthropic.AI_PROMPT} {msg.content}"
            elif msg.role == MessageRole.none:
                message += f"\n\n{msg.content}"

        return message

    def pretty_print(self, responses: list[LLMResponse]) -> None:
        for msg in self.messages:
            if msg.role != MessageRole.none:
                cprint(f"=={msg.role.upper()}:", "white")
            cprint(msg.content, PRINT_COLORS[msg.role])
        for i, response in enumerate(responses):
            cprint(f"==RESPONSE {i + 1} ({response.model_id}):", "white")
            cprint(response.completion, PRINT_COLORS["assistant"], attrs=["bold"])
        print()

    def truncate_messages(self, max_length: int = 1000) -> "Prompt":
        truncated_prompt = Prompt(messages=[])
        for msg in self.messages:
            if len(msg.content) > max_length:
                msg = ChatMessage(role=msg.role, content=msg.content[:max_length])
            truncated_prompt.messages.append(msg)
        return truncated_prompt
