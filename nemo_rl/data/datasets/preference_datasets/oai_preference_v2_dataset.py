# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.datasets.response_datasets.oai_format_v2_dataset import (
    load_data_paths,
    merge_template_kwargs,
)

# Arguments of the other preference datasets or of the SFT `openai_format_v2`
# that this class does not support. They would otherwise be swallowed by **kwargs.
_UNSUPPORTED_ARGS = (
    "system_key",
    "system_prompt",
    "use_preserving_dataset",
    "split_validation_size",
    "dataset_paths",
)


def _as_messages(value: Any, role: str) -> list[dict[str, Any]]:
    """A list of messages, or a string wrapped as a single `role` message."""
    if isinstance(value, str):
        return [{"role": role, "content": value}]
    return list(value)


class OpenAIPreferenceDatasetV2(RawDataset):
    """DPO / RM dataset in the OpenAI format, with per-file options.

    Each sample looks like:
    {
        "prompt": [{"role": "user", "content": "What is 2+2?"}],     # or a string
        "chosen": [{"role": "assistant", "content": "4"}],            # or a string
        "rejected": [{"role": "assistant", "content": "5"}],          # or a string
        "tools": [...],                                               # optional
        "chat_template_kwargs": {"enable_thinking": true}             # optional
    }

    It is converted to the `PreferenceDataset` format (``context`` and two ranked
    ``completions``). The prompt can hold several turns, and the system prompt
    goes in the data.

    Without a ``prompt_key`` column (or when it is null), ``chosen`` and
    ``rejected`` are full conversations, as in
    OpenLLM-France/Luciole-PostTraining-Dataset-1.1, that differ only in their
    last message: the other messages are the context.

    Each completion is a single assistant message: DPO only trains the last
    message of each conversation, so a difference in an earlier turn would be
    ignored.

    ``data_path``, ``tool_key``, ``task_name``, ``seed`` and ``subsample_seed``
    work as in the SFT `OpenAIFormatDatasetV2`: per-file ``weight``,
    ``chat_template_kwargs`` and ``use_tools``, globs, json / jsonl(.gz) /
    parquet files, and the same ``chat_template_kwargs`` priorities. The
    validation data is given in ``data.validation``.

    Args:
        prompt_key: Key for the prompt messages (default: "prompt")
        chosen_key: Key for the preferred completion (default: "chosen")
        rejected_key: Key for the rejected completion (default: "rejected")
    """

    def __init__(
        self,
        data_path: "str | dict | list",
        prompt_key: str = "prompt",
        chosen_key: str = "chosen",
        rejected_key: str = "rejected",
        tool_key: str | None = "tools",
        task_name: str = "openai_preference_v2",
        seed: int = 42,
        subsample_seed: int | None = None,
        **kwargs,
    ):
        unsupported = [k for k in _UNSUPPORTED_ARGS if k in kwargs]
        if unsupported:
            raise TypeError(
                f"openai_preference_v2 does not support {unsupported}: put the files "
                "in `data_path`, the validation files in `data.validation`, and "
                "the system prompt in the data."
            )

        self.prompt_key = prompt_key
        self.chosen_key = chosen_key
        self.rejected_key = rejected_key
        self.tool_key = tool_key
        self.task_name = task_name

        columns = [prompt_key, chosen_key, rejected_key, "chat_template_kwargs"]
        if tool_key is not None:
            columns.append(tool_key)
        self.dataset = load_data_paths(
            data_path,
            columns,
            self.format_data,
            tool_key,
            subsample_seed=seed if subsample_seed is None else subsample_seed,
        )
        print(f"Loaded {self.task_name}: {len(self.dataset)} samples.")

        self.val_dataset = None

    def format_data(
        self,
        data: dict[str, Any],
        file_kwargs: dict[str, Any],
        use_tools: bool = False,
    ) -> dict[str, Any]:
        """Map one raw sample to ``{context, completions, chat_template_kwargs, task_name}``."""
        chosen = _as_messages(data[self.chosen_key], "assistant")
        rejected = _as_messages(data[self.rejected_key], "assistant")
        prompt = data.get(self.prompt_key)
        if prompt is None:
            context = chosen[:-1]
            assert context == rejected[:-1], (
                "chosen and rejected must differ only in their last message, "
                "the only one DPO trains on.\n\n"
                f"chosen: {chosen}\n\nrejected: {rejected}"
            )
            chosen, rejected = chosen[-1:], rejected[-1:]
        else:
            context = _as_messages(prompt, "user")
        for name, completion in (("chosen", chosen), ("rejected", rejected)):
            assert len(completion) == 1, (
                f"The {name} completion must be a single message, the only one "
                f"DPO trains on; got {completion}"
            )
            assert completion[0]["role"] == "assistant", (
                f"The last {name} message ({completion[0]}) is not from assistant!"
            )
        if chosen == rejected:
            raise ValueError(f"chosen and rejected are identical: {chosen!r}")
        return {
            "context": context,
            "completions": [
                {"rank": 0, "completion": chosen},
                {"rank": 1, "completion": rejected},
            ],
            "chat_template_kwargs": merge_template_kwargs(
                data, file_kwargs, use_tools, self.tool_key
            ),
            "task_name": self.task_name,
        }
