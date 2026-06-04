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

import json
from typing import Any

from datasets import load_dataset, concatenate_datasets

from nemo_rl.data.interfaces import TaskDataSpec


def to_preference_data_format(
    data: dict[str, Any],
) -> dict[
    str, list[dict[str, int | list[dict[str, str | Any]]]] | list[dict[str, str]]
]:
    chosen_conversation = data["chosen"]
    rejected_conversation = data["rejected"]

    context = chosen_conversation[:-1]

    # We assume that except last assistant response, all messages in
    # chosen and rejected conversations are similar. Validating this...
    assert json.dumps(context, ensure_ascii=False) == json.dumps(
        rejected_conversation[:-1], ensure_ascii=False
    ), (
        f"Context mismatch.\n\nchosen: {chosen_conversation}\n\n rejected: {rejected_conversation}"
    )

    # We assume that last response is always from the assistant. Validating this...
    assert chosen_conversation[-1]["role"] == "assistant", (
        f"The last chosen response ({chosen_conversation[-1]}) is not from assistant!"
    )
    assert rejected_conversation[-1]["role"] == "assistant", (
        f"The last rejected response ({rejected_conversation[-1]}) is not from assistant!"
    )

    chosen_response = chosen_conversation[-1]["content"]
    rejected_response = rejected_conversation[-1]["content"]

    return {
        "context": context,
        "completions": [
            {
                "rank": 0,
                "completion": [{"role": "assistant", "content": chosen_response}],
            },
            {
                "rank": 1,
                "completion": [{"role": "assistant", "content": rejected_response}],
            },
        ],
    }


class LocalPreferenceDataset:
    """Local preference dataset for DPO training."""

    def __init__(
            self,
            dataset_paths: str | list[str] = [],
            split: str = "train",
        ) -> None:

        if isinstance(dataset_paths, str):
            dataset_paths = [dataset_paths]

        datasets = [
            load_dataset("json", data_files=path, split=split) if ".json" in path
            else load_dataset(path, split=split)
            for path in dataset_paths
        ]

        ds = concatenate_datasets(datasets)

        self.formatted_ds = {
            "train": ds.map(to_preference_data_format),
            "validation": None,
        }

        self.task_spec = TaskDataSpec(
            task_name="LocalPreference",
        )