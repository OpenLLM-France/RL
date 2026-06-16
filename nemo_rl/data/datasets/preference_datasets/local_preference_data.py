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
from typing import Any, Callable, Union, List

from datasets import load_dataset, concatenate_datasets
from pathlib import Path
import math
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

def _normalize_weighted_paths(paths) -> list[tuple[str, float]]:
    """Normalize a path or list of entries to `[(path, weight), ...]`.

    Accepts:
      - a bare string
      - a list of strings
      - a list of ``{"path": ..., "weight": ...}`` dicts (weight optional,
        defaults to 1.0)
      - a list of ``[path, weight]`` / ``(path, weight)`` pairs (legacy)
      - a mix of the above

    Each weight must satisfy ``0 < weight <= 1.0``.
    """
    if isinstance(paths, str):
        paths = [paths]

    out: list[tuple[str, float]] = []
    for entry in paths:
        if isinstance(entry, str):
            path, weight = entry, 1.0
        elif isinstance(entry, dict):
            if "path" not in entry:
                raise ValueError(
                    f"Dict entry must have a 'path' key; got {entry!r}."
                )
            path = entry["path"]
            weight = entry.get("weight", 1.0)
        elif isinstance(entry, (tuple, list)) and len(entry) == 2:
            path, weight = entry
        else:
            raise ValueError(
                f"Each train_ds_path entry must be a path, a dict with "
                f"'path'/'weight', or a (path, weight) pair; got {entry!r}."
            )
        if weight > 1.0:
            raise ValueError(
                f"weight for {path!r} must be <= 1.0; got {weight}. "
                "Oversampling (weight > 1) is not supported."
            )
        if weight <= 0:
            raise ValueError(
                f"weight for {path!r} must be > 0; got {weight}."
            )
        out.append((str(path), float(weight)))
    return out

class LocalPreferenceDataset:
    """Local preference dataset for DPO training."""

    def __init__(
            self,
            dataset_paths: Union[str, List[Union[str, tuple[str, float]]]],
            split: str = "train",
            subsample_seed: int = 42,
        ) -> None:

        if isinstance(dataset_paths, str):
            dataset_paths = [dataset_paths]

        weighted_dataset_paths = _normalize_weighted_paths(dataset_paths)

        datasets = []
        cols_to_keep = ["chosen","rejected"]
        for train_path, weight in weighted_dataset_paths:
            ds = load_dataset("json", data_files=train_path)["train"]
            if weight < 1.0:
                n = math.floor(len(ds) * weight)
                ds = ds.shuffle(seed=subsample_seed).select(range(n))
            print(f"  - {Path(train_path).stem} ({train_path}): {len(ds)} samples (weight={weight})")
            ds = ds.remove_columns([c for c in ds.column_names if c not in cols_to_keep])
            datasets.append(ds)

        ds = concatenate_datasets(datasets)

        self.formatted_ds = {
            "train": ds.map(to_preference_data_format),
            "validation": None,
        }

        self.task_spec = TaskDataSpec(
            task_name="LocalPreference",
        )
