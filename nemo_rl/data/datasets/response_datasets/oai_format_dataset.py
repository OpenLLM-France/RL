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
import math
import random
import warnings
from typing import Any, Callable, Union, List

from datasets import load_dataset, concatenate_datasets
from pathlib import Path

from nemo_rl.data.interfaces import TaskDataSpec


class PreservingDataset:
    """A dataset wrapper that preserves original dict structure without None-filling.

    Unlike HuggingFace's Dataset class which enforces schema uniformity across all samples
    (filling missing keys with None), this class maintains the exact structure of each sample.
    This is critical for heterogeneous data like tool calls where different samples may have
    different argument structures.
    """

    def __init__(self, data: list[dict[str, Any]]):
        """Initialize the dataset with a list of dictionaries.

        Args:
            data: List of dictionary samples, each can have different keys
        """
        self.data = data
        self.features = None  # For compatibility with HF Dataset interface

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, idx: Union[int, slice, list]
    ) -> Union[dict[str, Any], list[dict[str, Any]]]:
        """Support integer indexing, slicing, and list indexing."""
        if isinstance(idx, slice):
            return [self.data[i] for i in range(*idx.indices(len(self.data)))]
        elif isinstance(idx, int):
            # Handle negative indices
            if idx < 0:
                idx = len(self.data) + idx
            if idx < 0 or idx >= len(self.data):
                raise IndexError(
                    f"Index {idx} out of range for dataset of size {len(self.data)}"
                )
            return self.data[idx]
        elif isinstance(idx, list):
            return [self.data[i] for i in idx]
        else:
            raise TypeError(
                f"Indices must be integers, slices, or lists, not {type(idx)}"
            )

    def __iter__(self):
        return iter(self.data)

    def map(self, function: Callable, *args, **kwargs) -> "PreservingDataset":
        """Apply a function to each sample in the dataset.

        Args:
            function: Function to apply to each sample
            with_indices: If True, pass index as second argument to function

        Returns:
            New PreservingDataset with transformed samples
        """
        if kwargs.get("with_indices", False):
            mapped_data = [function(item, i) for i, item in enumerate(self.data)]
        else:
            mapped_data = [function(item) for item in self.data]
        return PreservingDataset(mapped_data)

import glob
import gzip
import json
import os


def load_jsonl_files(paths) -> list:
    """
    Load records from one or more file paths or glob patterns.
    Supports plain .jsonl and gzip-compressed .jsonl.gz files.
    
    Args:
        paths: A str/Path, or a list of str/Path glob patterns.
    
    Returns:
        A flat list of parsed JSON records.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]

    records = []
    for pattern in paths:
        matched = sorted(glob.glob(str(pattern), recursive=True))
        if not matched:
            print(f"Warning: no files matched pattern '{pattern}'")
        for filepath in matched:
            open_fn = gzip.open if filepath.endswith(".gz") else open
            with open_fn(filepath, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
    return records


def _normalize_weighted_paths(paths) -> list[tuple[str, float]]:
    """Normalize a path or list of paths/`(path, weight)` entries to `[(path, weight), ...]`.

    Accepts a bare string, a list of strings, a list of `(path, weight)` tuples,
    a list of `[path, weight]` lists (as produced by YAML), or a mix. Bare paths
    default to weight 1.0. Each weight must satisfy ``0 < weight <= 1.0``.
    """
    if isinstance(paths, str):
        paths = [paths]

    out: list[tuple[str, float]] = []
    for entry in paths:
        if isinstance(entry, str):
            path, weight = entry, 1.0
        elif isinstance(entry, (tuple, list)) and len(entry) == 2:
            path, weight = entry
        else:
            raise ValueError(
                f"Each train_ds_path entry must be a path or a (path, weight) pair; got {entry!r}."
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


class OpenAIFormatDataset:
    """This class is used to load an SFT dataset in the OpenAI format.

    The dataset should be in the following format:
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."}
        ]
    }

    Args:
        train_ds_path: Path(s) to the training dataset JSON file(s). Can be:
            - a single path (str)
            - a list of paths
            - a list mixing paths and ``(path, weight)`` pairs, where
              ``0 < weight <= 1.0`` randomly subsamples that file to
              ``floor(len * weight)`` records. Bare paths default to weight 1.0.
              Oversampling (weight > 1) is not supported.
        val_ds_path: Path to the validation dataset JSON file. Pass ``None``
            (or YAML ``null``) to skip validation; ``formatted_ds["validation"]``
            will then be ``None``.
        chat_key: Key for the messages list in the dataset (default: "messages")
        system_key: Optional key for system prompt in the dataset
        system_prompt: Optional system prompt to add if not in the dataset
        tool_key: Key for tools in the dataset (default: "tools")
        use_preserving_dataset: If True, uses PreservingDataset to maintain
            heterogeneous schemas (e.g., for tool calls with varying argument
            structures). If False, uses standard HuggingFace dataset loading.
            Default is False for backward compatibility.
        subsample_seed: Seed used when randomly subsampling files with weight < 1.0.
            Same seed yields the same kept records across runs.

    Notes:
        - system_key and system_prompt are optional. If provided, it will be added
          to the beginning of the dataset.
        - chat_key should be the key of the messages list. Multi-turn conversations
          are supported.
        - The last message in the conversation must be from the assistant.
        - When use_preserving_dataset=True, the dataset preserves the exact structure
          of each sample without None-filling for missing keys, which is useful for
          heterogeneous tool argument schemas.
    """

    def __init__(
        self,
        train_ds_path: Union[str, List[Union[str, tuple[str, float]]]],
        val_ds_path: str | None,
        chat_key: str = "messages",
        system_key: str | None = None,
        system_prompt: str | None = None,
        tool_key: str | None = "tools",
        use_preserving_dataset: bool = False,
        subsample_seed: int = 42,
    ):
        self.chat_key = chat_key
        self.system_key = system_key
        self.system_prompt = system_prompt
        self.tool_key = tool_key

        weighted_train_paths = _normalize_weighted_paths(train_ds_path)

        if not use_preserving_dataset:

            train_datasets = []

            for train_path, weight in weighted_train_paths:
                ds = load_dataset("json", data_files=train_path)["train"]
                if weight < 1.0:
                    n = math.floor(len(ds) * weight)
                    ds = ds.shuffle(seed=subsample_seed).select(range(n))
                print(f"  - {Path(train_path).stem} ({train_path}): {len(ds)} samples (weight={weight})")
                cols_to_keep = [chat_key]
                if system_key is not None:
                    cols_to_keep.append(system_key)
                if tool_key is not None:
                    cols_to_keep.append(tool_key)

                ds = ds.remove_columns(
                    [c for c in ds.column_names if c not in cols_to_keep]
                )

                ds = ds.map(self.add_messages_key)
                train_datasets.append(ds)

            formatted_train_dataset = concatenate_datasets(train_datasets)

            # validation
            if val_ds_path is None:
                formatted_val_dataset = None
            else:
                val_ds = load_dataset("json", data_files=val_ds_path)["train"]
                formatted_val_dataset = val_ds.map(self.add_messages_key)

            val_len = len(formatted_val_dataset) if formatted_val_dataset is not None else 0
            print(
                f"Loaded dataset using standard approach (train: {len(formatted_train_dataset)}, val: {val_len})"
            )

            # Warn if tools are present in the dataset
            if self.tool_key and any(
                self.tool_key in sample for sample in formatted_train_dataset
            ):
                warnings.warn(
                    "Tools detected in dataset. Set use_preserving_dataset=True to preserve heterogeneous tool schemas. "
                    "Current mode may add None values for missing tool arguments, making samples invalid.",
                    UserWarning,
                    stacklevel=2,
                )

        else:
            # Use custom loading for heterogeneous schemas
            # Issue: When tool calls have varying argument structures across samples,
            # HuggingFace's Dataset.from_list enforces uniform schema by adding None
            # values for missing keys. Example:
            #   Sample 1: {"tools": [{"name": "search", "args": {"query": "x"}}]}
            #   Sample 2: {"tools": [{"name": "calc", "args": {"expr": "y", "precision": 2}}]}
            # Standard loading would add "precision: None" to Sample 1 and "query: None" to Sample 2.
            # PreservingDataset maintains exact structure without None-filling.

            print(
                "Using PreservingDataset to preserve heterogeneous tool argument schemas without None-filling."
            )

            train_data: list = []
            for train_path, weight in weighted_train_paths:
                records = load_jsonl_files(train_path)
                if weight < 1.0:
                    n = math.floor(len(records) * weight)
                    records = random.Random(subsample_seed).sample(records, n)
                print(f"  - {Path(train_path).stem} ({train_path}): {len(records)} samples (weight={weight})")
                train_data.extend(records)

            formatted_train_data = [self.add_messages_key(item) for item in train_data]
            if val_ds_path is None:
                formatted_val_data = None
            else:
                val_data = load_jsonl_files(val_ds_path)
                formatted_val_data = [self.add_messages_key(item) for item in val_data]

            formatted_train_dataset = PreservingDataset(formatted_train_data)
            formatted_val_dataset = (
                PreservingDataset(formatted_val_data) if formatted_val_data is not None else None
            )

            val_len = len(formatted_val_dataset) if formatted_val_dataset is not None else 0
            print(
                f"Loaded dataset using PreservingDataset "
                f"(train: {len(formatted_train_dataset)}, val: {val_len})"
            )

        self.formatted_ds = {
            "train": formatted_train_dataset,
            "validation": formatted_val_dataset,
        }

        self.task_spec = TaskDataSpec(
            "json_dataset",
        )

    def add_messages_key(
        self,
        example: dict[str, Any],
    ) -> dict[str, list[dict[str, Any]]]:
        messages = [message for message in example[self.chat_key]]
        if self.system_key is not None and self.system_key in example:
            messages = [
                {"role": "system", "content": example[self.system_key]}
            ] + messages
        elif self.system_prompt:
            messages = [{"role": "system", "content": self.system_prompt}] + messages
        assert messages[-1]["role"] == "assistant"

        # Preserve tools if they exist in the data
        result = {"messages": messages}
        if self.tool_key and self.tool_key in example:
            result["tools"] = example[self.tool_key]

        return result


class OpenAIFormatDatasetMultiFiles:
    """This class is used to load an SFT dataset in the OpenAI format.

    The dataset should be in the following format:
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."}
        ]
    }

    Args:
        train_ds_path_folder: Path to train folder containing training dataset JSON files
        val_ds_path_folder: Path to train folder containing validation dataset JSON files
        chat_key: Key for the messages list in the dataset (default: "messages")
        system_key: Optional key for system prompt in the dataset
        system_prompt: Optional system prompt to add if not in the dataset
        tool_key: Key for tools in the dataset (default: "tools")
        use_preserving_dataset: If True, uses PreservingDataset to maintain
            heterogeneous schemas (e.g., for tool calls with varying argument
            structures). If False, uses standard HuggingFace dataset loading.
            Default is False for backward compatibility.

    Notes:
        - system_key and system_prompt are optional. If provided, it will be added
          to the beginning of the dataset.
        - chat_key should be the key of the messages list. Multi-turn conversations
          are supported.
        - The last message in the conversation must be from the assistant.
        - When use_preserving_dataset=True, the dataset preserves the exact structure
          of each sample without None-filling for missing keys, which is useful for
          heterogeneous tool argument schemas.
    """

    def __init__(
        self,
        train_ds_path_folder: str,
        val_ds_path_folder: list,
        chat_key: str = "messages",
        system_key: str | None = None,
        system_prompt: str | None = None,
        tool_key: str | None = "tools",
        use_preserving_dataset: bool = False,
    ):
        warnings.warn(
            "OpenAIFormatDatasetMultiFiles is deprecated. "
            "Use OpenAIFormatDataset with a recursive glob pattern instead "
            "(e.g. train_data_path='my_folder/**/*.jsonl').",
            DeprecationWarning,
            stacklevel=2,
        )

        self.chat_key = chat_key
        self.system_key = system_key
        self.system_prompt = system_prompt
        self.tool_key = tool_key
        train_ds_path_folder = Path(train_ds_path_folder)
        val_ds_path_folder = Path(val_ds_path_folder)


        if not use_preserving_dataset:
            # Use the standard HuggingFace approach (faster and more standard)            
            train_original_dataset = concatenate_datasets([load_dataset("json",data_files={"train":str(file)})["train"] 
                                         for file in train_ds_path_folder.rglob("*.jsonl")])

            val_original_dataset = concatenate_datasets([load_dataset("json",data_files={"train":str(file)})["train"] 
                                         for file in val_ds_path_folder.rglob("*.jsonl")])

            formatted_train_dataset = train_original_dataset.map(self.add_messages_key)
            formatted_val_dataset = val_original_dataset.map(self.add_messages_key)

            print(
                f"Loaded dataset using standard approach (train: {len(formatted_train_dataset)}, val: {len(formatted_val_dataset)})"
            )

            # Warn if tools are present in the dataset
            if self.tool_key and any(
                self.tool_key in sample for sample in formatted_train_dataset
            ):
                warnings.warn(
                    "Tools detected in dataset. Set use_preserving_dataset=True to preserve heterogeneous tool schemas. "
                    "Current mode may add None values for missing tool arguments, making samples invalid.",
                    UserWarning,
                    stacklevel=2,
                )

        else:
            raise NotImplementedError

        self.formatted_ds = {
            "train": formatted_train_dataset,
            "validation": formatted_val_dataset,
        }

        self.task_spec = TaskDataSpec(
            "json_dataset",
        )

    def add_messages_key(
        self,
        example: dict[str, Any],
    ) -> dict[str, list[dict[str, Any]]]:
        messages = [message for message in example[self.chat_key]]
        if self.system_key is not None and self.system_key in example:
            messages = [
                {"role": "system", "content": example[self.system_key]}
            ] + messages
        elif self.system_prompt:
            messages = [{"role": "system", "content": self.system_prompt}] + messages
        assert messages[-1]["role"] == "assistant", messages

        # Preserve tools if they exist in the data
        result = {"messages": messages}
        if self.tool_key and self.tool_key in example:
            result["tools"] = example[self.tool_key]

        return result
