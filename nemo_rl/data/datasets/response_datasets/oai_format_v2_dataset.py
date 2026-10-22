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

import glob
import gzip
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson
from datasets import Dataset, concatenate_datasets

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.datasets.utils import load_dataset_from_path

# Arguments of `OpenAIFormatDataset` (or of the Luciole v0.1 `OpenAIFormatDatasetV2`)
# that this class does not support. They would otherwise be swallowed by **kwargs.
_UNSUPPORTED_ARGS = (
    "system_key",
    "system_prompt",
    "use_preserving_dataset",
    "train_data_path",
    "val_data_path",
)


@dataclass
class DatasetSpec:
    """One entry of ``data_path``.

    Attributes:
        path: file or glob pattern ending in .json/.jsonl (optionally
            compressed) or .parquet -- local or remote, e.g.
            "hf://datasets/OpenLLM-France/ReAct/data/**/*.parquet" -- or the
            name of a dataset on the HuggingFace Hub.
        weight: ``0 < weight <= 1.0``; randomly subsamples the file to
            ``floor(len * weight)`` records.
        chat_template_kwargs: passed to ``apply_chat_template`` for every sample
            of the file, e.g. ``{"enable_thinking": False}``. Tools are given to
            the template only if it defines ``tools``: ``tools: true`` passes
            the ``tool_key`` column of each sample, a list is the tool set of
            every sample of the file.
    """

    path: str
    weight: float = 1.0
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not 0 < self.weight <= 1.0:
            raise ValueError(
                f"weight for {self.path!r} must satisfy 0 < weight <= 1.0; "
                f"got {self.weight} (oversampling is not supported)."
            )

    @classmethod
    def parse(cls, entry: "str | dict | DatasetSpec") -> "DatasetSpec":
        if isinstance(entry, DatasetSpec):
            return entry
        if isinstance(entry, str):
            return cls(path=entry)
        if isinstance(entry, dict):  # also covers omegaconf DictConfig after to_container
            # an unknown key (e.g. a typo) raises TypeError here
            return cls(**entry)
        raise ValueError(
            f"A dataset entry must be a path or a dict with 'path' and optional "
            f"'weight' / 'chat_template_kwargs'; got {entry!r}."
        )


def parse_specs(paths: "str | dict | list | None") -> list[DatasetSpec]:
    if paths is None:
        return []
    if isinstance(paths, (str, dict, DatasetSpec)):
        paths = [paths]
    return [DatasetSpec.parse(p) for p in paths]


def load_jsonl_files(paths) -> list:
    """Load records from one or more file paths or glob patterns.

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
            # orjson.loads accepts bytes directly -- skip text decoding for speed.
            with open_fn(filepath, "rb") as f:
                for line in f:
                    if line.strip():
                        records.append(orjson.loads(line))
    return records


def _load_columns(path: str, columns: list[str]) -> Dataset:
    """Load `path` as a HuggingFace Dataset holding only `columns`.

    json files are read with orjson and pruned to `columns` before Arrow sees
    them: the json builder infers one schema per 10MB block and casts every
    block onto the first one, which blows up on the `tools` column of the tool
    calling datasets (each tool's `parameters` is a different struct, and the
    same key can hold different types across samples).

    A column that only some json records carry is kept (filled with None),
    whereas `Dataset.from_list` would take the schema from the first record
    and drop e.g. a `chat_template_kwargs` missing there for the whole file.

    parquet files and HuggingFace Hub datasets keep their columns pruned after
    loading.
    """
    if ".json" not in str(path):
        ds = load_dataset_from_path(path)
        return ds.remove_columns([c for c in ds.column_names if c not in columns])
    records = [
        {key: record.get(key) for key in columns} for record in load_jsonl_files(path)
    ]
    # a column absent from every record would be an all-null column
    present = [c for c in columns if any(r[c] is not None for r in records)]
    return Dataset.from_list([{c: r[c] for c in present} for r in records])


def _sample_tools(value: Any) -> list | None:
    """Per-sample tools column: a list or json string; None when empty."""
    if isinstance(value, str):
        value = json.loads(value) if value else None
    return list(value) if value else None


def _sample_template_kwargs(value: Any) -> dict[str, Any]:
    """Per-sample `chat_template_kwargs` column: a dict or json string.

    Arrow fills the keys missing from a sample with None; those are dropped.
    """
    if not value:
        return {}
    if isinstance(value, str):
        value = json.loads(value)
    return {k: v for k, v in value.items() if v is not None}


class OpenAIFormatDatasetV2(RawDataset):
    """SFT dataset in the OpenAI format, with per-file options.

    Each sample looks like:
    {
        "messages": [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."}
        ],
        "tools": [...],                                   # optional
        "chat_template_kwargs": {"enable_thinking": true}  # optional
    }

    Tools are passed to ``apply_chat_template`` as the ``tools`` key of
    ``chat_template_kwargs``, only for the files whose ``chat_template_kwargs``
    define ``tools`` (see `DatasetSpec`).

    Args:
        data_path: a path, a dict, or a list of paths / dicts. A dict has the
            keys of `DatasetSpec`: ``path``, ``weight`` (default 1.0) and
            ``chat_template_kwargs`` (default {}), e.g.
            ``{path: data/*.parquet, weight: 0.5,
            chat_template_kwargs: {enable_thinking: false}}``.
        chat_key: Key for the messages list in the dataset (default: "messages")
        tool_key: Key for tools in the dataset (default: "tools"); None drops them.
        task_name: Name used to bind the task spec and processor. Give two
            entries of `data.train` different names if they need different
            `prompt_file` / `system_prompt_file` / `processor`.
        split_validation_size: Size of the validation data taken from this
            dataset, default is 0 (use `data.validation` instead).
        seed: Seed for the train/validation split, and for the subsampling
            of files with weight < 1.0 unless `subsample_seed` is given.
        subsample_seed: Seed for the subsampling of files with weight < 1.0.

    The kwargs given to ``apply_chat_template`` are, by increasing priority:
    the global ``policy.tokenizer.chat_template_kwargs``, the
    ``chat_template_kwargs`` of the file (its ``tools`` resolved as above), then the per-sample
    ``chat_template_kwargs`` column (as in smoltalk2), merged key by key.
    """

    def __init__(
        self,
        data_path: "str | dict | list",
        chat_key: str = "messages",
        tool_key: str | None = "tools",
        task_name: str = "openai_format_v2",
        split_validation_size: float = 0,
        seed: int = 42,
        subsample_seed: int | None = None,
        **kwargs,
    ):
        unsupported = [k for k in _UNSUPPORTED_ARGS if k in kwargs]
        if unsupported:
            raise TypeError(
                f"openai_format_v2 does not support {unsupported}: put the files "
                "in `data_path`, the validation files in `data.validation`, and "
                "the system prompt in the data."
            )

        self.chat_key = chat_key
        self.tool_key = tool_key
        self.task_name = task_name
        self.subsample_seed = seed if subsample_seed is None else subsample_seed

        self.cols_to_keep = [chat_key, "chat_template_kwargs"]
        if tool_key is not None:
            self.cols_to_keep.append(tool_key)

        specs = parse_specs(data_path)
        if not specs:
            raise ValueError("openai_format_v2 needs at least one `data_path`.")
        self.dataset = self._load(specs)
        print(f"Loaded {self.task_name}: {len(self.dataset)} samples.")

        # `self.val_dataset` is used (not None) only when current dataset is used for both training and validation
        self.val_dataset = None
        self.split_train_validation(split_validation_size, seed)

    def _load(self, specs: list[DatasetSpec]) -> Dataset:
        datasets = []
        for spec in specs:
            print(f"  - loading {spec.path} (weight={spec.weight})", flush=True)
            if spec.chat_template_kwargs.get("tools") is True and self.tool_key is None:
                raise ValueError(
                    f"{spec.path!r} sets tools: true but tool_key is None."
                )
            ds = _load_columns(spec.path, self.cols_to_keep)
            if len(ds) == 0:
                # e.g. a glob that matches no file only warns in the loader
                raise ValueError(f"No sample found for {spec.path!r}.")
            if spec.weight < 1.0:
                n = math.floor(len(ds) * spec.weight)
                ds = ds.shuffle(seed=self.subsample_seed).select(range(n))
            if len(ds) == 0:
                raise ValueError(
                    f"weight={spec.weight} leaves no sample of {spec.path!r}."
                )
            print(f"    {Path(spec.path).stem}: {len(ds)} samples")
            datasets.append(
                ds.map(
                    self.format_data,
                    fn_kwargs={"file_kwargs": spec.chat_template_kwargs},
                    remove_columns=ds.column_names,
                )
            )
        try:
            return concatenate_datasets(datasets)
        except Exception:
            # the datasets loaded fine on their own but disagree on the schema
            print("Cannot concatenate the datasets, schemas are:")
            for spec, ds in zip(specs, datasets):
                print(f"  - {spec.path}\n      {ds.features}")
            raise

    def format_data(
        self, data: dict[str, Any], file_kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Map one raw sample to ``{messages, chat_template_kwargs, task_name}``.

        ``chat_template_kwargs`` is the file-level ``file_kwargs`` updated by the
        sample's own ``chat_template_kwargs`` column, serialized as a json string.
        A file-level ``tools: true`` is replaced by the sample's ``tool_key``
        column; without ``tools`` in ``file_kwargs``, no tools are passed.
        """
        messages = list(data[self.chat_key])
        kwargs = dict(file_kwargs)
        tools = kwargs.pop("tools", None)
        if tools is True:
            tools = _sample_tools(data.get(self.tool_key))
        if tools:
            kwargs["tools"] = tools
        kwargs.update(_sample_template_kwargs(data.get("chat_template_kwargs")))
        # Stored as a json string: samples with different keys would otherwise
        # be None-filled by the arrow schema.
        return {
            "messages": messages,
            "chat_template_kwargs": json.dumps(kwargs),
            "task_name": self.task_name,
        }
