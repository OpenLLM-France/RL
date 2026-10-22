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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets

from nemo_rl.data.datasets.response_datasets.oai_format_dataset import (
    load_dataset_columns,
    load_jsonl_files,
)
from nemo_rl.data.interfaces import TaskDataSpec


@dataclass
class DatasetSpec:
    """One entry of ``train_ds_path`` / ``val_ds_path``.

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
    specs = [DatasetSpec.parse(p) for p in paths]
    seen: set[str] = set()
    for spec in specs:
        # the same path twice would be subsampled with the same seed, i.e. to
        # the very same samples, which would then be seen twice
        if spec.path in seen:
            raise ValueError(
                f"Dataset {spec.path!r} appears more than once; "
                "list each dataset only once."
            )
        seen.add(spec.path)
    return specs


def _sample_template_kwargs(value: Any) -> dict[str, Any]:
    """Per-sample `chat_template_kwargs` column: a dict or json string.

    Arrow fills the keys missing from a sample with None; those are dropped.
    """
    if not value:
        return {}
    if isinstance(value, str):
        value = json.loads(value)
    return {k: v for k, v in value.items() if v is not None}


def _sample_tools(value: Any) -> list | None:
    """Per-sample tools column: a list or json string; None when empty."""
    if isinstance(value, str):
        value = json.loads(value) if value else None
    return list(value) if value else None


def _load_columns(path: str, columns: list[str], tool_key: str | None) -> Dataset:
    """`load_dataset_columns`, but a column that only some json records carry is kept.

    The shared loader takes the schema of `Dataset.from_list` from the first
    record, so e.g. a `chat_template_kwargs` missing there is dropped for the
    whole file. Missing values are filled with None here.

    The `tool_key` column is kept as a json string: as a struct column, arrow
    would give every tool the parameters of all the others, set to None.
    """
    if ".json" not in str(path):
        return load_dataset_columns(path, columns)
    records = [
        {key: record.get(key) for key in columns}
        for record in load_jsonl_files(path)
    ]
    if tool_key is not None:
        for r in records:
            if r.get(tool_key) is not None and not isinstance(r[tool_key], str):
                r[tool_key] = json.dumps(r[tool_key])
    # a column absent from every record would be an all-null column
    present = [c for c in columns if any(r[c] is not None for r in records)]
    return Dataset.from_list([{c: r[c] for c in present} for r in records])


class OpenAIFormatDatasetV2:
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
        train_ds_path: a path, a dict, or a list of paths / dicts. A dict has the
            keys of `DatasetSpec`: ``path``, ``weight`` (default 1.0) and
            ``chat_template_kwargs`` (default {}), e.g.
            ``{path: data/*.parquet, weight: 0.5,
            chat_template_kwargs: {enable_thinking: false}}``.
        val_ds_path: same forms as ``train_ds_path`` (weights apply too), or
            None to skip validation.
        chat_key / tool_key: as in `OpenAIFormatDataset`.
        subsample_seed: seed used when subsampling files with weight < 1.0.

    The kwargs given to ``apply_chat_template`` are, by increasing priority:
    the global ``tokenizer.chat_template_kwargs``, the ``chat_template_kwargs``
    of the file (its ``tools`` resolved as above), then the per-sample
    ``chat_template_kwargs`` column (as in smoltalk2), merged key by key.
    """

    def __init__(
        self,
        train_ds_path: "str | dict | list",
        val_ds_path: "str | dict | list | None" = None,
        chat_key: str = "messages",
        tool_key: str | None = "tools",
        subsample_seed: int = 42,
    ):
        self.chat_key = chat_key
        self.tool_key = tool_key
        self.subsample_seed = subsample_seed

        self.cols_to_keep = [chat_key, "chat_template_kwargs"]
        if tool_key is not None:
            self.cols_to_keep.append(tool_key)

        train = self._load(parse_specs(train_ds_path), "train")
        val_specs = parse_specs(val_ds_path)
        val = self._load(val_specs, "validation") if val_specs else None
        print(
            f"Loaded dataset (train: {len(train)}, "
            f"val: {len(val) if val is not None else 0})"
        )

        self.formatted_ds = {"train": train, "validation": val}
        self.task_spec = TaskDataSpec("json_dataset")

    def _load(self, specs: list[DatasetSpec], name: str):
        datasets = []
        for spec in specs:
            print(f"  - loading {spec.path} ({name}, weight={spec.weight})", flush=True)
            if spec.chat_template_kwargs.get("tools") is True and self.tool_key is None:
                raise ValueError(
                    f"{spec.path!r} sets tools: true but tool_key is None."
                )
            ds = _load_columns(spec.path, self.cols_to_keep, self.tool_key)
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
                    self.add_messages_key,
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

    def add_messages_key(
        self, example: dict[str, Any], file_kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Map one raw sample to ``{messages, chat_template_kwargs}``.

        ``chat_template_kwargs`` is the file-level ``file_kwargs`` updated by the
        sample's own ``chat_template_kwargs`` column, serialized as a json string.
        A file-level ``tools: true`` is replaced by the sample's ``tool_key``
        column; without ``tools`` in ``file_kwargs``, no tools are passed.
        """
        messages = list(example[self.chat_key])
        kwargs = dict(file_kwargs)
        tools = kwargs.pop("tools", None)
        if tools is True:
            tools = _sample_tools(example.get(self.tool_key))
        if tools:
            kwargs["tools"] = tools
        kwargs.update(_sample_template_kwargs(example.get("chat_template_kwargs")))
        # Stored as a json string: samples with different keys would otherwise
        # be None-filled by the arrow schema.
        return {"messages": messages, "chat_template_kwargs": json.dumps(kwargs)}
