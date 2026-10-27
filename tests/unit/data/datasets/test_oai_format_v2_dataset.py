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

import gzip
import json
from functools import partial

import pytest
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from nemo_rl.data.datasets.response_datasets import (
    OpenAIFormatDatasetV2,
    load_response_dataset,
)
from nemo_rl.data.datasets.response_datasets.oai_format_v2_dataset import (
    DatasetSpec,
    parse_specs,
)
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.llm_message_utils import get_formatted_message_log


def _sample(i, **extra):
    return {
        "messages": [
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": f"a{i}"},
        ],
        **extra,
    }


def _write_jsonl(path, records):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return str(path)


def _kwargs(ds):
    return [json.loads(x) for x in ds["chat_template_kwargs"]]


# --------------------------------------------------------------------------
# DatasetSpec / parse_specs
# --------------------------------------------------------------------------
class TestDatasetSpec:
    def test_string(self):
        spec = DatasetSpec.parse("a.jsonl")
        assert (spec.path, spec.weight, spec.chat_template_kwargs) == (
            "a.jsonl",
            1.0,
            {},
        )

    def test_full_dict(self):
        spec = DatasetSpec.parse(
            {
                "path": "a",
                "weight": 0.5,
                "chat_template_kwargs": {"enable_thinking": 0},
                "use_tools": True,
            }
        )
        assert spec == DatasetSpec("a", 0.5, {"enable_thinking": 0}, True)

    def test_dict_defaults(self):
        assert DatasetSpec.parse({"path": "a"}) == DatasetSpec("a")

    def test_passthrough(self):
        spec = DatasetSpec("a")
        assert DatasetSpec.parse(spec) is spec

    def test_unknown_key_rejected(self):
        with pytest.raises(TypeError, match="wieght"):
            DatasetSpec.parse({"path": "a", "wieght": 0.5})

    def test_missing_path_rejected(self):
        with pytest.raises(TypeError):
            DatasetSpec.parse({"weight": 0.5})

    @pytest.mark.parametrize("bad", [0, -0.1, 1.01, 2, float("nan")])
    def test_bad_weight(self, bad):
        with pytest.raises(ValueError, match="weight"):
            DatasetSpec("a", bad)

    @pytest.mark.parametrize("ok", [1e-6, 0.5, 1, 1.0])
    def test_good_weight(self, ok):
        assert DatasetSpec("a", ok).weight == ok

    @pytest.mark.parametrize("bad", [3, 1.5, ("a", 0.5), ["a", 0.5], None])
    def test_bad_type(self, bad):
        with pytest.raises(ValueError, match="dataset entry"):
            DatasetSpec.parse(bad)

    def test_default_kwargs_not_shared(self):
        a, b = DatasetSpec("a"), DatasetSpec("b")
        a.chat_template_kwargs["x"] = 1
        assert b.chat_template_kwargs == {}


class TestParseSpecs:
    def test_none(self):
        assert parse_specs(None) == []

    def test_empty_list(self):
        assert parse_specs([]) == []

    def test_single_str_and_dict(self):
        assert parse_specs("a") == [DatasetSpec("a")]
        assert parse_specs({"path": "a", "weight": 0.2}) == [DatasetSpec("a", 0.2)]

    def test_mixed_list(self):
        specs = parse_specs(["a", {"path": "b", "weight": 0.5}, DatasetSpec("c")])
        assert [s.path for s in specs] == ["a", "b", "c"]
        assert [s.weight for s in specs] == [1.0, 0.5, 1.0]

    @pytest.mark.parametrize(
        "entries",
        [
            ["a", "a"],
            ["a", {"path": "a", "weight": 0.5}],
            [{"path": "a", "weight": 0.5}, {"path": "b"}, DatasetSpec("a", 0.5)],
        ],
    )
    def test_duplicate_path_rejected(self, entries):
        with pytest.raises(ValueError, match="more than once"):
            parse_specs(entries)

    def test_duplicate_path_rejected_in_dataset(self, tmp_path):
        p = _write_jsonl(tmp_path / "a.jsonl", [_sample(0)])
        with pytest.raises(ValueError, match="more than once"):
            OpenAIFormatDatasetV2([p, p])
        with pytest.raises(ValueError, match="more than once"):
            OpenAIFormatDatasetV2(p, [p, {"path": p, "weight": 0.5}])


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
class TestLoading:
    def test_plain_string_path(self, tmp_path):
        p = _write_jsonl(tmp_path / "a.jsonl", [_sample(i) for i in range(3)])
        ds = OpenAIFormatDatasetV2(p).formatted_ds
        assert len(ds["train"]) == 3
        assert ds["validation"] is None
        assert ds["train"].column_names == ["messages", "chat_template_kwargs"]
        assert ds["train"][1]["messages"][0]["content"] == "q1"
        assert _kwargs(ds["train"]) == [{}, {}, {}]

    def test_task_spec(self, tmp_path):
        p = _write_jsonl(tmp_path / "a.jsonl", [_sample(0)])
        assert isinstance(OpenAIFormatDatasetV2(p).task_spec, TaskDataSpec)

    def test_list_concatenated_in_order(self, tmp_path):
        a = _write_jsonl(tmp_path / "a.jsonl", [_sample(i) for i in range(2)])
        b = _write_jsonl(tmp_path / "b.jsonl", [_sample(10 + i) for i in range(3)])
        ds = OpenAIFormatDatasetV2([a, b]).formatted_ds["train"]
        got = [m["messages"][0]["content"] for m in ds]
        assert got == ["q0", "q1", "q10", "q11", "q12"]

    def test_glob_and_gzip(self, tmp_path):
        _write_jsonl(tmp_path / "x1.jsonl.gz", [_sample(0), _sample(1)])
        _write_jsonl(tmp_path / "x2.jsonl", [_sample(2)])
        ds = OpenAIFormatDatasetV2(str(tmp_path / "x*.json*")).formatted_ds["train"]
        assert len(ds) == 3

    def test_val_none_and_val_specs(self, tmp_path):
        t = _write_jsonl(tmp_path / "t.jsonl", [_sample(0)])
        v = _write_jsonl(tmp_path / "v.jsonl", [_sample(i) for i in range(10)])
        assert OpenAIFormatDatasetV2(t, None).formatted_ds["validation"] is None
        ds = OpenAIFormatDatasetV2(
            t, {"path": v, "weight": 0.5, "chat_template_kwargs": {"k": 1}}
        ).formatted_ds
        assert len(ds["validation"]) == 5
        assert _kwargs(ds["validation"]) == [{"k": 1}] * 5

    def test_custom_chat_key(self, tmp_path):
        rec = {"conv": _sample(0)["messages"]}
        p = _write_jsonl(tmp_path / "a.jsonl", [rec])
        ds = OpenAIFormatDatasetV2(p, chat_key="conv").formatted_ds["train"]
        assert ds[0]["messages"][1]["content"] == "a0"

    def test_missing_file_gives_no_samples_or_error(self, tmp_path):
        with pytest.raises(Exception):
            OpenAIFormatDatasetV2(str(tmp_path / "nope*.jsonl"))

    def test_system_message_in_data_untouched(self, tmp_path):
        rec = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ]
        }
        p = _write_jsonl(tmp_path / "a.jsonl", [rec])
        msgs = OpenAIFormatDatasetV2(p).formatted_ds["train"][0]["messages"]
        assert [m["role"] for m in msgs] == ["system", "user", "assistant"]

    def test_unused_columns_dropped(self, tmp_path):
        p = _write_jsonl(tmp_path / "a.jsonl", [_sample(0, source="x", score=3)])
        ds = OpenAIFormatDatasetV2(p).formatted_ds["train"]
        assert ds.column_names == ["messages", "chat_template_kwargs"]

    def test_legacy_args_not_accepted(self, tmp_path):
        p = _write_jsonl(tmp_path / "a.jsonl", [_sample(0)])
        for kw in ({"system_key": "s"}, {"system_prompt": "s"}, {"use_preserving_dataset": False}):
            with pytest.raises(TypeError):
                OpenAIFormatDatasetV2(p, **kw)

    def test_legacy_pair_not_accepted(self, tmp_path):
        p = _write_jsonl(tmp_path / "a.jsonl", [_sample(0)])
        with pytest.raises(ValueError):
            OpenAIFormatDatasetV2([(p, 0.5)])


# --------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------
class TestWeights:
    @pytest.fixture
    def path(self, tmp_path):
        return _write_jsonl(tmp_path / "a.jsonl", [_sample(i) for i in range(100)])

    @pytest.mark.parametrize("w,n", [(1.0, 100), (0.5, 50), (0.333, 33), (0.01, 1), (0.001, 0)])
    def test_floor(self, path, w, n):
        if n == 0:
            # an empty selection must not silently produce a dataset
            with pytest.raises(Exception):
                OpenAIFormatDatasetV2({"path": path, "weight": w})
        else:
            ds = OpenAIFormatDatasetV2({"path": path, "weight": w}).formatted_ds
            assert len(ds["train"]) == n

    def test_no_duplicates(self, path):
        ds = OpenAIFormatDatasetV2({"path": path, "weight": 0.5}).formatted_ds["train"]
        got = [m["messages"][0]["content"] for m in ds]
        assert len(set(got)) == len(got)

    def test_seed_reproducible_and_different(self, path):
        spec = {"path": path, "weight": 0.5}
        get = lambda seed: [
            m["messages"][0]["content"]
            for m in OpenAIFormatDatasetV2(spec, subsample_seed=seed).formatted_ds["train"]
        ]
        assert get(1) == get(1)
        assert get(1) != get(2)

    def test_weight_applies_per_file(self, tmp_path):
        a = _write_jsonl(tmp_path / "a.jsonl", [_sample(i) for i in range(10)])
        b = _write_jsonl(tmp_path / "b.jsonl", [_sample(i) for i in range(10)])
        ds = OpenAIFormatDatasetV2(
            [{"path": a, "weight": 0.5}, {"path": b}]
        ).formatted_ds["train"]
        assert len(ds) == 15


# --------------------------------------------------------------------------
# chat_template_kwargs merging
# --------------------------------------------------------------------------
class TestChatTemplateKwargs:
    def test_file_level(self, tmp_path):
        p = _write_jsonl(tmp_path / "a.jsonl", [_sample(0), _sample(1)])
        ds = OpenAIFormatDatasetV2(
            {"path": p, "chat_template_kwargs": {"enable_thinking": False, "a": [1]}}
        ).formatted_ds["train"]
        assert _kwargs(ds) == [{"enable_thinking": False, "a": [1]}] * 2

    def test_per_file_independent(self, tmp_path):
        a = _write_jsonl(tmp_path / "a.jsonl", [_sample(0)])
        b = _write_jsonl(tmp_path / "b.jsonl", [_sample(1)])
        ds = OpenAIFormatDatasetV2(
            [
                {"path": a, "chat_template_kwargs": {"enable_thinking": True}},
                {"path": b, "chat_template_kwargs": {"enable_thinking": False}},
            ]
        ).formatted_ds["train"]
        assert _kwargs(ds) == [{"enable_thinking": True}, {"enable_thinking": False}]

    def test_sample_overrides_file_key_by_key(self, tmp_path):
        p = _write_jsonl(
            tmp_path / "a.jsonl",
            [
                _sample(0, chat_template_kwargs={"enable_thinking": True}),
                _sample(1, chat_template_kwargs={"other": 1}),
            ],
        )
        ds = OpenAIFormatDatasetV2(
            {"path": p, "chat_template_kwargs": {"enable_thinking": False, "a": 1}}
        ).formatted_ds["train"]
        k = _kwargs(ds)
        assert k[0] == {"enable_thinking": True, "a": 1}
        assert k[1] == {"enable_thinking": False, "a": 1, "other": 1}
        # the file-level dict is not mutated by the merge
        assert len({json.dumps(x, sort_keys=True) for x in k}) == 2

    def test_sample_null_value_does_not_override(self, tmp_path):
        p = _write_jsonl(
            tmp_path / "a.jsonl",
            [_sample(0, chat_template_kwargs={"enable_thinking": None, "x": 2})],
        )
        ds = OpenAIFormatDatasetV2(
            {"path": p, "chat_template_kwargs": {"enable_thinking": False}}
        ).formatted_ds["train"]
        assert _kwargs(ds) == [{"enable_thinking": False, "x": 2}]

    def test_sample_json_string(self, tmp_path):
        p = _write_jsonl(
            tmp_path / "a.jsonl",
            [_sample(0, chat_template_kwargs=json.dumps({"enable_thinking": True}))],
        )
        ds = OpenAIFormatDatasetV2(p).formatted_ds["train"]
        assert _kwargs(ds) == [{"enable_thinking": True}]

    @pytest.mark.parametrize("empty", [{}, None, ""])
    def test_empty_sample_values(self, tmp_path, empty):
        p = _write_jsonl(tmp_path / "a.jsonl", [_sample(0, chat_template_kwargs=empty)])
        assert _kwargs(OpenAIFormatDatasetV2(p).formatted_ds["train"]) == [{}]

    def test_heterogeneous_keys_not_none_filled(self, tmp_path):
        p = _write_jsonl(
            tmp_path / "a.jsonl",
            [
                _sample(0, chat_template_kwargs={"a": 1}),
                _sample(1, chat_template_kwargs={"b": "x"}),
            ],
        )
        assert _kwargs(OpenAIFormatDatasetV2(p).formatted_ds["train"]) == [
            {"a": 1},
            {"b": "x"},
        ]

    def test_stored_as_string_column(self, tmp_path):
        p = _write_jsonl(tmp_path / "a.jsonl", [_sample(0)])
        ds = OpenAIFormatDatasetV2(p).formatted_ds["train"]
        assert isinstance(ds[0]["chat_template_kwargs"], str)

    def test_mixed_files_with_and_without_column(self, tmp_path):
        a = _write_jsonl(tmp_path / "a.jsonl", [_sample(0, chat_template_kwargs={"k": 1})])
        b = _write_jsonl(tmp_path / "b.jsonl", [_sample(1)])
        ds = OpenAIFormatDatasetV2([a, b]).formatted_ds["train"]
        assert _kwargs(ds) == [{"k": 1}, {}]
        ds = OpenAIFormatDatasetV2([b, a]).formatted_ds["train"]
        assert _kwargs(ds) == [{}, {"k": 1}]

    def test_parquet_smoltalk2_like_struct(self, tmp_path):
        """smoltalk2 stores a struct column with custom_instructions / enable_thinking /
        python_tools / xml_tools; absent entries come back as None."""
        rows = [
            _sample(
                0,
                chat_template_kwargs={
                    "custom_instructions": "be nice",
                    "enable_thinking": False,
                    "python_tools": None,
                    "xml_tools": None,
                },
            ),
            _sample(
                1,
                chat_template_kwargs={
                    "custom_instructions": None,
                    "enable_thinking": True,
                    "python_tools": ["def f(): pass"],
                    "xml_tools": None,
                },
            ),
        ]
        path = tmp_path / "s.parquet"
        Dataset.from_list(rows).to_parquet(str(path))
        ds = OpenAIFormatDatasetV2(
            {"path": str(path), "chat_template_kwargs": {"enable_thinking": True, "z": 0}}
        ).formatted_ds["train"]
        assert _kwargs(ds) == [
            {"enable_thinking": False, "z": 0, "custom_instructions": "be nice"},
            {"enable_thinking": True, "z": 0, "python_tools": ["def f(): pass"]},
        ]

    def test_parquet_and_jsonl_together(self, tmp_path):
        pq = tmp_path / "s.parquet"
        Dataset.from_list([_sample(0, chat_template_kwargs={"enable_thinking": True})]).to_parquet(str(pq))
        js = _write_jsonl(tmp_path / "a.jsonl", [_sample(1)])
        ds = OpenAIFormatDatasetV2([str(pq), js]).formatted_ds["train"]
        assert _kwargs(ds) == [{"enable_thinking": True}, {}]


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------
class TestTools:
    TOOL = {"type": "function", "function": {"name": "f", "parameters": {"x": 1}}}
    TOOL2 = {"type": "function", "function": {"name": "g", "parameters": {"y": 2}}}
    ON = {"use_tools": True}

    @pytest.mark.parametrize("as_str", [False, True])
    def test_tools_true_uses_column(self, tmp_path, as_str):
        tools = json.dumps([self.TOOL]) if as_str else [self.TOOL]
        p = _write_jsonl(tmp_path / "a.jsonl", [_sample(0, tools=tools)])
        ds = OpenAIFormatDatasetV2({"path": p, **self.ON}).formatted_ds["train"]
        assert ds.column_names == ["messages", "chat_template_kwargs"]
        assert _kwargs(ds) == [{"tools": [self.TOOL]}]

    @pytest.mark.parametrize("extra", [{}, {"use_tools": False}])
    def test_use_tools_off_drops_column(self, tmp_path, extra):
        p = _write_jsonl(tmp_path / "a.jsonl", [_sample(0, tools=[self.TOOL])])
        ds = OpenAIFormatDatasetV2({"path": p, **extra}).formatted_ds["train"]
        assert _kwargs(ds) == [{}]

    @pytest.mark.parametrize("empty", [None, [], ""])
    def test_empty_tools_not_passed(self, tmp_path, empty):
        p = _write_jsonl(tmp_path / "a.jsonl", [_sample(0, tools=empty)])
        ds = OpenAIFormatDatasetV2({"path": p, **self.ON}).formatted_ds["train"]
        assert _kwargs(ds) == [{}]

    def test_tools_not_none_filled(self, tmp_path):
        # as a struct column, f would get g's parameter y = None and vice versa
        p = _write_jsonl(
            tmp_path / "a.jsonl",
            [_sample(0, tools=[self.TOOL]), _sample(1, tools=[self.TOOL2])],
        )
        ds = OpenAIFormatDatasetV2({"path": p, **self.ON}).formatted_ds["train"]
        assert _kwargs(ds) == [{"tools": [self.TOOL]}, {"tools": [self.TOOL2]}]

    def test_file_tools_list(self, tmp_path):
        p = _write_jsonl(tmp_path / "a.jsonl", [_sample(0), _sample(1, tools=[self.TOOL2])])
        ds = OpenAIFormatDatasetV2(
            {"path": p, "chat_template_kwargs": {"tools": [self.TOOL]}}
        ).formatted_ds["train"]
        assert _kwargs(ds) == [{"tools": [self.TOOL]}] * 2

    def test_sample_kwargs_override_tools(self, tmp_path):
        p = _write_jsonl(
            tmp_path / "a.jsonl",
            [_sample(0, tools=[self.TOOL], chat_template_kwargs={"tools": [self.TOOL2]})],
        )
        ds = OpenAIFormatDatasetV2({"path": p, **self.ON}).formatted_ds["train"]
        assert _kwargs(ds) == [{"tools": [self.TOOL2]}]

    def test_custom_tool_key(self, tmp_path):
        p = _write_jsonl(tmp_path / "a.jsonl", [_sample(0, fns=json.dumps([self.TOOL]))])
        ds = OpenAIFormatDatasetV2({"path": p, **self.ON}, tool_key="fns").formatted_ds["train"]
        assert _kwargs(ds) == [{"tools": [self.TOOL]}]

    def test_tools_true_without_tool_key_rejected(self, tmp_path):
        p = _write_jsonl(tmp_path / "a.jsonl", [_sample(0)])
        with pytest.raises(ValueError):
            OpenAIFormatDatasetV2({"path": p, **self.ON}, tool_key=None)

    def test_tool_key_none_drops_tools(self, tmp_path):
        p = _write_jsonl(tmp_path / "a.jsonl", [_sample(0, tools=[self.TOOL])])
        ds = OpenAIFormatDatasetV2(p, tool_key=None).formatted_ds["train"]
        assert _kwargs(ds) == [{}]

    def test_per_file(self, tmp_path):
        a = _write_jsonl(tmp_path / "a.jsonl", [_sample(0, tools=[self.TOOL])])
        b = _write_jsonl(tmp_path / "b.jsonl", [_sample(1, tools=[self.TOOL2])])
        c = _write_jsonl(tmp_path / "c.jsonl", [_sample(2)])
        ds = OpenAIFormatDatasetV2(
            [{"path": a, **self.ON}, b, {"path": c, **self.ON}]
        ).formatted_ds["train"]
        assert _kwargs(ds) == [{"tools": [self.TOOL]}, {}, {}]

    def test_parquet_tools(self, tmp_path):
        path = tmp_path / "t.parquet"
        Dataset.from_list(
            [_sample(0, tools=json.dumps([self.TOOL])), _sample(1, tools=None)]
        ).to_parquet(str(path))
        ds = OpenAIFormatDatasetV2({"path": str(path), **self.ON}).formatted_ds["train"]
        assert _kwargs(ds) == [{"tools": [self.TOOL]}, {}]


# --------------------------------------------------------------------------
# Registration in load_response_dataset
# --------------------------------------------------------------------------
class TestRegistration:
    def test_load_response_dataset(self, tmp_path):
        p = _write_jsonl(tmp_path / "a.jsonl", [_sample(i) for i in range(4)])
        data = load_response_dataset(
            {
                "dataset_name": "openai_format_v2",
                "train_data_path": [
                    {"path": p, "weight": 0.5, "chat_template_kwargs": {"enable_thinking": False}}
                ],
                "val_data_path": p,
            },
            seed=0,
        )
        assert len(data.formatted_ds["train"]) == 2
        assert len(data.formatted_ds["validation"]) == 4

    def test_minimal_config(self, tmp_path):
        p = _write_jsonl(tmp_path / "a.jsonl", [_sample(0)])
        data = load_response_dataset(
            {"dataset_name": "openai_format_v2", "train_data_path": p}
        )
        assert data.formatted_ds["validation"] is None

    def test_seed_falls_back_to_global_seed(self, tmp_path):
        p = _write_jsonl(tmp_path / "a.jsonl", [_sample(i) for i in range(50)])
        cfg = {"dataset_name": "openai_format_v2", "train_data_path": {"path": p, "weight": 0.5}}
        get = lambda s: [m["messages"][0]["content"] for m in load_response_dataset(cfg, s).formatted_ds["train"]]
        assert get(1) == get(1)
        assert get(1) != get(2)

    def test_omegaconf_config(self, tmp_path):
        omegaconf = pytest.importorskip("omegaconf")
        p = _write_jsonl(tmp_path / "a.jsonl", [_sample(0), _sample(1)])
        cfg = omegaconf.OmegaConf.create(
            {
                "dataset_name": "openai_format_v2",
                "train_data_path": [
                    {"path": p, "chat_template_kwargs": {"enable_thinking": False}}
                ],
            }
        )
        cfg = omegaconf.OmegaConf.to_container(cfg)
        data = load_response_dataset(cfg)
        assert _kwargs(data.formatted_ds["train"]) == [{"enable_thinking": False}] * 2


# --------------------------------------------------------------------------
# End to end with the chat template (offline tokenizer, no download)
# --------------------------------------------------------------------------
TEMPLATE = (
    "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}<|end|>{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
    "{% if enable_thinking is defined %}[think={{ enable_thinking }}]{% endif %}"
    "{% if extra is defined %}[extra={{ extra }}]{% endif %}"
)


@pytest.fixture(scope="module")
def tokenizer():
    vocab = {"<unk>": 0, "<eos>": 1}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    t = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>"
    )
    t.chat_template = TEMPLATE
    return t


def _fmt(tokenizer, kwargs, **extra):
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
    log = get_formatted_message_log(
        msgs,
        tokenizer,
        TaskDataSpec("t"),
        add_bos_token=False,
        add_eos_token=False,
        chat_template_kwargs=kwargs,
        **extra,
    )
    return "".join(m["content"] for m in log)


class TestEndToEnd:
    def test_default_no_kwargs(self, tokenizer):
        assert "[think=" not in _fmt(tokenizer, None)

    @pytest.mark.parametrize("empty", [None, {}, "", "{}"])
    def test_empty_equivalent_to_none(self, tokenizer, empty):
        assert _fmt(tokenizer, empty) == _fmt(tokenizer, None)

    def test_dict_kwargs(self, tokenizer):
        assert "[think=False]" in _fmt(tokenizer, {"enable_thinking": False})
        assert "[think=True]" in _fmt(tokenizer, {"enable_thinking": True})

    def test_json_string_kwargs(self, tokenizer):
        assert _fmt(tokenizer, json.dumps({"enable_thinking": False})) == _fmt(
            tokenizer, {"enable_thinking": False}
        )

    def test_caller_dict_not_mutated(self, tokenizer):
        kw = {"enable_thinking": True}
        _fmt(tokenizer, kw)
        assert kw == {"enable_thinking": True}

    def test_overrides_global_tokenizer_kwargs(self, tokenizer):
        tok = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer._tokenizer, unk_token="<unk>", eos_token="<eos>"
        )
        tok.chat_template = TEMPLATE
        tok.apply_chat_template = partial(
            tok.apply_chat_template, enable_thinking=True, extra="g"
        )
        out_global = _fmt(tok, None)
        assert "[think=True]" in out_global and "[extra=g]" in out_global
        out = _fmt(tok, {"enable_thinking": False})
        assert "[think=False]" in out and "[extra=g]" in out

    def test_different_samples_different_output(self, tokenizer):
        assert _fmt(tokenizer, {"enable_thinking": True}) != _fmt(
            tokenizer, {"enable_thinking": False}
        )

    def test_dataset_to_formatted_log(self, tokenizer, tmp_path):
        p = _write_jsonl(
            tmp_path / "a.jsonl",
            [_sample(0), _sample(1, chat_template_kwargs={"enable_thinking": True})],
        )
        ds = OpenAIFormatDatasetV2(
            {"path": p, "chat_template_kwargs": {"enable_thinking": False}}
        ).formatted_ds["train"]
        outs = []
        for row in ds:
            log = get_formatted_message_log(
                row["messages"],
                tokenizer,
                TaskDataSpec("t"),
                add_bos_token=False,
                add_eos_token=False,
                chat_template_kwargs=row["chat_template_kwargs"],
            )
            outs.append("".join(m["content"] for m in log))
        assert "[think=False]" in outs[0]
        assert "[think=True]" in outs[1]
