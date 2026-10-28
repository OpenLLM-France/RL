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

import pytest
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from nemo_rl.data.datasets.preference_datasets import (
    OpenAIPreferenceDatasetV2,
    load_preference_dataset,
)
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.processors import preference_preprocessor
from nemo_rl.data.utils import setup_preference_data

SYSTEM = {"role": "system", "content": "sys"}
TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {}}}]


def _user(i):
    return {"role": "user", "content": f"q{i}"}


def _assistant(text):
    return {"role": "assistant", "content": text}


def _full_sample(i, **extra):
    """Luciole-like: chosen / rejected are full conversations."""
    return {
        "chosen": [SYSTEM, _user(i), _assistant(f"good{i}")],
        "rejected": [SYSTEM, _user(i), _assistant(f"bad{i}")],
        **extra,
    }


def _write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return str(path)


def _kwargs(ds):
    return [json.loads(x) for x in ds["chat_template_kwargs"]]


class TestFormat:
    def test_full_conversations_split_on_last_message(self, tmp_path):
        p = _write_jsonl(tmp_path / "a.jsonl", [_full_sample(0)])
        row = OpenAIPreferenceDatasetV2(p).dataset[0]
        assert row["context"] == [SYSTEM, _user(0)]
        assert row["completions"] == [
            {"rank": 0, "completion": [_assistant("good0")]},
            {"rank": 1, "completion": [_assistant("bad0")]},
        ]
        assert row["task_name"] == "openai_preference_v2"

    def test_multi_turn_context(self, tmp_path):
        rec = {
            "chosen": [_user(0), _assistant("x"), _user(1), _assistant("good")],
            "rejected": [_user(0), _assistant("x"), _user(1), _assistant("bad")],
        }
        p = _write_jsonl(tmp_path / "a.jsonl", [rec])
        row = OpenAIPreferenceDatasetV2(p).dataset[0]
        assert row["context"] == [_user(0), _assistant("x"), _user(1)]
        assert row["completions"][0]["completion"] == [_assistant("good")]

    @pytest.mark.parametrize(
        "chosen,rejected",
        [
            # an earlier assistant turn differs: it would not be trained
            (
                [_user(0), _assistant("x"), _user(1), _assistant("good")],
                [_user(0), _assistant("y"), _user(1), _assistant("bad")],
            ),
            # different lengths
            (
                [_user(0), _assistant("a"), _assistant("b")],
                [_user(0), _assistant("a")],
            ),
        ],
    )
    def test_earlier_turns_must_match(self, tmp_path, chosen, rejected):
        p = _write_jsonl(
            tmp_path / "a.jsonl", [{"chosen": chosen, "rejected": rejected}]
        )
        with pytest.raises(AssertionError, match="differ only in their last message"):
            OpenAIPreferenceDatasetV2(p)

    def test_multi_message_completion_with_prompt(self, tmp_path):
        rec = {
            "prompt": "q",
            "chosen": [_assistant("a"), _user(1), _assistant("b")],
            "rejected": "c",
        }
        p = _write_jsonl(tmp_path / "a.jsonl", [rec])
        with pytest.raises(AssertionError, match="chosen completion must be a single"):
            OpenAIPreferenceDatasetV2(p)

    def test_identical_rejected(self, tmp_path):
        rec = {"chosen": [_user(0), _assistant("a")], "rejected": [_user(0), _assistant("a")]}
        p = _write_jsonl(tmp_path / "a.jsonl", [rec])
        with pytest.raises(ValueError, match="identical"):
            OpenAIPreferenceDatasetV2(p)

    @pytest.mark.parametrize("side", ["chosen", "rejected"])
    def test_last_message_not_assistant(self, tmp_path, side):
        rec = _full_sample(0)
        rec[side][-1] = _user(1)
        p = _write_jsonl(tmp_path / "a.jsonl", [rec])
        with pytest.raises(AssertionError, match=f"last {side} message"):
            OpenAIPreferenceDatasetV2(p)

    def test_prompt_column(self, tmp_path):
        rec = {
            "prompt": [SYSTEM, _user(0)],
            "chosen": [_assistant("good")],
            "rejected": "bad",
        }
        p = _write_jsonl(tmp_path / "a.jsonl", [rec])
        row = OpenAIPreferenceDatasetV2(p).dataset[0]
        assert row["context"] == [SYSTEM, _user(0)]
        assert row["completions"][0]["completion"] == [_assistant("good")]
        assert row["completions"][1]["completion"] == [_assistant("bad")]

    def test_string_prompt(self, tmp_path):
        rec = {"prompt": "q", "chosen": "a", "rejected": "b"}
        p = _write_jsonl(tmp_path / "a.jsonl", [rec])
        row = OpenAIPreferenceDatasetV2(p).dataset[0]
        assert row["context"] == [{"role": "user", "content": "q"}]

    def test_custom_keys(self, tmp_path):
        rec = {"ctx": "q", "win": "a", "lose": "b"}
        p = _write_jsonl(tmp_path / "a.jsonl", [rec])
        ds = OpenAIPreferenceDatasetV2(
            p, prompt_key="ctx", chosen_key="win", rejected_key="lose"
        ).dataset
        assert ds[0]["completions"][1]["completion"] == [_assistant("b")]

    def test_parquet(self, tmp_path):
        p = str(tmp_path / "a.parquet")
        Dataset.from_list([_full_sample(i) for i in range(3)]).to_parquet(p)
        ds = OpenAIPreferenceDatasetV2(p).dataset
        assert len(ds) == 3
        assert ds[2]["context"] == [SYSTEM, _user(2)]

    @pytest.mark.parametrize("kw", ["split_validation_size", "system_prompt", "dataset_paths"])
    def test_unsupported_args(self, tmp_path, kw):
        p = _write_jsonl(tmp_path / "a.jsonl", [_full_sample(0)])
        with pytest.raises(TypeError, match=kw):
            OpenAIPreferenceDatasetV2(p, **{kw: 0.1})


class TestFileOptions:
    def test_weights_and_kwargs_per_file(self, tmp_path):
        a = _write_jsonl(tmp_path / "a.jsonl", [_full_sample(i) for i in range(10)])
        b = _write_jsonl(tmp_path / "b.jsonl", [_full_sample(i) for i in range(4)])
        ds = OpenAIPreferenceDatasetV2(
            [
                {"path": a, "weight": 0.5, "chat_template_kwargs": {"enable_thinking": True}},
                b,
            ],
            seed=0,
        ).dataset
        assert len(ds) == 9
        assert _kwargs(ds) == [{"enable_thinking": True}] * 5 + [{}] * 4

    def test_sample_kwargs_override_file(self, tmp_path):
        p = _write_jsonl(
            tmp_path / "a.jsonl",
            [_full_sample(0, chat_template_kwargs={"enable_thinking": False})],
        )
        ds = OpenAIPreferenceDatasetV2(
            {"path": p, "chat_template_kwargs": {"enable_thinking": True, "x": 1}}
        ).dataset
        assert _kwargs(ds) == [{"enable_thinking": False, "x": 1}]


class TestTools:
    @pytest.mark.parametrize("as_str", [False, True])
    def test_use_tools(self, tmp_path, as_str):
        tools = json.dumps(TOOLS) if as_str else TOOLS
        p = _write_jsonl(tmp_path / "a.jsonl", [_full_sample(0, tools=tools)])
        ds = OpenAIPreferenceDatasetV2({"path": p, "use_tools": True}).dataset
        assert _kwargs(ds) == [{"tools": TOOLS}]

    def test_tools_ignored_without_use_tools(self, tmp_path):
        p = _write_jsonl(tmp_path / "a.jsonl", [_full_sample(0, tools=TOOLS)])
        ds = OpenAIPreferenceDatasetV2(p).dataset
        assert _kwargs(ds) == [{}]
        assert "tools" not in ds.column_names

    def test_mixed_tool_and_plain_files(self, tmp_path):
        a = _write_jsonl(tmp_path / "a.jsonl", [_full_sample(0, tools=json.dumps(TOOLS))])
        b = _write_jsonl(tmp_path / "b.jsonl", [_full_sample(1)])
        ds = OpenAIPreferenceDatasetV2([{"path": a, "use_tools": True}, b]).dataset
        assert _kwargs(ds) == [{"tools": TOOLS}, {}]


class TestRegistration:
    def test_load_preference_dataset(self, tmp_path):
        p = _write_jsonl(tmp_path / "a.jsonl", [_full_sample(0)])
        data = load_preference_dataset(
            {
                "dataset_name": "openai_preference_v2",
                "data_path": [{"path": p, "use_tools": False}],
                "prompt_file": None,
                "system_prompt_file": None,
            }
        )
        assert isinstance(data, OpenAIPreferenceDatasetV2)
        assert data.task_spec.task_name == "openai_preference_v2"
        assert data.val_dataset is None


TEMPLATE = (
    "{% if tools is defined %}[tools={{ tools | length }}]{% endif %}"
    "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}<|end|>{% endfor %}"
    "{% if enable_thinking is defined %}[think={{ enable_thinking }}]{% endif %}"
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


class TestEndToEnd:
    def test_preference_preprocessor_passes_kwargs(self, tokenizer, tmp_path):
        p = _write_jsonl(tmp_path / "a.jsonl", [_full_sample(0, tools=TOOLS)])
        datum = OpenAIPreferenceDatasetV2(
            {
                "path": p,
                "use_tools": True,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ).dataset[0]
        out = preference_preprocessor(
            datum, TaskDataSpec("t"), tokenizer, max_seq_length=10_000, idx=0
        )
        chosen = "".join(m["content"] for m in out["message_log_chosen"])
        rejected = "".join(m["content"] for m in out["message_log_rejected"])
        assert chosen.startswith("[tools=1]<|system|>sys")
        assert "good0" in chosen and "bad0" in rejected
        assert "[think=False]" in chosen and "[think=False]" in rejected


# Luciole-like: the generation prompt is the assistant header, plus an empty
# think block when thinking is disabled.
GEN_TEMPLATE = (
    "{% for m in messages %}"
    "{% if m.role == 'assistant' %}<|im_start|>assistant\n<think>\n\n</think>\n\n"
    "{{ m.content }}<|im_end|>\n"
    "{% else %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is false %}"
    "<think>\n\n</think>\n\n{% endif %}{% endif %}"
)


@pytest.fixture(scope="module")
def gen_tokenizer(tokenizer):
    t = tokenizer.__class__(tokenizer_object=tokenizer.backend_tokenizer)
    t.eos_token, t.pad_token, t.unk_token = "<eos>", "<eos>", "<unk>"
    t.chat_template = GEN_TEMPLATE
    return t


class TestAddGenerationPrompt:
    @pytest.fixture
    def datum(self, tmp_path):
        p = _write_jsonl(tmp_path / "a.jsonl", [_full_sample(0)])
        return OpenAIPreferenceDatasetV2(
            {"path": p, "chat_template_kwargs": {"enable_thinking": False}}
        ).dataset[0]

    def _process(self, datum, tok, **kwargs):
        return preference_preprocessor(
            datum, TaskDataSpec("t"), tok, max_seq_length=10_000, idx=0, **kwargs
        )

    def test_default_header_in_completions(self, datum, gen_tokenizer):
        out = self._process(datum, gen_tokenizer)
        for side in ("chosen", "rejected"):
            final = out[f"message_log_{side}"][-1]["content"]
            assert final.startswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")

    def test_header_moved_to_context(self, datum, gen_tokenizer):
        out = self._process(datum, gen_tokenizer, add_generation_prompt=True)
        assert out["message_log_chosen"][-1]["content"].startswith("good0<|im_end|>")
        assert out["message_log_rejected"][-1]["content"].startswith("bad0<|im_end|>")
        for side in ("chosen", "rejected"):
            context = out[f"message_log_{side}"][-2]
            assert context["role"] == "user"
            assert context["content"].endswith(
                "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
        # the full sequence is unchanged
        default = self._process(datum, gen_tokenizer)
        for side in ("chosen", "rejected"):
            key = f"message_log_{side}"
            assert "".join(m["content"] for m in out[key]) == "".join(
                m["content"] for m in default[key]
            )

    def test_setup_preference_data_reads_config(self, tmp_path, gen_tokenizer):
        p = _write_jsonl(tmp_path / "a.jsonl", [_full_sample(0)])
        dataset, _ = setup_preference_data(
            gen_tokenizer,
            {
                "max_input_seq_length": 10_000,
                "add_generation_prompt": True,
                "train": {"dataset_name": "openai_preference_v2", "data_path": p},
            },
        )
        final = dataset[0]["message_log_chosen"][-1]["content"]
        assert final.startswith("<think>\n\n</think>\n\ngood0")
