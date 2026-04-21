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

"""Spec (via tests) for BankingRunner's tool-response message shape and
its interaction with Nemotron 3 Nano's chat template.

Two layers of checks:

1. **Shape checks (always run)** — what role/content keys does the runner
   produce? The runner must emit ``{"role": "tool", "content": <plain>}``
   rather than pre-wrapping the content with ``<environment>`` tags. The
   tokenizer's chat template owns the ``<tool_response>`` wrapping.

2. **Tokenizer integration (opt-in)** — when the Nemotron 3 Nano
   tokenizer is available (env var ``NEMOTRON_TOKENIZER_DIR`` points at
   a directory containing tokenizer_config.json and tokenizer.json), we
   render a realistic multi-turn tool chain through
   ``apply_chat_template`` and assert:

     - tool-role messages are emitted as
       ``<|im_start|>user\\n<tool_response>\\n{content}\\n</tool_response>...``
     - the generation prompt opens a fresh ``<think>`` so the model can
       reason again at the next turn
     - ``tokenize=True`` does not raise
"""

import os

import pytest

from nemo_rl.environments.banking.runner import BankingRunner
from nemo_rl.environments.banking.state import (
    canonical_hash,
    empty_state,
    seed_state,
)
from nemo_rl.environments.banking.tools import apply_actions


# ---- Shape checks -------------------------------------------------------


def _metadata_for(gold_actions):
    initial = {"accounts": {"data": {"a1": {"status": "ACTIVE"}}}}
    predicted = seed_state(initial)
    gold = seed_state(initial)
    apply_actions(gold, gold_actions)
    return {
        "predicted_state": predicted,
        "gold_hash": canonical_hash(gold),
        "num_turns": 0,
        "max_turns": 5,
    }


def _render_tool_call(name, arguments):
    import json as _json

    lines = []
    for key, value in arguments.items():
        if isinstance(value, str):
            rendered = value
        elif value is None:
            rendered = "null"
        elif isinstance(value, bool):
            rendered = "true" if value else "false"
        else:
            rendered = _json.dumps(value, ensure_ascii=False)
        lines.append(f"<parameter={key}>{rendered}</parameter>")
    return (
        "<tool_call>\n<function="
        + name
        + ">\n"
        + "\n".join(lines)
        + "\n</function>\n</tool_call>"
    )


def _turn(name, args):
    return [
        {"role": "user", "content": "do the thing"},
        {
            "role": "assistant",
            "content": "<think>planning</think>\n" + _render_tool_call(name, args),
        },
    ]


def test_runner_emits_tool_role_with_plain_content_on_successful_apply():
    metadata = _metadata_for(
        [{"name": "close_account_or_service", "arguments": {"target_id": "a1"}}]
    )
    obs, _, _, _, _, _ = BankingRunner().process_turn(
        _turn("close_account_or_service", {"target_id": "a1"}), metadata
    )
    assert obs["role"] == "tool"
    # Content is plain text — no <environment>...</environment> wrapping.
    assert "<environment>" not in obs["content"]
    assert "</environment>" not in obs["content"]


def test_runner_emits_tool_role_on_intermediate_step():
    metadata = _metadata_for(
        [{"name": "close_account_or_service", "arguments": {"target_id": "a1"}}]
    )
    obs, _, terminated, *_ = BankingRunner().process_turn(
        _turn("KB_search", {"query": "x"}), metadata
    )
    assert terminated is False
    assert obs["role"] == "tool"
    assert "<environment>" not in obs["content"]


def test_runner_emits_tool_role_on_malformed_tool_call():
    metadata = _metadata_for(
        [{"name": "close_account_or_service", "arguments": {"target_id": "a1"}}]
    )
    malformed = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "<tool_call>garbage</tool_call>"},
    ]
    obs, _, _, _, _, _ = BankingRunner().process_turn(malformed, metadata)
    assert obs["role"] == "tool"


# ---- Tokenizer integration ---------------------------------------------


_TOKENIZER_DIR = os.environ.get("NEMOTRON_TOKENIZER_DIR")


@pytest.mark.skipif(
    not _TOKENIZER_DIR, reason="NEMOTRON_TOKENIZER_DIR not set; skipping integration."
)
def test_tool_response_wrapping_and_fresh_think_via_chat_template():
    from transformers import AutoTokenizer  # type: ignore

    tok = AutoTokenizer.from_pretrained(_TOKENIZER_DIR)

    # Build a two-round tool chain that mirrors what a BankingEnvironment
    # rollout produces: assistant thinks + tool_call → tool response →
    # assistant thinks + tool_call → tool response → generation prompt.
    messages = [
        {"role": "user", "content": "입출금통장 해지해줘"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "먼저 KB를 확인.",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "KB_search",
                        "arguments": {"query": "입출금통장 해지"},
                    },
                }
            ],
        },
        # BankingRunner output shape.
        {"role": "tool", "content": "Applied KB_search."},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "조건 충족. 계좌 종료.",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "close_account_or_service",
                        "arguments": {"target_id": "a1"},
                    },
                }
            ],
        },
        {"role": "tool", "content": "Applied close_account_or_service."},
    ]

    rendered = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Tool role is wrapped by the template as a <tool_response> block.
    assert "<tool_response>\nApplied KB_search.\n</tool_response>" in rendered
    assert (
        "<tool_response>\nApplied close_account_or_service.\n</tool_response>"
        in rendered
    )

    # After the tool response the template opens a fresh <think> for the
    # next assistant turn. This is the behavior we depend on so the agent
    # can reason again between tool calls.
    assert rendered.endswith("<|im_start|>assistant\n<think>\n")

    # And tokenization itself must succeed without raising, and yield a
    # non-empty input_ids sequence.
    encoded = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    if hasattr(encoded, "input_ids"):
        input_ids = list(encoded.input_ids)
    elif isinstance(encoded, dict):
        input_ids = list(encoded["input_ids"])
    else:
        input_ids = list(encoded)
    assert len(input_ids) > 0


@pytest.mark.skipif(
    not _TOKENIZER_DIR, reason="NEMOTRON_TOKENIZER_DIR not set; skipping integration."
)
def test_new_user_turn_strips_all_prior_reasoning_but_keeps_tool_calls():
    """Once a new user turn appears, every assistant ``reasoning_content``
    from BEFORE that last user turn is stripped from the rendered
    history — but the tool_call XML itself is preserved so the DB state
    trace remains visible to the model.
    """
    from transformers import AutoTokenizer  # type: ignore

    tok = AutoTokenizer.from_pretrained(_TOKENIZER_DIR)
    messages = [
        {"role": "user", "content": "round1"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "OLD_REASONING_1",
            "tool_calls": [
                {"type": "function", "function": {"name": "A", "arguments": {"x": "1"}}}
            ],
        },
        {"role": "tool", "content": "r1"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "OLD_REASONING_2",
            "tool_calls": [
                {"type": "function", "function": {"name": "B", "arguments": {"y": "2"}}}
            ],
        },
        {"role": "tool", "content": "r2"},
        # New user turn pushes the "last_user_idx" forward; everything
        # above loses its reasoning in the rendered history.
        {"role": "user", "content": "round2"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "CURRENT_REASONING",
            "tool_calls": [
                {"type": "function", "function": {"name": "C", "arguments": {"z": "3"}}}
            ],
        },
        {"role": "tool", "content": "r3"},
    ]
    rendered = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Prior-round reasoning is gone.
    assert "OLD_REASONING_1" not in rendered
    assert "OLD_REASONING_2" not in rendered
    # Current-round reasoning (after last user) is kept so the model can
    # continue its in-progress thinking chain.
    assert "CURRENT_REASONING" in rendered
    # Tool calls themselves are still part of the rendered trace — the
    # DB-state history remains visible; only <think> blocks are pruned.
    assert "<function=A>" in rendered
    assert "<function=B>" in rendered
    assert "<function=C>" in rendered


@pytest.mark.skipif(
    not _TOKENIZER_DIR, reason="NEMOTRON_TOKENIZER_DIR not set; skipping integration."
)
def test_prior_turn_thinking_is_truncated_by_default():
    """With ``truncate_history_thinking`` default True, any reasoning from
    assistant turns BEFORE the last user message is stripped out of the
    rendered history. Only the current (in-progress) reasoning chain is
    exposed to the model.
    """
    from transformers import AutoTokenizer  # type: ignore

    tok = AutoTokenizer.from_pretrained(_TOKENIZER_DIR)
    messages = [
        {"role": "user", "content": "1"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "THOUGHT_THAT_MUST_BE_STRIPPED",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "KB_search", "arguments": {"query": "q"}},
                }
            ],
        },
        {"role": "tool", "content": "ok"},
        {"role": "user", "content": "keep going"},
    ]
    rendered = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    assert "THOUGHT_THAT_MUST_BE_STRIPPED" not in rendered
