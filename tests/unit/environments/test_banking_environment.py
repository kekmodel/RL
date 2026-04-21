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

"""Spec (via tests) for BankingRunner, the plain-python core of the Ray
actor ``BankingEnvironment``.

The tool-call wire format is the Nemotron 3 Nano / Qwen3-coder XML shape,
per ``megatron.core.tokenizers.text.parsers.qwen3_coder_tool_parser``:

    <tool_call>
    <function=tool_name>
    <parameter=key>value</parameter>
    ...
    </function>
    </tool_call>

BankingRunner.process_turn(message_log, metadata):
  - Parses the last ``<tool_call>...</tool_call>`` block in the last
    assistant message.
  - Parameter values are coerced with json.loads / ast.literal_eval;
    failures degrade to a trimmed string.
  - A parsed tool call is applied to metadata['predicted_state'] via the
    tool dispatcher; the runner then compares canonical_hash against
    metadata['gold_hash']. On match → reward 1.0 and metadata None.
  - When the assistant emits text with no <tool_call> block, the episode
    terminates (the model has finished tool usage) and reward is 1.0 iff
    the current predicted_state already hashes to the gold hash.
  - Reaching max_turns terminates with the same hash-based reward.
  - next_stop_strings is ``["</tool_call>"]`` while live, None on
    termination.
"""

from nemo_rl.environments.banking.runner import BankingRunner
from nemo_rl.environments.banking.state import (
    canonical_hash,
    empty_state,
    seed_state,
)
from nemo_rl.environments.banking.tools import apply_actions


def _build_metadata(
    initial_state: dict,
    gold_actions: list[dict],
    *,
    num_turns: int = 0,
    max_turns: int = 5,
) -> dict:
    predicted = seed_state(initial_state)
    gold = seed_state(initial_state)
    apply_actions(gold, gold_actions)
    return {
        "predicted_state": predicted,
        "gold_hash": canonical_hash(gold),
        "num_turns": num_turns,
        "max_turns": max_turns,
    }


def _render_tool_call(name: str, arguments: dict) -> str:
    import json as _json

    param_lines = []
    for key, value in arguments.items():
        if isinstance(value, str):
            rendered = value
        elif value is None:
            rendered = "null"
        elif isinstance(value, bool):
            rendered = "true" if value else "false"
        else:
            rendered = _json.dumps(value, ensure_ascii=False)
        param_lines.append(f"<parameter={key}>{rendered}</parameter>")
    body = "\n".join(param_lines)
    return (
        f"<tool_call>\n<function={name}>\n{body}\n</function>\n</tool_call>"
    )


def _assistant_turn(name: str, arguments: dict) -> list[dict]:
    return [
        {"role": "user", "content": "do the thing"},
        {
            "role": "assistant",
            "content": (
                "<think>planning</think>\n" + _render_tool_call(name, arguments)
            ),
        },
    ]


def test_tool_call_matching_gold_state_terminates_with_full_reward():
    initial = {"accounts": {"data": {"a1": {"status": "ACTIVE"}}}}
    gold_actions = [
        {"name": "close_account_or_service", "arguments": {"target_id": "a1"}}
    ]
    metadata = _build_metadata(initial, gold_actions)
    runner = BankingRunner()
    obs, reward, terminated, stop, next_meta, _ = runner.process_turn(
        _assistant_turn("close_account_or_service", {"target_id": "a1"}),
        metadata,
    )
    assert terminated is True
    assert reward == 1.0
    assert next_meta is None
    assert stop is None
    assert obs["role"] == "environment"


def test_wrong_tool_call_does_not_terminate():
    initial = {"accounts": {"data": {"a1": {"status": "ACTIVE"}}}}
    gold_actions = [
        {"name": "close_account_or_service", "arguments": {"target_id": "a1"}}
    ]
    metadata = _build_metadata(initial, gold_actions, max_turns=5)
    runner = BankingRunner()
    _, reward, terminated, stop, next_meta, _ = runner.process_turn(
        _assistant_turn("KB_search", {"query": "x"}),
        metadata,
    )
    assert terminated is False
    assert reward == 0.0
    assert next_meta is not None
    assert next_meta["num_turns"] == 1
    assert stop == ["</tool_call>"]


def test_plain_text_without_tool_call_terminates_with_hash_reward_zero():
    initial = {"accounts": {"data": {"a1": {"status": "ACTIVE"}}}}
    gold_actions = [
        {"name": "close_account_or_service", "arguments": {"target_id": "a1"}}
    ]
    metadata = _build_metadata(initial, gold_actions)
    runner = BankingRunner()
    _, reward, terminated, _, next_meta, _ = runner.process_turn(
        [
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": "I'm done, have a nice day."},
        ],
        metadata,
    )
    assert terminated is True
    assert reward == 0.0
    assert next_meta is None


def test_plain_text_when_state_already_matches_terminates_with_reward_one():
    # Gold action sequence is empty; predicted_state already matches gold.
    initial = {"accounts": {"data": {"a1": {"status": "ACTIVE"}}}}
    metadata = _build_metadata(initial, gold_actions=[])
    runner = BankingRunner()
    _, reward, terminated, _, next_meta, _ = runner.process_turn(
        [
            {"role": "user", "content": "done?"},
            {"role": "assistant", "content": "All done."},
        ],
        metadata,
    )
    assert terminated is True
    assert reward == 1.0
    assert next_meta is None


def test_max_turns_terminates_and_returns_hash_reward():
    initial = {"accounts": {"data": {"a1": {"status": "ACTIVE"}}}}
    gold_actions = [
        {"name": "close_account_or_service", "arguments": {"target_id": "a1"}}
    ]
    metadata = _build_metadata(initial, gold_actions, num_turns=5, max_turns=5)
    runner = BankingRunner()
    _, reward, terminated, _, next_meta, _ = runner.process_turn(
        _assistant_turn("close_account_or_service", {"target_id": "a1"}),
        metadata,
    )
    assert terminated is True
    assert reward == 0.0  # turn budget exhausted; predicted_state had not matched
    assert next_meta is None


def test_parameter_values_are_coerced_for_json_numbers_and_dicts():
    # create_loan_application's requested_amount_krw must be int, not "45000000".
    initial = {}
    gold_actions = [
        {
            "name": "create_loan_application",
            "arguments": {
                "application_id": "app_1",
                "customer_id": "c_1",
                "product_name": "개인사업자 신용대출",
                "requested_amount_krw": 45000000,
                "purpose": "WORKING_CAPITAL",
                "expected_status": "SUBMITTED_FOR_SCREENING",
            },
        }
    ]
    metadata = _build_metadata(initial, gold_actions)
    runner = BankingRunner()
    _, reward, terminated, *_ = runner.process_turn(
        _assistant_turn(
            "create_loan_application",
            {
                "application_id": "app_1",
                "customer_id": "c_1",
                "product_name": "개인사업자 신용대출",
                "requested_amount_krw": 45000000,
                "purpose": "WORKING_CAPITAL",
                "expected_status": "SUBMITTED_FOR_SCREENING",
            },
        ),
        metadata,
    )
    assert terminated is True
    assert reward == 1.0


def test_runner_does_not_mutate_caller_metadata():
    initial = {"accounts": {"data": {"a1": {"status": "ACTIVE"}}}}
    gold_actions = [
        {"name": "close_account_or_service", "arguments": {"target_id": "a1"}}
    ]
    metadata = _build_metadata(initial, gold_actions)
    before_hash = canonical_hash(metadata["predicted_state"])
    runner = BankingRunner()
    runner.process_turn(
        _assistant_turn("KB_search", {"query": "x"}),
        metadata,
    )
    assert canonical_hash(metadata["predicted_state"]) == before_hash
    assert metadata["num_turns"] == 0


def test_only_the_last_tool_call_is_applied():
    # If the model emits two tool_call blocks (e.g. a self-corrected call),
    # the runner acts on the LAST one, matching Qwen3-coder convention.
    initial = {
        "accounts": {
            "data": {
                "a_wrong": {"status": "ACTIVE"},
                "a_right": {"status": "ACTIVE"},
            }
        }
    }
    gold_actions = [
        {"name": "close_account_or_service", "arguments": {"target_id": "a_right"}}
    ]
    metadata = _build_metadata(initial, gold_actions)
    two_calls = (
        _render_tool_call("close_account_or_service", {"target_id": "a_wrong"})
        + "\n"
        + _render_tool_call("close_account_or_service", {"target_id": "a_right"})
    )
    runner = BankingRunner()
    _, reward, terminated, *_ = runner.process_turn(
        [
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": two_calls},
        ],
        metadata,
    )
    assert terminated is True
    assert reward == 1.0
