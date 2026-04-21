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

BankingRunner.process_turn(message_log, metadata):
  - Parses the last assistant message for a ``<action>{...}</action>`` tag
    whose body is a JSON object ``{"name": str, "arguments": dict}``.
  - Applies the action to ``metadata['predicted_state']`` via the tool
    dispatcher, then compares canonical_hash(predicted_state) against
    ``metadata['gold_hash']``.
  - On match: terminates with reward 1.0 and metadata None.
  - On unparseable action: leaves state unchanged, increments turn count,
    does not terminate.
  - When the incoming turn count already hits max_turns: terminates with
    reward equal to whether the current state already matches the gold
    hash (1.0 if it does, else 0.0), metadata None.
  - ``<action>{"name": "done"}</action>`` terminates the episode, with
    reward computed against the current predicted_state.

Observation body is always a ``{"role": "environment", "content": str}``
dict. next_stop_strings is ``["</action>"]`` while the episode is live,
None on termination.
"""

import json

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


def _assistant_turn(action: dict) -> list[dict]:
    return [
        {"role": "user", "content": "do the thing"},
        {"role": "assistant", "content": f"thinking. <action>{json.dumps(action)}</action>"},
    ]


def test_action_that_matches_gold_state_terminates_with_full_reward():
    initial = {"accounts": {"data": {"a1": {"status": "ACTIVE"}}}}
    gold_actions = [
        {"name": "close_account_or_service", "arguments": {"target_id": "a1"}}
    ]
    metadata = _build_metadata(initial, gold_actions)
    action = {
        "name": "close_account_or_service",
        "arguments": {"target_id": "a1"},
    }
    runner = BankingRunner()
    obs, reward, terminated, stop, next_meta, _ = runner.process_turn(
        _assistant_turn(action), metadata
    )
    assert terminated is True
    assert reward == 1.0
    assert next_meta is None
    assert stop is None
    assert obs["role"] == "environment"


def test_wrong_action_does_not_terminate_and_keeps_metadata_live():
    initial = {"accounts": {"data": {"a1": {"status": "ACTIVE"}}}}
    gold_actions = [
        {"name": "close_account_or_service", "arguments": {"target_id": "a1"}}
    ]
    metadata = _build_metadata(initial, gold_actions, max_turns=5)
    action = {"name": "KB_search", "arguments": {"query": "x"}}
    runner = BankingRunner()
    _, reward, terminated, stop, next_meta, _ = runner.process_turn(
        _assistant_turn(action), metadata
    )
    assert terminated is False
    assert reward == 0.0
    assert next_meta is not None
    assert next_meta["num_turns"] == 1
    assert stop == ["</action>"]


def test_unparseable_action_leaves_state_unchanged():
    initial = {"accounts": {"data": {"a1": {"status": "ACTIVE"}}}}
    metadata = _build_metadata(initial, [])
    before_hash = canonical_hash(metadata["predicted_state"])
    runner = BankingRunner()
    _, reward, terminated, _, next_meta, _ = runner.process_turn(
        [
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": "no action tag here"},
        ],
        metadata,
    )
    assert terminated is False
    assert reward == 0.0
    assert canonical_hash(next_meta["predicted_state"]) == before_hash
    assert next_meta["num_turns"] == 1


def test_max_turns_terminates_and_returns_match_based_reward():
    initial = {"accounts": {"data": {"a1": {"status": "ACTIVE"}}}}
    gold_actions = [
        {"name": "close_account_or_service", "arguments": {"target_id": "a1"}}
    ]
    metadata = _build_metadata(initial, gold_actions, num_turns=5, max_turns=5)
    runner = BankingRunner()
    _, reward, terminated, _, next_meta, _ = runner.process_turn(
        _assistant_turn({"name": "close_account_or_service", "arguments": {"target_id": "a1"}}),
        metadata,
    )
    assert terminated is True
    assert reward == 0.0  # state had not yet been mutated to match gold
    assert next_meta is None


def test_done_action_terminates_and_compares_hash():
    initial = {"accounts": {"data": {"a1": {"status": "ACTIVE"}}}}
    gold_actions = [
        {"name": "close_account_or_service", "arguments": {"target_id": "a1"}}
    ]
    metadata = _build_metadata(initial, gold_actions)
    runner = BankingRunner()
    _, reward, terminated, _, next_meta, _ = runner.process_turn(
        _assistant_turn({"name": "done"}),
        metadata,
    )
    assert terminated is True
    assert reward == 0.0  # agent called done before achieving the gold state
    assert next_meta is None


def test_runner_does_not_mutate_caller_metadata():
    initial = {"accounts": {"data": {"a1": {"status": "ACTIVE"}}}}
    gold_actions = [
        {"name": "close_account_or_service", "arguments": {"target_id": "a1"}}
    ]
    metadata = _build_metadata(initial, gold_actions)
    before_hash = canonical_hash(metadata["predicted_state"])
    runner = BankingRunner()
    runner.process_turn(
        _assistant_turn({"name": "KB_search", "arguments": {"query": "x"}}),
        metadata,
    )
    assert canonical_hash(metadata["predicted_state"]) == before_hash
    assert metadata["num_turns"] == 0
