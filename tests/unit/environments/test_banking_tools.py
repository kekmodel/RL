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

"""Spec (via tests) for the kakaobank action dispatcher.

Contract:
  apply_action(state, action):
    - action is a dict with keys {"name": str, "arguments": dict}
    - dispatches to the named handler and mutates state in place
    - read tools (KB_search, get_customer_profile, get_account_or_contract)
      leave state unchanged
    - unknown or missing tool names are no-ops (forward-compat; the reward
      stays decidable because both gold replay and agent rollout treat
      unknown calls identically)
    - missing or None arguments are treated as {} without raising

  apply_actions(state, actions):
    - applies each action in order, mutating state in place
"""

import copy

from nemo_rl.environments.banking.state import canonical_hash, empty_state
from nemo_rl.environments.banking.tools import apply_action, apply_actions


def test_read_tools_do_not_mutate_state():
    state = empty_state()
    state["customers"]["data"]["c1"] = {"name": "X"}
    before = copy.deepcopy(state)
    for tool in ("KB_search", "get_customer_profile", "get_account_or_contract"):
        apply_action(state, {"name": tool, "arguments": {"query": "q"}})
    assert state == before


def test_unknown_tool_is_noop():
    state = empty_state()
    before_hash = canonical_hash(state)
    apply_action(state, {"name": "not_a_tool", "arguments": {"x": 1}})
    assert canonical_hash(state) == before_hash


def test_missing_arguments_is_treated_as_empty():
    state = empty_state()
    before = copy.deepcopy(state)
    apply_action(state, {"name": "KB_search"})
    apply_action(state, {"name": "KB_search", "arguments": None})
    assert state == before


def test_apply_actions_runs_in_order_and_is_idempotent_for_read_only_sequences():
    state = empty_state()
    state["customers"]["data"]["c1"] = {"name": "Y"}
    before = copy.deepcopy(state)
    apply_actions(
        state,
        [
            {"name": "KB_search", "arguments": {"query": "q1"}},
            {"name": "get_customer_profile", "arguments": {"customer_id": "c1"}},
            {"name": "get_account_or_contract", "arguments": {"record_id": "x"}},
        ],
    )
    assert state == before
