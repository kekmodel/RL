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

"""Action dispatcher for the kakaobank_knowledge action families.

Handlers are registered in ``HANDLERS``. Each handler is deterministic in
the pair ``(name, arguments)`` so that gold replay and agent rollout agree
on the resulting state whenever the action sequences are semantically
equivalent.

This module currently provides the dispatcher plumbing and the three read
tools. Write-tool handlers are added incrementally in follow-up changes.
"""

from typing import Any, Callable

from nemo_rl.environments.banking.state import KakaoBankState


def _noop(_state: KakaoBankState, _args: dict[str, Any]) -> None:
    return None


HANDLERS: dict[str, Callable[[KakaoBankState, dict[str, Any]], None]] = {
    # Read tools: state-invariant by construction.
    "KB_search": _noop,
    "get_customer_profile": _noop,
    "get_account_or_contract": _noop,
}


def apply_action(state: KakaoBankState, action: dict[str, Any]) -> None:
    """Apply a single ``{name, arguments}`` action to ``state`` in place.

    Unknown tool names are treated as no-ops so that future task schemas
    and speculative agent calls do not destabilize replay.
    """
    name = action.get("name")
    args = action.get("arguments") or {}
    handler = HANDLERS.get(name, _noop)
    handler(state, args)


def apply_actions(state: KakaoBankState, actions: list[dict[str, Any]]) -> None:
    for action in actions:
        apply_action(state, action)
