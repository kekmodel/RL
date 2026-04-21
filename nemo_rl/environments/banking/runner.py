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

"""Pure-python core of BankingEnvironment.

Kept free of ``ray`` and ``torch`` imports so that the turn logic can be
exercised directly by unit tests. The Ray-wrapped actor in
``environment.py`` delegates each per-turn step to
:meth:`BankingRunner.process_turn`.
"""

import copy
import json
from typing import Any, Optional, TypedDict

from nemo_rl.environments.banking.state import KakaoBankState, canonical_hash
from nemo_rl.environments.banking.tools import apply_action


class BankingMetadata(TypedDict):
    predicted_state: KakaoBankState
    gold_hash: str
    num_turns: int
    max_turns: int


_ACTION_STOP = ["</action>"]


def _parse_action(text: str) -> Optional[dict[str, Any]]:
    prefix = "<action>"
    suffix = "</action>"
    start = text.rfind(prefix)
    if start == -1:
        return None
    end = text.find(suffix, start + len(prefix))
    if end == -1:
        return None
    try:
        payload = json.loads(text[start + len(prefix) : end].strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or "name" not in payload:
        return None
    return payload


def _terminate(
    reward: float, message: str
) -> tuple[
    dict[str, str],
    float,
    bool,
    Optional[list[str]],
    Optional[BankingMetadata],
    Optional[str],
]:
    return (
        {
            "role": "environment",
            "content": f"<environment>\n{message}\n</environment>\n",
        },
        reward,
        True,
        None,
        None,
        None,
    )


class BankingRunner:
    """Processes a single agent turn against a banking task."""

    def process_turn(
        self,
        message_log: list[dict[str, Any]],
        metadata: BankingMetadata,
    ) -> tuple[
        dict[str, str],
        float,
        bool,
        Optional[list[str]],
        Optional[BankingMetadata],
        Optional[str],
    ]:
        num_turns = metadata["num_turns"]
        max_turns = metadata["max_turns"]
        gold_hash = metadata["gold_hash"]

        if num_turns >= max_turns:
            reward = (
                1.0
                if canonical_hash(metadata["predicted_state"]) == gold_hash
                else 0.0
            )
            return _terminate(reward, f"Maximum turns ({max_turns}) reached.")

        last_content = ""
        if message_log and message_log[-1]["role"] == "assistant":
            last_content = str(message_log[-1]["content"])

        action = _parse_action(last_content)
        if action is None:
            return (
                {
                    "role": "environment",
                    "content": (
                        "<environment>\nInvalid response: expected "
                        "<action>{\"name\": ..., \"arguments\": ...}</action>"
                        " JSON.\n</environment>\n"
                    ),
                },
                0.0,
                False,
                _ACTION_STOP,
                {
                    **metadata,
                    "predicted_state": copy.deepcopy(metadata["predicted_state"]),
                    "num_turns": num_turns + 1,
                },
                None,
            )

        if action.get("name") == "done":
            reward = (
                1.0
                if canonical_hash(metadata["predicted_state"]) == gold_hash
                else 0.0
            )
            return _terminate(reward, "done")

        next_state = copy.deepcopy(metadata["predicted_state"])
        apply_action(next_state, action)
        if canonical_hash(next_state) == gold_hash:
            return _terminate(1.0, f"Applied {action['name']} — gold state reached.")

        return (
            {
                "role": "environment",
                "content": f"<environment>\nApplied {action['name']}.\n</environment>\n",
            },
            0.0,
            False,
            _ACTION_STOP,
            {
                **metadata,
                "predicted_state": next_state,
                "num_turns": num_turns + 1,
            },
            None,
        )
