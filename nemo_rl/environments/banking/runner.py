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

The tool-call wire format is the Nemotron 3 Nano / Qwen3-coder XML shape,
mirroring ``megatron.core.tokenizers.text.parsers.qwen3_coder_tool_parser``:

    <tool_call>
    <function=tool_name>
    <parameter=key>value</parameter>
    ...
    </function>
    </tool_call>

Parameter values are free-form text. They are coerced via
:func:`json.loads` and, on failure, :func:`ast.literal_eval` before
falling back to the raw string. This matches the Qwen3-coder parser's
no-schema path and allows numbers, booleans, nulls, and nested JSON to
round-trip faithfully into the tool dispatcher.
"""

import ast
import copy
import json
import re
from typing import Any, Optional, TypedDict

from nemo_rl.environments.banking.state import KakaoBankState, canonical_hash
from nemo_rl.environments.banking.tools import apply_action


class BankingMetadata(TypedDict):
    predicted_state: KakaoBankState
    gold_hash: str
    num_turns: int
    max_turns: int


_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=(.*?)</function>", re.DOTALL)
_PARAMETER_RE = re.compile(
    r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)", re.DOTALL
)

_ACTION_STOP = ["</tool_call>"]


def _coerce_param_value(raw: str) -> Any:
    """Parse ``raw`` as JSON, then as a Python literal, else return the
    trimmed string. Matches the no-schema path of Qwen3CoderToolParser.
    """
    stripped = raw
    if stripped.startswith("\n"):
        stripped = stripped[1:]
    if stripped.endswith("\n"):
        stripped = stripped[:-1]
    if stripped.strip().lower() == "null":
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(stripped)
    except (ValueError, SyntaxError, TypeError):
        pass
    return stripped


def _parse_action(text: str) -> Optional[dict[str, Any]]:
    """Extract the last ``<tool_call>`` block. Returns ``None`` when the
    message contains no tool call at all (the signal that the agent has
    finished acting and is addressing the user).
    """
    tool_calls = _TOOL_CALL_RE.findall(text)
    if not tool_calls:
        return None
    function_match = _FUNCTION_RE.search(tool_calls[-1])
    if not function_match:
        return {"name": "__malformed__", "arguments": {}}
    body = function_match.group(1)
    name_end = body.find(">")
    if name_end == -1:
        return {"name": "__malformed__", "arguments": {}}
    name = body[:name_end]
    params_body = body[name_end + 1 :]
    arguments: dict[str, Any] = {}
    for match in _PARAMETER_RE.findall(params_body):
        sep = match.find(">")
        if sep == -1:
            continue
        arguments[match[:sep]] = _coerce_param_value(match[sep + 1 :])
    return {"name": name, "arguments": arguments}


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
    # Role ``tool`` is recognized by the Nemotron 3 Nano chat template and
    # gets wrapped as ``<tool_response>...</tool_response>``. We pass plain
    # text and let the template own the wrapping, so content composes
    # correctly when the agent is re-prompted for the next turn.
    return (
        {"role": "tool", "content": message},
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

        # No tool call at all: the agent is addressing the user; terminate.
        if action is None:
            reward = (
                1.0
                if canonical_hash(metadata["predicted_state"]) == gold_hash
                else 0.0
            )
            return _terminate(reward, "Assistant finished without further tool use.")

        if action["name"] == "__malformed__":
            return (
                {
                    "role": "tool",
                    "content": (
                        "Malformed tool_call: expected "
                        "<tool_call><function=NAME>...</function></tool_call>."
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

        next_state = copy.deepcopy(metadata["predicted_state"])
        apply_action(next_state, action)
        if canonical_hash(next_state) == gold_hash:
            return _terminate(1.0, f"Applied {action['name']} — gold state reached.")

        return (
            {
                "role": "tool",
                "content": f"Applied {action['name']}.",
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
