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

import copy
from typing import Any, Callable

from nemo_rl.environments.banking.state import KakaoBankState


def _noop(_state: KakaoBankState, _args: dict[str, Any]) -> None:
    return None


# Tables that ``close_account_or_service`` may target. Searched in this order
# to resolve a target_id when the caller does not name the table explicitly.
_CLOSABLE_TABLES: tuple[str, ...] = (
    "accounts",
    "deposit_contracts",
    "savings_boxes",
    "group_memberships",
    "pockets",
    "service_enrollments",
    "prepaid_wallets",
)


def _find_record_table(
    state: KakaoBankState,
    record_id: str,
    search: tuple[str, ...],
) -> str | None:
    for table in search:
        data = state.get(table, {}).get("data", {})
        if record_id in data:
            return table
    return None


def _upsert(
    state: KakaoBankState,
    table: str,
    record_id: str,
    fields: dict[str, Any],
) -> None:
    state.setdefault(table, {"data": {}})
    data = state[table]["data"]
    existing = data.get(record_id)
    if existing is None:
        data[record_id] = copy.deepcopy(fields)
    else:
        for key, value in fields.items():
            existing[key] = copy.deepcopy(value)


def _close_account_or_service(state: KakaoBankState, args: dict[str, Any]) -> None:
    target_id = args.get("target_id")
    if not target_id:
        return
    table = _find_record_table(state, target_id, _CLOSABLE_TABLES)
    if table is None:
        return
    updates: dict[str, Any] = {"status": "CLOSED"}
    if args.get("close_type") is not None:
        updates["close_type"] = args["close_type"]
    if args.get("reason") is not None:
        updates["close_reason"] = args["reason"]
    _upsert(state, table, target_id, updates)


def _open_or_enroll_product(state: KakaoBankState, args: dict[str, Any]) -> None:
    target_table = args.get("target_table")
    target_id = args.get("target_id")
    if not target_table or not target_id:
        return
    fields: dict[str, Any] = {
        "customer_id": args.get("customer_id"),
        "product_name": args.get("product_name"),
        "status": args.get("status", "ACTIVE"),
    }
    options = args.get("options") or {}
    if isinstance(options, dict):
        fields.update(options)
    _upsert(state, target_table, target_id, fields)


HANDLERS: dict[str, Callable[[KakaoBankState, dict[str, Any]], None]] = {
    # Read tools: state-invariant by construction.
    "KB_search": _noop,
    "get_customer_profile": _noop,
    "get_account_or_contract": _noop,
    # Write tools.
    "close_account_or_service": _close_account_or_service,
    "open_or_enroll_product": _open_or_enroll_product,
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
