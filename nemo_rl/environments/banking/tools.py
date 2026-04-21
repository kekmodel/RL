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


def _first_present(args: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = args.get(key)
        if value is not None:
            return value
    return None


def _update_card_state(state: KakaoBankState, args: dict[str, Any]) -> None:
    order_id = _first_present(args, "order_id", "card_order_id")
    if not order_id:
        return
    op = args.get("operation")
    if op == "REJECT_NEW_ISSUE":
        order_status = "REJECTED"
    elif op in ("APPROVE_NEW_ISSUE", "APPROVE_REISSUE", "REISSUE_CARD", "ISSUE"):
        order_status = "APPROVED"
    elif op == "CANCEL":
        order_status = "CANCELLED"
    else:
        order_status = op or "PROCESSED"
    _upsert(
        state,
        "card_orders",
        order_id,
        {
            "card_order_id": order_id,
            "customer_id": args.get("customer_id"),
            "card_id": args.get("card_id"),
            "status": order_status,
            "reason": args.get("reason"),
            "operation": op,
        },
    )


def _update_loan_contract_state(state: KakaoBankState, args: dict[str, Any]) -> None:
    loan_id = _first_present(args, "loan_id", "target_id")
    if not loan_id:
        return
    op = args.get("operation") or ""
    updates: dict[str, Any] = {"operation": op}
    if "ACCELERATE" in op or op == "IMMEDIATE_REPAYMENT":
        updates["status"] = "ACCELERATED"
        updates["acceleration_reason"] = args.get("reason")
    elif op == "EXECUTE":
        updates["status"] = "EXECUTED"
    elif op == "WITHDRAW":
        updates["status"] = "WITHDRAWN"
        updates["withdrawal_reason"] = args.get("reason")
    elif op == "SUSPEND":
        updates["status"] = "SUSPENDED"
    elif op == "REJECT_EXECUTION":
        updates["status"] = "REJECTED"
        updates["rejection_reason"] = args.get("reason")
    if args.get("effective_at") is not None:
        updates["effective_at"] = args["effective_at"]
    _upsert(state, "loans", loan_id, updates)


def _process_refinance_request(state: KakaoBankState, args: dict[str, Any]) -> None:
    refi_id = args.get("refinance_id")
    if not refi_id:
        return
    op = args.get("operation") or ""
    if op.startswith("CANCEL"):
        status = "CANCELLED"
    elif op.startswith("COMPLETE"):
        status = "COMPLETED"
    else:
        status = op or "PENDING"
    _upsert(
        state,
        "refinance_requests",
        refi_id,
        {
            "refinance_id": refi_id,
            "status": status,
            "old_loan_repayment_status": args.get("old_loan_repayment_status"),
            "operation": op,
        },
    )


def _request_maturity_or_extension(state: KakaoBankState, args: dict[str, Any]) -> None:
    target_id = args.get("target_id")
    if not target_id:
        return
    table = _find_record_table(state, target_id, ("deposit_contracts", "loans"))
    if table is None:
        return
    op = args.get("operation") or ""
    options = args.get("options") or {}
    updates: dict[str, Any] = {"maturity_decision": op}
    if op.startswith("REJECT"):
        # Decision recorded; record's active status remains unchanged.
        pass
    elif "MATURE" in op or op == "MATURITY_CLOSE":
        updates["status"] = "CLOSED"
        if options.get("close_type"):
            updates["close_type"] = options["close_type"]
        else:
            updates["close_type"] = "MATURITY"
    elif "AUTO_CLOSE" in op:
        updates["status"] = "CLOSED"
        updates["close_type"] = "AUTO"
    elif "EXTEND" in op:
        updates["status"] = "EXTENDED"
    _upsert(state, table, target_id, updates)


def _execute_remittance_case(state: KakaoBankState, args: dict[str, Any]) -> None:
    options = args.get("options") or {}
    case_id = _first_present(args, "remittance_id") or options.get("remittance_id")
    if not case_id:
        return
    _upsert(
        state,
        "remittance_cases",
        case_id,
        {
            "remittance_id": case_id,
            "customer_id": args.get("customer_id"),
            "direction": args.get("direction"),
            "amount": args.get("amount"),
            "currency": args.get("currency"),
            "country": args.get("country"),
            "purpose_code": args.get("purpose_code"),
        },
    )


def _deterministic_transaction_id(args: dict[str, Any]) -> str:
    # A stable composite id so that agent and gold replay agree when the
    # same canonical args are applied.
    parts = [
        str(args.get("transaction_type") or args.get("transfer_type") or ""),
        str(
            args.get("source_id")
            or args.get("source_account_id")
            or ""
        ),
        str(args.get("target_id") or ""),
        str(args.get("amount") or ""),
        str(args.get("currency") or ""),
    ]
    return "txn::" + "::".join(parts)


def _execute_deposit_or_box_transfer(state: KakaoBankState, args: dict[str, Any]) -> None:
    # At least one of target_id / source_id must exist for a transaction to
    # have a meaningful composite id.
    if not (args.get("target_id") or args.get("source_id") or args.get("source_account_id")):
        return
    txn_id = args.get("transaction_id") or _deterministic_transaction_id(args)
    kind = args.get("transaction_type") or args.get("transfer_type") or ""
    status = "REJECTED" if kind.startswith("REJECT") else "POSTED"
    _upsert(
        state,
        "transactions",
        txn_id,
        {
            "transaction_id": txn_id,
            "source_id": args.get("source_id") or args.get("source_account_id"),
            "target_id": args.get("target_id"),
            "amount": args.get("amount"),
            "currency": args.get("currency"),
            "transaction_type": kind,
            "status": status,
            "reason": args.get("reason"),
        },
    )


def _configure_auto_transfer(state: KakaoBankState, args: dict[str, Any]) -> None:
    options = args.get("options") or {}
    rule_id = _first_present(args, "auto_transfer_id", "existing_auto_transfer_id") or options.get("auto_transfer_id")
    if not rule_id:
        return
    op = args.get("operation") or "CREATE"
    if op == "CANCEL":
        status = "CANCELLED"
    elif op.startswith("REJECT"):
        status = "REJECTED"
    else:
        status = args.get("status") or "ACTIVE"
    _upsert(
        state,
        "auto_transfer_rules",
        rule_id,
        {
            "auto_transfer_id": rule_id,
            "source_account_id": args.get("source_account_id"),
            "target_id": args.get("target_id"),
            "amount_krw": args.get("amount_krw"),
            "status": status,
            "operation": op,
        },
    )


def _request_interest_payment(state: KakaoBankState, args: dict[str, Any]) -> None:
    target_id = args.get("target_id")
    if not target_id:
        return
    options = args.get("options") or {}
    txn_id = f"interest::{target_id}::{options.get('interest_amount_krw', '')}"
    _upsert(
        state,
        "transactions",
        txn_id,
        {
            "transaction_id": txn_id,
            "target_id": target_id,
            "transaction_type": "INTEREST_PAYMENT",
            "amount": options.get("interest_amount_krw"),
            "status": "POSTED",
        },
    )


def _create_loan_application(state: KakaoBankState, args: dict[str, Any]) -> None:
    app_id = args.get("application_id")
    if not app_id:
        return
    _upsert(
        state,
        "loan_applications",
        app_id,
        {
            "application_id": app_id,
            "customer_id": args.get("customer_id"),
            "product_name": args.get("product_name"),
            "requested_amount_krw": args.get("requested_amount_krw"),
            "purpose": args.get("purpose"),
            "partner_id": args.get("partner_id"),
            "comparison_id": args.get("comparison_id"),
            "status": args.get("expected_status") or args.get("status", "SUBMITTED"),
        },
    )


def _file_dispute_or_objection(state: KakaoBankState, args: dict[str, Any]) -> None:
    customer_id = args.get("customer_id")
    target_id = args.get("target_id")
    if not (customer_id and target_id):
        return
    dispute_id = args.get("dispute_id") or f"dispute::{customer_id}::{target_id}"
    _upsert(
        state,
        "disputes",
        dispute_id,
        {
            "dispute_id": dispute_id,
            "customer_id": customer_id,
            "target_type": args.get("target_type"),
            "target_id": target_id,
            "reason": args.get("reason"),
            "status": args.get("status", "FILED"),
        },
    )


HANDLERS: dict[str, Callable[[KakaoBankState, dict[str, Any]], None]] = {
    # Read tools: state-invariant by construction.
    "KB_search": _noop,
    "get_customer_profile": _noop,
    "get_account_or_contract": _noop,
    # Write tools.
    "close_account_or_service": _close_account_or_service,
    "open_or_enroll_product": _open_or_enroll_product,
    "update_card_state": _update_card_state,
    "update_loan_contract_state": _update_loan_contract_state,
    "process_refinance_request": _process_refinance_request,
    "request_maturity_or_extension": _request_maturity_or_extension,
    "execute_remittance_case": _execute_remittance_case,
    "execute_deposit_or_box_transfer": _execute_deposit_or_box_transfer,
    "configure_auto_transfer": _configure_auto_transfer,
    "request_interest_payment": _request_interest_payment,
    "create_loan_application": _create_loan_application,
    "file_dispute_or_objection": _file_dispute_or_objection,
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
