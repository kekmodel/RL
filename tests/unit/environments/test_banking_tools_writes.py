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

"""Spec (via tests) for the 10 remaining write-tool handlers used by the
DB-decidable kakaobank_manual_v0 task subset.

Each handler must be deterministic in ``(state, args)``; the DB-hash reward
depends on it. The checks below assert the primary record a handler writes
(table, key, status field) using argument shapes observed in real tasks.
Two shared invariants are asserted for every tool: the handler is a no-op
when its primary id arg is missing, and applying the same action twice
leaves the state identical.
"""

import copy

import pytest

from nemo_rl.environments.banking.state import canonical_hash, empty_state, seed_state
from nemo_rl.environments.banking.tools import apply_action


# ---- update_card_state --------------------------------------------------


def test_update_card_state_rejects_new_issue_via_card_order():
    state = empty_state()
    apply_action(
        state,
        {
            "name": "update_card_state",
            "arguments": {
                "card_id": None,
                "operation": "REJECT_NEW_ISSUE",
                "reason": "ADULT_NEW_ISSUE_NOT_ALLOWED",
                "order_id": "card_order_062",
                "customer_id": "cust_062",
            },
        },
    )
    rec = state["card_orders"]["data"]["card_order_062"]
    assert rec["status"] == "REJECTED"
    assert rec["reason"] == "ADULT_NEW_ISSUE_NOT_ALLOWED"
    assert rec["customer_id"] == "cust_062"


def test_update_card_state_reissue_records_approved_order():
    state = empty_state()
    apply_action(
        state,
        {
            "name": "update_card_state",
            "arguments": {
                "card_id": "mini_card_001",
                "operation": "REISSUE_CARD",
                "reason": "DAMAGED_CARD",
                "order_id": "card_order_001",
            },
        },
    )
    assert state["card_orders"]["data"]["card_order_001"]["status"] == "APPROVED"


# ---- update_loan_contract_state ----------------------------------------


def test_update_loan_contract_state_acceleration():
    state = seed_state({"loans": {"data": {"biz_credit_loan_056": {"status": "ACTIVE"}}}})
    apply_action(
        state,
        {
            "name": "update_loan_contract_state",
            "arguments": {
                "loan_id": "biz_credit_loan_056",
                "operation": "ACCELERATE_IMMEDIATE_REPAYMENT",
                "reason": "POST_USE_INSPECTION_STATEMENT_NOT_SUBMITTED_WITHIN_3_MONTHS",
                "effective_at": "2026-04-20T15:10:00+09:00",
            },
        },
    )
    rec = state["loans"]["data"]["biz_credit_loan_056"]
    assert rec["status"] == "ACCELERATED"
    assert rec["acceleration_reason"].startswith("POST_USE")


# ---- process_refinance_request -----------------------------------------


def test_process_refinance_request_cancel_on_old_repayment_failure():
    state = empty_state()
    apply_action(
        state,
        {
            "name": "process_refinance_request",
            "arguments": {
                "refinance_id": "biz_refi_001",
                "operation": "CANCEL_NEW_LOAN_AFTER_OLD_REPAYMENT_FAILURE",
                "old_loan_repayment_status": "FAILED",
            },
        },
    )
    rec = state["refinance_requests"]["data"]["biz_refi_001"]
    assert rec["status"] == "CANCELLED"
    assert rec["old_loan_repayment_status"] == "FAILED"


def test_process_refinance_request_complete():
    state = empty_state()
    apply_action(
        state,
        {
            "name": "process_refinance_request",
            "arguments": {
                "refinance_id": "biz_refi_002",
                "operation": "COMPLETE_OLD_LOAN_REPAYMENT_AND_ACTIVATE_NEW_LOAN",
                "old_loan_repayment_status": "SUCCEEDED",
            },
        },
    )
    assert state["refinance_requests"]["data"]["biz_refi_002"]["status"] == "COMPLETED"


# ---- request_maturity_or_extension -------------------------------------


def test_request_maturity_or_extension_matures_deposit():
    state = seed_state({"deposit_contracts": {"data": {"dep_26w_082": {"status": "ACTIVE"}}}})
    apply_action(
        state,
        {
            "name": "request_maturity_or_extension",
            "arguments": {
                "target_id": "dep_26w_082",
                "operation": "MATURE_HOLIDAY_PREVIOUS_BUSINESS_DAY_CLOSE",
                "options": {"close_type": "MATURITY_HOLIDAY_PREVIOUS_BUSINESS_DAY"},
            },
        },
    )
    rec = state["deposit_contracts"]["data"]["dep_26w_082"]
    assert rec["status"] == "CLOSED"
    assert rec["close_type"] == "MATURITY_HOLIDAY_PREVIOUS_BUSINESS_DAY"


def test_request_maturity_or_extension_rejects_auto_close():
    state = seed_state({"deposit_contracts": {"data": {"dep_26w_081": {"status": "ACTIVE"}}}})
    apply_action(
        state,
        {
            "name": "request_maturity_or_extension",
            "arguments": {
                "target_id": "dep_26w_081",
                "operation": "REJECT_AUTO_CLOSE",
                "options": {"reason": "LEGAL_RESTRICTION_AUTO_CLOSE_BLOCKED"},
            },
        },
    )
    rec = state["deposit_contracts"]["data"]["dep_26w_081"]
    assert rec["status"] == "ACTIVE"  # unchanged because auto-close was rejected
    assert rec["maturity_decision"] == "REJECT_AUTO_CLOSE"


# ---- execute_remittance_case -------------------------------------------


def test_execute_remittance_case_creates_case_record():
    state = empty_state()
    apply_action(
        state,
        {
            "name": "execute_remittance_case",
            "arguments": {
                "customer_id": "cust_016",
                "direction": "DOLLARBOX_GIFT_AUTO_CANCEL",
                "amount": 320,
                "currency": "USD",
                "country": "KR",
                "options": {"remittance_id": "remit_gift_016"},
            },
        },
    )
    rec = state["remittance_cases"]["data"]["remit_gift_016"]
    assert rec["customer_id"] == "cust_016"
    assert rec["amount"] == 320
    assert rec["currency"] == "USD"
    assert rec["direction"] == "DOLLARBOX_GIFT_AUTO_CANCEL"


def test_execute_remittance_case_uses_top_level_remittance_id_when_present():
    state = empty_state()
    apply_action(
        state,
        {
            "name": "execute_remittance_case",
            "arguments": {
                "customer_id": "c1",
                "direction": "OUTBOUND",
                "amount": 1000,
                "currency": "KRW",
                "remittance_id": "remit_alpha",
            },
        },
    )
    assert "remit_alpha" in state["remittance_cases"]["data"]


# ---- execute_deposit_or_box_transfer -----------------------------------


def test_execute_deposit_or_box_transfer_records_transaction():
    state = empty_state()
    apply_action(
        state,
        {
            "name": "execute_deposit_or_box_transfer",
            "arguments": {
                "source_account_id": "acct_base_003",
                "target_id": "contract_26w_001",
                "amount": 8000,
                "currency": "KRW",
                "transfer_type": "MISSED_WEEK_FILL",
            },
        },
    )
    assert len(state["transactions"]["data"]) == 1
    rec = next(iter(state["transactions"]["data"].values()))
    assert rec["amount"] == 8000
    assert rec["target_id"] == "contract_26w_001"


def test_execute_deposit_or_box_transfer_rejection_records_rejected_status():
    state = empty_state()
    apply_action(
        state,
        {
            "name": "execute_deposit_or_box_transfer",
            "arguments": {
                "source_id": "dep_26w_072",
                "target_id": "acct_base_072",
                "amount": 120000,
                "currency": "KRW",
                "transaction_type": "REJECT_26W_EMERGENCY_WITHDRAWAL",
                "reason": "EMERGENCY_WITHDRAWAL_REMAINING_BALANCE_BELOW_100000",
            },
        },
    )
    rec = next(iter(state["transactions"]["data"].values()))
    assert rec["status"] == "REJECTED"


# ---- configure_auto_transfer --------------------------------------------


def test_configure_auto_transfer_create_uses_options_id():
    state = empty_state()
    apply_action(
        state,
        {
            "name": "configure_auto_transfer",
            "arguments": {
                "source_account_id": "acct_piggy_base_226",
                "target_id": "box_piggy_226",
                "amount_krw": None,
                "operation": "CREATE",
                "options": {"auto_transfer_id": "auto_piggy_226"},
            },
        },
    )
    assert state["auto_transfer_rules"]["data"]["auto_piggy_226"]["status"] == "ACTIVE"


def test_configure_auto_transfer_reject_status():
    state = empty_state()
    apply_action(
        state,
        {
            "name": "configure_auto_transfer",
            "arguments": {
                "source_account_id": "acct_base_084",
                "target_id": "dep_26w_084",
                "amount_krw": 10000,
                "operation": "REJECT_RE_REGISTER_AFTER_26W_SERVICE_CANCEL",
                "existing_auto_transfer_id": "auto_rule_084",
            },
        },
    )
    assert state["auto_transfer_rules"]["data"]["auto_rule_084"]["status"] == "REJECTED"


# ---- request_interest_payment -------------------------------------------


def test_request_interest_payment_creates_interest_transaction():
    state = empty_state()
    apply_action(
        state,
        {
            "name": "request_interest_payment",
            "arguments": {
                "target_id": "box_safe_151",
                "options": {
                    "interest_amount_krw": 5260,
                    "add_to_principal": True,
                },
            },
        },
    )
    rec = next(iter(state["transactions"]["data"].values()))
    assert rec["transaction_type"] == "INTEREST_PAYMENT"
    assert rec["target_id"] == "box_safe_151"


# ---- create_loan_application -------------------------------------------


def test_create_loan_application_records_application():
    state = empty_state()
    apply_action(
        state,
        {
            "name": "create_loan_application",
            "arguments": {
                "application_id": "biz_credit_app_054",
                "customer_id": "cust_054",
                "product_name": "개인사업자 신용대출",
                "requested_amount_krw": 45000000,
                "purpose": "WORKING_CAPITAL",
                "expected_status": "SUBMITTED_FOR_SCREENING",
            },
        },
    )
    rec = state["loan_applications"]["data"]["biz_credit_app_054"]
    assert rec["status"] == "SUBMITTED_FOR_SCREENING"
    assert rec["product_name"] == "개인사업자 신용대출"
    assert rec["requested_amount_krw"] == 45000000


# ---- file_dispute_or_objection ------------------------------------------


def test_file_dispute_or_objection_records_dispute():
    state = empty_state()
    apply_action(
        state,
        {
            "name": "file_dispute_or_objection",
            "arguments": {
                "customer_id": "cust_093",
                "target_type": "affiliate_service_content",
                "target_id": "svc_biz_credit_info_093",
                "reason": "NICE_CONTENT_VALIDITY_NOT_KAKAOBANK_GUARANTEED",
            },
        },
    )
    assert len(state["disputes"]["data"]) == 1
    rec = next(iter(state["disputes"]["data"].values()))
    assert rec["customer_id"] == "cust_093"
    assert rec["reason"] == "NICE_CONTENT_VALIDITY_NOT_KAKAOBANK_GUARANTEED"


# ---- Shared invariants --------------------------------------------------


_WRITE_HANDLERS_WITH_MINIMAL_ID_ARG = (
    ("update_card_state", {"order_id": "ord"}),
    ("update_loan_contract_state", {"loan_id": "l1"}),
    ("process_refinance_request", {"refinance_id": "r1"}),
    ("request_maturity_or_extension", {"target_id": "d1"}),
    ("execute_remittance_case", {"remittance_id": "m1", "customer_id": "c"}),
    ("execute_deposit_or_box_transfer", {"target_id": "t", "source_id": "s", "amount": 1}),
    ("configure_auto_transfer", {"options": {"auto_transfer_id": "a1"}}),
    ("request_interest_payment", {"target_id": "x"}),
    ("create_loan_application", {"application_id": "app1"}),
    ("file_dispute_or_objection", {"customer_id": "c", "target_id": "t"}),
)


@pytest.mark.parametrize("name,args", _WRITE_HANDLERS_WITH_MINIMAL_ID_ARG)
def test_write_handler_is_idempotent(name, args):
    state = empty_state()
    apply_action(state, {"name": name, "arguments": args})
    h1 = canonical_hash(state)
    apply_action(state, {"name": name, "arguments": copy.deepcopy(args)})
    assert canonical_hash(state) == h1


@pytest.mark.parametrize(
    "name",
    [
        "update_card_state",
        "update_loan_contract_state",
        "process_refinance_request",
        "request_maturity_or_extension",
        "execute_remittance_case",
        "configure_auto_transfer",
        "request_interest_payment",
        "create_loan_application",
        "file_dispute_or_objection",
    ],
)
def test_write_handler_is_noop_without_primary_id(name):
    state = empty_state()
    h0 = canonical_hash(state)
    apply_action(state, {"name": name, "arguments": {}})
    assert canonical_hash(state) == h0
