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

"""Spec (via tests) for close_account_or_service and open_or_enroll_product.

close_account_or_service(args):
  - locates ``args['target_id']`` across the 7 closable tables
    (accounts, deposit_contracts, savings_boxes, group_memberships,
     pockets, service_enrollments, prepaid_wallets) in that order
  - sets the record's ``status`` to "CLOSED"
  - records ``close_type`` and ``close_reason`` when provided in args
  - is a no-op when target_id is missing from every closable table
  - is idempotent: applying the same close twice yields the same state

open_or_enroll_product(args):
  - creates a record in ``args['target_table']`` keyed by ``args['target_id']``
  - the record carries ``customer_id``, ``product_name``, and ``status``
    (default "ACTIVE") plus any additional key/value pairs in
    ``args['options']``
  - is a no-op when either target_table or target_id is missing
  - merges additional fields into an existing record rather than
    overwriting unrelated keys
"""

import copy

from nemo_rl.environments.banking.state import canonical_hash, empty_state, seed_state
from nemo_rl.environments.banking.tools import apply_action


def test_close_sets_status_on_accounts():
    state = seed_state({"accounts": {"data": {"a1": {"status": "ACTIVE", "balance_krw": 0}}}})
    apply_action(
        state,
        {
            "name": "close_account_or_service",
            "arguments": {
                "customer_id": "c1",
                "target_id": "a1",
                "close_type": "CLOSE_DEMAND_DEPOSIT_ACCOUNT",
                "reason": "NO_LINKS",
            },
        },
    )
    rec = state["accounts"]["data"]["a1"]
    assert rec["status"] == "CLOSED"
    assert rec["close_type"] == "CLOSE_DEMAND_DEPOSIT_ACCOUNT"
    assert rec["close_reason"] == "NO_LINKS"
    assert rec["balance_krw"] == 0  # unrelated field preserved


def test_close_finds_target_across_tables():
    state = seed_state(
        {
            "deposit_contracts": {"data": {"dep_1": {"status": "ACTIVE"}}},
            "savings_boxes": {"data": {"box_1": {"status": "ACTIVE"}}},
        }
    )
    apply_action(
        state,
        {"name": "close_account_or_service", "arguments": {"target_id": "dep_1"}},
    )
    apply_action(
        state,
        {"name": "close_account_or_service", "arguments": {"target_id": "box_1"}},
    )
    assert state["deposit_contracts"]["data"]["dep_1"]["status"] == "CLOSED"
    assert state["savings_boxes"]["data"]["box_1"]["status"] == "CLOSED"


def test_close_missing_target_is_noop():
    state = empty_state()
    before = canonical_hash(state)
    apply_action(
        state,
        {"name": "close_account_or_service", "arguments": {"target_id": "ghost"}},
    )
    assert canonical_hash(state) == before


def test_close_is_idempotent():
    state = seed_state({"accounts": {"data": {"a1": {"status": "ACTIVE"}}}})
    apply_action(
        state,
        {
            "name": "close_account_or_service",
            "arguments": {"target_id": "a1", "close_type": "X", "reason": "R"},
        },
    )
    after_first = canonical_hash(copy.deepcopy(state))
    apply_action(
        state,
        {
            "name": "close_account_or_service",
            "arguments": {"target_id": "a1", "close_type": "X", "reason": "R"},
        },
    )
    assert canonical_hash(state) == after_first


def test_open_creates_record_with_default_status():
    state = empty_state()
    apply_action(
        state,
        {
            "name": "open_or_enroll_product",
            "arguments": {
                "customer_id": "c1",
                "product_name": "입출금통장",
                "target_table": "accounts",
                "target_id": "acct_new",
            },
        },
    )
    rec = state["accounts"]["data"]["acct_new"]
    assert rec["customer_id"] == "c1"
    assert rec["product_name"] == "입출금통장"
    assert rec["status"] == "ACTIVE"


def test_open_merges_options_into_record():
    state = empty_state()
    apply_action(
        state,
        {
            "name": "open_or_enroll_product",
            "arguments": {
                "customer_id": "c2",
                "product_name": "세이프박스",
                "target_table": "savings_boxes",
                "target_id": "box_99",
                "options": {"limit_amount": 10_000_000, "currency": "KRW"},
            },
        },
    )
    rec = state["savings_boxes"]["data"]["box_99"]
    assert rec["limit_amount"] == 10_000_000
    assert rec["currency"] == "KRW"


def test_open_without_target_identifiers_is_noop():
    state = empty_state()
    before = canonical_hash(state)
    apply_action(state, {"name": "open_or_enroll_product", "arguments": {}})
    assert canonical_hash(state) == before


def test_open_merges_into_existing_record_without_dropping_other_fields():
    state = seed_state({"accounts": {"data": {"a9": {"legacy_field": "keep"}}}})
    apply_action(
        state,
        {
            "name": "open_or_enroll_product",
            "arguments": {
                "customer_id": "c3",
                "product_name": "입출금통장",
                "target_table": "accounts",
                "target_id": "a9",
                "options": {"balance_krw": 0},
            },
        },
    )
    rec = state["accounts"]["data"]["a9"]
    assert rec["legacy_field"] == "keep"
    assert rec["balance_krw"] == 0
    assert rec["product_name"] == "입출금통장"
