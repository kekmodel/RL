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

"""KakaoBank mutable state model and canonical hasher.

The state is a nested ``dict`` keyed by table name. Each table has a ``data``
sub-dict keyed by record id, mirroring the shape used by tau2-style task
initialization data. The canonical hash is a SHA-256 of the JSON encoding
with sorted keys, which is stable across runs and independent of insertion
order.
"""

import copy
import hashlib
import json
from typing import Any

# All 26 tables declared by the kakaobank_action_verifier_state.v0 schema.
# Tables absent from a task's ``initial_state`` are seeded as empty.
KAKAOBANK_TABLES: tuple[str, ...] = (
    "customers",
    "businesses",
    "consents",
    "accounts",
    "deposit_contracts",
    "savings_boxes",
    "auto_transfer_rules",
    "group_memberships",
    "pockets",
    "child_relationships",
    "cards",
    "card_orders",
    "prepaid_wallets",
    "loans",
    "loan_applications",
    "refinance_requests",
    "required_documents",
    "mortgage_collateral",
    "lease_contracts",
    "vehicle_purchase_cases",
    "comparison_sessions",
    "remittance_profiles",
    "remittance_cases",
    "service_enrollments",
    "transactions",
    "disputes",
)


KakaoBankState = dict[str, dict[str, dict[str, Any]]]


def empty_state() -> KakaoBankState:
    """Return a fresh KakaoBank state with every table present but empty."""
    return {table: {"data": {}} for table in KAKAOBANK_TABLES}


def seed_state(initial_state: dict[str, Any]) -> KakaoBankState:
    """Build a KakaoBank state by overlaying a task's ``initial_state`` onto
    an empty base. Unknown tables are accepted verbatim so that future schema
    extensions do not break replay.
    """
    state = empty_state()
    for table_name, table_body in initial_state.items():
        if not isinstance(table_body, dict):
            continue
        records = table_body.get("data", {})
        if not isinstance(records, dict):
            continue
        state.setdefault(table_name, {"data": {}})
        state[table_name]["data"] = copy.deepcopy(records)
    return state


def canonical_hash(state: KakaoBankState) -> str:
    """Return a stable SHA-256 hex digest of the state.

    The hash ignores insertion order (sort_keys=True) and separator whitespace.
    Any value that is JSON-serializable via ``default=str`` is permitted; the
    ``default=str`` fallback converts datetimes and other structured values
    consistently.
    """
    encoded = json.dumps(
        state, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
