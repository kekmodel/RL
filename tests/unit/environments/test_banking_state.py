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

"""Spec (via tests) for nemo_rl.environments.banking.state.

Contract:
  empty_state():
    - returns a dict with exactly the 26 KAKAOBANK_TABLES as keys
    - each value is {"data": {}}

  seed_state(initial):
    - starts from an empty_state base
    - overlays tables present in initial (keeping the {"data": {...}} shape)
    - leaves tables absent from initial as empty {"data": {}}
    - deep-copies record bodies so later mutation of the return value does not
      affect the input

  canonical_hash(state):
    - is a deterministic SHA-256 hex digest of the state
    - is independent of dict insertion order (sort_keys=True)
    - differs when any record field changes
    - tolerates non-JSON-native scalars (e.g. dates) via default=str
"""

import copy

import pytest

from nemo_rl.environments.banking.state import (
    KAKAOBANK_TABLES,
    canonical_hash,
    empty_state,
    seed_state,
)


def test_empty_state_has_all_24_tables():
    state = empty_state()
    assert set(state.keys()) == set(KAKAOBANK_TABLES)
    assert len(KAKAOBANK_TABLES) == 26  # see note below
    for table in KAKAOBANK_TABLES:
        assert state[table] == {"data": {}}


def test_seed_state_overlays_records_and_leaves_others_empty():
    initial = {
        "customers": {
            "data": {"c1": {"name": "A", "age": 30}},
        },
        "accounts": {
            "data": {"a1": {"status": "ACTIVE"}},
        },
    }
    state = seed_state(initial)
    assert state["customers"]["data"] == {"c1": {"name": "A", "age": 30}}
    assert state["accounts"]["data"] == {"a1": {"status": "ACTIVE"}}
    assert state["cards"] == {"data": {}}
    assert state["loans"] == {"data": {}}


def test_seed_state_deep_copies_records():
    initial = {"customers": {"data": {"c1": {"tags": ["k"]}}}}
    state = seed_state(initial)
    state["customers"]["data"]["c1"]["tags"].append("mutated")
    assert initial["customers"]["data"]["c1"]["tags"] == ["k"]


def test_canonical_hash_is_stable_under_insertion_order():
    a = empty_state()
    b = empty_state()
    a["customers"]["data"]["c1"] = {"name": "X", "age": 30}
    # insert the same logical content but populate field keys in a different
    # order on the b side
    b["customers"]["data"]["c1"] = {}
    b["customers"]["data"]["c1"]["age"] = 30
    b["customers"]["data"]["c1"]["name"] = "X"
    assert canonical_hash(a) == canonical_hash(b)


def test_canonical_hash_differs_on_field_change():
    a = empty_state()
    b = empty_state()
    a["accounts"]["data"]["acct_1"] = {"status": "ACTIVE", "balance_krw": 100}
    b["accounts"]["data"]["acct_1"] = {"status": "CLOSED", "balance_krw": 100}
    assert canonical_hash(a) != canonical_hash(b)


def test_canonical_hash_is_deterministic_across_calls():
    state = empty_state()
    state["customers"]["data"]["c1"] = {"name": "Y"}
    h1 = canonical_hash(state)
    h2 = canonical_hash(copy.deepcopy(state))
    assert h1 == h2


def test_canonical_hash_tolerates_non_json_scalars():
    import datetime as dt

    a = empty_state()
    a["loans"]["data"]["l1"] = {"maturity_date": dt.date(2026, 4, 20)}
    # Must not raise; and must differ from the empty state's hash.
    assert canonical_hash(a) != canonical_hash(empty_state())
