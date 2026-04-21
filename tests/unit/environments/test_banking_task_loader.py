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

"""Spec (via tests) for nemo_rl.environments.banking.task_loader.

Contract:
  is_db_decidable(raw_task):
    - True iff 'DB' in reward_basis AND no expected_actions entry has
      requestor == 'user'
    - False if 'DB' is absent, or if any expected action is user-requested

  load_task(path):
    - reads JSON and returns a KakaoBankTask dict with keys
      {task_id, product_names, task_type, user_prompt, initial_state,
       expected_actions, required_document_ids}
    - required_document_ids is sourced from the task's required_documents
      list, never None

  load_tasks(dir, db_decidable_only=True, product_names=None):
    - enumerates only files matching kb_manual_*.json
    - when db_decidable_only=True, filters via is_db_decidable on the raw JSON
    - when product_names is provided, keeps tasks whose product_names intersect
    - returns tasks in sorted-by-filename order
"""

import json
from pathlib import Path

from nemo_rl.environments.banking.task_loader import (
    is_db_decidable,
    load_task,
    load_tasks,
)


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _minimal_task(task_id: str, **overrides) -> dict:
    base = {
        "task_id": task_id,
        "product_names": ["입출금통장"],
        "task_type": "success",
        "user_prompt": "hi",
        "initial_state": {"accounts": {"data": {"a1": {"status": "ACTIVE"}}}},
        "expected_actions": [
            {"name": "KB_search", "requestor": "assistant", "arguments": {"query": "q"}}
        ],
        "reward_basis": ["DB", "ACTION", "COMMUNICATE"],
        "required_documents": ["doc_1"],
    }
    base.update(overrides)
    return base


def test_is_db_decidable_requires_db_reward_basis():
    assert is_db_decidable(_minimal_task("t1"))
    no_db = _minimal_task("t1", reward_basis=["ACTION", "COMMUNICATE"])
    assert not is_db_decidable(no_db)


def test_is_db_decidable_excludes_user_requestor_actions():
    user_tool = _minimal_task(
        "t2",
        expected_actions=[
            {"name": "KB_search", "requestor": "assistant", "arguments": {}},
            {"name": "submit_required_document", "requestor": "user", "arguments": {}},
        ],
    )
    assert not is_db_decidable(user_tool)


def test_load_task_normalizes_fields(tmp_path):
    path = tmp_path / "kb_manual_demo.json"
    _write(path, _minimal_task("t3"))
    task = load_task(path)
    assert task["task_id"] == "t3"
    assert task["user_prompt"] == "hi"
    assert task["required_document_ids"] == ["doc_1"]
    assert task["expected_actions"][0]["name"] == "KB_search"
    assert task["initial_state"]["accounts"]["data"]["a1"]["status"] == "ACTIVE"


def test_load_tasks_filters_db_decidable_by_default(tmp_path):
    _write(tmp_path / "kb_manual_a.json", _minimal_task("a"))
    _write(
        tmp_path / "kb_manual_b.json",
        _minimal_task("b", reward_basis=["ACTION", "COMMUNICATE"]),
    )
    _write(
        tmp_path / "kb_manual_c.json",
        _minimal_task(
            "c",
            expected_actions=[
                {"name": "submit_required_document", "requestor": "user"}
            ],
        ),
    )
    loaded = load_tasks(tmp_path)
    assert [t["task_id"] for t in loaded] == ["a"]


def test_load_tasks_can_include_all_when_decidable_filter_off(tmp_path):
    _write(tmp_path / "kb_manual_a.json", _minimal_task("a"))
    _write(
        tmp_path / "kb_manual_b.json",
        _minimal_task("b", reward_basis=["ACTION", "COMMUNICATE"]),
    )
    loaded = load_tasks(tmp_path, db_decidable_only=False)
    assert {t["task_id"] for t in loaded} == {"a", "b"}


def test_load_tasks_filters_by_product_name(tmp_path):
    _write(
        tmp_path / "kb_manual_a.json",
        _minimal_task("a", product_names=["입출금통장"]),
    )
    _write(
        tmp_path / "kb_manual_b.json",
        _minimal_task("b", product_names=["mini카드"]),
    )
    mini_only = load_tasks(tmp_path, product_names=["mini카드"])
    assert [t["task_id"] for t in mini_only] == ["b"]


def test_load_tasks_ignores_non_kb_manual_files(tmp_path):
    _write(tmp_path / "kb_manual_a.json", _minimal_task("a"))
    _write(tmp_path / "other_task.json", _minimal_task("other"))
    loaded = load_tasks(tmp_path)
    assert [t["task_id"] for t in loaded] == ["a"]


def test_load_tasks_sorts_by_filename(tmp_path):
    _write(tmp_path / "kb_manual_b.json", _minimal_task("b"))
    _write(tmp_path / "kb_manual_a.json", _minimal_task("a"))
    loaded = load_tasks(tmp_path)
    assert [t["task_id"] for t in loaded] == ["a", "b"]
