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

"""Task JSON loader for kakaobank_manual_v0 tasks.

Filters to the subset whose binary reward is decidable by canonical DB-hash
equality after replaying ``expected_actions`` against ``initial_state``.
"""

import json
from pathlib import Path
from typing import Any, Optional, TypedDict


class KakaoBankTask(TypedDict):
    task_id: str
    product_names: list[str]
    task_type: str
    user_prompt: str
    initial_state: dict[str, Any]
    expected_actions: list[dict[str, Any]]
    required_document_ids: list[str]


def _has_user_requestor(expected_actions: list[dict[str, Any]]) -> bool:
    return any(a.get("requestor") == "user" for a in expected_actions)


def is_db_decidable(task: dict[str, Any]) -> bool:
    """A task is DB-decidable iff its reward_basis includes ``DB`` and its
    expected_actions can be replayed without a user-side tool. The second
    condition excludes the 4 documented user-tool-only tasks whose semantics
    depend on a separate user simulator.
    """
    reward_basis = task.get("reward_basis", [])
    if "DB" not in reward_basis:
        return False
    if _has_user_requestor(task.get("expected_actions", [])):
        return False
    return True


def load_task(path: Path) -> KakaoBankTask:
    """Load and normalize a single task JSON."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "task_id": raw["task_id"],
        "product_names": raw.get("product_names", []),
        "task_type": raw.get("task_type", ""),
        "user_prompt": raw["user_prompt"],
        "initial_state": raw.get("initial_state", {}),
        "expected_actions": raw.get("expected_actions", []),
        "required_document_ids": list(raw.get("required_documents", [])),
    }


def load_tasks(
    tasks_dir: Path,
    *,
    db_decidable_only: bool = True,
    product_names: Optional[list[str]] = None,
) -> list[KakaoBankTask]:
    """Load every ``kb_manual_*.json`` task from ``tasks_dir``.

    When ``db_decidable_only`` is true (default), tasks whose reward cannot
    be expressed as a DB-hash equality are dropped. Optional ``product_names``
    restricts to tasks whose ``product_names`` intersect the provided set.
    """
    tasks: list[KakaoBankTask] = []
    for path in sorted(Path(tasks_dir).glob("kb_manual_*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if db_decidable_only and not is_db_decidable(raw):
            continue
        if product_names is not None and not (
            set(raw.get("product_names", [])) & set(product_names)
        ):
            continue
        tasks.append(load_task(path))
    return tasks
