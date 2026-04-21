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

"""Ray-actor wrapper around :class:`BankingRunner`."""

from typing import Optional

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.banking.runner import BankingMetadata, BankingRunner
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)


@ray.remote  # pragma: no cover
class BankingEnvironment(EnvironmentInterface[BankingMetadata]):
    """Batched Ray-actor wrapper around :class:`BankingRunner`."""

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}
        self.runner = BankingRunner()

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[BankingMetadata],
    ) -> EnvironmentReturn[BankingMetadata]:
        results = [
            self.runner.process_turn(log, meta)
            for log, meta in zip(message_log_batch, metadata)
        ]
        observations: list[dict[str, str]] = []
        rewards: list[float] = []
        terminateds: list[bool] = []
        stops: list[list[str] | None] = []
        next_metadata: list[Optional[BankingMetadata]] = []
        answers: list[Optional[str]] = []
        for obs, rew, term, stop, nm, ans in results:
            observations.append(obs)
            rewards.append(rew)
            terminateds.append(term)
            stops.append(stop)
            next_metadata.append(nm)
            answers.append(ans)
        return EnvironmentReturn(
            observations=observations,
            metadata=next_metadata,
            next_stop_strings=stops,
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.tensor(terminateds, dtype=torch.bool),
            answers=answers,
        )

    def shutdown(self):
        pass

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        total_reward = batch.get(
            "total_reward", torch.tensor([0.0] * len(batch.get("idx", [])))
        )
        if len(total_reward) > 0:
            success_rate = (total_reward == 1.0).float().mean().item()
        else:
            success_rate = 0.0
        return batch, {"banking_success_rate": success_rate}
