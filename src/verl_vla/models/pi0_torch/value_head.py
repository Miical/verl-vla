# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from __future__ import annotations

import torch
from torch import nn

from verl_vla.utils.models.mlp import MLP


class PI0FPOValueHead(nn.Module):
    """Vanilla-FPO state-value head over frozen PI0 observation features."""

    def __init__(self, input_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        self.mlp = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation="relu",
            init_method="kaiming",
        )

    @staticmethod
    def pool_state_features(state_features) -> torch.Tensor:
        prefix_features, states = state_features
        prefix_embs, prefix_pad_masks, _ = prefix_features
        mask = prefix_pad_masks.to(dtype=prefix_embs.dtype).unsqueeze(-1)
        pooled_prefix = (prefix_embs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return torch.cat([pooled_prefix, states], dim=-1)

    def forward(self, state_features) -> torch.Tensor:
        return self.mlp(self.pool_state_features(state_features)).squeeze(-1)


__all__ = ["PI0FPOValueHead"]
