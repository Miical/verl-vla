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

from types import SimpleNamespace

import numpy as np
import torch

from verl_vla.models.openvla_oft.trainable_model import OpenVLATrainableModel, _load_action_statistics


class _Policy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace()
        self.bins = np.linspace(-1.0, 1.0, 256)
        self.norm_stats = {
            "libero_spatial_no_noops": {
                "action": {
                    "q01": [0.0] * 7,
                    "q99": [10.0] * 6 + [1.0],
                    "mask": [True] * 6 + [False],
                }
            }
        }

    def get_action_stats(self, unnorm_key: str) -> dict:
        return self.norm_stats[unnorm_key]["action"]


def test_libero_actions_follow_openvla_token_convention() -> None:
    model = OpenVLATrainableModel(_Policy(), processor=None)
    actions = torch.tensor(
        [
            [[0.0] * 6 + [1.0]],
            [[10.0] * 6 + [-1.0]],
        ]
    )

    normalized = model._normalize_sft_actions(actions)
    classes = model._action_classes(normalized)

    torch.testing.assert_close(normalized[0, 0], torch.tensor([-1.0] * 6 + [0.0]))
    torch.testing.assert_close(normalized[1, 0], torch.tensor([1.0] * 7))
    assert torch.equal(classes[0, 0, :6], torch.full((6,), 255))
    assert torch.equal(classes[1, 0], torch.zeros(7, dtype=torch.long))


def test_libero_rollout_uses_robosuite_gripper_convention() -> None:
    model = OpenVLATrainableModel(_Policy(), processor=None)
    actions = np.zeros((2, 8, 7))
    actions[:, 1::2, -1] = 1.0

    transformed = model._environment_actions(actions, batch_size=2)

    assert transformed.shape == (2, 8, 7)
    torch.testing.assert_close(transformed[0, :, -1], torch.tensor([1.0, -1.0] * 4))


def test_lerobot_statistics_are_converted_to_native_libero_actions(tmp_path) -> None:
    statistics_path = tmp_path / "norm_stats.json"
    statistics_path.write_text(
        """{
            "action": {
                "min": [-1, -1, -1, -1, -1, -1, -1],
                "max": [1, 1, 1, 1, 1, 1, 1],
                "mean": [0, 0, 0, 0, 0, 0, 0.2],
                "std": [1, 1, 1, 1, 1, 1, 0.8],
                "q01": [-0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.8],
                "q99": [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.6]
            }
        }""",
        encoding="utf-8",
    )

    statistics = _load_action_statistics(statistics_path, action_transform="libero")

    assert statistics["min"][-1] == 0.0
    assert statistics["max"][-1] == 1.0
    assert statistics["mean"][-1] == 0.4
    assert statistics["std"][-1] == 0.4
    assert statistics["q01"][-1] == 0.2
    assert statistics["q99"][-1] == 0.9
    assert statistics["mask"] == [True, True, True, True, True, True, False]
