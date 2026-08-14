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

from dataclasses import dataclass

from verl.base_config import BaseConfig


@dataclass
class OpenVLAAdapterConfig(BaseConfig):
    """Runtime inputs and action convention for the native OpenVLA policy."""

    image_key: str = "observation.images.image"
    task_key: str = "task"
    unnorm_key: str = "libero_spatial_no_noops"
    norm_stats_path: str | None = None
    prompt_template: str = "In: What action should the robot take to {task}?\nOut:"
    action_transform: str = "libero"

    def __post_init__(self) -> None:
        if self.action_transform not in {"identity", "libero"}:
            raise ValueError(f"Unsupported OpenVLA action_transform: {self.action_transform}")


__all__ = ["OpenVLAAdapterConfig"]
