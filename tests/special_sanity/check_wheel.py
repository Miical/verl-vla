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

from importlib.metadata import distribution
from importlib.resources import files

from hydra import compose, initialize_config_module

EXPECTED_CONFIG_FILES = (
    "dagger",
    "eval",
    "record",
    "replay",
    "teleop",
    "train/fpo",
    "train/recap",
    "train/sac",
    "train/sft",
)
SELF_CONTAINED_CONFIGS = EXPECTED_CONFIG_FILES
EXPECTED_ENTRY_POINTS = {
    "vvla-dagger",
    "vvla-eval",
    "vvla-record",
    "vvla-replay",
    "vvla-teleop",
    "vvla-train-recap",
    "vvla-train-fpo",
    "vvla-train-sac",
    "vvla-train-sft",
}


def main() -> None:
    package_root = files("verl_vla")
    config_root = package_root / "workflows/config"
    for config_name in EXPECTED_CONFIG_FILES:
        assert (config_root / f"{config_name}.yaml").is_file(), f"wheel is missing {config_name}.yaml"

    actor_config_root = package_root / "workflows/config/model/adapter/dsrl/actor"
    for filename in ("cnn.yaml", "mlp.yaml", "transformer.yaml"):
        assert (actor_config_root / filename).is_file(), f"wheel is missing {filename}"

    entry_points = {entry_point.name for entry_point in distribution("verl-vla").entry_points}
    assert entry_points == EXPECTED_ENTRY_POINTS

    with initialize_config_module(version_base=None, config_module="verl_vla.workflows.config"):
        for config_name in SELF_CONTAINED_CONFIGS:
            compose(config_name=config_name)


if __name__ == "__main__":
    main()
