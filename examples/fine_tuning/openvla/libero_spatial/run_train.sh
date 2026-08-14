#!/usr/bin/env bash
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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/.data/libero_spatial_image}"
NORM_STATS_PATH="${NORM_STATS_PATH:-${DATASET_ROOT}/norm_stats.json}"
OPENVLA_MODEL="${OPENVLA_MODEL:-openvla/openvla-7b}"

cd "$REPO_ROOT"

if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
  hf download lerobot/libero_spatial_image \
    --repo-type dataset \
    --local-dir "$DATASET_ROOT"
fi

if [[ ! -f "$NORM_STATS_PATH" ]]; then
  python scripts/compute_norm_stats.py \
    --repo-id lerobot/libero_spatial_image \
    --root "$DATASET_ROOT" \
    --output-path "$NORM_STATS_PATH"
fi

vvla-train-sft \
  --config-dir "$SCRIPT_DIR" \
  --config-name openvla_sft \
  cluster.actor_rollout_ref.model.path="$OPENVLA_MODEL" \
  cluster.actor_rollout_ref.model.adapter.norm_stats_path="$NORM_STATS_PATH" \
  data.root="$DATASET_ROOT" \
  "$@"
