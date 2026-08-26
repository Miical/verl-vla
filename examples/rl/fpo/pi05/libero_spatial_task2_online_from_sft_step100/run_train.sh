#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
OUTPUT_DIR="${FPO_OUTPUT_DIR:-${REPO_ROOT}/outputs/rl/fpo/pi05/libero-spatial-task2-online-from-sft-step100}"

cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec vvla-train-fpo \
  --config-dir "$SCRIPT_DIR" \
  --config-name fpo \
  output_dir="$OUTPUT_DIR" \
  cluster.actor_rollout_ref.model.path=Miical/pi05-libero-spatial-sft-step-100 \
  cluster.resource.model.gpus_per_node=8 \
  cluster.resource.model.workers_per_node=8 \
  cluster.resource.env.device=cpu \
  cluster.resource.env.workers_per_node=4 \
  cluster.env.env_worker.num_envs=8 \
  ray_kwargs.ray_init.runtime_env.env_vars.MUJOCO_GL=osmesa \
  ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="$PYTHONPATH" \
  ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR="$OUTPUT_DIR/tensorboard" \
  "$@"
