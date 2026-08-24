# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from pprint import pprint
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm
from verl import DataProto
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics

from verl_vla.train_cluster import TrainCluster
from verl_vla.utils.keys import OBS_KEY
from verl_vla.utils.rlpd import pad_dataproto_to_divisor_with_valid_mask

from .config import FPOTrainerConfig


def prepare_fpo_actor_input(
    rollout: DataProto,
    rollout_end_obs: DataProto,
    *,
    trainer_config: FPOTrainerConfig,
    global_steps: int,
) -> DataProto:
    """Convert an env-loop window into chunk-level vanilla-FPO transitions.

    The env loop returns ``S`` aligned observation/action/feedback slots plus
    the observation after the final action as a separate ``rollout_end_obs``.
    This function builds the matching next observations and folds each chunk's
    low-level feedback into the rewards, discounts, and masks consumed by the
    FPO worker.
    """
    obs_prefix = f"{OBS_KEY}."
    next_obs_prefix = "next_obs."
    rollout_steps = int(rollout.batch["action.action"].shape[1])
    batch_size = len(rollout)

    # Align every tensor observation with its exact successor:
    #   obs      = [s0, s1, ..., s(S-1)]
    #   next_obs = [s1, s2, ..., sS]
    # Intermediate successors come from the next rollout slot; only sS needs
    # the separate final observation returned by EnvLoop.
    for obs_key in [key for key in rollout.batch.keys() if key.startswith(obs_prefix)]:
        field = obs_key[len(obs_prefix) :]
        obs = rollout.batch[obs_key]
        end_obs = rollout_end_obs.batch[field]
        if obs.shape[:2] != (batch_size, rollout_steps) or end_obs.shape != (batch_size, *obs.shape[2:]):
            raise ValueError(
                f"FPO rollout observations must have shape [B, S, ...] and the final observation [B, ...], "
                f"got {obs_key}={tuple(obs.shape)} and {field}={tuple(end_obs.shape)}."
            )
        rollout.batch[f"{next_obs_prefix}{field}"] = torch.cat([obs[:, 1:], end_obs.unsqueeze(1)], dim=1)

    # Tasks, task ids, and other object-valued observations follow the same
    # temporal alignment as tensor observations.
    for obs_key in [key for key in rollout.non_tensor_batch if key.startswith(obs_prefix)]:
        field = obs_key[len(obs_prefix) :]
        obs = rollout.non_tensor_batch[obs_key]
        end_obs = rollout_end_obs.non_tensor_batch[field]
        if obs.shape[:2] != (batch_size, rollout_steps) or end_obs.shape != (batch_size, *obs.shape[2:]):
            raise ValueError(
                f"FPO rollout observations must have shape [B, S, ...] and the final observation [B, ...], "
                f"got {obs_key}={obs.shape} and {field}={end_obs.shape}."
            )
        rollout.non_tensor_batch[f"{next_obs_prefix}{field}"] = np.concatenate(
            [obs[:, 1:], np.expand_dims(end_obs, axis=1)],
            axis=1,
        )

    # Environment feedback is emitted per executed low-level action, whereas
    # FPO treats one policy action chunk as one semi-MDP transition.
    terminated_substeps = rollout.batch.pop("next.terminated").bool()
    truncated_substeps = rollout.batch.pop("next.truncated").bool()
    reward_substeps = rollout.batch.pop("next.reward").float()
    rollout.batch.pop("next.success")
    done_substeps = terminated_substeps | truncated_substeps

    action_substeps = rollout.batch["action.action"].shape[2]
    native_action_horizon = rollout.batch["action.full_action"].shape[2]
    if reward_substeps.shape[2] != action_substeps:
        raise ValueError(
            f"Reward chunk length {reward_substeps.shape[2]} must match executed action length {action_substeps}."
        )
    if native_action_horizon < action_substeps:
        raise ValueError(
            f"Native action horizon {native_action_horizon} cannot be shorter than executed length {action_substeps}."
        )

    # Include the substep that ends the episode, but mask any padded feedback
    # after the first termination/truncation inside the same action chunk.
    valid_action_substeps = (done_substeps.cumsum(dim=2) - done_substeps.long()) == 0

    # Discount and sum low-level rewards within each action chunk. The step
    # penalty is charged once per policy decision rather than once per substep.
    substep_indices = torch.arange(action_substeps, device=reward_substeps.device, dtype=torch.float32)
    reward_discounts = float(trainer_config.gamma) ** substep_indices
    rewards = (reward_substeps * valid_action_substeps * reward_discounts).sum(dim=2)
    rewards -= float(trainer_config.step_penalty)
    dones = done_substeps.any(dim=2).float()

    # A chunk may stop early, so its semi-MDP duration is the number of actions
    # actually executed. Terminal chunks must not bootstrap V(next_obs), and
    # their GAE recursion must not cross into an auto-reset episode.
    executed_steps = valid_action_substeps.sum(dim=2)
    discounts = float(trainer_config.gamma) ** executed_steps
    gae_discounts = (float(trainer_config.gamma) * float(trainer_config.gae_lambda)) ** executed_steps
    discounts *= 1.0 - dones
    gae_discounts *= 1.0 - dones

    # Pi0 predicts its full native action horizon H, while the environment may
    # execute only K <= H actions. Preserve the H dimension so the worker can
    # exclude unexecuted CFM-loss positions when constructing the log ratio.
    action_valids = torch.zeros(
        *valid_action_substeps.shape[:2],
        native_action_horizon,
        device=valid_action_substeps.device,
        dtype=torch.float32,
    )
    action_valids[:, :, :action_substeps] = valid_action_substeps.float()

    rollout.batch["info.rewards"] = rewards
    rollout.batch["info.dones"] = dones
    rollout.batch["info.discounts"] = discounts
    rollout.batch["info.gae_discounts"] = gae_discounts
    # All collected slots are real here. Distributed padding later appends
    # slots with info.valids=0 so they do not contribute to any loss or metric.
    rollout.batch["info.valids"] = torch.ones_like(rewards)
    rollout.batch["info.action_valids"] = action_valids
    rollout.meta_info.update(
        {
            "global_steps": global_steps,
            "gamma": float(trainer_config.gamma),
            "gae_lambda": float(trainer_config.gae_lambda),
            "global_token_num": [0] * len(rollout),
        }
    )
    return rollout


class FPORayTrainer:
    def __init__(self, trainer_config, cluster: TrainCluster, tracking_config: dict[str, Any]):
        self.cluster = cluster
        self.trainer_config: FPOTrainerConfig = instantiate(trainer_config)
        self.config = OmegaConf.create(tracking_config)

    def _prepare_actor_input(self, rollout_output: DataProto, rollout_end_obs: DataProto) -> DataProto:
        actor_input = prepare_fpo_actor_input(
            rollout_output,
            rollout_end_obs,
            trainer_config=self.trainer_config,
            global_steps=self.global_steps,
        )
        return pad_dataproto_to_divisor_with_valid_mask(
            actor_input,
            int(self.cluster.actor_worker_group.world_size),
            valid_key="info.valids",
        )

    def fit(self) -> None:
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.trainer_config.project_name,
            experiment_name=self.trainer_config.experiment_name,
            default_backend=self.trainer_config.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.global_steps = 0
        checkpoint_state = self.cluster.load_checkpoint()
        if checkpoint_state is not None:
            self.global_steps, _checkpoint_dir = checkpoint_state

        eval_episodes = self.trainer_config.eval_episodes
        eval_episodes = eval_episodes if eval_episodes > 0 else None
        if self.trainer_config.val_before_train:
            val_metrics = self.cluster.eval(max_episodes=eval_episodes)
            pprint(f"Initial evaluation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.trainer_config.val_only:
                return

        total_steps = self.trainer_config.total_training_steps
        progress = tqdm(total=total_steps, initial=self.global_steps, desc="FPO training")
        self.global_steps += 1
        max_step_duration = 0.0
        last_val_metrics = None

        while self.global_steps <= total_steps:
            metrics: dict[str, Any] = {}
            timing_raw: dict[str, float] = {}
            with marked_timer("step", timing_raw):
                with marked_timer("rollout", timing_raw, color="red"):
                    rollout_output, rollout_end_obs, _datasets, rollout_metrics = self.cluster.rollout(
                        async_rollout=self.trainer_config.async_rollout
                    )
                    metrics.update(rollout_metrics)
                    actor_input = self._prepare_actor_input(rollout_output, rollout_end_obs)

                with marked_timer("update_actor", timing_raw, color="red"):
                    actor_output = self.cluster.train(actor_input, async_update=False)
                    metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

            is_last_step = self.global_steps >= total_steps
            if self.trainer_config.test_freq > 0 and (
                is_last_step or self.global_steps % self.trainer_config.test_freq == 0
            ):
                with marked_timer("testing", timing_raw, color="green"):
                    last_val_metrics = self.cluster.eval(max_episodes=eval_episodes)
                    metrics.update(last_val_metrics)

            max_step_duration = max(max_step_duration, timing_raw["step"])
            esi_close = should_save_ckpt_esi(
                max_steps_duration=max_step_duration,
                redundant_time=self.trainer_config.esi_redundant_time,
            )
            if self.trainer_config.save_freq > 0 and (
                is_last_step or self.global_steps % self.trainer_config.save_freq == 0 or esi_close
            ):
                with marked_timer("save_checkpoint", timing_raw, color="green"):
                    self.cluster.save_checkpoint(self.global_steps)

            metrics["training/global_step"] = self.global_steps
            metrics.update({f"timing_s/{name}": value for name, value in timing_raw.items()})
            logger.log(data=metrics, step=self.global_steps)
            progress.update(1)
            self.global_steps += 1
            if is_last_step:
                pprint(f"Final evaluation metrics: {last_val_metrics}")
                progress.close()
                return
