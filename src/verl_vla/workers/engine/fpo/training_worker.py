# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import torch
from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import get_device_id, get_device_name
from verl.workers.config import TrainingWorkerConfig
from verl.workers.engine_workers import TrainingWorker

from verl_vla.utils.data import flatten_trajectories, get_dataproto_from_prefix
from verl_vla.workers.config import FPOActorConfig

logger = logging.getLogger(__name__)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    discounts: torch.Tensor,
    gae_discounts: torch.Tensor,
    valids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE over rollout chunks without crossing padded or terminal slots."""
    advantages = torch.zeros_like(values)
    last_advantage = torch.zeros_like(values[:, 0])
    for step in reversed(range(values.shape[1])):
        active = valids[:, step]
        delta = rewards[:, step] + discounts[:, step] * next_values[:, step] - values[:, step]
        last_advantage = (delta + gae_discounts[:, step] * last_advantage) * active
        advantages[:, step] = last_advantage
    return advantages, advantages + values


def clipped_policy_loss(
    log_ratio: torch.Tensor,
    advantages: torch.Tensor,
    valids: torch.Tensor,
    clip_coef: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Vanilla-FPO's PPO surrogate, where CFM loss differences are log ratios."""
    ratio = log_ratio.exp()
    unclipped = -advantages * ratio
    clipped = -advantages * ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef)
    valid_count = valids.sum().clamp_min(1.0)
    loss = (torch.maximum(unclipped, clipped) * valids).sum() / valid_count
    with torch.no_grad():
        approx_kl = (((ratio - 1.0) - log_ratio) * valids).sum() / valid_count
        clip_fraction = (((ratio - 1.0).abs() > clip_coef).float() * valids).sum() / valid_count
    return loss, {"ratio": ratio.detach(), "approx_kl": approx_kl, "clip_fraction": clip_fraction}


class FPOTrainingWorker(TrainingWorker):
    """On-policy vanilla Flow Policy Optimization worker."""

    def __init__(self, config: TrainingWorkerConfig, actor_config: FPOActorConfig, tokenizer=None):
        super().__init__(config=config)
        self.actor_config = actor_config
        self.tokenizer = tokenizer or self.model_config.tokenizer
        self.local_mini_batch_size = self._global_to_local_batch_size(actor_config.mini_batch_size)
        self._fpo_initialized = False

    @staticmethod
    def _global_to_local_batch_size(global_batch_size: int) -> int:
        world_size = torch.distributed.get_world_size()
        if global_batch_size % world_size:
            raise ValueError(f"FPO mini_batch_size={global_batch_size} must be divisible by world_size={world_size}.")
        return global_batch_size // world_size

    def _ensure_fpo_initialized(self) -> None:
        if self._fpo_initialized:
            return
        self.engine.module.fpo_init()
        self.value_parameters = self.engine.module.fpo_get_value_parameters()
        self.value_optimizer = torch.optim.Adam(
            self.value_parameters,
            lr=self.actor_config.value.lr,
            weight_decay=self.actor_config.value.weight_decay,
        )
        self.value_scheduler = torch.optim.lr_scheduler.ConstantLR(self.value_optimizer, factor=1.0)
        self._fpo_initialized = True

    def _value_optimizer_checkpoint_path(self, local_path: str) -> Path:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        return Path(local_path) / f"fpo_value_optim_world_size_{world_size}_rank_{rank}.pt"

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        self._ensure_fpo_initialized()
        super().save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)
        checkpoint_path = self._value_optimizer_checkpoint_path(local_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "optimizer": self.value_optimizer.state_dict(),
                "scheduler": self.value_scheduler.state_dict(),
            },
            checkpoint_path,
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        self._ensure_fpo_initialized()
        checkpoint_path = self._value_optimizer_checkpoint_path(local_path)
        if checkpoint_path.exists():
            state = torch.load(
                checkpoint_path,
                map_location=torch.device(get_device_name(), get_device_id()),
                weights_only=True,
            )
            self.value_optimizer.load_state_dict(state["optimizer"])
            self.value_scheduler.load_state_dict(state["scheduler"])
        else:
            logger.warning(
                "Checkpoint %s predates FPO value-optimizer persistence; resuming value optimization "
                "with fresh Adam state.",
                checkpoint_path,
            )
        return super().load_checkpoint(local_path, hdfs_path, del_local_after_load)

    @staticmethod
    def _global_normalize(values: torch.Tensor, valids: torch.Tensor) -> torch.Tensor:
        valid_values = values * valids
        stats = torch.stack([(valid_values).sum(), (valid_values.square()).sum(), valids.sum()])
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        mean = stats[0] / stats[2].clamp_min(1.0)
        variance = stats[1] / stats[2].clamp_min(1.0) - mean.square()
        return (values - mean) / variance.clamp_min(1e-8).sqrt()

    def _forward_old_statistics(self, data: DataProto) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = flatten_trajectories(data, reference_key="action.action")
        old_values = []
        next_values = []
        old_cfm_losses = []
        for micro_batch in flat.split(self.actor_config.micro_batch_size):
            micro_batch = micro_batch.to(get_device_id())
            obs = get_dataproto_from_prefix(micro_batch, "obs.")
            next_obs = get_dataproto_from_prefix(micro_batch, "next_obs.")
            with torch.no_grad(), torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                old_values.append(self.engine.module.fpo_forward_value(obs, self.tokenizer))
                next_values.append(self.engine.module.fpo_forward_value(next_obs, self.tokenizer))
                old_cfm_losses.append(
                    self.engine.module.fpo_cfm_loss(
                        obs,
                        self.tokenizer,
                        micro_batch.batch["action.full_action"],
                        micro_batch.batch["fpo.timesteps"],
                        micro_batch.batch["fpo.noise"],
                    )
                )
        batch_size, rollout_steps = data.batch["info.valids"].shape
        return (
            torch.cat(old_values).reshape(batch_size, rollout_steps),
            torch.cat(next_values).reshape(batch_size, rollout_steps),
            torch.cat(old_cfm_losses).reshape(batch_size, rollout_steps, *old_cfm_losses[0].shape[1:]),
        )

    def _prepare_training_batch(self, data: DataProto) -> DataProto:
        full_actions = data.batch["action.full_action"]
        batch_size, rollout_steps, action_horizon, action_dim = full_actions.shape
        sample_count = self.actor_config.n_action_samples
        device = get_device_id()
        data.batch["fpo.timesteps"] = torch.rand(
            batch_size, rollout_steps, sample_count, device=device, dtype=torch.float32
        )
        data.batch["fpo.noise"] = torch.randn(
            batch_size,
            rollout_steps,
            sample_count,
            action_horizon,
            action_dim,
            device=device,
            dtype=full_actions.dtype,
        )

        old_values, next_values, old_cfm_loss = self._forward_old_statistics(data)
        valids = data.batch["info.valids"].to(device=device, dtype=torch.float32)
        advantages, returns = compute_gae(
            rewards=data.batch["info.rewards"].to(device),
            values=old_values,
            next_values=next_values,
            discounts=data.batch["info.discounts"].to(device),
            gae_discounts=data.batch["info.gae_discounts"].to(device),
            valids=valids,
        )
        if self.actor_config.normalize_advantages:
            advantages = self._global_normalize(advantages, valids)

        data.batch["fpo.old_values"] = old_values
        data.batch["fpo.old_cfm_loss"] = old_cfm_loss
        data.batch["fpo.advantages"] = advantages
        data.batch["fpo.returns"] = returns
        return flatten_trajectories(data, reference_key="action.action")

    def _update_fpo_policy(self, data: DataProto) -> dict[str, float]:
        data = self._prepare_training_batch(data)
        if len(data) < self.local_mini_batch_size:
            raise ValueError(
                f"Each FPO rank needs at least {self.local_mini_batch_size} valid/padded rollout slots, "
                f"got {len(data)}."
            )

        value_only = int(data.meta_info["global_steps"]) <= self.actor_config.value_only_updates
        metric_lists: dict[str, list[float]] = defaultdict(list)
        epochs_run = 0

        for _epoch in range(self.actor_config.update_epochs):
            permutation = torch.randperm(len(data))
            epoch_kls = []
            for start in range(0, len(data), self.local_mini_batch_size):
                indices = permutation[start : start + self.local_mini_batch_size]
                mini_batch = data.select_idxs(indices)
                micro_batches = mini_batch.split(self.actor_config.micro_batch_size)
                grad_accum_steps = len(micro_batches)
                self.engine.optimizer_zero_grad()
                self.value_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    obs = get_dataproto_from_prefix(micro_batch, "obs.")
                    valids = micro_batch.batch["info.valids"].float()
                    with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                        values = self.engine.module.fpo_forward_value(obs, self.tokenizer)
                        value_loss = (
                            0.5
                            * ((values - micro_batch.batch["fpo.returns"]).square() * valids).sum()
                            / valids.sum().clamp_min(1.0)
                        )

                        if value_only:
                            policy_loss = values.new_zeros(())
                            policy_metrics = {
                                "ratio": torch.ones_like(values),
                                "approx_kl": values.new_zeros(()),
                                "clip_fraction": values.new_zeros(()),
                            }
                        else:
                            current_cfm_loss = self.engine.module.fpo_cfm_loss(
                                obs,
                                self.tokenizer,
                                micro_batch.batch["action.full_action"],
                                micro_batch.batch["fpo.timesteps"],
                                micro_batch.batch["fpo.noise"],
                            )
                            action_valids = micro_batch.batch["info.action_valids"].unsqueeze(-1)
                            log_ratio = (
                                ((micro_batch.batch["fpo.old_cfm_loss"] - current_cfm_loss) * action_valids)
                                .sum(dim=1)
                                .mean(dim=-1)
                            )
                            policy_loss, policy_metrics = clipped_policy_loss(
                                log_ratio,
                                micro_batch.batch["fpo.advantages"],
                                valids,
                                self.actor_config.clip_coef,
                            )

                        loss = policy_loss + self.actor_config.vf_coef * value_loss
                    (loss / grad_accum_steps).backward()

                    metric_lists["actor/loss"].append(float(policy_loss.detach()))
                    metric_lists["value/loss"].append(float(value_loss.detach()))
                    metric_lists["fpo/approx_kl"].append(float(policy_metrics["approx_kl"]))
                    metric_lists["fpo/clip_fraction"].append(float(policy_metrics["clip_fraction"]))
                    metric_lists["fpo/ratio_mean"].append(float(policy_metrics["ratio"].mean()))
                    epoch_kls.append(float(policy_metrics["approx_kl"]))

                value_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.value_parameters, max_norm=self.actor_config.value.clip_grad
                )
                self.value_optimizer.step()
                self.value_scheduler.step()
                for parameter in self.value_parameters:
                    parameter.grad = None

                if not value_only:
                    actor_grad_norm = self.engine.optimizer_step()
                    self.engine.lr_scheduler_step()
                    metric_lists["actor/grad_norm"].append(float(actor_grad_norm))
                metric_lists["value/grad_norm"].append(float(value_grad_norm))

            epochs_run += 1
            mean_epoch_kl = sum(epoch_kls) / max(len(epoch_kls), 1)
            if self.actor_config.target_kl is not None and mean_epoch_kl > self.actor_config.target_kl:
                break

        returns = data.batch["fpo.returns"].float()
        metric_device = returns.device
        valids = data.batch["info.valids"].to(metric_device).float()
        rewards = data.batch["info.rewards"].to(metric_device).float()
        old_values = data.batch["fpo.old_values"].to(metric_device).float()
        valid_count = valids.sum().clamp_min(1.0)
        return_mean = (returns * valids).sum() / valid_count
        return_variance = ((returns - return_mean).square() * valids).sum() / valid_count
        residual_variance = ((returns - old_values).square() * valids).sum() / valid_count
        explained_variance = 1.0 - residual_variance / return_variance.clamp_min(1e-8)

        metrics = {key: sum(values) / len(values) for key, values in metric_lists.items() if values}
        metrics.update(
            {
                "data/reward_mean": float((rewards * valids).sum() / valid_count),
                "data/valid_ratio": float(valids.mean()),
                "value/explained_variance": float(explained_variance),
                "value/lr": float(self.value_optimizer.param_groups[0]["lr"]),
                "actor/lr": float(self.engine.optimizer.param_groups[0]["lr"]),
                "fpo/value_only": float(value_only),
                "fpo/epochs_run": float(epochs_run),
            }
        )
        return metrics

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def train_mini_batch(self, data: DataProto) -> DataProto:
        self._ensure_fpo_initialized()
        with self.engine.train_mode():
            metrics = self._update_fpo_policy(data)
        return DataProto(meta_info={"metrics": metrics})
