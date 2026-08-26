# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import json
import math
import os
import shutil
from pathlib import Path
from typing import Literal, cast

import torch
import torch.nn.functional as F
from onnx_ir import Tensor
from torch import nn
from torch.distributed.fsdp import register_fsdp_forward_method
from typing_extensions import override
from verl.protocol import DataProto
from verl.utils.device import get_device_name

from ..base import ModelOutput, SupportFPOTraining, SupportSACTraining, SupportSFTTraining, TrainableVLAModelBase
from ..dsrl import DSRLSteering
from .adapter_config import PI0AdapterConfig
from .critic import (
    CrossAttentionCriticBackend,
    MeanPoolCriticBackend,
    MultiCrossAttentionCriticBackend,
    PI0CNNCriticBackend,
)
from .embodiments import get_pi0_embodiment_classes
from .embodiments.base import Pi0Output
from .model.modeling_pi0 import PI0Policy, make_att_2d_masks
from .pi0_utils import (
    ImageTransform,
    Normalize,
    PromptTokenizerTransform,
    Unnormalize,
)
from .value_head import PI0FPOValueHead

CRITIC_BACKENDS = {
    "cnn": PI0CNNCriticBackend(),
    "cross_attn": CrossAttentionCriticBackend(),
    "mean_pool": MeanPoolCriticBackend(),
    "multi_cross_attn": MultiCrossAttentionCriticBackend(),
}


def load_pi0_norm_stats(path: str | os.PathLike[str]) -> tuple[dict, dict]:
    stats_path = Path(path).expanduser()
    if not stats_path.is_file():
        raise FileNotFoundError(f"Pi0 normalization statistics do not exist: {stats_path}")

    with stats_path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Pi0 normalization statistics must be a JSON object: {stats_path}")

    state_stats = payload.get("state")
    action_stats = payload.get("action")
    for name, stats in (("state", state_stats), ("action", action_stats)):
        if not isinstance(stats, dict):
            raise ValueError(f"Pi0 normalization statistics are missing a {name!r} object: {stats_path}")
        for field in ("mean", "std", "q01", "q99"):
            values = stats.get(field)
            if not isinstance(values, list) or not values:
                raise ValueError(f"Pi0 {name} normalization field {field!r} must be a non-empty list: {stats_path}")

    return state_stats, action_stats


class PI0TrainableModel(
    TrainableVLAModelBase,
    SupportFPOTraining,
    SupportSACTraining,
    SupportSFTTraining,
):
    def __init__(self, config: PI0AdapterConfig, policy: PI0Policy | None = None):
        if policy is None:
            policy = PI0Policy(
                max_state_dim=int(getattr(config, "max_state_dim", 32)),
                max_action_dim=int(getattr(config, "max_action_dim", 32)),
                proj_width=int(getattr(config, "proj_width", 1024)),
                n_action_steps=int(getattr(config, "n_action_steps", 50)),
                num_steps=int(getattr(config, "num_steps", 10)),
                use_cache=bool(getattr(config, "use_cache", True)),
                pi05_enabled=bool(getattr(config, "pi05_enabled", False)),
            )
        super().__init__(policy=policy)
        self.config = config
        SupportSFTTraining.__init__(self, config)
        norm_stats_path = getattr(config, "norm_stats_path", None)
        if hasattr(config, "norm_stats_path"):
            delattr(config, "norm_stats_path")
        if not norm_stats_path:
            model_path = config.model_path
            checkpoint_stats_path = Path(model_path).expanduser() / "norm_stats.json" if model_path else None
            if checkpoint_stats_path and checkpoint_stats_path.is_file():
                norm_stats_path = checkpoint_stats_path.resolve()
        if norm_stats_path:
            norm_stats_path = Path(norm_stats_path).expanduser()
            if not norm_stats_path.is_absolute():
                model_path = config.model_path
                if model_path:
                    norm_stats_path = Path(model_path).expanduser() / norm_stats_path
            self.state_norm_stats, self.action_norm_stats = load_pi0_norm_stats(norm_stats_path)
            self.norm_stats_path = norm_stats_path
        else:
            self.state_norm_stats = config.state_norm_stats
            self.action_norm_stats = config.action_norm_stats
            self.norm_stats_path = None
        self.pi05_enabled = config.pi05_enabled
        self.embodiment = config.embodiment
        self.action_chunk_size = int(getattr(config, "action_chunk_size", 10))
        self.critic_type = config.critic.type
        self.critic = None

        assert self.state_norm_stats, "state_norm_stats must be provided for the PI0 adapter"
        assert self.action_norm_stats, "action_norm_stats must be provided for the PI0 adapter"
        assert isinstance(self.pi05_enabled, bool), "pi05_enabled must be provided by the native PI0 policy config"

        self.fpo_value_head = None
        if config.fpo.enabled:
            self.fpo_value_head = PI0FPOValueHead(
                input_dim=int(config.fpo.value_input_dim),
                hidden_dims=[int(dim) for dim in config.fpo.value_hidden_dims],
            )

        # Input transforms
        self.state_normalize_transform = Normalize(self.state_norm_stats, use_quantiles=self.pi05_enabled)
        self.action_normalize_transform = Normalize(self.action_norm_stats, use_quantiles=self.pi05_enabled)
        self.image_transform = ImageTransform(resize_imgs_with_padding=(224, 224), enable_image_aug=False)
        max_length = 200 if self.pi05_enabled else 48
        self.prompt_tokenizer_transform = PromptTokenizerTransform(max_length=max_length, discrete_state_input=False)

        # Output transforms
        self.state_unnormalize_transform = Unnormalize(self.state_norm_stats, use_quantiles=self.pi05_enabled)
        self.action_unnormalize_transform = Unnormalize(self.action_norm_stats, use_quantiles=self.pi05_enabled)

        # Flow SDE parameters
        self._to(get_device_name())
        self.flow_sde_enable = bool(getattr(config, "flow_sde_enable", True))
        self.flow_sde_noise_level = float(getattr(config, "flow_sde_noise_level", 0.5))
        self.flow_sde_task_noise_level = self._parse_task_noise_levels(config.flow_sde_task_noise_level)
        self.flow_sde_rollout_noise_scale = float(getattr(config, "flow_sde_rollout_noise_scale", 1.0))
        self.flow_sde_train_noise_scale = float(getattr(config, "flow_sde_train_noise_scale", 1.0))

        ##### SAC Algorithm Support #####
        if self.config.critic.enabled:
            if self.critic_type not in CRITIC_BACKENDS:
                raise ValueError(f"Unsupported critic_type: {self.critic_type}")
            self.critic_api = CRITIC_BACKENDS[self.critic_type]
            self.critic_api.init(self)

        self.dsrl = None
        if config.dsrl.enabled:
            if self.flow_sde_enable:
                raise ValueError("DSRL noise steering and Flow-SDE are mutually exclusive; set flow_sde_enable=False.")
            if not config.critic.enabled:
                raise ValueError("DSRL requires the SAC critic; set adapter.critic.enabled=True.")
            self.dsrl = DSRLSteering(
                config.dsrl,
                feature_dim=config.critic.prefix_embed_dim,
                state_dim=len(self.state_norm_stats["mean"]),
                noise_dim=self.policy.max_action_dim,
                noise_horizon=self.policy.n_action_steps,
            )
            self.policy.requires_grad_(False)
            self.policy.eval()

    def _get_pi0_embodiment_classes(self):
        return get_pi0_embodiment_classes(self.embodiment)

    def _to(self, device: torch.device | str):
        self.state_normalize_transform.to(device)
        self.state_unnormalize_transform.to(device)
        self.action_normalize_transform.to(device)
        self.action_unnormalize_transform.to(device)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.dsrl is not None:
            self.policy.eval()
        return self

    def forward(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> Tensor:
        """Full forward pass for one diffusion denoising step.

        Args:
            images: List of image tensors, each shaped (B, C, H, W) after batching.
            img_masks: List of boolean masks corresponding to images, each (B,).
            lang_tokens: Language token ids (B, L).
            lang_masks: Language attention mask (B, L) with True for valid tokens.
            state: State tensor (B, state_dim) if pi05 is disabled else ignored.
            x_t: Noisy action tokens (B, n_action_steps, action_dim).
            timestep: Diffusion timestep as float tensor (B,).

        Returns:
            Predicted v_t with shape (B, n_action_steps, action_dim).
        """

        if self.policy is None:
            raise RuntimeError("PI0TrainableModel.policy is not initialized. Did from_pretrained() run?")

        return self.policy(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            x_t,
            timestep,
        )

    @torch.no_grad()
    def sample_actions(
        self,
        env_obs: DataProto,
        tokenizer,
        eval: bool = False,
    ) -> Pi0Output:
        """Run one forward pass from environment observations to actions."""

        pi0_input_cls, pi0_output_cls = self._get_pi0_embodiment_classes()
        pi0_input = pi0_input_cls.from_env_obs(env_obs)

        # Input transforms
        state = self.state_normalize_transform(pi0_input.state)
        images, _ = self.image_transform.call_batch(pi0_input.images)
        lang_tokens, lang_masks = self.prompt_tokenizer_transform.call_batch(
            {"task": pi0_input.task, "observation.state": state}, tokenizer
        )
        prefix_features = self.policy.embed_prefix(
            images=images,
            img_masks=pi0_input.img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
        )
        state_features = (prefix_features, state)
        if self.dsrl is not None and self.dsrl.actor_type == "cnn":
            pixels = env_obs.batch["observation.images.image"]
            raw_state = env_obs.batch["observation.state"]
            state_features = (prefix_features, state, (pixels, raw_state))
        task_ids = torch.tensor(env_obs.non_tensor_batch["task_id"], device=state.device, dtype=torch.long)

        steering_noise = None
        if self.dsrl is not None:
            # DSRL: sample steering noise from the small SAC actor and let the
            # frozen flow head integrate it deterministically into an action.
            features, dsrl_state = self._dsrl_actor_inputs(state_features)
            rollout_noise_scale = None if eval else self.dsrl.config.rollout_noise_scale
            steering_noise, rollout_log_probs, _ = self.dsrl.sample(
                features,
                dsrl_state,
                deterministic=eval,
                noise_scale=rollout_noise_scale,
            )
            initial_noise = steering_noise
        else:
            shape = (state.shape[0], self.policy.n_action_steps, self.policy.max_action_dim)
            initial_noise = self.policy.sample_noise(shape, state.device)

        noise_scale = self.flow_sde_rollout_noise_scale if self.flow_sde_enable and not eval else 0.0
        pred_action, flow_log_probs = self._sample_actions_flow_sde(
            state_features=state_features,
            initial_noise=initial_noise,
            noise_scale=noise_scale,
            requires_grad=False,
            return_log_prob=self.flow_sde_enable and not eval,
            task_ids=task_ids,
        )
        if self.dsrl is None:
            rollout_log_probs = flow_log_probs

        # Output transforms
        full_action = self.action_unnormalize_transform(pred_action)
        pi0_output = pi0_output_cls.from_model_output(
            {
                "full_action": full_action,
                "log_probs": rollout_log_probs,
                "action_chunk_size": self.action_chunk_size,
            }
        )
        pi0_output.full_action = full_action
        if steering_noise is not None:
            pi0_output.steering_noise = steering_noise.detach().float()

        return pi0_output

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        del model_args
        config = kwargs.pop("config", None)
        adapter_config = kwargs.pop("adapter_config", None)
        policy_config_overrides = dict(kwargs.pop("policy_config_overrides", {}) or {})
        policy_load_keys = {"cache_dir", "force_download", "local_files_only", "proxies", "revision", "token"}
        policy_load_kwargs = {key: kwargs.pop(key) for key in tuple(kwargs) if key in policy_load_keys}
        torch_dtype = kwargs.pop("torch_dtype", None)
        if torch_dtype is not None:
            policy_load_kwargs["torch_dtype"] = torch_dtype

        policy = PI0Policy.from_pretrained(
            pretrained_model_name_or_path,
            **policy_load_kwargs,
            **policy_config_overrides,
        )
        policy_config = dict(policy.config)
        if config is None:
            overrides = dict(adapter_config or {})
            overrides.update(kwargs)
            config = PI0AdapterConfig(
                policy_config=policy_config,
                model_path=pretrained_model_name_or_path,
                **overrides,
            )
        elif not isinstance(config, PI0AdapterConfig):
            config = PI0AdapterConfig(
                policy_config=policy_config,
                model_path=pretrained_model_name_or_path,
                **dict(config),
            )
        return cls(config=config, policy=policy)

    def save_pretrained(self, save_directory, *args, state_dict=None, **kwargs):
        os.makedirs(save_directory, exist_ok=True)

        policy_to_save = self.native_policy
        if state_dict is not None:
            policy_state_dict = self.extract_policy_state_dict(state_dict)
            policy_to_save = PI0Policy.from_config(dict(self.native_policy.config))
            policy_to_save.load_state_dict(policy_state_dict, strict=True)
        policy_to_save.save_pretrained(save_directory, *args, **kwargs)

        norm_stats_path = self.norm_stats_path
        if not norm_stats_path:
            return

        exported_stats_path = Path(save_directory) / "norm_stats.json"
        source_stats_path = Path(norm_stats_path).expanduser()
        if not source_stats_path.is_absolute():
            model_path = self.config.model_path
            if model_path:
                source_stats_path = Path(model_path).expanduser() / source_stats_path

        # Keep exported checkpoints self-contained without retaining stale
        # embedded statistics or an absolute path from the training host.
        if source_stats_path.resolve() != exported_stats_path.resolve():
            shutil.copyfile(source_stats_path, exported_stats_path)

    def export_policy(self, output_dir, *, state_dict=None) -> None:
        """Export PI0 weights together with its normalization statistics."""

        self.save_pretrained(output_dir, state_dict=state_dict)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        # Preserve complete adapter checkpoints, including critic/target state.
        normalized_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("critic_backend."):
                key = f"critic.{key.removeprefix('critic_backend.')}"
            elif key.startswith("auxiliary_modules.critic."):
                key = f"critic.{key.removeprefix('auxiliary_modules.critic.')}"
            normalized_state_dict[key] = value
        return nn.Module.load_state_dict(self, normalized_state_dict, strict=strict, assign=assign)

    def can_generate(self) -> bool:
        return False

    def freeze_vision_tower(self) -> None:
        """Freeze the vision tower parameters."""

        if self.policy is None:
            raise RuntimeError("PI0TrainableModel.policy is not initialized. Did from_pretrained() run?")
        vision_tower = self.policy.paligemma_with_expert.vision_tower
        vision_tower.requires_grad_(False)
        vision_tower.eval()

    def _dsrl_actor_inputs(
        self,
        state_features,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.dsrl.actor_type == "cnn":
            _, _, cnn_inputs = state_features
            return cnn_inputs
        prefix_features, states = state_features[:2]
        prefix_embs, prefix_pad_masks, _ = prefix_features
        prefix_mask = prefix_pad_masks.to(dtype=prefix_embs.dtype).unsqueeze(-1)
        pooled = (prefix_embs * prefix_mask).sum(dim=1) / prefix_mask.sum(dim=1).clamp_min(1.0)
        return pooled, states[..., : self.dsrl.noise_actor.state_dim]

    @override
    def sft_init(self):
        """Override SupportSFTTraining.sft_init for PI0 SFT setup."""
        self.freeze_vision_tower()
        register_fsdp_forward_method(self, "sft_loss")

    @override
    def sft_loss(
        self,
        obs: DataProto,
        tokenizer: torch.nn.Module,
        actions: dict[str, torch.Tensor],
        valids: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        target_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Override SupportSFTTraining.sft_loss for PI0 BC training."""

        del target_values
        pi0_input_cls, _ = self._get_pi0_embodiment_classes()
        action_tensor = actions["action"]
        action_horizon = self.policy.n_action_steps
        action_length = min(action_tensor.shape[1], action_horizon)
        action_tensor = action_tensor[:, :action_horizon, : self.policy.max_action_dim]
        if action_mask is not None:
            action_mask = action_mask[:, :action_horizon]
        if action_tensor.shape[1] < action_horizon:
            action_tensor = torch.nn.functional.pad(
                action_tensor,
                (
                    0,
                    0,
                    0,
                    action_horizon - action_tensor.shape[1],
                ),
                value=0.0,
            )
            if action_mask is not None:
                action_mask = torch.nn.functional.pad(
                    action_mask,
                    (0, action_horizon - action_mask.shape[1]),
                    value=0.0,
                )
        action_tensor = torch.nn.functional.pad(
            action_tensor,
            (
                0,
                self.policy.max_action_dim - action_tensor.shape[-1],
            ),
            value=0.0,
        )
        action_tensor = self.action_normalize_transform(action_tensor)

        with torch.no_grad():
            pi0_input = pi0_input_cls.from_env_obs(obs)
            states = self.state_normalize_transform(pi0_input.state)
            images, _ = self.image_transform.call_batch(pi0_input.images)
            img_masks = pi0_input.img_masks
            lang_tokens, lang_masks = self.prompt_tokenizer_transform.call_batch(
                {"task": pi0_input.task, "observation.state": states}, tokenizer
            )

        noise = self.policy.sample_noise(action_tensor.shape, device=action_tensor.device)
        gamma1 = torch.empty((action_tensor.shape[0],), device=action_tensor.device).uniform_(0, 1).pow(1 / 1.5)
        gamma2 = torch.empty((action_tensor.shape[0],), device=action_tensor.device).uniform_(0, 1).pow(1 / 1.0)
        time = (gamma1 / (gamma1 + gamma2)) * 0.999 + 0.001
        time = time.to(dtype=torch.float32, device=action_tensor.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1.0 - time_expanded) * action_tensor
        u_t = noise - action_tensor

        model_pred = self.policy(images, img_masks, lang_tokens, lang_masks, states, x_t, time)
        loss = F.mse_loss(u_t, model_pred, reduction="none").mean(dim=-1)
        valids = valids.to(device=loss.device, dtype=loss.dtype)
        if action_mask is None:
            action_mask = (
                (torch.arange(action_horizon, device=loss.device) < action_length)
                .to(dtype=loss.dtype)
                .unsqueeze(0)
                .expand_as(loss)
            )
        else:
            action_mask = action_mask.to(device=loss.device, dtype=loss.dtype)
            if action_mask.shape[1] != action_horizon:
                raise ValueError(f"SFT action mask length must be {action_horizon}, got {action_mask.shape[1]}.")

        sample_loss = (loss * action_mask).sum(dim=-1) / action_mask.sum(dim=-1).clamp_min(1.0)
        return (sample_loss * valids).sum() / valids.sum().clamp_min(1.0)

    # --- Vanilla FPO Algorithm Support ---

    @override
    def fpo_init(self) -> None:
        if self.fpo_value_head is None:
            raise RuntimeError("FPO requires adapter.fpo.enabled=true.")
        if self.dsrl is not None:
            raise ValueError("FPO and DSRL noise steering cannot be enabled together.")
        if not self.policy.use_cache:
            raise ValueError("FPO requires the PI0 prefix KV cache (model use_cache=true).")
        self.freeze_vision_tower()
        register_fsdp_forward_method(self, "fpo_cfm_loss")
        register_fsdp_forward_method(self, "fpo_forward_value")

    def _fpo_policy_inputs(self, obs: DataProto, tokenizer: torch.nn.Module):
        pi0_input_cls, _ = self._get_pi0_embodiment_classes()
        with torch.no_grad():
            pi0_input = pi0_input_cls.from_env_obs(obs)
            states = self.state_normalize_transform(pi0_input.state)
            images, _ = self.image_transform.call_batch(pi0_input.images)
            lang_tokens, lang_masks = self.prompt_tokenizer_transform.call_batch(
                {"task": pi0_input.task, "observation.state": states}, tokenizer
            )
        return states, images, pi0_input.img_masks, lang_tokens, lang_masks

    @override
    def fpo_cfm_loss(
        self,
        obs: DataProto,
        tokenizer: torch.nn.Module,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        if timesteps.ndim != 2:
            raise ValueError(f"FPO timesteps must have shape [B, N], got {tuple(timesteps.shape)}.")
        if noise.ndim != 4:
            raise ValueError(f"FPO noise must have shape [B, N, H, D], got {tuple(noise.shape)}.")
        batch_size, n_action_samples = timesteps.shape
        if actions.shape[0] != batch_size or noise.shape[:2] != (batch_size, n_action_samples):
            raise ValueError("FPO actions, timesteps, and noise batch dimensions must match.")

        action_horizon = self.policy.n_action_steps
        action_tensor = actions[:, :action_horizon, : self.policy.max_action_dim]
        if action_tensor.shape[1] != action_horizon or action_tensor.shape[2] != self.policy.max_action_dim:
            raise ValueError(
                "FPO requires the full native action sample with shape "
                f"[B, {action_horizon}, {self.policy.max_action_dim}], got {tuple(actions.shape)}."
            )
        action_tensor = self.action_normalize_transform(action_tensor)
        if noise.shape[2:] != action_tensor.shape[1:]:
            raise ValueError(
                f"FPO noise tail shape {tuple(noise.shape[2:])} does not match "
                f"actions {tuple(action_tensor.shape[1:])}."
            )

        states, images, img_masks, lang_tokens, lang_masks = self._fpo_policy_inputs(obs, tokenizer)

        def repeat_samples(tensor: torch.Tensor) -> torch.Tensor:
            return (
                tensor.unsqueeze(1)
                .expand(-1, n_action_samples, *tensor.shape[1:])
                .reshape(batch_size * n_action_samples, *tensor.shape[1:])
            )

        repeated_actions = repeat_samples(action_tensor)
        repeated_states = repeat_samples(states)
        repeated_timesteps = timesteps.reshape(-1).to(device=actions.device, dtype=torch.float32)
        repeated_noise = noise.reshape(batch_size * n_action_samples, *noise.shape[2:]).to(
            device=actions.device, dtype=repeated_actions.dtype
        )

        time = repeated_timesteps[:, None, None]
        x_t = time * repeated_noise + (1.0 - time) * repeated_actions
        target_velocity = repeated_noise - repeated_actions
        # The image/language prefix is independent of the sampled CFM time and
        # noise. Build its differentiable KV cache once per observation, then
        # share it across the N action samples. This preserves gradients into
        # the trainable PaliGemma prefix while avoiding N identical prefix
        # forwards per rollout slot.
        prefix_features = self.policy.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        _, prefix_pad_masks, _ = prefix_features
        past_key_values = self._build_kv_cache_from_prefix(prefix_features)
        repeated_past_key_values = {
            layer_idx: {name: repeat_samples(value) for name, value in layer_cache.items()}
            for layer_idx, layer_cache in past_key_values.items()
        }
        predicted_velocity = self.policy.denoise_step(
            repeated_states,
            repeat_samples(prefix_pad_masks),
            repeated_past_key_values,
            x_t,
            repeated_timesteps,
        )
        # PI0 pads every embodiment to ``max_action_dim``, but only dimensions
        # covered by the checkpoint's action statistics belong to the action
        # executed by the environment.  Padding dimensions are latent model
        # plumbing, not part of the behavior-policy ratio used by FPO.
        action_stats = self.action_norm_stats["q01" if self.pi05_enabled else "mean"]
        action_dim = len(action_stats)
        if action_dim > self.policy.max_action_dim:
            raise ValueError(
                f"FPO action statistics define {action_dim} dimensions, exceeding the policy maximum "
                f"of {self.policy.max_action_dim}."
            )
        per_action_loss = F.mse_loss(
            target_velocity[..., :action_dim],
            predicted_velocity[..., :action_dim],
            reduction="none",
        ).mean(dim=-1)
        return per_action_loss.reshape(batch_size, n_action_samples, action_horizon).permute(0, 2, 1)

    @override
    def fpo_forward_value(
        self,
        obs: DataProto,
        tokenizer: torch.nn.Module,
    ) -> torch.Tensor:
        if self.fpo_value_head is None:
            raise RuntimeError("FPO requires adapter.fpo.enabled=true.")
        states, images, img_masks, lang_tokens, lang_masks = self._fpo_policy_inputs(obs, tokenizer)
        with torch.no_grad():
            prefix_features = self.policy.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        return self.fpo_value_head((prefix_features, states)).float()

    @override
    def fpo_get_value_parameters(self) -> list[torch.nn.Parameter]:
        if self.fpo_value_head is None:
            raise RuntimeError("FPO requires adapter.fpo.enabled=true.")
        return list(self.fpo_value_head.parameters())

    # --- SAC Algorithm Support ---

    def _gaussian_log_prob(
        self,
        sample: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        std_safe = std.clamp_min(1e-6)
        log_prob = -0.5 * (((sample - mean) / std_safe) ** 2 + 2.0 * torch.log(std_safe) + math.log(2.0 * math.pi))
        return log_prob.mean(dim=(-1, -2))

    def _parse_task_noise_levels(
        self,
        task_noise_levels: str,
    ) -> dict[int, float]:
        normalized: dict[int, float] = {}
        if not task_noise_levels:
            return normalized
        for item in task_noise_levels.split(","):
            task_id, noise_level = item.split(":", 1)
            normalized_task_id = int(task_id)
            normalized_noise_level = float(noise_level)
            if normalized_noise_level < 0:
                raise ValueError(
                    f"flow_sde_task_noise_level[{normalized_task_id}] must be non-negative, "
                    f"got {normalized_noise_level}."
                )
            normalized[normalized_task_id] = normalized_noise_level
        return normalized

    def _resolve_flow_sde_noise_levels(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        task_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        noise_levels = torch.full((batch_size,), self.flow_sde_noise_level, device=device, dtype=dtype)
        task_ids = task_ids.to(device=device, dtype=torch.long).reshape(-1)
        if task_ids.shape[0] != batch_size:
            raise ValueError(f"task_ids batch size {task_ids.shape[0]} does not match batch size {batch_size}")

        for task_id, task_noise_level in self.flow_sde_task_noise_level.items():
            task_mask = task_ids == task_id
            if task_mask.any():
                noise_levels = noise_levels.masked_fill(task_mask, task_noise_level)
        normal_noise_factors = (torch.randn_like(noise_levels) / 6.0 + 0.5).clamp(0.0, 1.0)
        return noise_levels * normal_noise_factors

    def _sample_actions_flow_sde(
        self,
        state_features: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        initial_noise: torch.Tensor,
        noise_scale: float,
        requires_grad: bool,
        return_log_prob: bool,
        task_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        add noise to the action sampling process using Flow-SDE method.
        see https://arxiv.org/abs/2510.25889
        """

        prefix_features, states = state_features[:2]
        prefix_embs, prefix_pad_masks, _ = prefix_features
        batch_size = prefix_embs.shape[0]
        device = prefix_embs.device
        noise_levels = self._resolve_flow_sde_noise_levels(
            batch_size=batch_size,
            device=device,
            dtype=prefix_embs.dtype,
            task_ids=task_ids,
        )

        with torch.set_grad_enabled(requires_grad):
            past_key_values = self._build_kv_cache_from_prefix(prefix_features)
        x_t = initial_noise.to(device=device, dtype=prefix_embs.dtype)

        timesteps = torch.linspace(1.0, 0.0, self.policy.num_steps + 1, dtype=torch.float32, device=device)
        step_log_probs: list[torch.Tensor] = []

        for idx in range(self.policy.num_steps):
            t_cur = timesteps[idx]
            t_next = timesteps[idx + 1]
            delta = (t_cur - t_next).clamp_min(1e-6)

            if requires_grad:
                v_t = self.policy.denoise_step(
                    states,
                    prefix_pad_masks,
                    past_key_values,
                    x_t,
                    t_cur.expand(batch_size),
                )
            else:
                with torch.no_grad():
                    v_t = self.policy.denoise_step(
                        states,
                        prefix_pad_masks,
                        past_key_values,
                        x_t,
                        t_cur.expand(batch_size),
                    )

            t_cur_safe = t_cur.clamp(min=1e-4, max=1.0 - 1e-4)
            t_cur_exp = t_cur_safe.view(1, 1, 1)
            t_next_exp = t_next.view(1, 1, 1)
            delta_exp = delta.view(1, 1, 1)

            x0_pred = x_t - v_t * t_cur_exp
            x1_pred = x_t + v_t * (1.0 - t_cur_exp)

            if noise_scale > 0:
                sigma = noise_levels * noise_scale * torch.sqrt(t_cur_safe / (1.0 - t_cur_safe))
                sigma_exp = sigma.view(batch_size, 1, 1)
                x0_weight = 1.0 - t_next_exp
                x1_weight = t_next_exp - sigma_exp.pow(2) * delta_exp / (2.0 * t_cur_exp)
                x_mean = x0_pred * x0_weight + x1_pred * x1_weight
                sigma_t = torch.sqrt(delta_exp) * sigma_exp
                eps = torch.randn_like(x_t)
                x_prev = x_mean + sigma_t * eps
            else:
                x0_weight = 1.0 - t_next_exp
                x1_weight = t_next_exp
                x_mean = x0_pred * x0_weight + x1_pred * x1_weight
                sigma_t = torch.zeros_like(x_mean)
                x_prev = x_mean

            if return_log_prob:
                step_log_probs.append(self._gaussian_log_prob(x_prev, x_mean, sigma_t))

            x_t = x_prev

        if return_log_prob:
            log_probs = torch.stack(step_log_probs, dim=1).sum(dim=1)
        else:
            log_probs = None

        return x_t, log_probs

    def _build_kv_cache_from_prefix(
        self,
        prefix_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ):
        """Build a prefix KV cache under the caller's gradient context."""
        prefix_embs, prefix_pad_masks, prefix_att_masks = prefix_features
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        _, past_key_values = self.policy.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.policy.use_cache,
            fill_kv_cache=True,
            adarms_cond=[None, None],
        )
        return past_key_values

    @override
    def sac_init(self):
        """Initialize SAC-related components."""

        if self.dsrl is not None:
            self.policy.requires_grad_(False)
            self.policy.eval()
        self.freeze_vision_tower()
        forward_methods = [
            "sft_loss",
            "sac_sample_actions",
            "sac_forward_critic",
            "sac_forward_actor",
            "sac_forward_state_features",
        ]
        for method in forward_methods:
            register_fsdp_forward_method(self, method)

    @torch.no_grad()
    def sac_sample_actions(
        self,
        obs: DataProto,
        tokenizer: torch.nn.Module | None = None,
        eval: bool = False,
    ) -> Pi0Output:
        return self.sample_actions(obs, tokenizer, eval)

    @torch.no_grad()
    def sac_get_critic_value(
        self,
        obs: DataProto,
        actions: ModelOutput,
        tokenizer: nn.Module | None = None,
    ) -> torch.Tensor:
        actions = cast(Pi0Output, actions)
        state_features = self.sac_forward_state_features(obs, tokenizer)
        task_ids = None
        if self.critic_api.uses_task_ids:
            task_ids = torch.tensor(obs.non_tensor_batch["task_id"], device=actions.action.device, dtype=torch.long)
        a = {"action": actions.action}
        if actions.steering_noise is not None:
            a["steering_noise"] = actions.steering_noise
        critic_q_values = self.sac_forward_critic(
            a=a,
            state_features=state_features,
            task_ids=task_ids,
            use_target_network=False,
            method="min",
            requires_grad=False,
        )
        return critic_q_values.detach().float()

    @override
    def sac_forward_actor(
        self,
        state_features: tuple[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            torch.Tensor,
        ],
        task_ids: torch.Tensor | None = None,
        is_first_micro_batch: bool = False,
        noise_scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, float]]:
        actor_metrics: dict[str, float] = {}
        if self.dsrl is not None:
            features, state = self._dsrl_actor_inputs(state_features)
            return self.dsrl.sample(
                features,
                state,
                noise_scale=noise_scale,
            )
        prefix_features, _ = state_features
        batch_size = prefix_features[0].shape[0]
        shape = (batch_size, self.policy.n_action_steps, self.policy.max_action_dim)
        initial_noise = self.policy.sample_noise(shape, prefix_features[0].device)
        if self.flow_sde_enable:
            resolved_noise_scale = self.flow_sde_train_noise_scale if noise_scale is None else noise_scale
            actions, log_probs = self._sample_actions_flow_sde(
                state_features=state_features,
                initial_noise=initial_noise,
                noise_scale=resolved_noise_scale,
                requires_grad=True,
                return_log_prob=True,
                task_ids=task_ids,
            )
            actor_metrics = {
                "flow_sde_noise_level": self.flow_sde_noise_level,
                "flow_sde_noise_scale": float(resolved_noise_scale),
            }
        else:
            actions, log_probs = self._sample_actions_flow_sde(
                state_features=state_features,
                initial_noise=initial_noise,
                noise_scale=0.0,
                requires_grad=True,
                return_log_prob=False,
                task_ids=task_ids,
            )
        _, pi0_output_cls = self._get_pi0_embodiment_classes()
        pi0_output = pi0_output_cls.from_model_output(
            {
                "full_action": self.action_unnormalize_transform(actions),
                "log_probs": log_probs,
                "action_chunk_size": self.action_chunk_size,
            }
        )
        return pi0_output.action, pi0_output.log_prob, actor_metrics

    @override
    def sac_forward_critic(
        self,
        a: dict[str, torch.Tensor],
        state_features: tuple[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            torch.Tensor,
        ],
        task_ids: torch.Tensor | None = None,
        *,
        use_target_network: bool = False,
        method: Literal["cat", "min"] = "cat",
        requires_grad: bool = False,
    ):
        if self.dsrl is not None:
            a = self.dsrl.select_critic_noise(a)
        if self.critic_api.uses_task_ids and task_ids is None:
            raise ValueError(f"critic_type={self.critic_type} requires task_ids for critic forward.")
        return self.critic_api.forward(
            self,
            a=a,
            state_features=state_features,
            task_ids=task_ids,
            use_target_network=use_target_network,
            method=method,
            requires_grad=requires_grad,
        )

    @override
    def sac_get_critic_parameters(self) -> list[torch.nn.Parameter]:
        return self.critic_api.get_critic_parameters(self)

    @override
    def sac_get_named_actor_parameters(self) -> list[tuple[str, torch.nn.Parameter]]:
        if self.dsrl is not None:
            return self.dsrl.named_actor_parameters()
        named_parameters = [(name, param) for name, param in self.policy.named_parameters() if param.requires_grad]
        return named_parameters

    @override
    def sac_forward_state_features(
        self,
        obs: DataProto,
        tokenizer: torch.nn.Module,
    ):
        pi0_input_cls, _ = self._get_pi0_embodiment_classes()
        pi0_input = pi0_input_cls.from_env_obs(obs)

        with torch.no_grad():
            state = self.state_normalize_transform(pi0_input.state)
            needs_prefix = self.dsrl is None or self.dsrl.actor_type != "cnn" or self.critic_type != "cnn"
            if not needs_prefix:
                pixels = obs.batch["observation.images.image"]
                critic_state = obs.batch["observation.state"]
                return ((), state, (pixels, critic_state))

            images, _ = self.image_transform.call_batch(pi0_input.images)
            lang_tokens, lang_masks = self.prompt_tokenizer_transform.call_batch(
                {"task": pi0_input.task, "observation.state": state}, tokenizer
            )
            prefix_features = self.policy.embed_prefix(
                images=images,
                img_masks=pi0_input.img_masks,
                lang_tokens=lang_tokens,
                lang_masks=lang_masks,
            )
        if self.critic_type == "cnn" or (self.dsrl is not None and self.dsrl.actor_type == "cnn"):
            pixels = obs.batch["observation.images.image"]
            critic_state = obs.batch["observation.state"]
            return (prefix_features, state, (pixels, critic_state))
        return (prefix_features, state)

    @override
    @torch.no_grad()
    def sac_update_target_network(self, tau: float):
        self.critic_api.update_target_network(self, tau)
