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

"""Trainable adapter around the native OpenVLA-OFT policy."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import functional as TVF
from verl import DataProto

from verl_vla.models.base import ModelOutput, SupportSFTTraining, TrainableVLAModelBase
from verl_vla.utils.envs.action import center_crop_image, resize_image

from .adapter_config import OpenVLAAdapterConfig
from .constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from .modeling_prismatic import OpenVLAForActionPrediction
from .processing_prismatic import PrismaticProcessor

_IMAGE_SIZE = (224, 224)
_CROP_SCALE = 0.9


def _load_action_statistics(path: str | Path, *, action_transform: str) -> dict[str, list[float] | list[bool]]:
    with Path(path).expanduser().open(encoding="utf-8") as file:
        statistics = json.load(file)["action"]

    fields = ("min", "max", "mean", "std", "q01", "q99")
    if any(len(statistics[field]) != ACTION_DIM for field in fields):
        raise ValueError(f"OpenVLA action statistics must have {ACTION_DIM} values per field")

    action_statistics = {field: list(statistics[field]) for field in fields}
    if action_transform == "libero":
        minimum = action_statistics["min"][-1]
        maximum = action_statistics["max"][-1]
        q01 = action_statistics["q01"][-1]
        q99 = action_statistics["q99"][-1]
        action_statistics["min"][-1] = (1.0 - maximum) / 2.0
        action_statistics["max"][-1] = (1.0 - minimum) / 2.0
        action_statistics["mean"][-1] = (1.0 - action_statistics["mean"][-1]) / 2.0
        action_statistics["std"][-1] /= 2.0
        action_statistics["q01"][-1] = (1.0 - q99) / 2.0
        action_statistics["q99"][-1] = (1.0 - q01) / 2.0
        action_statistics["mask"] = [True] * (ACTION_DIM - 1) + [False]
    else:
        action_statistics["mask"] = [True] * ACTION_DIM
    return action_statistics


def _random_crop_image(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    crop_height = int(height * _CROP_SCALE)
    crop_width = int(width * _CROP_SCALE)
    top = int(torch.randint(height - crop_height + 1, ()).item())
    left = int(torch.randint(width - crop_width + 1, ()).item())
    return TVF.resized_crop(
        image,
        top,
        left,
        crop_height,
        crop_width,
        (height, width),
        antialias=True,
    )


class OpenVLAOutput(ModelOutput):
    """Actions produced for the normalized environment boundary."""

    def __init__(self, action: torch.Tensor) -> None:
        self.action = action

    def to_data_proto(self) -> DataProto:
        return DataProto.from_dict(tensors={"action": self.action.float()})


class OpenVLATrainableModel(TrainableVLAModelBase, SupportSFTTraining):
    """Expose OpenVLA's native policy through verl-vla training contracts."""

    def __init__(
        self,
        policy: OpenVLAForActionPrediction,
        *,
        processor: PrismaticProcessor,
        adapter_config: Mapping | None = None,
        artifact_source_dir: str | Path | None = None,
    ) -> None:
        TrainableVLAModelBase.__init__(self, policy=policy)
        SupportSFTTraining.__init__(self, policy.config)
        self.adapter_config = OpenVLAAdapterConfig(**dict(adapter_config or {}))
        self.processor = processor
        self.artifact_source_dir = Path(artifact_source_dir) if artifact_source_dir is not None else None
        if self.adapter_config.norm_stats_path is not None:
            self.policy.norm_stats = {
                self.adapter_config.unnorm_key: {
                    "action": _load_action_statistics(
                        self.adapter_config.norm_stats_path,
                        action_transform=self.adapter_config.action_transform,
                    )
                }
            }

    def forward(self, *args, **kwargs):
        return self.policy(*args, **kwargs)

    def can_generate(self) -> bool:
        return False

    def predict_action(self, *args, **kwargs):
        return self.policy.predict_action(*args, **kwargs)

    def generate_action_verl(self, *args, **kwargs):
        return self.policy.generate_action_verl(*args, **kwargs)

    def _prepare_rollout_policy_input(self, obs: DataProto) -> dict[str, torch.Tensor]:
        images = obs.batch[self.adapter_config.image_key]
        tasks = obs.non_tensor_batch[self.adapter_config.task_key]
        if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != torch.uint8:
            raise ValueError(f"OpenVLA rollout expects BHWC uint8 images, got {tuple(images.shape)} {images.dtype}")
        if len(tasks) != len(images):
            raise ValueError(f"OpenVLA task batch has {len(tasks)} entries for {len(images)} images")

        input_ids = []
        attention_masks = []
        pixel_values = []
        for task, image in zip(tasks, images, strict=True):
            resized = resize_image(image.detach().cpu().numpy(), _IMAGE_SIZE)
            pil_image = center_crop_image(Image.fromarray(resized).convert("RGB"))
            prompt = self.adapter_config.prompt_template.format(task=str(task).lower())
            features = self.processor(prompt, pil_image)
            ids = features["input_ids"]
            mask = features["attention_mask"]
            if not torch.all(ids[:, -1] == 29871):
                ids = torch.cat([ids, ids.new_tensor([[29871]])], dim=1)
                mask = torch.cat([mask, mask.new_ones((1, 1))], dim=1)
            input_ids.append(ids.transpose(0, 1))
            attention_masks.append(mask.transpose(0, 1))
            pixel_values.append(features["pixel_values"])

        parameter = next(self.policy.parameters())
        ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
        ).squeeze(-1)
        masks = pad_sequence(attention_masks, batch_first=True, padding_value=0).squeeze(-1)
        padding = ids.eq(self.processor.tokenizer.pad_token_id).int()
        left_padding_order = torch.argsort(padding, dim=1, descending=True, stable=True)
        return {
            "input_ids": torch.gather(ids, 1, left_padding_order).to(parameter.device),
            "attention_mask": torch.gather(masks, 1, left_padding_order).to(parameter.device),
            "pixel_values": torch.cat(pixel_values).to(device=parameter.device, dtype=parameter.dtype),
        }

    def _environment_actions(self, actions: np.ndarray, *, batch_size: int) -> torch.Tensor:
        actions = np.asarray(actions)
        expected_shape = (batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM)
        if actions.shape != expected_shape:
            raise ValueError(f"OpenVLA generated actions must have shape {expected_shape}, got {actions.shape}")
        actions = actions.copy()
        if self.adapter_config.action_transform == "libero":
            # Native LIBERO RLDS actions use 0=close and 1=open; robosuite
            # expects +1=close and -1=open.
            actions[..., -1] = -np.sign(2.0 * actions[..., -1] - 1.0)
        return torch.as_tensor(actions, device=next(self.policy.parameters()).device, dtype=torch.float32)

    @torch.no_grad()
    def sample_actions(self, obs: DataProto, *, eval: bool = True) -> OpenVLAOutput:
        policy_input = self._prepare_rollout_policy_input(obs)
        actions, _responses = self.policy.generate_action_verl(
            **policy_input,
            padding_idx=self.processor.tokenizer.pad_token_id,
            do_sample=not eval,
            unnorm_key=self.adapter_config.unnorm_key,
            temperature=1.0,
        )
        return OpenVLAOutput(self._environment_actions(actions, batch_size=len(obs)))

    def sac_sample_actions(
        self,
        obs: DataProto,
        tokenizer: torch.nn.Module | None = None,
        eval: bool = False,
    ) -> OpenVLAOutput:
        """Serve deterministic evaluation through the shared HF rollout RPC."""

        del tokenizer
        return self.sample_actions(obs, eval=eval)

    def _prepare_sft_policy_input(self, obs: DataProto) -> dict[str, torch.Tensor]:
        images = obs.batch[self.adapter_config.image_key]
        tasks = obs.non_tensor_batch[self.adapter_config.task_key]
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"OpenVLA SFT expects BCHW float images, got {tuple(images.shape)}")
        if len(tasks) != len(images):
            raise ValueError(f"OpenVLA task batch has {len(tasks)} entries for {len(images)} images")

        input_ids = []
        attention_masks = []
        pixel_values = []
        for task, image in zip(tasks, images, strict=True):
            image_array = image.detach().cpu().permute(1, 2, 0).clamp(0, 1).mul(255).to(torch.uint8).numpy()
            resized = resize_image(image_array, _IMAGE_SIZE)
            pil_image = _random_crop_image(Image.fromarray(resized))
            prompt = self.adapter_config.prompt_template.format(task=str(task).lower())
            features = self.processor(prompt, pil_image)
            ids = features["input_ids"]
            mask = features["attention_mask"]
            if not torch.all(ids[:, -1] == 29871):
                ids = torch.cat([ids, ids.new_tensor([[29871]])], dim=1)
                mask = torch.cat([mask, mask.new_ones((1, 1))], dim=1)
            input_ids.append(ids.transpose(0, 1))
            attention_masks.append(mask.transpose(0, 1))
            pixel_values.append(features["pixel_values"])

        parameter = next(self.policy.parameters())
        ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
        ).squeeze(-1)
        masks = pad_sequence(attention_masks, batch_first=True, padding_value=0).squeeze(-1)
        padding = ids.eq(self.processor.tokenizer.pad_token_id).int()
        left_padding_order = torch.argsort(padding, dim=1, descending=True, stable=True)
        return {
            "input_ids": torch.gather(ids, 1, left_padding_order).to(parameter.device),
            "attention_mask": torch.gather(masks, 1, left_padding_order).to(parameter.device),
            "pixel_values": torch.cat(pixel_values).to(device=parameter.device, dtype=parameter.dtype),
        }

    def _normalize_sft_actions(self, actions: torch.Tensor) -> torch.Tensor:
        stats = self.native_policy.get_action_stats(self.adapter_config.unnorm_key)
        actions = actions.clone()
        if self.adapter_config.action_transform == "libero":
            # LeRobot stores robosuite gripper commands (+1 close, -1 open),
            # while the native LIBERO RLDS data stores 0 close and 1 open.
            actions[..., -1] = (1.0 - actions[..., -1]) / 2.0

        low = actions.new_tensor(stats["q01"])
        high = actions.new_tensor(stats["q99"])
        mask = torch.as_tensor(stats.get("mask", np.ones(len(low), dtype=bool)), device=actions.device)
        normalized = 2.0 * (actions - low) / (high - low + 1e-8) - 1.0
        return torch.where(mask, normalized, actions).clamp(-1.0, 1.0)

    def _action_classes(self, normalized_actions: torch.Tensor) -> torch.Tensor:
        bins = normalized_actions.new_tensor(self.native_policy.bins)
        discretized = torch.bucketize(normalized_actions, bins, right=True).clamp(1, len(bins))
        return len(bins) - discretized

    def sft_loss(
        self,
        obs: DataProto,
        tokenizer: torch.nn.Module,
        actions: dict[str, torch.Tensor],
        valids: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        target_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del tokenizer, target_values
        action_tensor = actions["action"]
        if action_tensor.ndim != 3 or action_tensor.shape[1:] != (8, 7):
            raise ValueError(f"OpenVLA SFT actions must have shape (batch, 8, 7), got {tuple(action_tensor.shape)}")

        policy_input = self._prepare_sft_policy_input(obs)
        logits = self.policy(**policy_input)
        action_logits = logits[..., -320:-64]
        targets = self._action_classes(self._normalize_sft_actions(action_tensor)).reshape(action_tensor.shape[0], -1)
        token_loss = F.cross_entropy(action_logits.transpose(1, 2), targets, reduction="none")

        if action_mask is None:
            token_mask = torch.ones_like(token_loss)
        else:
            if action_mask.shape != action_tensor.shape[:2]:
                raise ValueError(
                    f"OpenVLA SFT action mask must have shape {tuple(action_tensor.shape[:2])}, "
                    f"got {tuple(action_mask.shape)}"
                )
            token_mask = action_mask.to(token_loss.dtype).repeat_interleave(action_tensor.shape[-1], dim=1)
        token_mask = token_mask * valids.to(token_loss.dtype).unsqueeze(1)
        loss = (token_loss * token_mask).sum() / token_mask.sum().clamp_min(1.0)

        predictions = action_logits.argmax(dim=-1)
        self.sft_metrics = {
            "action_token_accuracy": ((predictions == targets) * token_mask.bool()).sum()
            / token_mask.sum().clamp_min(1.0)
        }
        return loss

    def export_policy(self, output_dir, *, state_dict=None) -> None:
        output_dir = Path(output_dir)
        policy_state = self.extract_policy_state_dict(state_dict) if state_dict is not None else None
        self.native_policy.save_pretrained(output_dir, state_dict=policy_state, safe_serialization=True)
        self.processor.save_pretrained(output_dir)
        with (output_dir / "dataset_statistics.json").open("w", encoding="utf-8") as file:
            json.dump(self.native_policy.norm_stats, file)

        module_dir = Path(__file__).parent
        for filename in (
            "configuration_prismatic.py",
            "constants.py",
            "modeling_prismatic.py",
            "processing_prismatic.py",
            "train_utils.py",
        ):
            shutil.copy2(module_dir / filename, output_dir / filename)
        if self.artifact_source_dir is not None:
            for filename in ("added_tokens.json", "tokenizer.model"):
                source = self.artifact_source_dir / filename
                if source.is_file():
                    shutil.copy2(source, output_dir / filename)


__all__ = ["OpenVLATrainableModel"]
