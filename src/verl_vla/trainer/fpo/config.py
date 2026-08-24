# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

from dataclasses import dataclass, field

from verl.base_config import BaseConfig

__all__ = ["FPOTrainerConfig"]


@dataclass
class FPOTrainerConfig(BaseConfig):
    """End-to-end on-policy collection and evaluation settings for vanilla FPO."""

    _target_: str = "verl_vla.trainer.fpo.config.FPOTrainerConfig"

    project_name: str = "vla-fpo"
    experiment_name: str = "libero-spatial"
    logger: list[str] = field(default_factory=lambda: ["console"])
    total_training_steps: int = 1000
    gamma: float = 0.995
    gae_lambda: float = 0.99
    step_penalty: float = 0.0
    async_rollout: bool = False
    save_freq: int = 50
    test_freq: int = 25
    eval_episodes: int = 50
    val_before_train: bool = True
    val_only: bool = False
    esi_redundant_time: int = 0

    def __post_init__(self):
        if self.total_training_steps <= 0:
            raise ValueError(f"total_training_steps must be positive, got {self.total_training_steps}")
        if not 0 < self.gamma <= 1:
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        if not 0 <= self.gae_lambda <= 1:
            raise ValueError(f"gae_lambda must be in [0, 1], got {self.gae_lambda}")
        if self.eval_episodes == 0:
            raise ValueError("eval_episodes must be positive or negative to use the benchmark default")
