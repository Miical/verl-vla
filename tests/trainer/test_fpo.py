# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

from types import SimpleNamespace

import numpy as np
import torch
from verl import DataProto

from verl_vla.trainer.fpo import prepare_fpo_actor_input
from verl_vla.workers.engine.fpo.training_worker import clipped_policy_loss, compute_gae


def test_prepare_fpo_actor_input_preserves_temporal_contract_and_masks_terminal_chunk():
    obs_states = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    rollout_end_states = torch.tensor([[100, 101, 102, 103], [200, 201, 202, 203]], dtype=torch.float32)
    rollout = DataProto.from_dict(
        tensors={
            "obs.state": obs_states,
            "action.action": torch.zeros(2, 3, 3, 2),
            "action.full_action": torch.zeros(2, 3, 5, 4),
            "next.reward": torch.tensor(
                [
                    [[1.0, 2.0, 99.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                    [[0.0, 1.0, 2.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                ]
            ),
            "next.terminated": torch.tensor(
                [
                    [[False, True, True], [False, False, False], [False, False, False]],
                    [[False, False, False], [False, False, False], [False, False, False]],
                ]
            ),
            "next.truncated": torch.zeros(2, 3, 3, dtype=torch.bool),
            "next.success": torch.zeros(2, 3, 3, dtype=torch.bool),
        },
        non_tensors={
            "obs.task": np.array([["a0", "a1", "a2"], ["b0", "b1", "b2"]], dtype=object),
        },
    )

    output = prepare_fpo_actor_input(
        rollout,
        DataProto.from_dict(
            tensors={"state": rollout_end_states},
            non_tensors={"task": np.array(["a3", "b3"], dtype=object)},
        ),
        trainer_config=SimpleNamespace(step_penalty=0.25, gamma=0.995, gae_lambda=0.99),
        global_steps=7,
    )

    assert output.batch["info.valids"].shape == (2, 3)
    assert output.batch["info.action_valids"].shape == (2, 3, 5)
    torch.testing.assert_close(
        output.batch["next_obs.state"],
        torch.cat([obs_states[:, 1:], rollout_end_states.unsqueeze(1)], dim=1),
    )
    np.testing.assert_array_equal(
        output.non_tensor_batch["next_obs.task"],
        np.array([["a1", "a2", "a3"], ["b1", "b2", "b3"]], dtype=object),
    )
    gamma = 0.995
    torch.testing.assert_close(
        output.batch["info.rewards"][0],
        torch.tensor([1 + gamma * 2 - 0.25, 3 + gamma * 4 + gamma**2 * 5 - 0.25, 6 + gamma * 7 + gamma**2 * 8 - 0.25]),
    )
    torch.testing.assert_close(output.batch["info.dones"][0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(output.batch["info.discounts"][0], torch.tensor([0.0, gamma**3, gamma**3]))
    torch.testing.assert_close(output.batch["info.action_valids"][0, 0], torch.tensor([1, 1, 0, 0, 0]).float())
    assert output.meta_info["global_steps"] == 7
    assert output.meta_info["gamma"] == 0.995


def test_compute_gae_stops_at_terminal_and_padding_boundaries():
    rewards = torch.tensor([[1.0, 2.0, 100.0, 100.0]])
    values = torch.zeros_like(rewards)
    next_values = torch.tensor([[10.0, 10.0, 10.0, 10.0]])
    discounts = torch.tensor([[0.5, 0.0, 0.5, 0.5]])
    gae_discounts = torch.tensor([[0.5, 0.0, 0.5, 0.5]])
    valids = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    advantages, returns = compute_gae(
        rewards,
        values,
        next_values,
        discounts,
        gae_discounts,
        valids,
    )

    torch.testing.assert_close(advantages, torch.tensor([[7.0, 2.0, 0.0, 0.0]]))
    torch.testing.assert_close(returns, advantages)


def test_clipped_policy_loss_uses_cfm_loss_difference_as_log_ratio():
    log_ratio = torch.log(torch.tensor([1.2, 0.8, 4.0]))
    advantages = torch.tensor([1.0, -1.0, 100.0])
    valids = torch.tensor([1.0, 1.0, 0.0])

    loss, metrics = clipped_policy_loss(log_ratio, advantages, valids, clip_coef=0.1)

    # positive advantage clips 1.2 to 1.1; negative advantage clips 0.8 to 0.9.
    torch.testing.assert_close(loss, torch.tensor((-1.1 + 0.9) / 2))
    torch.testing.assert_close(metrics["clip_fraction"], torch.tensor(1.0))
