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

from __future__ import annotations

from torch import nn

from verl_vla.models.pi0_torch import PI0AdapterConfig
from verl_vla.models.pi0_torch.model.paligemma_with_expert import PaliGemmaWithExpertModel


def _linear() -> nn.Linear:
    return nn.Linear(2, 2)


def _dual_stream_layer() -> nn.Module:
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.q_proj = nn.ModuleList([_linear(), _linear()])
    layer.self_attn.k_proj = nn.ModuleList([_linear(), _linear()])
    layer.self_attn.v_proj = nn.ModuleList([_linear(), _linear()])
    layer.self_attn.o_proj = nn.ModuleList([_linear(), _linear()])
    layer.mlps = nn.ModuleList([_linear(), _linear()])
    layer.input_layernorms = nn.ModuleList([nn.LayerNorm(2), nn.LayerNorm(2)])
    layer.post_attention_layernorms = nn.ModuleList([nn.LayerNorm(2), nn.LayerNorm(2)])
    return layer


def _tiny_paligemma_with_expert() -> PaliGemmaWithExpertModel:
    model = PaliGemmaWithExpertModel.__new__(PaliGemmaWithExpertModel)
    nn.Module.__init__(model)
    model.vision_tower = _linear()
    model.multi_modal_projector = _linear()
    model.embed_tokens = nn.Embedding(4, 2)
    model.layers = nn.ModuleList([_dual_stream_layer()])
    model.norms = nn.ModuleList([nn.LayerNorm(2), nn.LayerNorm(2)])
    return model


def test_pi0_vlm_freezing_is_enabled_by_default() -> None:
    default_config = PI0AdapterConfig(policy_config={"pi05_enabled": True})
    unfrozen_config = PI0AdapterConfig(
        policy_config={"pi05_enabled": True},
        freeze_vlm_backbone=False,
    )

    assert default_config.freeze_vlm_backbone is True
    assert unfrozen_config.freeze_vlm_backbone is False


def test_freeze_vlm_backbone_preserves_action_expert_parameters() -> None:
    model = _tiny_paligemma_with_expert()
    layer = model.layers[0]
    vlm_modules = [
        model.vision_tower,
        model.multi_modal_projector,
        model.embed_tokens,
        model.norms[0],
        layer.self_attn.q_proj[0],
        layer.self_attn.k_proj[0],
        layer.self_attn.v_proj[0],
        layer.self_attn.o_proj[0],
        layer.mlps[0],
        layer.input_layernorms[0],
        layer.post_attention_layernorms[0],
    ]
    expert_modules = [
        model.norms[1],
        layer.self_attn.q_proj[1],
        layer.self_attn.k_proj[1],
        layer.self_attn.v_proj[1],
        layer.self_attn.o_proj[1],
        layer.mlps[1],
        layer.input_layernorms[1],
        layer.post_attention_layernorms[1],
    ]

    model.freeze_vlm_backbone()

    assert all(not parameter.requires_grad for module in vlm_modules for parameter in module.parameters())
    assert all(not module.training for module in vlm_modules)
    assert all(parameter.requires_grad for module in expert_modules for parameter in module.parameters())
    assert all(module.training for module in expert_modules)
