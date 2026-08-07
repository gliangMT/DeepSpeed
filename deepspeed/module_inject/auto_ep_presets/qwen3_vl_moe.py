# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Qwen3-VL-MoE AutoEP preset."""

from __future__ import annotations

from deepspeed.module_inject.auto_ep_presets.base import MoEModelPreset, TransformersTopLevelRouterLogitsAdapter

PRESET_NAME = "qwen3_vl_moe"

PRESET = MoEModelPreset(
    moe_layer_pattern=r"model\.language_model\.layers\.\d+\.mlp",
    router_pattern="gate",
    experts_pattern="experts",
    expert_storage="fused_3d",
    expert_w1="gate_up_proj",
    expert_w2="down_proj",
    expert_w3=None,
    num_experts_attr="num_experts",
    top_k_attr="num_experts_per_tok",
    score_func="softmax",
    score_apply="post",
    route_norm=True,
    gate_bias=False,
    has_shared_experts=False,
    preset_adapter="qwen3_vl_moe",
    hf_model_types=("qwen3_vl_moe", "qwen3_vl_moe_text"),
    min_transformers_version="5.2.0",
    docs_support_notes="Supports the Qwen3-VL-MoE text backbone in Transformers 5.2 or newer.",
)


class Qwen3VLMoePresetAdapter(TransformersTopLevelRouterLogitsAdapter):
    """Resolve routing attributes from the Qwen3-VL text-backbone config."""

    def resolve_model_config(self, model_config):
        return getattr(model_config, "text_config", model_config)

PRESET_ADAPTERS = {
    "qwen3_vl_moe":
    Qwen3VLMoePresetAdapter(
        display_name="Qwen3-VL-MoE",
        hf_model_types=("qwen3_vl_moe", "qwen3_vl_moe_text"),
        class_name_fragments=("Qwen3VLMoe", ),
    ),
}
