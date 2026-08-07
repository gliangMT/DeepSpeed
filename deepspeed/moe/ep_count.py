# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Helpers for expert token counting in AutoEP routing paths."""

import torch

from deepspeed.accelerator import get_accelerator


def _requires_host_bincount(indices: torch.Tensor) -> bool:
    torch_version = torch.__version__.split("+", maxsplit=1)[0]
    return indices.device.type == "musa" and torch_version.startswith("2.7.")


def count_tokens_per_expert(
    selected_experts_indices: torch.Tensor,
    num_experts: int,
    *,
    out_dtype: torch.dtype = torch.float32,
    deterministic_safe: bool = False,
) -> torch.Tensor:
    """Count routed tokens per expert.

    Fast path uses ``torch.bincount`` on the current device. MUSA torch 2.7.x
    falls back to CPU because its device bincount can silently undercount
    production-sized routing tensors. ``deterministic_safe=True`` also uses
    the CPU when deterministic algorithms reject the accelerator kernel.
    """
    flat_indices = selected_experts_indices.reshape(-1).to(torch.int64)

    requires_deterministic_fallback = (deterministic_safe and torch.are_deterministic_algorithms_enabled()
                                       and get_accelerator().on_accelerator(flat_indices))
    use_host_bincount = _requires_host_bincount(flat_indices) or requires_deterministic_fallback
    if use_host_bincount:
        counts = torch.bincount(flat_indices.detach().cpu(), minlength=num_experts)
        if counts.numel() > num_experts or int(counts.sum().item()) != flat_indices.numel():
            raise RuntimeError("Expert token counting produced invalid routing metadata.")
        counts = counts.to(selected_experts_indices.device)
    else:
        counts = torch.bincount(flat_indices, minlength=num_experts)

    if counts.numel() < num_experts:
        pad = torch.zeros(num_experts - counts.numel(), device=counts.device, dtype=counts.dtype)
        counts = torch.cat([counts, pad], dim=0)
    elif counts.numel() > num_experts:
        counts = counts[:num_experts]

    return counts.to(out_dtype)
