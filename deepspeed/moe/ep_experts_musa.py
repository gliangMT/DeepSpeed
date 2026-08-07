# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""MUSA Transformer Engine expert backend for AutoEP."""

from __future__ import annotations

import importlib.util

import torch
import torch.nn as nn
import torch.nn.functional as F


def is_musa_te_grouped_gemm_available() -> bool:
    """Return whether the explicit MUSA TE backend can be constructed."""
    return (hasattr(torch, "musa") and importlib.util.find_spec("torch_musa") is not None
            and importlib.util.find_spec("transformer_engine") is not None)


def _te_grouped_gemm_api():
    # The MUSA TE build patches CUDA-named compatibility APIs during import.
    import torch_musa  # noqa: F401, I001

    from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm
    from transformer_engine.pytorch.module.base import get_multi_stream_cublas_workspace

    return general_grouped_gemm, get_multi_stream_cublas_workspace


def _split_sizes(tokens_per_expert: torch.Tensor, total_tokens: int) -> list[int]:
    split_sizes = tokens_per_expert.detach().to(device="cpu", dtype=torch.int64).tolist()
    if any(size < 0 for size in split_sizes):
        raise ValueError(f"Grouped GEMM token counts must be non-negative, got {split_sizes}.")
    if sum(split_sizes) > total_tokens:
        raise ValueError(f"Grouped GEMM token counts sum to {sum(split_sizes)}, exceeding {total_tokens} input rows.")
    return split_sizes


def _validate_grouped_linear_inputs(
    input: torch.Tensor,
    weight: torch.Tensor,
    tokens_per_expert: torch.Tensor,
) -> None:
    if input.device.type != "musa" or weight.device.type != "musa":
        raise RuntimeError("expert_backend='musa_te' requires MUSA input and expert weights.")
    if input.ndim != 2 or weight.ndim != 3:
        raise ValueError(f"Expected 2D input and 3D weight, got input={input.shape} and weight={weight.shape}.")
    if input.size(-1) != weight.size(-1):
        raise ValueError(f"Grouped GEMM K dimensions do not match: {input.size(-1)} != {weight.size(-1)}.")
    if weight.size(0) != tokens_per_expert.numel():
        raise ValueError(f"Expected one token count per expert, got {tokens_per_expert.numel()} counts for "
                         f"{weight.size(0)} experts.")
    if input.dtype not in (torch.float16, torch.bfloat16) or weight.dtype != input.dtype:
        raise TypeError(f"MUSA TE grouped GEMM requires matching fp16/bf16 tensors, got "
                        f"input={input.dtype} and weight={weight.dtype}.")


def _te_grouped_forward(input: torch.Tensor, weight: torch.Tensor, split_sizes: list[int]) -> torch.Tensor:
    general_grouped_gemm, get_workspaces = _te_grouped_gemm_api()
    output = torch.empty((input.size(0), weight.size(1)), device=input.device, dtype=input.dtype)
    general_grouped_gemm(
        list(weight.unbind(dim=0)),
        list(torch.split(input, split_sizes, dim=0)),
        [output],
        input.dtype,
        get_workspaces(),
        m_splits=split_sizes,
        single_output=True,
    )
    return output


def _te_grouped_input_grad(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    split_sizes: list[int],
) -> torch.Tensor:
    general_grouped_gemm, get_workspaces = _te_grouped_gemm_api()
    grad_input = torch.empty((grad_output.size(0), weight.size(2)),
                             device=grad_output.device,
                             dtype=grad_output.dtype)
    general_grouped_gemm(
        list(weight.unbind(dim=0)),
        list(torch.split(grad_output, split_sizes, dim=0)),
        list(torch.split(grad_input, split_sizes, dim=0)),
        grad_output.dtype,
        get_workspaces(),
        layout="NN",
        m_splits=split_sizes,
        grad=True,
    )
    return grad_input


def _te_grouped_weight_grad(
    input: torch.Tensor,
    grad_output: torch.Tensor,
    split_sizes: list[int],
    weight: torch.Tensor,
) -> torch.Tensor:
    general_grouped_gemm, get_workspaces = _te_grouped_gemm_api()
    grad_weight = torch.empty_like(weight)
    general_grouped_gemm(
        list(torch.split(input, split_sizes, dim=0)),
        list(torch.split(grad_output, split_sizes, dim=0)),
        list(grad_weight.unbind(dim=0)),
        grad_output.dtype,
        get_workspaces(),
        layout="NT",
        m_splits=split_sizes,
        grad=True,
    )
    return grad_weight


class _MusaTEGroupedLinear(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        _validate_grouped_linear_inputs(input, weight, tokens_per_expert)
        split_sizes = _split_sizes(tokens_per_expert, input.size(0))
        input_rows = input.size(0)
        active_rows = sum(split_sizes)
        input = input[:active_rows].contiguous()
        weight = weight.contiguous()
        ctx.save_for_backward(input, weight)
        ctx.split_sizes = split_sizes
        ctx.input_rows = input_rows
        if active_rows == 0:
            return input.new_zeros((input_rows, weight.size(1)))
        output = _te_grouped_forward(input, weight, split_sizes)
        if active_rows < input_rows:
            output = torch.vstack((output, output.new_zeros((input_rows - active_rows, output.size(1)))))
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, weight = ctx.saved_tensors
        grad_output = grad_output[:input.size(0)].contiguous()
        grad_input = grad_weight = None
        if input.size(0) == 0:
            if ctx.needs_input_grad[0]:
                grad_input = weight.new_zeros((ctx.input_rows, weight.size(2)))
            if ctx.needs_input_grad[1]:
                grad_weight = torch.zeros_like(weight)
            return grad_input, grad_weight, None
        if ctx.needs_input_grad[0]:
            grad_input = _te_grouped_input_grad(grad_output, weight, ctx.split_sizes)
            if input.size(0) < ctx.input_rows:
                grad_input = torch.vstack((grad_input,
                                           grad_input.new_zeros((ctx.input_rows - input.size(0), grad_input.size(1)))))
        if ctx.needs_input_grad[1]:
            grad_weight = _te_grouped_weight_grad(input, grad_output, ctx.split_sizes, weight)
        return grad_input, grad_weight, None


def musa_te_grouped_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    return _MusaTEGroupedLinear.apply(input, weight, tokens_per_expert)


def _swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    if hasattr(F, "swish_glu"):
        return F.swish_glu(gate_up)
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


class MusaTEGroupedExperts(nn.Module):
    """Fused gate-up AutoEP experts backed by MUSA Transformer Engine."""

    def __init__(self, dim: int, hidden_dim: int, num_experts: int) -> None:
        super().__init__()
        if not is_musa_te_grouped_gemm_available():
            raise RuntimeError("expert_backend='musa_te' requires torch_musa and a MUSA-compatible "
                               "Transformer Engine build.")
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * hidden_dim, dim))
        self.down_proj = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.gate_up_proj.is_expert_group = True
        self.down_proj.is_expert_group = True

    def forward(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        gate_up = musa_te_grouped_linear(x, self.gate_up_proj, num_tokens_per_expert)
        hidden = _swiglu(gate_up)
        return musa_te_grouped_linear(hidden, self.down_proj, num_tokens_per_expert).type_as(x)
