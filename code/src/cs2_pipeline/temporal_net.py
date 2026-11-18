"""Small FP32 Transformer and CUDA training loop for the trailing-window ablation."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from cs2_pipeline.schema import PipelineError


class WindowTransformer(nn.Module):
    """Use the last token for prediction; every unmasked token is at or before the target."""

    def __init__(self, n_features: int, max_length: int, spec: dict[str, Any]) -> None:
        super().__init__()
        width = spec["hidden_dim"]
        self.project = nn.Linear(n_features, width)
        frequency = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
        angle = torch.arange(1 - max_length, 1, dtype=torch.float32)[:, None] * frequency
        position = torch.empty(max_length, width)
        position[:, 0::2] = angle.sin()
        position[:, 1::2] = angle.cos()
        self.register_buffer("position", position, persistent=False)
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=width,
                nhead=spec["heads"],
                dim_feedforward=spec["feed_forward_dim"],
                dropout=spec["dropout"],
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(spec["layers"])
        )
        self.norm = nn.LayerNorm(width)
        self.readout = nn.Linear(width, 1)

    def forward(self, snapshots: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        tokens = self.project(snapshots) + self.position[-snapshots.size(1) :]
        for block in self.blocks:
            tokens = block(tokens, src_key_padding_mask=padding)
        return self.readout(self.norm(tokens[:, -1])).squeeze(-1)


@dataclass(frozen=True)
class DeviceWindows:
    values: torch.Tensor
    rows: torch.Tensor
    targets: torch.Tensor
    weights: torch.Tensor

    @classmethod
    def from_arrays(
        cls,
        values: np.ndarray,
        rows: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray,
        device: torch.device,
    ) -> DeviceWindows:
        features = torch.as_tensor(values, device=device, dtype=torch.float32)
        return cls(
            functional.pad(features, (0, 0, 1, 0)),
            torch.as_tensor(rows, device=device, dtype=torch.long),
            torch.tensor(targets, device=device, dtype=torch.float32),
            torch.as_tensor(weights, device=device, dtype=torch.float32),
        )

    def batch(self, selection: slice | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows = self.rows[selection]
        return self.values[rows + 1], rows < 0


def cuda_environment(device_name: str) -> tuple[torch.device, dict[str, Any]]:
    if not device_name.startswith("cuda:") or not torch.cuda.is_available():
        raise PipelineError(
            "CUDA is unavailable. Install the temporal extra and check the NVIDIA driver; "
            "this experiment will not fall back to the CPU."
        )
    device = torch.device(device_name)
    properties = torch.cuda.get_device_properties(device)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    return device, {
        "device": str(device),
        "gpu_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "device_total_mib": properties.total_memory / 2**20,
        "torch": str(torch.__version__),
        "torch_cuda_runtime": torch.version.cuda,
        "dtype": "float32",
        "matmul_precision": torch.get_float32_matmul_precision(),
        "attention_backend": "math",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def train_epoch(
    model: WindowTransformer,
    optimizer: torch.optim.Optimizer,
    data: DeviceWindows,
    batch_size: int,
    shuffle: torch.Generator,
    *,
    verify_step: bool = False,
) -> dict[str, Any]:
    model.train()
    device = data.values.device
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    order = torch.randperm(len(data.targets), generator=shuffle).to(device)
    weighted_total = torch.zeros((), device=device)
    normalizer = data.weights.mean() * batch_size
    evidence = None
    for step, rows in enumerate(order.split(batch_size)):
        snapshots, padding = data.batch(rows)
        checking = verify_step and step == 0
        if checking:
            torch.cuda.synchronize(device)
            step_start = time.perf_counter()
            before = model.readout.weight.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        logits = model(snapshots, padding)
        losses = functional.binary_cross_entropy_with_logits(
            logits, data.targets[rows], reduction="none"
        )
        weighted = (losses * data.weights[rows]).sum()
        loss = weighted / normalizer
        loss.backward()
        optimizer.step()
        weighted_total += weighted.detach()
        if checking:
            grad = model.readout.weight.grad
            if grad is None or not all(
                value.device == device and value.dtype == torch.float32
                for value in (snapshots, logits, loss, grad)
            ):
                raise PipelineError("The forward or backward pass was not FP32 on CUDA.")
            grad_norm = grad.norm().item()
            update = (model.readout.weight.detach() - before).abs().max().item()
            loss_value = loss.item()
            torch.cuda.synchronize(device)
            if (
                not all(map(math.isfinite, (loss_value, grad_norm, update)))
                or grad_norm <= 0
                or update <= 0
            ):
                raise PipelineError("The CUDA smoke batch did not produce a finite optimizer step.")
            evidence = {
                "forward_device": str(logits.device),
                "gradient_device": str(grad.device),
                "gradient_l2": grad_norm,
                "max_parameter_change": update,
                "loss": loss_value,
                "seconds": time.perf_counter() - step_start,
                "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            }
    log_loss = (weighted_total / data.weights.sum()).item()
    torch.cuda.synchronize(device)
    if not math.isfinite(log_loss):
        raise PipelineError("Training produced a nonfinite epoch loss.")
    return {
        "train_log_loss": log_loss,
        "steps": math.ceil(len(data.targets) / batch_size),
        "seconds": time.perf_counter() - start,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        "first_step": evidence,
    }


@torch.inference_mode()
def predict(model: WindowTransformer, data: DeviceWindows, batch_size: int) -> np.ndarray:
    model.eval()
    logits = torch.cat(
        [
            model(*data.batch(slice(start, start + batch_size)))
            for start in range(0, len(data.targets), batch_size)
        ]
    )
    return logits.to(device="cpu", dtype=torch.float64).numpy()
