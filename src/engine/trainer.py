"""Minimal single-device NAFNet-small baseline trainer."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

import torch

from src.engine.checkpoint import ResumeState, load_checkpoint, save_checkpoint
from src.metrics.image_metrics import rgb_psnr_per_image, rgb_ssim_per_image


def _make_grad_scaler(enabled: bool) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_context(enabled: bool) -> Any:
    if not enabled:
        return nullcontext()
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def _require_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor.")
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} contains NaN or Inf.")


def _require_finite_gradients(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"Gradient for parameter {name!r} contains NaN or Inf.")


def _gradients_are_finite(model: torch.nn.Module) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def _string_batch(value: Any, batch_size: int, field_name: str) -> List[str]:
    if isinstance(value, str):
        values = [value]
    else:
        values = [str(item) for item in value]
    if len(values) != batch_size:
        raise ValueError(
            f"Batch field {field_name!r} has {len(values)} values for batch size {batch_size}."
        )
    return values


@torch.no_grad()
def validate_model(
    model: torch.nn.Module,
    data_loader: Iterable[Mapping[str, Any]],
    *,
    device: Union[str, torch.device],
) -> Dict[str, Any]:
    """Evaluate validation only and retain every per-image RGB metric."""

    resolved_device = torch.device(device)
    was_training = model.training
    model.eval()
    records: List[Dict[str, Any]] = []
    try:
        for batch in data_loader:
            inputs = batch["input"].to(resolved_device, non_blocking=True)
            targets = batch["target"].to(resolved_device, non_blocking=True)
            _require_finite_tensor("validation input", inputs)
            _require_finite_tensor("validation target", targets)
            predictions = model(inputs)
            if predictions.shape != targets.shape:
                raise ValueError(
                    f"Validation prediction/target shape mismatch: {predictions.shape} "
                    f"versus {targets.shape}."
                )
            _require_finite_tensor("validation prediction", predictions)
            psnr_values = rgb_psnr_per_image(predictions, targets)
            ssim_values = rgb_ssim_per_image(predictions, targets)
            batch_size = predictions.shape[0]
            sample_ids = _string_batch(batch["sample_id"], batch_size, "sample_id")
            input_paths = _string_batch(
                batch["input_relative_path"], batch_size, "input_relative_path"
            )
            gt_paths = _string_batch(
                batch["gt_relative_path"], batch_size, "gt_relative_path"
            )
            for index in range(batch_size):
                records.append(
                    {
                        "sample_id": sample_ids[index],
                        "input_relative_path": input_paths[index],
                        "gt_relative_path": gt_paths[index],
                        "psnr_rgb": float(psnr_values[index].detach().cpu()),
                        "ssim_rgb": float(ssim_values[index].detach().cpu()),
                    }
                )
    finally:
        model.train(was_training)
    if not records:
        raise ValueError("Validation data loader produced no samples.")
    return {
        "num_samples": len(records),
        "psnr_rgb": sum(record["psnr_rgb"] for record in records) / len(records),
        "ssim_rgb": sum(record["ssim_rgb"] for record in records) / len(records),
        "per_image": records,
    }


class BaselineTrainer:
    """Single-device baseline trainer with optional CUDA AMP and strict resume."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_function: torch.nn.Module,
        device: Union[str, torch.device],
        amp: bool,
        gradient_clip_norm: Optional[float] = None,
        scheduler: Optional[Any] = None,
        config: Optional[Mapping[str, Any]] = None,
        random_seed: int = 0,
        git_commit: str = "unknown",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.scheduler = scheduler
        self.amp_requested = bool(amp)
        self.amp_enabled = self.amp_requested and self.device.type == "cuda"
        self.scaler = _make_grad_scaler(self.amp_enabled)
        if gradient_clip_norm is not None and gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive or None.")
        self.gradient_clip_norm = gradient_clip_norm
        self.config = dict(config or {})
        self.random_seed = random_seed
        self.git_commit = git_commit
        self.epoch = 0
        self.global_step = 0
        self.best_validation_psnr = float("-inf")

    def train_step(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        self.model.train()
        inputs = batch["input"].to(self.device, non_blocking=True)
        targets = batch["target"].to(self.device, non_blocking=True)
        _require_finite_tensor("training input", inputs)
        _require_finite_tensor("training target", targets)
        self.optimizer.zero_grad(set_to_none=True)
        with _autocast_context(self.amp_enabled):
            predictions = self.model(inputs)
            if predictions.shape != targets.shape:
                raise ValueError(
                    f"Training prediction/target shape mismatch: {predictions.shape} "
                    f"versus {targets.shape}."
                )
            _require_finite_tensor("training prediction", predictions)
            loss = self.loss_function(predictions, targets)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise FloatingPointError("Training loss must be a finite scalar.")
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        gradients_finite = _gradients_are_finite(self.model)
        if not gradients_finite and not self.amp_enabled:
            _require_finite_gradients(self.model)
        if gradients_finite and self.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_norm, error_if_nonfinite=True
            )
            _require_finite_gradients(self.model)
        amp_scale_before = float(self.scaler.get_scale())
        self.scaler.step(self.optimizer)
        self.scaler.update()
        amp_scale_after = float(self.scaler.get_scale())
        amp_overflow_detected = (
            self.amp_enabled and amp_scale_after < amp_scale_before
        )
        optimizer_step_applied = not amp_overflow_detected
        if optimizer_step_applied:
            self.global_step += 1
        if amp_overflow_detected:
            print(
                "AMP_OVERFLOW_DETECTED "
                "optimizer_step_applied=false "
                f"global_step={self.global_step} "
                f"amp_scale_before={amp_scale_before} "
                f"amp_scale_after={amp_scale_after}"
            )
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        return {
            "loss": float(loss.detach().cpu()),
            "learning_rate": learning_rate,
            "global_step": self.global_step,
            "optimizer_step_applied": optimizer_step_applied,
            "amp_overflow_detected": amp_overflow_detected,
            "amp_scale_before": amp_scale_before,
            "amp_scale_after": amp_scale_after,
        }

    def validation_epoch(
        self, data_loader: Iterable[Mapping[str, Any]]
    ) -> Dict[str, Any]:
        return validate_model(self.model, data_loader, device=self.device)

    def step_scheduler(self) -> None:
        if self.scheduler is not None:
            self.scheduler.step()

    def save(self, path: Union[str, Path], *, epoch: int) -> Path:
        return save_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=epoch,
            global_step=self.global_step,
            best_validation_psnr=self.best_validation_psnr,
            config=self.config,
            random_seed=self.random_seed,
            git_commit=self.git_commit,
        )

    def resume(self, path: Union[str, Path]) -> ResumeState:
        state = load_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            map_location="cpu",
            strict=True,
            require_training_state=True,
        )
        self.epoch = state.epoch
        self.global_step = state.global_step
        self.best_validation_psnr = state.best_validation_psnr
        return state
