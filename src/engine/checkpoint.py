"""Validated, atomic baseline checkpoint persistence."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import torch


CHECKPOINT_SCHEMA_VERSION = 1
REQUIRED_CHECKPOINT_FIELDS = {
    "schema_version",
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "scaler_state_dict",
    "epoch",
    "global_step",
    "best_validation_psnr",
    "config",
    "random_seed",
    "git_commit",
    "creation_timestamp_utc",
}


@dataclass(frozen=True)
class ResumeState:
    epoch: int
    global_step: int
    best_validation_psnr: float
    config: Mapping[str, Any]
    random_seed: int
    git_commit: str
    creation_timestamp_utc: str
    optimizer_restored: bool
    scheduler_restored: bool
    scaler_restored: bool


def _timestamp_utc() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _state_dict_or_none(component: Optional[Any]) -> Optional[Dict[str, Any]]:
    return None if component is None else component.state_dict()


def save_checkpoint(
    path: Union[str, Path],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[Any],
    epoch: int,
    global_step: int,
    best_validation_psnr: float,
    config: Mapping[str, Any],
    random_seed: int,
    git_commit: str,
) -> Path:
    """Atomically save all fields required for strict training resume."""

    if type(epoch) is not int or epoch < 0:
        raise ValueError(f"epoch must be a non-negative integer, got {epoch!r}.")
    if type(global_step) is not int or global_step < 0:
        raise ValueError(
            f"global_step must be a non-negative integer, got {global_step!r}."
        )
    if type(random_seed) is not int or random_seed < 0:
        raise ValueError(f"random_seed must be non-negative, got {random_seed!r}.")
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping.")
    if not isinstance(git_commit, str) or not git_commit:
        raise ValueError("git_commit must be a non-empty string.")

    destination = Path(path).expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": _state_dict_or_none(scheduler),
        "scaler_state_dict": _state_dict_or_none(scaler),
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_psnr": float(best_validation_psnr),
        "config": dict(config),
        "random_seed": random_seed,
        "git_commit": git_commit,
        "creation_timestamp_utc": _timestamp_utc(),
    }

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return destination


def _torch_load(path: Path, map_location: Union[str, torch.device]) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_checkpoint(
    path: Union[str, Path],
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = True,
    require_training_state: bool = False,
) -> ResumeState:
    """Load a validated checkpoint and optionally restore all training state."""

    checkpoint_path = Path(path).expanduser().resolve(strict=False)
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = _torch_load(checkpoint_path, map_location)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint root must be a mapping.")
    missing = sorted(REQUIRED_CHECKPOINT_FIELDS - set(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {missing}")
    if checkpoint["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported checkpoint schema_version={checkpoint['schema_version']!r}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}."
        )
    if not isinstance(checkpoint["model_state_dict"], Mapping):
        raise ValueError("model_state_dict must be a mapping.")
    if not isinstance(checkpoint["optimizer_state_dict"], Mapping):
        raise ValueError("optimizer_state_dict must be a mapping.")
    if not isinstance(checkpoint["config"], Mapping):
        raise ValueError("config must be a mapping.")

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    optimizer_restored = False
    scheduler_restored = False
    scaler_restored = False

    if require_training_state and optimizer is None:
        raise ValueError("optimizer is required when require_training_state=True.")
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        optimizer_restored = True

    scheduler_state = checkpoint["scheduler_state_dict"]
    if scheduler_state is not None and not isinstance(scheduler_state, Mapping):
        raise ValueError("scheduler_state_dict must be a mapping or None.")
    if scheduler is not None:
        if scheduler_state is None:
            raise ValueError("A scheduler was supplied but the checkpoint has no scheduler state.")
        scheduler.load_state_dict(scheduler_state)
        scheduler_restored = True
    elif require_training_state and scheduler_state is not None:
        raise ValueError(
            "Checkpoint contains scheduler state but no scheduler was supplied for resume."
        )

    scaler_state = checkpoint["scaler_state_dict"]
    if scaler_state is not None and not isinstance(scaler_state, Mapping):
        raise ValueError("scaler_state_dict must be a mapping or None.")
    if scaler is not None:
        if scaler_state is None:
            raise ValueError("A scaler was supplied but the checkpoint has no scaler state.")
        scaler.load_state_dict(scaler_state)
        scaler_restored = True
    elif require_training_state and scaler_state is not None:
        raise ValueError(
            "Checkpoint contains scaler state but no scaler was supplied for resume."
        )

    epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]
    random_seed = checkpoint["random_seed"]
    if type(epoch) is not int or epoch < 0:
        raise ValueError(f"Invalid checkpoint epoch: {epoch!r}.")
    if type(global_step) is not int or global_step < 0:
        raise ValueError(f"Invalid checkpoint global_step: {global_step!r}.")
    if type(random_seed) is not int or random_seed < 0:
        raise ValueError(f"Invalid checkpoint random_seed: {random_seed!r}.")
    return ResumeState(
        epoch=epoch,
        global_step=global_step,
        best_validation_psnr=float(checkpoint["best_validation_psnr"]),
        config=dict(checkpoint["config"]),
        random_seed=random_seed,
        git_commit=str(checkpoint["git_commit"]),
        creation_timestamp_utc=str(checkpoint["creation_timestamp_utc"]),
        optimizer_restored=optimizer_restored,
        scheduler_restored=scheduler_restored,
        scaler_restored=scaler_restored,
    )
