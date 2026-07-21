#!/usr/bin/env python3
"""Train the single-path NAFNet-small baseline on formal train/validation manifests."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXED_MODEL_CONFIG = {
    "img_channel": 3,
    "width": 32,
    "enc_blk_nums": [2, 2, 2],
    "middle_blk_num": 4,
    "dec_blk_nums": [2, 2, 2],
}
FORMAL_TRAIN_MANIFEST = "splits/lsui19/train.tsv"
FORMAL_VALIDATION_MANIFEST = "splits/lsui19/validation.tsv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the single-path NAFNet-small LSUI baseline. This entry point "
            "never loads the formal test manifest."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or an explicit torch device"
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser


def _require_mapping(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"Config section {name!r} must be a mapping.")
    return value


def load_and_validate_config(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read --config.") from exc

    config_path = path.expanduser().resolve(strict=False)
    if not config_path.exists() or not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("YAML config root must be a mapping.")
    experiment = _require_mapping(config, "experiment")
    data = _require_mapping(config, "data")
    model = _require_mapping(config, "model")
    loss = _require_mapping(config, "loss")
    optimizer = _require_mapping(config, "optimizer")
    training = _require_mapping(config, "training")
    metrics = _require_mapping(config, "metrics")

    if dict(model) != FIXED_MODEL_CONFIG:
        raise ValueError(
            f"model must equal the fixed NAFNet-small config {FIXED_MODEL_CONFIG}; "
            f"got {dict(model)}."
        )
    if data.get("train_manifest") != FORMAL_TRAIN_MANIFEST:
        raise ValueError(f"data.train_manifest must be {FORMAL_TRAIN_MANIFEST!r}.")
    if data.get("validation_manifest") != FORMAL_VALIDATION_MANIFEST:
        raise ValueError(
            f"data.validation_manifest must be {FORMAL_VALIDATION_MANIFEST!r}."
        )
    if "test_manifest" in data:
        raise ValueError("The Phase B2b training config must not contain a test manifest.")
    if str(loss.get("name", "")).lower() != "charbonnier":
        raise ValueError("Only Charbonnier loss is authorized for this baseline.")
    if str(optimizer.get("name", "")).lower() != "adamw":
        raise ValueError("Only AdamW is authorized for this baseline.")
    if float(metrics.get("data_range", -1)) != 1.0:
        raise ValueError("metrics.data_range must be 1.0.")
    if metrics.get("crop_border") != 0:
        raise ValueError("metrics.crop_border must be 0.")
    if metrics.get("ssim_window_size") != 11:
        raise ValueError("metrics.ssim_window_size must be 11.")
    if float(metrics.get("ssim_sigma", -1)) != 1.5:
        raise ValueError("metrics.ssim_sigma must be 1.5.")
    if type(experiment.get("seed")) is not int or experiment["seed"] < 0:
        raise ValueError("experiment.seed must be a non-negative integer.")
    for key in ("patch_size", "batch_size", "num_workers"):
        if type(data.get(key)) is not int or data[key] < (0 if key == "num_workers" else 1):
            raise ValueError(f"data.{key} has an invalid value: {data.get(key)!r}.")
    for key in ("epochs", "validate_every", "save_every"):
        if type(training.get(key)) is not int or training[key] <= 0:
            raise ValueError(f"training.{key} must be a positive integer.")
    if not isinstance(training.get("amp"), bool):
        raise ValueError("training.amp must be boolean.")
    return config


def resolve_formal_manifest(relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError("Formal manifest paths in config must be repository-relative.")
    resolved = (PROJECT_ROOT / path).resolve(strict=False)
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"Manifest path escapes the repository: {relative_path!r}.") from exc
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"Formal manifest is missing: {resolved}")
    return resolved


def current_git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def resolve_device(device_text: str, torch_module: Any) -> Any:
    if device_text == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    device = torch_module.device(device_text)
    if device.type == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError(f"CUDA device {device} was requested but CUDA is unavailable.")
    return device


def _append_json_line(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), sort_keys=True, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> int:
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive when provided.")
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    config = load_and_validate_config(args.config)
    dataset_root = args.dataset_root.expanduser().resolve(strict=False)
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(f"--dataset-root is not a directory: {dataset_root}")
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    resume_path = args.resume.expanduser().resolve(strict=False) if args.resume else None
    if output_dir.exists() and resume_path is None:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Use --resume explicitly."
        )
    if resume_path is not None and (not resume_path.exists() or not resume_path.is_file()):
        raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")

    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required at runtime and must be installed separately for the "
            "cloud CUDA environment."
        ) from exc
    from src.data.paired_image_dataset import PairedImageDataset
    from src.engine.trainer import BaselineTrainer
    from src.losses.charbonnier import CharbonnierLoss
    from src.models.backbones.nafnet_small import build_nafnet_small
    from src.utils.seed import seed_worker, set_global_seed

    checkpoint_config = copy.deepcopy(config)
    worker_count = (
        args.num_workers
        if args.num_workers is not None
        else int(checkpoint_config["data"]["num_workers"])
    )
    checkpoint_config["data"]["num_workers"] = worker_count
    device = resolve_device(args.device, torch)
    actual_config = copy.deepcopy(checkpoint_config)
    actual_config["runtime"] = {
        "dataset_root": str(dataset_root),
        "device": str(device),
        "output_dir": str(output_dir),
        "max_steps": args.max_steps,
    }
    seed = int(checkpoint_config["experiment"]["seed"])
    set_global_seed(
        seed,
        deterministic=bool(checkpoint_config["experiment"].get("deterministic", False)),
    )

    train_manifest = resolve_formal_manifest(checkpoint_config["data"]["train_manifest"])
    validation_manifest = resolve_formal_manifest(
        checkpoint_config["data"]["validation_manifest"]
    )
    data_config = checkpoint_config["data"]
    train_dataset = PairedImageDataset(
        str(dataset_root),
        str(train_manifest),
        training=True,
        patch_size=int(data_config["patch_size"]),
        enable_hflip=bool(data_config["hflip"]),
        enable_vflip=bool(data_config["vflip"]),
        enable_rot90=bool(data_config["rot90"]),
        pad_if_smaller=bool(data_config["pad_if_smaller"]),
    )
    validation_dataset = PairedImageDataset(
        str(dataset_root), str(validation_manifest), training=False
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(data_config["batch_size"]),
        shuffle=True,
        num_workers=worker_count,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=worker_count,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    model = build_nafnet_small(**checkpoint_config["model"])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer_config = checkpoint_config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
    )
    loss_function = CharbonnierLoss(
        epsilon=float(checkpoint_config["loss"]["epsilon"]), reduction="mean"
    )
    git_commit = current_git_commit()
    trainer = BaselineTrainer(
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        device=device,
        amp=bool(checkpoint_config["training"]["amp"]),
        gradient_clip_norm=checkpoint_config["training"]["gradient_clip_norm"],
        scheduler=None,
        config=checkpoint_config,
        random_seed=seed,
        git_commit=git_commit,
    )

    start_epoch = 0
    if resume_path is not None:
        resume_state = trainer.resume(resume_path)
        if dict(resume_state.config) != checkpoint_config:
            raise ValueError("Resume checkpoint config does not match the actual config.")
        start_epoch = resume_state.epoch + 1

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(actual_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Git commit: {git_commit}")
    print(f"Device: {device}")
    print(f"Parameter count: {parameter_count}")
    print("Resolved config:")
    print(json.dumps(actual_config, indent=2, sort_keys=True))

    training_config = checkpoint_config["training"]
    log_path = output_dir / "training_log.jsonl"
    stop_requested = False
    for epoch in range(start_epoch, int(training_config["epochs"])):
        trainer.epoch = epoch
        losses = []
        last_learning_rate = float(optimizer.param_groups[0]["lr"])
        for batch in train_loader:
            if args.max_steps is not None and trainer.global_step >= args.max_steps:
                stop_requested = True
                break
            step_result = trainer.train_step(batch)
            losses.append(step_result["loss"])
            last_learning_rate = step_result["learning_rate"]
        if not losses and stop_requested:
            break
        mean_train_loss = sum(losses) / len(losses)
        should_validate = (
            (epoch + 1) % int(training_config["validate_every"]) == 0
            or stop_requested
            or (args.max_steps is not None and trainer.global_step >= args.max_steps)
        )
        validation_result: Optional[Dict[str, Any]] = None
        is_best = False
        if should_validate:
            validation_result = trainer.validation_epoch(validation_loader)
            validation_psnr = float(validation_result["psnr_rgb"])
            if validation_psnr > trainer.best_validation_psnr:
                trainer.best_validation_psnr = validation_psnr
                is_best = True
                trainer.save(output_dir / "best.pt", epoch=epoch)
        log_record = {
            "epoch": epoch,
            "global_step": trainer.global_step,
            "learning_rate": last_learning_rate,
            "train_loss": mean_train_loss,
            "validation_psnr": (
                None if validation_result is None else validation_result["psnr_rgb"]
            ),
            "validation_ssim": (
                None if validation_result is None else validation_result["ssim_rgb"]
            ),
            "best_validation_psnr": trainer.best_validation_psnr,
            "is_best": is_best,
        }
        print(json.dumps(log_record, sort_keys=True))
        _append_json_line(log_path, log_record)
        if (epoch + 1) % int(training_config["save_every"]) == 0:
            trainer.save(output_dir / f"epoch_{epoch + 1:04d}.pt", epoch=epoch)
        trainer.save(output_dir / "last.pt", epoch=epoch)
        trainer.step_scheduler()
        if args.max_steps is not None and trainer.global_step >= args.max_steps:
            break
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
