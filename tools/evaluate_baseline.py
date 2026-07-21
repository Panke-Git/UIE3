#!/usr/bin/env python3
"""Evaluate a baseline checkpoint on validation only during Phase B2b."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class EvaluationManifestSelection:
    """Resolved validation manifest and provenance for one evaluation."""

    run_mode: str
    formal_experiment: bool
    validation_manifest: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a NAFNet-small baseline checkpoint on validation only."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or an explicit torch device"
    )
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Enable an explicitly non-formal validation smoke evaluation.",
    )
    parser.add_argument("--validation-manifest-override", type=Path, default=None)
    return parser


def validate_evaluation_run_mode(args: argparse.Namespace) -> None:
    """Enforce validation-only evaluation and guarded smoke overrides."""

    if args.split != "validation":
        if args.split == "test":
            raise ValueError(
                "Phase B2b-r1 does not authorize final test evaluation; "
                "--split must be validation even in smoke-test mode."
            )
        raise ValueError(f"Unsupported --split {args.split!r}; only validation is allowed.")
    if not args.smoke_test and args.validation_manifest_override is not None:
        raise ValueError(
            "--validation-manifest-override requires explicit --smoke-test; formal "
            "evaluation always uses the canonical validation manifest."
        )


def resolve_evaluation_manifest_selection(
    args: argparse.Namespace,
) -> EvaluationManifestSelection:
    """Resolve canonical validation or one explicit non-test smoke manifest."""

    validate_evaluation_run_mode(args)
    from tools.train_baseline import (
        FORMAL_VALIDATION_MANIFEST,
        resolve_formal_manifest,
        resolve_smoke_manifest_override,
    )

    if args.smoke_test:
        validation_manifest = (
            resolve_formal_manifest(FORMAL_VALIDATION_MANIFEST)
            if args.validation_manifest_override is None
            else resolve_smoke_manifest_override(
                args.validation_manifest_override,
                "--validation-manifest-override",
            )
        )
        return EvaluationManifestSelection(
            run_mode="SMOKE_TEST",
            formal_experiment=False,
            validation_manifest=validation_manifest,
        )
    return EvaluationManifestSelection(
        run_mode="FORMAL",
        formal_experiment=True,
        validation_manifest=resolve_formal_manifest(FORMAL_VALIDATION_MANIFEST),
    )


def _write_csv_atomic(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "sample_id",
                    "input_relative_path",
                    "gt_relative_path",
                    "psnr_rgb",
                    "ssim_rgb",
                ),
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def run(args: argparse.Namespace) -> int:
    validate_evaluation_run_mode(args)
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")

    sys.path.insert(0, str(PROJECT_ROOT))
    from tools.train_baseline import (
        load_and_validate_config,
        resolve_device,
    )

    config = load_and_validate_config(args.config)
    manifest_selection = resolve_evaluation_manifest_selection(args)
    dataset_root = args.dataset_root.expanduser().resolve(strict=False)
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(f"--dataset-root is not a directory: {dataset_root}")
    checkpoint_path = args.checkpoint.expanduser().resolve(strict=False)
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"--checkpoint does not exist: {checkpoint_path}")

    print(f"RUN_MODE={manifest_selection.run_mode}")
    print(f"FORMAL_EXPERIMENT={str(manifest_selection.formal_experiment).lower()}")
    if args.validation_manifest_override is not None:
        print(
            "validation_manifest_override="
            f"{manifest_selection.validation_manifest}"
        )

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required at runtime and must be installed separately for the "
            "cloud CUDA environment."
        ) from exc
    from src.data.paired_image_dataset import PairedImageDataset
    from src.engine.checkpoint import load_checkpoint
    from src.engine.trainer import validate_model
    from src.models.backbones.nafnet_small import build_nafnet_small
    from src.utils.seed import seed_worker, set_global_seed

    device = resolve_device(args.device, torch)
    seed = int(config["experiment"]["seed"])
    set_global_seed(seed, deterministic=bool(config["experiment"].get("deterministic", False)))
    worker_count = args.num_workers if args.num_workers is not None else int(
        config["data"]["num_workers"]
    )
    validation_manifest = manifest_selection.validation_manifest
    dataset = PairedImageDataset(
        str(dataset_root), str(validation_manifest), training=False
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=worker_count,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )
    model = build_nafnet_small(**config["model"])
    load_checkpoint(
        checkpoint_path,
        model=model,
        map_location="cpu",
        strict=True,
        require_training_state=False,
    )
    model.to(device)
    result = validate_model(model, loader, device=device)
    output_csv = args.output_csv.expanduser().resolve(strict=False)
    _write_csv_atomic(output_csv, result["per_image"])
    summary = {
        "split": "validation",
        "run_mode": manifest_selection.run_mode.lower(),
        "formal_experiment": manifest_selection.formal_experiment,
        "actual_validation_manifest": str(manifest_selection.validation_manifest),
        "num_samples": result["num_samples"],
        "mean_psnr_rgb": result["psnr_rgb"],
        "mean_ssim_rgb": result["ssim_rgb"],
        "output_csv": str(output_csv),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
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
