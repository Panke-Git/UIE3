"""Runtime smoke tests for a single NAFNet-small baseline update and resume."""

from argparse import Namespace
from pathlib import Path
from typing import Optional

import pytest
import torch

from src.engine.trainer import BaselineTrainer
from src.losses.charbonnier import CharbonnierLoss
from src.models.backbones.nafnet_small import build_nafnet_small
from tools.evaluate_baseline import (
    resolve_evaluation_manifest_selection,
    validate_evaluation_run_mode,
)
from tools.train_baseline import (
    FIXED_MODEL_CONFIG,
    FORMAL_TRAIN_MANIFEST,
    FORMAL_VALIDATION_MANIFEST,
    PROJECT_ROOT,
    build_checkpoint_config,
    resolve_training_manifest_selection,
    validate_baseline_config,
    validate_training_run_mode,
)


def _batch(batch_size: int = 1):
    inputs = torch.rand((batch_size, 3, 16, 16), dtype=torch.float32)
    targets = torch.rand((batch_size, 3, 16, 16), dtype=torch.float32)
    return {
        "sample_id": [f"sample-{index}" for index in range(batch_size)],
        "input": inputs,
        "target": targets,
        "input_relative_path": [f"input/{index}.png" for index in range(batch_size)],
        "gt_relative_path": [f"gt/{index}.png" for index in range(batch_size)],
    }


def _trainer() -> BaselineTrainer:
    model = build_nafnet_small()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    return BaselineTrainer(
        model=model,
        optimizer=optimizer,
        loss_function=CharbonnierLoss(),
        device="cpu",
        amp=True,
        config={"test": True},
        random_seed=3407,
        git_commit="test-commit",
    )


def _training_args(
    *,
    smoke_test: bool = False,
    max_steps: Optional[int] = None,
    train_override: Optional[Path] = None,
    validation_override: Optional[Path] = None,
) -> Namespace:
    return Namespace(
        smoke_test=smoke_test,
        max_steps=max_steps,
        train_manifest_override=train_override,
        validation_manifest_override=validation_override,
    )


def _evaluation_args(
    *,
    smoke_test: bool = False,
    split: str = "validation",
    validation_override: Optional[Path] = None,
) -> Namespace:
    return Namespace(
        smoke_test=smoke_test,
        split=split,
        validation_manifest_override=validation_override,
    )


def _write_smoke_manifest(path: Path) -> Path:
    path.write_text(
        "smoke-sample\tTrain/input/smoke.png\tTrain/GT/smoke.png\n",
        encoding="utf-8",
    )
    return path


def _valid_baseline_config() -> dict:
    return {
        "experiment": {"name": "test", "seed": 3407},
        "data": {
            "train_manifest": FORMAL_TRAIN_MANIFEST,
            "validation_manifest": FORMAL_VALIDATION_MANIFEST,
            "patch_size": 256,
            "batch_size": 1,
            "num_workers": 0,
        },
        "model": dict(FIXED_MODEL_CONFIG),
        "loss": {"name": "charbonnier", "epsilon": 1.0e-3},
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 2.0e-4,
            "weight_decay": 0.0,
            "betas": [0.9, 0.999],
        },
        "training": {
            "epochs": 1,
            "amp": False,
            "validate_every": 1,
            "save_every": 1,
        },
        "metrics": {
            "data_range": 1.0,
            "crop_border": 0,
            "ssim_window_size": 11,
            "ssim_sigma": 1.5,
        },
    }


def test_single_train_step_updates_parameters_and_disables_cpu_amp() -> None:
    trainer = _trainer()
    before = [parameter.detach().clone() for parameter in trainer.model.parameters()]
    result = trainer.train_step(_batch())
    assert not trainer.amp_enabled
    assert torch.isfinite(torch.tensor(result["loss"]))
    assert trainer.global_step == 1
    assert any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, trainer.model.parameters())
    )
    for parameter in trainer.model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_validation_returns_every_per_image_psnr_and_ssim() -> None:
    trainer = _trainer()
    result = trainer.validation_epoch([_batch(batch_size=2)])
    assert result["num_samples"] == 2
    assert len(result["per_image"]) == 2
    assert all("psnr_rgb" in row and "ssim_rgb" in row for row in result["per_image"])


def test_checkpoint_resume_preserves_continuous_global_step(tmp_path: Path) -> None:
    first = _trainer()
    first.train_step(_batch())
    checkpoint = tmp_path / "resume.pt"
    first.save(checkpoint, epoch=0)

    resumed = _trainer()
    state = resumed.resume(checkpoint)
    assert state.global_step == 1
    assert resumed.global_step == 1
    resumed.train_step(_batch())
    assert resumed.global_step == 2


@pytest.mark.parametrize(
    ("key", "noncanonical"),
    [
        ("train_manifest", "tmp/smoke-train.tsv"),
        ("validation_manifest", "tmp/smoke-validation.tsv"),
    ],
)
def test_formal_config_still_rejects_noncanonical_manifest(
    key: str, noncanonical: str
) -> None:
    config = _valid_baseline_config()
    config["data"][key] = noncanonical
    with pytest.raises(ValueError, match=rf"data\.{key} must be"):
        validate_baseline_config(config)


def test_formal_training_rejects_manifest_override(tmp_path: Path) -> None:
    override = _write_smoke_manifest(tmp_path / "smoke.tsv")
    args = _training_args(train_override=override)
    with pytest.raises(ValueError, match="require explicit --smoke-test"):
        validate_training_run_mode(args)


def test_smoke_training_requires_max_steps(tmp_path: Path) -> None:
    train = _write_smoke_manifest(tmp_path / "smoke-train.tsv")
    validation = _write_smoke_manifest(tmp_path / "smoke-validation.tsv")
    args = _training_args(
        smoke_test=True,
        train_override=train,
        validation_override=validation,
    )
    with pytest.raises(ValueError, match="requires a positive --max-steps"):
        validate_training_run_mode(args)


@pytest.mark.parametrize("max_steps", [0, -1])
def test_smoke_training_rejects_nonpositive_max_steps(
    tmp_path: Path, max_steps: int
) -> None:
    train = _write_smoke_manifest(tmp_path / "smoke-train.tsv")
    validation = _write_smoke_manifest(tmp_path / "smoke-validation.tsv")
    args = _training_args(
        smoke_test=True,
        max_steps=max_steps,
        train_override=train,
        validation_override=validation,
    )
    with pytest.raises(ValueError, match="positive integer"):
        validate_training_run_mode(args)


@pytest.mark.parametrize("missing", ["train", "validation"])
def test_smoke_training_requires_both_overrides(
    tmp_path: Path, missing: str
) -> None:
    train = _write_smoke_manifest(tmp_path / "smoke-train.tsv")
    validation = _write_smoke_manifest(tmp_path / "smoke-validation.tsv")
    args = _training_args(
        smoke_test=True,
        max_steps=1,
        train_override=None if missing == "train" else train,
        validation_override=None if missing == "validation" else validation,
    )
    with pytest.raises(ValueError, match=f"requires --{missing}-manifest-override"):
        validate_training_run_mode(args)


@pytest.mark.parametrize("test_override", ["train", "validation"])
def test_smoke_training_rejects_test_manifest(
    tmp_path: Path, test_override: str
) -> None:
    smoke = _write_smoke_manifest(tmp_path / "smoke.tsv")
    test_manifest = PROJECT_ROOT / "splits/lsui19/test.tsv"
    args = _training_args(
        smoke_test=True,
        max_steps=1,
        train_override=test_manifest if test_override == "train" else smoke,
        validation_override=test_manifest if test_override == "validation" else smoke,
    )
    with pytest.raises(ValueError, match="must not reference test.tsv"):
        resolve_training_manifest_selection(args)


def test_smoke_training_rejects_missing_override_file(tmp_path: Path) -> None:
    validation = _write_smoke_manifest(tmp_path / "smoke-validation.tsv")
    args = _training_args(
        smoke_test=True,
        max_steps=1,
        train_override=tmp_path / "missing.tsv",
        validation_override=validation,
    )
    with pytest.raises(FileNotFoundError, match="does not exist or is not a file"):
        resolve_training_manifest_selection(args)


def test_legal_smoke_training_overrides_are_accepted(tmp_path: Path) -> None:
    train = _write_smoke_manifest(tmp_path / "smoke-train.tsv")
    validation = _write_smoke_manifest(tmp_path / "smoke-validation.tsv")
    selection = resolve_training_manifest_selection(
        _training_args(
            smoke_test=True,
            max_steps=2,
            train_override=train,
            validation_override=validation,
        )
    )
    assert selection.run_mode == "SMOKE_TEST"
    assert not selection.formal_experiment
    assert selection.train_manifest == train.resolve()
    assert selection.validation_manifest == validation.resolve()


def test_smoke_checkpoint_config_records_actual_manifest_provenance(
    tmp_path: Path,
) -> None:
    train = _write_smoke_manifest(tmp_path / "smoke-train.tsv")
    validation = _write_smoke_manifest(tmp_path / "smoke-validation.tsv")
    selection = resolve_training_manifest_selection(
        _training_args(
            smoke_test=True,
            max_steps=2,
            train_override=train,
            validation_override=validation,
        )
    )
    formal_config = _valid_baseline_config()
    checkpoint_config = build_checkpoint_config(
        formal_config, selection, worker_count=0
    )
    assert checkpoint_config["run_mode"] == "smoke_test"
    assert checkpoint_config["formal_experiment"] is False
    assert checkpoint_config["actual_train_manifest"] == str(train.resolve())
    assert checkpoint_config["actual_validation_manifest"] == str(validation.resolve())
    assert "run_mode" not in formal_config


def test_formal_evaluation_rejects_validation_override(tmp_path: Path) -> None:
    override = _write_smoke_manifest(tmp_path / "smoke-validation.tsv")
    with pytest.raises(ValueError, match="requires explicit --smoke-test"):
        validate_evaluation_run_mode(
            _evaluation_args(validation_override=override)
        )


def test_validation_smoke_override_is_accepted(tmp_path: Path) -> None:
    override = _write_smoke_manifest(tmp_path / "smoke-validation.tsv")
    selection = resolve_evaluation_manifest_selection(
        _evaluation_args(smoke_test=True, validation_override=override)
    )
    assert selection.run_mode == "SMOKE_TEST"
    assert not selection.formal_experiment
    assert selection.validation_manifest == override.resolve()


def test_validation_smoke_rejects_test_manifest_override() -> None:
    test_manifest = PROJECT_ROOT / "splits/lsui19/test.tsv"
    with pytest.raises(ValueError, match="must not reference test.tsv"):
        resolve_evaluation_manifest_selection(
            _evaluation_args(smoke_test=True, validation_override=test_manifest)
        )


def test_test_split_is_rejected_even_in_smoke_mode(tmp_path: Path) -> None:
    override = _write_smoke_manifest(tmp_path / "smoke-validation.tsv")
    with pytest.raises(ValueError, match="even in smoke-test mode"):
        validate_evaluation_run_mode(
            _evaluation_args(
                smoke_test=True,
                split="test",
                validation_override=override,
            )
        )


def test_manifest_selection_does_not_rewrite_formal_artifacts(tmp_path: Path) -> None:
    protected_paths = [
        PROJECT_ROOT / "configs/nafnet_small_lsui.yaml",
        PROJECT_ROOT / FORMAL_TRAIN_MANIFEST,
        PROJECT_ROOT / FORMAL_VALIDATION_MANIFEST,
        PROJECT_ROOT / "splits/lsui19/test.tsv",
    ]
    before = {path: path.read_bytes() for path in protected_paths}
    formal = resolve_training_manifest_selection(_training_args())
    train = _write_smoke_manifest(tmp_path / "smoke-train.tsv")
    validation = _write_smoke_manifest(tmp_path / "smoke-validation.tsv")
    smoke = resolve_training_manifest_selection(
        _training_args(
            smoke_test=True,
            max_steps=1,
            train_override=train,
            validation_override=validation,
        )
    )
    assert formal.formal_experiment
    assert formal.train_manifest == (PROJECT_ROOT / FORMAL_TRAIN_MANIFEST).resolve()
    assert formal.validation_manifest == (
        PROJECT_ROOT / FORMAL_VALIDATION_MANIFEST
    ).resolve()
    assert not smoke.formal_experiment
    assert {path: path.read_bytes() for path in protected_paths} == before
