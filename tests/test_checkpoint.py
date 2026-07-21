"""Runtime tests for strict atomic checkpoint save/load/resume."""

from pathlib import Path

import pytest
import torch

from src.engine.checkpoint import load_checkpoint, save_checkpoint


def _model_and_optimizer():
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    loss = model(torch.ones(2, 4)).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer


def _save_valid(path: Path):
    model, optimizer = _model_and_optimizer()
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        epoch=3,
        global_step=17,
        best_validation_psnr=21.25,
        config={"name": "test"},
        random_seed=3407,
        git_commit="0123456789abcdef",
    )
    return model, optimizer


def test_checkpoint_round_trip_restores_model_optimizer_and_counters(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    source_model, source_optimizer = _save_valid(path)
    target_model = torch.nn.Linear(4, 3)
    target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=9.0e-1)
    state = load_checkpoint(
        path,
        model=target_model,
        optimizer=target_optimizer,
        strict=True,
        require_training_state=True,
    )
    for source, target in zip(source_model.parameters(), target_model.parameters()):
        assert torch.equal(source, target)
    assert target_optimizer.state_dict()["state"]
    assert target_optimizer.state_dict()["param_groups"] == source_optimizer.state_dict()[
        "param_groups"
    ]
    assert state.epoch == 3
    assert state.global_step == 17
    assert state.best_validation_psnr == pytest.approx(21.25)
    assert state.optimizer_restored


def test_checkpoint_missing_required_field_is_rejected(tmp_path: Path) -> None:
    valid_path = tmp_path / "valid.pt"
    model, _ = _save_valid(valid_path)
    payload = torch.load(valid_path, map_location="cpu", weights_only=False)
    del payload["global_step"]
    invalid_path = tmp_path / "invalid.pt"
    torch.save(payload, invalid_path)
    with pytest.raises(ValueError, match="missing required fields"):
        load_checkpoint(invalid_path, model=model)


def test_checkpoint_strict_model_load_is_enforced(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    _save_valid(path)
    incompatible_model = torch.nn.Sequential(torch.nn.Linear(4, 3))
    with pytest.raises(RuntimeError):
        load_checkpoint(path, model=incompatible_model, strict=True)


def test_atomic_checkpoint_file_exists_without_temporary_residue(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    _save_valid(path)
    assert path.is_file()
    assert not list(tmp_path.glob(".checkpoint.pt.*.tmp"))
