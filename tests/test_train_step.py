"""Runtime smoke tests for a single NAFNet-small baseline update and resume."""

from pathlib import Path

import torch

from src.engine.trainer import BaselineTrainer
from src.losses.charbonnier import CharbonnierLoss
from src.models.backbones.nafnet_small import build_nafnet_small


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
