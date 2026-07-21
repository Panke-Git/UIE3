"""Runtime tests for Charbonnier loss and fixed RGB metrics."""

import math

import pytest
import torch

from src.losses.charbonnier import CharbonnierLoss
from src.metrics.image_metrics import (
    rgb_psnr,
    rgb_psnr_per_image,
    rgb_ssim,
    rgb_ssim_per_image,
)


def test_charbonnier_known_value_and_finite_backward() -> None:
    prediction = torch.zeros((1, 3, 4, 4), dtype=torch.float32, requires_grad=True)
    target = torch.zeros_like(prediction)
    loss = CharbonnierLoss(epsilon=1.0e-3)(prediction, target)
    assert float(loss) == pytest.approx(1.0e-3)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_charbonnier_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        CharbonnierLoss()(torch.zeros(1, 3, 4, 4), torch.zeros(1, 3, 4, 5))


def test_psnr_known_mse_and_identical_behavior() -> None:
    target = torch.ones((1, 3, 12, 12), dtype=torch.float32)
    prediction = torch.zeros_like(target)
    assert float(rgb_psnr(prediction, target)) == pytest.approx(0.0, abs=1.0e-6)
    identical = rgb_psnr_per_image(target, target)
    assert identical.shape == (1,)
    assert math.isinf(float(identical[0])) and float(identical[0]) > 0


def test_identical_image_ssim_is_one() -> None:
    image = torch.rand((2, 3, 16, 18), dtype=torch.float32)
    per_image = rgb_ssim_per_image(image, image)
    assert per_image.shape == (2,)
    assert torch.allclose(per_image, torch.ones_like(per_image), atol=1.0e-6)
    assert float(rgb_ssim(image, image)) == pytest.approx(1.0, abs=1.0e-6)


def test_batch_metrics_return_independent_per_image_values() -> None:
    target = torch.zeros((2, 3, 12, 12), dtype=torch.float32)
    prediction = target.clone()
    prediction[1] = 1.0
    psnr = rgb_psnr_per_image(prediction, target)
    ssim = rgb_ssim_per_image(prediction, target)
    assert psnr.shape == (2,)
    assert ssim.shape == (2,)
    assert torch.isinf(psnr[0])
    assert float(psnr[1]) == pytest.approx(0.0, abs=1.0e-6)
    assert float(ssim[0]) == pytest.approx(1.0, abs=1.0e-6)
    assert float(ssim[1]) < 1.0


def test_ssim_rejects_images_smaller_than_window() -> None:
    image = torch.zeros((1, 3, 10, 11), dtype=torch.float32)
    with pytest.raises(ValueError, match="H and W >= 11"):
        rgb_ssim_per_image(image, image)
