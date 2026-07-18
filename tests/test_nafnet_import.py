"""Phase B1b runtime tests for the minimal NAFNet import.

These tests require PyTorch and pytest.  They are created but deliberately not
executed during Phase B1a.  Set ``NAFNET_UPSTREAM_ROOT`` to the pinned official
NAFNet checkout; otherwise ``../NAFNet`` relative to the UIE3 repository is
used.
"""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import types
from typing import Tuple

import pytest
import torch

from src.models.backbones.nafnet_small import (
    NAFNET_SMALL_PADDER_SIZE,
    NAFNetSmall,
    build_nafnet_small,
)
from third_party.nafnet.nafnet_arch import NAFNet as ImportedNAFNet


UPSTREAM_COMMIT = "2b4af71ebe098a92a75910c233a3965a3e93ede4"
SEED = 3407
MAX_ABS_ERROR = 1.0e-6
MEAN_ABS_ERROR = 1.0e-7
MODEL_CONFIG = {
    "img_channel": 3,
    "width": 32,
    "enc_blk_nums": [2, 2, 2],
    "middle_blk_num": 4,
    "dec_blk_nums": [2, 2, 2],
}
EQUIVALENCE_INPUT_SHAPES = (
    (1, 3, 256, 256),
    (2, 3, 128, 192),
    (1, 3, 257, 259),
)


def _upstream_root() -> Path:
    configured_root = os.environ.get("NAFNET_UPSTREAM_ROOT")
    if configured_root:
        root = Path(configured_root).expanduser().resolve()
    else:
        root = (Path(__file__).resolve().parents[2] / "NAFNet").resolve()

    required_files = (
        root / "basicsr/models/archs/NAFNet_arch.py",
        root / "basicsr/models/archs/arch_util.py",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Official NAFNet sources were not found. Set NAFNET_UPSTREAM_ROOT "
            f"to the pinned checkout. Missing: {missing}"
        )

    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual_commit = result.stdout.strip()
    if actual_commit != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"Expected NAFNet commit {UPSTREAM_COMMIT}, got {actual_commit}."
        )
    return root


def _module_from_file(module_name: str, source_path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create an import spec for {source_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _package_stub(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _load_official_nafnet_class():
    """Load official source under an isolated name with framework-only stubs."""

    root = _upstream_root()
    arch_util_path = root / "basicsr/models/archs/arch_util.py"
    nafnet_arch_path = root / "basicsr/models/archs/NAFNet_arch.py"

    basicsr_module = _package_stub("basicsr")
    models_module = _package_stub("basicsr.models")
    archs_module = _package_stub("basicsr.models.archs")
    utils_module = types.ModuleType("basicsr.utils")
    utils_module.get_root_logger = lambda *args, **kwargs: None

    local_arch_module = types.ModuleType("basicsr.models.archs.local_arch")
    local_arch_module.Local_Base = type("Local_Base", (), {})

    temporary_names = (
        "basicsr",
        "basicsr.models",
        "basicsr.models.archs",
        "basicsr.utils",
        "basicsr.models.archs.local_arch",
        "basicsr.models.archs.arch_util",
        "_uie3_official_nafnet_reference.arch_util",
        "_uie3_official_nafnet_reference.NAFNet_arch",
    )
    missing_marker = object()
    previous_modules = {
        name: sys.modules.get(name, missing_marker) for name in temporary_names
    }

    try:
        sys.modules["basicsr"] = basicsr_module
        sys.modules["basicsr.models"] = models_module
        sys.modules["basicsr.models.archs"] = archs_module
        sys.modules["basicsr.utils"] = utils_module
        sys.modules["basicsr.models.archs.local_arch"] = local_arch_module

        official_arch_util = _module_from_file(
            "_uie3_official_nafnet_reference.arch_util", arch_util_path
        )
        sys.modules["basicsr.models.archs.arch_util"] = official_arch_util
        official_arch = _module_from_file(
            "_uie3_official_nafnet_reference.NAFNet_arch", nafnet_arch_path
        )

        if Path(official_arch.__file__).resolve() != nafnet_arch_path.resolve():
            raise AssertionError("Official NAFNet resolved to an unexpected source file.")
        official_class = official_arch.NAFNet
        if official_class is ImportedNAFNet:
            raise AssertionError("Official and imported NAFNet classes are aliased.")
        if not official_class.__module__.startswith(
            "_uie3_official_nafnet_reference"
        ):
            raise AssertionError("Official NAFNet was not loaded in isolation.")
        return official_class
    finally:
        for name, previous in previous_modules.items():
            if previous is missing_marker:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def _configure_determinism() -> None:
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    cuda_backend = getattr(torch.backends, "cuda", None)
    cuda_matmul = getattr(cuda_backend, "matmul", None)
    if cuda_matmul is not None:
        cuda_matmul.allow_tf32 = False
    cudnn_backend = getattr(torch.backends, "cudnn", None)
    if cudnn_backend is not None:
        cudnn_backend.allow_tf32 = False
        cudnn_backend.benchmark = False
        cudnn_backend.deterministic = True


def _parameter_counts(model) -> Tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def _new_equivalent_model_pair():
    _configure_determinism()
    official_class = _load_official_nafnet_class()
    official = official_class(**MODEL_CONFIG).cpu().float().eval()
    imported = ImportedNAFNet(**MODEL_CONFIG).cpu().float().eval()

    shared_state = {
        key: tensor.detach().clone() for key, tensor in official.state_dict().items()
    }
    load_result = imported.load_state_dict(shared_state, strict=True)
    assert not load_result.missing_keys
    assert not load_result.unexpected_keys
    return official, imported


def _assert_matching_structure(official, imported) -> None:
    assert _parameter_counts(official) == _parameter_counts(imported)

    official_state = official.state_dict()
    imported_state = imported.state_dict()
    assert list(official_state.keys()) == list(imported_state.keys())
    for key in official_state:
        assert official_state[key].shape == imported_state[key].shape
        assert official_state[key].dtype == imported_state[key].dtype


def _assert_finite_gradients(model, input_tensor) -> None:
    assert input_tensor.grad is not None
    assert torch.isfinite(input_tensor.grad).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"Missing gradient for {name}."
        assert torch.isfinite(parameter.grad).all(), f"Non-finite gradient for {name}."


@pytest.mark.parametrize("shape", EQUIVALENCE_INPUT_SHAPES)
def test_official_and_imported_output_equivalence(shape: Tuple[int, ...]) -> None:
    official, imported = _new_equivalent_model_pair()
    _assert_matching_structure(official, imported)
    assert next(official.parameters()).device.type == "cpu"
    assert next(imported.parameters()).device.type == "cpu"
    assert next(official.parameters()).dtype == torch.float32
    assert next(imported.parameters()).dtype == torch.float32

    torch.manual_seed(SEED + sum(shape))
    input_tensor = torch.randn(shape, dtype=torch.float32, device="cpu")
    with torch.no_grad():
        official_output = official(input_tensor)
        imported_output = imported(input_tensor)

    assert official_output.shape == input_tensor.shape
    assert imported_output.shape == input_tensor.shape
    absolute_error = (official_output - imported_output).abs()
    max_abs_error = absolute_error.max().item()
    mean_abs_error = absolute_error.mean().item()
    print(
        f"shape={shape} max_abs_error={max_abs_error:.12g} "
        f"mean_abs_error={mean_abs_error:.12g} "
        f"parameters={_parameter_counts(imported)}"
    )
    assert max_abs_error <= MAX_ABS_ERROR
    assert mean_abs_error <= MEAN_ABS_ERROR


def test_nafnet_small_output_shape() -> None:
    _configure_determinism()
    model = build_nafnet_small().cpu().float().eval()
    assert isinstance(model, NAFNetSmall)
    assert model.padder_size == NAFNET_SMALL_PADDER_SIZE

    input_tensor = torch.randn((2, 3, 64, 80), dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
    assert output.shape == input_tensor.shape

    with pytest.raises(ValueError, match="fixed width=32"):
        build_nafnet_small(width=16)


def test_nafnet_small_non_multiple_size() -> None:
    _configure_determinism()
    model = build_nafnet_small().cpu().float().eval()
    input_tensor = torch.randn((1, 3, 33, 35), dtype=torch.float32)
    padded = model.check_image_size(input_tensor)

    assert model.padder_size == 8
    assert padded.shape == (1, 3, 40, 40)
    torch.testing.assert_close(padded[:, :, :33, :35], input_tensor)
    assert torch.count_nonzero(padded[:, :, 33:, :]).item() == 0
    assert torch.count_nonzero(padded[:, :, :, 35:]).item() == 0

    with torch.no_grad():
        output = model(input_tensor)
    assert output.shape == input_tensor.shape


def test_nafnet_small_forward_backward() -> None:
    official, imported = _new_equivalent_model_pair()
    torch.manual_seed(SEED + 1)
    base_input = torch.randn((1, 3, 16, 24), dtype=torch.float32)
    official_input = base_input.detach().clone().requires_grad_(True)
    imported_input = base_input.detach().clone().requires_grad_(True)

    official_loss = official(official_input).square().mean()
    imported_loss = imported(imported_input).square().mean()
    assert torch.isfinite(official_loss)
    assert torch.isfinite(imported_loss)
    official_loss.backward()
    imported_loss.backward()

    _assert_finite_gradients(official, official_input)
    _assert_finite_gradients(imported, imported_input)
    torch.testing.assert_close(
        official_input.grad,
        imported_input.grad,
        rtol=0.0,
        atol=MAX_ABS_ERROR,
    )

    official_parameters = dict(official.named_parameters())
    imported_parameters = dict(imported.named_parameters())
    assert list(official_parameters) == list(imported_parameters)
    for name in official_parameters:
        try:
            torch.testing.assert_close(
                official_parameters[name].grad,
                imported_parameters[name].grad,
                rtol=0.0,
                atol=MAX_ABS_ERROR,
            )
        except AssertionError as exc:
            raise AssertionError(f"Gradient mismatch for {name}.") from exc


def test_nafnet_small_state_dict_compatibility() -> None:
    official, imported = _new_equivalent_model_pair()
    wrapper = build_nafnet_small().cpu().float()
    _assert_matching_structure(official, imported)
    _assert_matching_structure(official, wrapper)

    imported_result = imported.load_state_dict(official.state_dict(), strict=True)
    official_result = official.load_state_dict(imported.state_dict(), strict=True)
    wrapper_result = wrapper.load_state_dict(official.state_dict(), strict=True)
    for result in (imported_result, official_result, wrapper_result):
        assert not result.missing_keys
        assert not result.unexpected_keys


def test_nafnet_small_global_residual_not_duplicated() -> None:
    _configure_determinism()
    model = build_nafnet_small().cpu().float().eval()
    with torch.no_grad():
        model.ending.weight.zero_()
        model.ending.bias.zero_()

    input_tensor = torch.full((1, 3, 17, 19), 0.25, dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
    torch.testing.assert_close(output, input_tensor, rtol=0.0, atol=0.0)
