import pytest
import torch

from src.core.device import build_device_context, resolve_device
from src.core.exceptions import ConfigurationError


def test_device_resolution() -> None:
    auto_device = resolve_device("auto")
    assert auto_device.type in {"cpu", "cuda"}
    cpu_device = resolve_device("cpu")
    assert cpu_device.type == "cpu"


def test_device_context_defaults() -> None:
    ctx = build_device_context("cpu", use_amp=True)
    assert ctx.device == torch.device("cpu")
    assert ctx.use_amp is False


def test_explicit_cuda_without_cuda_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """resolve_device('cuda') must fail loudly, not defer to the first tensor op."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ConfigurationError, match="CUDA is not available"):
        resolve_device("cuda")


def test_explicit_cuda_index_without_cuda_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ConfigurationError, match="CUDA is not available"):
        resolve_device("cuda:0")


def test_explicit_cuda_with_cuda_available_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cuda").type == "cuda"


def test_auto_without_cuda_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """'auto' keeps its graceful CPU fallback — only explicit 'cuda' is strict."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"


def test_build_device_context_cuda_without_cuda_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ConfigurationError):
        build_device_context("cuda", use_amp=True)

