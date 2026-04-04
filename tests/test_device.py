import torch

from src.core.device import build_device_context, resolve_device


def test_device_resolution() -> None:
    auto_device = resolve_device("auto")
    assert auto_device.type in {"cpu", "cuda"}
    cpu_device = resolve_device("cpu")
    assert cpu_device.type == "cpu"


def test_device_context_defaults() -> None:
    ctx = build_device_context("cpu", use_amp=True)
    assert ctx.device == torch.device("cpu")
    assert ctx.use_amp is False

