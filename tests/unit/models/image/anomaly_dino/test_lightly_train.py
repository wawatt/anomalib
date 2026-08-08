# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the optional LightlyTrain ECViT feature extractor."""

from pathlib import Path

import pytest
import torch
from _pytest.monkeypatch import MonkeyPatch
from torch import nn

from anomalib.models.image.anomaly_dino import lightly_train


class _FakeBackbone(nn.Module):
    """Minimal ECViT-compatible backbone."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward_with_grid(self, input_tensor: torch.Tensor) -> tuple[list[torch.Tensor], tuple[int, int]]:
        batch_size = input_tensor.shape[0]
        layers = [
            torch.ones(batch_size, 4, 8) * self.scale,
            torch.full((batch_size, 4, 8), 2.0) * self.scale,
        ]
        return layers, (2, 2)


class _FakeWrapper(nn.Module):
    """Minimal LightlyTrain ECViT wrapper."""

    patch_size = 16

    def __init__(self) -> None:
        super().__init__()
        self.backbone = _FakeBackbone()


@pytest.mark.parametrize("model_name", sorted(lightly_train.SUPPORTED_ECVIT_MODELS))
def test_extracts_last_patch_token_layer(monkeypatch: MonkeyPatch, model_name: str) -> None:
    """Every public ECViT variant should expose its last patch-token layer."""
    requested_models = []

    def get_fake_wrapped_model(requested_model: str, load_weights: bool) -> _FakeWrapper:
        requested_models.append((requested_model, load_weights))
        return _FakeWrapper()

    monkeypatch.setattr(lightly_train, "_get_wrapped_model", get_fake_wrapped_model)
    extractor = lightly_train.LightlyTrainECViTFeatureExtractor(model_name)

    output = extractor.get_intermediate_layers(torch.randn(2, 3, 32, 32), n=1)

    assert extractor.patch_size == 16
    assert len(output) == 1
    assert output[0].shape == (2, 4, 8)
    assert torch.all(output[0] == 2)
    assert extractor.training is False
    assert requested_models == [(model_name, True)]


def test_rejects_unknown_ecvit_model() -> None:
    """Unknown EdgeCrafter aliases should fail before calling LightlyTrain internals."""
    with pytest.raises(ValueError, match="Unsupported LightlyTrain ECViT model"):
        lightly_train.LightlyTrainECViTFeatureExtractor("edgecrafter/ecvit-unknown")


def test_loads_lightly_train_wrapper_export(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """The adapter should load a LightlyTrain package-default state dict export."""
    source = _FakeWrapper()
    source.backbone.scale.data.fill_(3.0)
    weights_path = tmp_path / "exported_last.pt"
    torch.save(source.state_dict(), weights_path)
    target = _FakeWrapper()
    target.backbone.scale.data.zero_()
    monkeypatch.setattr(lightly_train, "_get_wrapped_model", lambda *_args, **_kwargs: target)

    lightly_train.LightlyTrainECViTFeatureExtractor("edgecrafter/ecvits", weights_path)

    assert torch.allclose(target.backbone.scale, torch.tensor(3.0))
