# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LightlyTrain ECViT feature extractor for AnomalyDINO."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from pathlib import Path

import torch
from torch import Tensor, nn

SUPPORTED_ECVIT_MODELS = frozenset({
    "edgecrafter/ecvitt",
    "edgecrafter/ecvittplus",
    "edgecrafter/ecvits",
    "edgecrafter/ecvitsplus",
})


def _get_wrapped_model(model_name: str, load_weights: bool) -> nn.Module:
    """Create a LightlyTrain model without importing it for DINOv2 users."""
    try:
        package_helpers = import_module("lightly_train._models.package_helpers")
    except ImportError as exc:
        msg = (
            "Using an EdgeCrafter ECViT encoder requires lightly-train>=0.17.0. "
            "Install it with `pip install 'lightly-train>=0.17.0'`."
        )
        raise ImportError(msg) from exc

    return package_helpers.get_wrapped_model(
        model=model_name,
        num_input_channels=3,
        load_weights=load_weights,
    )


def _unwrap_state_dict(checkpoint: object) -> Mapping[str, Tensor]:
    """Extract a tensor state dict from a lightweight exported checkpoint."""
    if not isinstance(checkpoint, Mapping):
        msg = f"Expected a state dict mapping, got {type(checkpoint)!r}."
        raise TypeError(msg)

    candidates = [checkpoint]
    candidates.extend(checkpoint[key] for key in ("state_dict", "model", "backbone") if key in checkpoint)
    for candidate in candidates:
        if isinstance(candidate, Mapping) and all(
            isinstance(key, str) and isinstance(value, Tensor) for key, value in candidate.items()
        ):
            return candidate

    msg = "Expected the checkpoint to contain a tensor state dict."
    raise TypeError(msg)


class LightlyTrainECViTFeatureExtractor(nn.Module):
    """Expose LightlyTrain ECViT patch tokens through the AnomalyDINO interface.

    Args:
        model_name: LightlyTrain model name. Supported values are
            ``"edgecrafter/ecvitt"``, ``"edgecrafter/ecvittplus"``,
            ``"edgecrafter/ecvits"``, and ``"edgecrafter/ecvitsplus"``.
        weights_path: Optional path to a LightlyTrain lightweight model export such
            as ``exported_last.pt`` or ``exported_best.pt``. When omitted,
            LightlyTrain loads the model's default pretrained weights.
    """

    def __init__(self, model_name: str, weights_path: str | Path | None = None) -> None:
        super().__init__()
        if model_name not in SUPPORTED_ECVIT_MODELS:
            supported_models = ", ".join(sorted(SUPPORTED_ECVIT_MODELS))
            msg = f"Unsupported LightlyTrain ECViT model '{model_name}'. Supported models: {supported_models}."
            raise ValueError(msg)

        wrapped_model = _get_wrapped_model(model_name, load_weights=weights_path is None)

        if not hasattr(wrapped_model, "backbone") or not hasattr(wrapped_model, "patch_size"):
            msg = f"LightlyTrain model '{model_name}' is not an ECViT backbone wrapper."
            raise TypeError(msg)

        if weights_path is not None:
            self._load_exported_weights(wrapped_model, Path(weights_path))

        self.backbone = wrapped_model.backbone
        self.patch_size = int(wrapped_model.patch_size)
        self.backbone.requires_grad_(requires_grad=False)
        self.eval()

    @staticmethod
    def _load_exported_weights(wrapped_model: nn.Module, weights_path: Path) -> None:
        """Load either a LightlyTrain wrapper export or raw ECViT backbone weights."""
        if not weights_path.is_file():
            msg = f"LightlyTrain ECViT weights file does not exist: {weights_path}"
            raise FileNotFoundError(msg)

        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        state_dict = _unwrap_state_dict(checkpoint)

        try:
            wrapped_model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            backbone_state = {
                key.removeprefix("backbone."): value for key, value in state_dict.items() if key.startswith("backbone.")
            }
            if not backbone_state:
                backbone_state = dict(state_dict)
            try:
                wrapped_model.backbone.load_state_dict(backbone_state, strict=True)
            except RuntimeError as backbone_error:
                msg = (
                    f"Could not load ECViT weights from '{weights_path}'. Use a LightlyTrain "
                    "lightweight export (exported_last.pt/exported_best.pt) or a raw ECViT "
                    "backbone state dict; full .ckpt training checkpoints are not supported."
                )
                raise ValueError(msg) from backbone_error
        else:
            return

    def train(self, mode: bool = True) -> LightlyTrainECViTFeatureExtractor:
        """Keep the frozen extractor in evaluation mode during memory-bank fitting."""
        del mode
        return super().train(mode=False)

    def get_intermediate_layers(self, input_tensor: Tensor, n: int = 1) -> tuple[Tensor, ...]:
        """Return the final ``n`` ECViT patch-token layers without the register token."""
        if n < 1:
            msg = f"n must be positive, got {n}."
            raise ValueError(msg)

        layers, _ = self.backbone.forward_with_grid(input_tensor)
        if len(layers) < n:
            msg = f"ECViT returned {len(layers)} layers, but {n} were requested."
            raise RuntimeError(msg)
        return tuple(layers[-n:])
