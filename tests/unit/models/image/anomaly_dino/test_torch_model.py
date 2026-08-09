# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the AnomalyDINO torch model."""

from pathlib import Path

import numpy as np
import pytest
import torch
from _pytest.monkeypatch import MonkeyPatch
from torch import nn

from anomalib.models.image.anomaly_dino.torch_model import AnomalyDINOModel


class TestAnomalyDINOModel:
    """Test the AnomalyDINO torch model."""

    @staticmethod
    def test_initialization_defaults() -> None:
        """Test initialization with default arguments."""
        model = AnomalyDINOModel()
        assert "dinov2" in model.encoder_name
        assert model.memory_bank.numel() == 0

    @staticmethod
    def test_invalid_encoder_name_raises() -> None:
        """Test that invalid encoder names raise an error."""
        with pytest.raises(ValueError, match="Encoder name must contain 'dino' or start with 'edgecrafter/'"):
            _ = AnomalyDINOModel(encoder_name="resnet50")

    @staticmethod
    def test_lightly_train_ecvit_initialization(monkeypatch: MonkeyPatch) -> None:
        """Test that EdgeCrafter model names use the LightlyTrain feature extractor."""

        class FakeECViTFeatureExtractor(nn.Module):
            """Minimal ECViT extractor used to avoid the optional test dependency."""

            patch_size = 16

            def __init__(self, model_name: str, weights_path: str | None) -> None:
                super().__init__()
                self.model_name = model_name
                self.weights_path = weights_path

            @staticmethod
            def get_intermediate_layers(input_tensor: torch.Tensor, n: int) -> tuple[torch.Tensor]:
                return (torch.zeros(input_tensor.shape[0], 4, 8),) * n

        monkeypatch.setattr(
            "anomalib.models.image.anomaly_dino.torch_model.LightlyTrainECViTFeatureExtractor",
            FakeECViTFeatureExtractor,
        )
        model = AnomalyDINOModel(
            encoder_name="edgecrafter/ecvits",
            encoder_weights="exported_last.pt",
        )

        assert model.feature_encoder.model_name == "edgecrafter/ecvits"
        assert model.feature_encoder.weights_path == "exported_last.pt"
        assert model.feature_encoder.patch_size == 16

    @staticmethod
    @pytest.mark.parametrize("official_dinov2_format", [False, True])
    def test_loads_local_dino_weights(
        monkeypatch: MonkeyPatch,
        tmp_path: Path,
        official_dinov2_format: bool,
    ) -> None:
        """Test loading timm-native and official DINOv2 local state dictionaries."""

        class FakeBackbone(nn.Module):
            """Minimal timm DINO backbone with register and position parameters."""

            def __init__(self) -> None:
                super().__init__()
                self.reg_token = nn.Parameter(torch.zeros(1, 4, 8))
                self.pos_embed = nn.Parameter(torch.zeros(1, 4, 8))

        class FakeTimmFeatureExtractor(nn.Module):
            """Minimal extractor exposing the underlying timm backbone."""

            patch_size = 14

            def __init__(self, *_args: object, pre_trained: bool, **_kwargs: object) -> None:
                super().__init__()
                self.pre_trained = pre_trained
                self.feature_extractor = FakeBackbone()

        weights_path = tmp_path / ("dinov2.pth" if official_dinov2_format else "dinov2.safetensors")
        weights_path.write_bytes(b"placeholder")
        pos_embed = torch.full((1, 4, 8), 2.0)
        if official_dinov2_format:
            state_dict = {
                "register_tokens": torch.ones(1, 4, 8),
                "mask_token": torch.zeros(1, 8),
                "pos_embed": torch.cat((torch.zeros(1, 1, 8), pos_embed), dim=1),
            }
        else:
            state_dict = {
                "reg_token": torch.ones(1, 4, 8),
                "pos_embed": pos_embed,
            }

        monkeypatch.setattr(
            "anomalib.models.image.anomaly_dino.torch_model.TimmFeatureExtractor",
            FakeTimmFeatureExtractor,
        )
        monkeypatch.setattr(
            "anomalib.models.image.anomaly_dino.torch_model.load_timm_state_dict",
            lambda *_args, **_kwargs: state_dict,
        )

        model = AnomalyDINOModel(
            encoder_name="vit_small_patch14_reg4_dinov2",
            encoder_weights=weights_path,
        )

        assert model.feature_encoder.pre_trained is False
        assert torch.all(model.feature_encoder.feature_extractor.reg_token == 1)
        assert torch.all(model.feature_encoder.feature_extractor.pos_embed == 2)

    @staticmethod
    def test_fit_raises_without_embeddings() -> None:
        """Test that fit raises when no embeddings have been collected."""
        model = AnomalyDINOModel()
        with pytest.raises(ValueError, match="No embeddings collected"):
            model.fit()

    @staticmethod
    def test_forward_train_adds_embeddings(monkeypatch: MonkeyPatch) -> None:
        """Test training mode collects embeddings into store."""
        model = AnomalyDINOModel()
        model.train()

        fake_features = torch.randn(2, 8, 128)
        monkeypatch.setattr(model, "extract_features", lambda _: fake_features)

        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert torch.is_tensor(output)
        assert output.requires_grad
        assert len(model.embedding_store) == 1
        assert model.embedding_store[0].ndim == 2

    @staticmethod
    def test_forward_eval_raises_with_empty_memory_bank(monkeypatch: MonkeyPatch) -> None:
        """Test that inference raises an error when memory bank is empty."""
        model = AnomalyDINOModel()
        model.eval()

        fake_features = torch.randn(1, 16, 64)
        monkeypatch.setattr(model, "extract_features", lambda _: fake_features)
        model.register_buffer("memory_bank", torch.empty(0, 64))

        x = torch.randn(1, 3, 224, 224)
        with pytest.raises(RuntimeError, match="Memory bank is empty"):
            _ = model(x)

    @staticmethod
    def test_compute_background_masks_runs() -> None:
        """Test that background mask computation produces boolean masks."""
        b, h, w, d = 2, 8, 8, 16
        features = np.random.randn(b, h * w, d).astype(np.float32)  # noqa: NPY002
        masks = AnomalyDINOModel.compute_background_masks(features, (h, w))
        assert masks.shape == (b, h * w)
        assert masks.dtype == bool

    @staticmethod
    def test_mean_top1p_computation() -> None:
        """Test that mean_top1p returns expected shape and value."""
        distances = torch.arange(0, 100, dtype=torch.float32).view(1, -1)
        result = AnomalyDINOModel.mean_top1p(distances)
        assert result.shape == (1, 1)
        assert torch.allclose(result, torch.tensor([[99.0]]))

    @staticmethod
    def test_forward_half_precision_eval(monkeypatch: MonkeyPatch) -> None:
        """Test inference in half precision (float16) using matmul cosine distance."""
        model = AnomalyDINOModel().half()
        model.eval()

        fake_features = torch.randn(1, 16, 64, dtype=torch.float16)
        monkeypatch.setattr(model, "extract_features", lambda _: fake_features)
        monkeypatch.setattr(model.anomaly_map_generator, "__call__", lambda x, __: x)

        model.register_buffer("memory_bank", torch.randn(16, 64, dtype=torch.float16))
        x = torch.randn(1, 3, 224, 224, dtype=torch.float16)
        out = model(x)

        assert hasattr(out, "pred_score")
        assert out.pred_score.shape == (1, 1)
        # outputs should be float16-safe with matmul
        assert out.pred_score.dtype == torch.float16
