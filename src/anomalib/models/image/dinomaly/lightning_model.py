# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dinomaly: Vision Transformer-based Anomaly Detection with Feature Reconstruction.

This module implements the Dinomaly model for anomaly detection using a Vision Transformer
encoder-decoder architecture. The model leverages pre-trained DINOv2 features and employs
a reconstruction-based approach to detect anomalies by comparing encoder and decoder features.

Dinomaly extracts features from multiple intermediate layers of a DINOv2 Vision Transformer,
compresses them through a bottleneck MLP, and reconstructs them using a Vision Transformer
decoder. Anomaly detection is performed by computing cosine similarity between encoder
and decoder features at multiple scales.

The model is particularly effective for visual anomaly detection tasks where the goal is
to identify regions or images that deviate from normal patterns learned during training.

Example:
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.models import Dinomaly
    >>> from anomalib.engine import Engine

    >>> datamodule = MVTecAD()
    >>> model = Dinomaly()
    >>> engine = Engine()

    >>> engine.fit(model, datamodule=datamodule)  # doctest: +SKIP
    >>> predictions = engine.predict(model, datamodule=datamodule)  # doctest: +SKIP

Notes:
    - The model uses DINOv2 Vision Transformer as the backbone encoder
    - Features are extracted from intermediate layers (typically layers 2-9 for base models)
    - A bottleneck MLP compresses multi-layer features before reconstruction
    - Anomaly maps are computed using cosine similarity between encoder-decoder features
    - The model supports both unsupervised anomaly detection and localization

See Also:
    :class:`anomalib.models.image.dinomaly.torch_model.DinomalyModel`:
        PyTorch implementation of the Dinomaly model.
"""

import logging
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch.nn.init import trunc_normal_
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, Resize

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.models.image.dinomaly.components import StableAdamW, WarmCosineScheduler
from anomalib.models.image.dinomaly.torch_model import DinomalyModel
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

logger = logging.getLogger(__name__)

# Training constants
DEFAULT_IMAGE_SIZE = 448
DEFAULT_CROP_SIZE = 392
MAX_STEPS_DEFAULT = 5000

# Default Training hyperparameters
TRAINING_CONFIG: dict[str, Any] = {
    "optimizer": {
        "lr": 2e-3,
        "betas": (0.9, 0.999),
        "weight_decay": 1e-4,
        "amsgrad": True,
        "eps": 1e-8,
    },
    "scheduler": {
        "base_value": 2e-3,
        "final_value": 2e-4,
        "total_iters": MAX_STEPS_DEFAULT,
        "warmup_iters": 100,
    },
    "trainer": {
        "gradient_clip_val": 0.1,
        "num_sanity_val_steps": 0,
        "max_steps": MAX_STEPS_DEFAULT,
    },
}


class Dinomaly(AnomalibModule):
    """Dinomaly Lightning Module for Vision Transformer-based Anomaly Detection.

    This lightning module trains the Dinomaly anomaly detection model (DinomalyModel).
    During training, the decoder learns to reconstruct normal features.
    During inference, the trained decoder is expected to successfully reconstruct normal
    regions of feature maps, but fail to reconstruct anomalous regions as
    it has not seen such patterns.

    Args:
        encoder_name (str): Name of the Vision Transformer encoder to use.
            Supports DINOv2 variants (small, base, large) with different patch sizes.
            Defaults to "dinov2reg_vit_base_14".
        bottleneck_dropout (float): Dropout rate for the bottleneck MLP layer.
            Helps prevent overfitting during feature compression. Defaults to 0.2.
        decoder_depth (int): Number of Vision Transformer decoder layers.
            More layers allow for more complex reconstruction. Defaults to 8.
        target_layers (list[int] | None): List of encoder layer indices to extract
            features from. If None, uses [2, 3, 4, 5, 6, 7, 8, 9] for base models
            and [4, 6, 8, 10, 12, 14, 16, 18] for large models.
        fuse_layer_encoder (list[list[int]] | None): Groupings of encoder layers
            for feature fusion. If None, uses [[0, 1, 2, 3], [4, 5, 6, 7]].
        fuse_layer_decoder (list[list[int]] | None): Groupings of decoder layers
            for feature fusion. If None, uses [[0, 1, 2, 3], [4, 5, 6, 7]].
        remove_class_token (bool): Whether to remove class token from features
            before processing. Defaults to False.
        pre_processor (PreProcessor | bool, optional): Pre-processor instance or
            flag to use default. Defaults to ``True``.
        post_processor (PostProcessor | bool, optional): Post-processor instance
            or flag to use default. Defaults to ``True``.
        evaluator (Evaluator | bool, optional): Evaluator instance or flag to use
            default. Defaults to ``True``.
        visualizer (Visualizer | bool, optional): Visualizer instance or flag to
            use default. Defaults to ``True``.

    Example:
        >>> from anomalib.data import MVTecAD
        >>> from anomalib.models import Dinomaly
        >>>
        >>> # Basic usage with default parameters
        >>> model = Dinomaly()
        >>>
        >>> # Custom configuration
        >>> model = Dinomaly(
        ...     encoder_name="dinov2reg_vit_large_14",
        ...     decoder_depth=12,
        ...     bottleneck_dropout=0.1,
        ...     mask_neighbor_size=3
        ... )
        >>>
        >>> # Training with datamodule
        >>> datamodule = MVTecAD()
        >>> engine = Engine()
        >>> engine.fit(model, datamodule=datamodule)

    Note:
        The model requires significant GPU memory due to the Vision Transformer
        architecture. Consider using gradient checkpointing or smaller model
        variants for memory-constrained environments.
    """

    def __init__(
        self,
        encoder_name: str = "dinov2reg_vit_base_14",
        bottleneck_dropout: float = 0.2,
        decoder_depth: int = 8,
        target_layers: list[int] | None = None,
        fuse_layer_encoder: list[list[int]] | None = None,
        fuse_layer_decoder: list[list[int]] | None = None,
        remove_class_token: bool = False,
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

        self.model: DinomalyModel = DinomalyModel(
            encoder_name=encoder_name,
            bottleneck_dropout=bottleneck_dropout,
            decoder_depth=decoder_depth,
            target_layers=target_layers,
            fuse_layer_encoder=fuse_layer_encoder,
            fuse_layer_decoder=fuse_layer_decoder,
            remove_class_token=remove_class_token,
        )

        # Set the trainable parameters for the model.
        # Only the bottleneck and decoder parameters are trained.

        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze bottleneck and decoder
        for param in self.model.bottleneck.parameters():
            param.requires_grad = True
        for param in self.model.decoder.parameters():
            param.requires_grad = True

        self.trainable_modules = torch.nn.ModuleList([self.model.bottleneck, self.model.decoder])
        self._initialize_trainable_modules(self.trainable_modules)

    @classmethod
    def configure_pre_processor(
        cls,
        image_size: tuple[int, int] | None = None,
        crop_size: int | None = None,
    ) -> PreProcessor:
        """Configure the default pre-processor for Dinomaly.

        Sets up image preprocessing pipeline including resizing, center cropping,
        and normalization with ImageNet statistics. The preprocessing is optimized
        for DINOv2 Vision Transformer models.

        Args:
            image_size (tuple[int, int] | None): Target size for image resizing
                as (height, width). Defaults to (448, 448).
            crop_size (int | None): Target size for center cropping (assumes square crop).
                Should be smaller than image_size. Defaults to 392.

        Returns:
            PreProcessor: Configured pre-processor with transforms for Dinomaly.

        Raises:
            ValueError: If crop_size is larger than the minimum dimension of image_size.

        Note:
            The default ImageNet normalization statistics are used:
            - Mean: [0.485, 0.456, 0.406]
            - Std: [0.229, 0.224, 0.225]
        """
        crop_size = crop_size or DEFAULT_CROP_SIZE
        image_size = image_size or (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)

        # Validate inputs
        if crop_size > min(image_size):
            msg = f"Crop size {crop_size} cannot be larger than image size {image_size}"
            raise ValueError(msg)

        data_transforms = Compose([
            Resize(image_size),
            CenterCrop(crop_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        return PreProcessor(transform=data_transforms)

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Training step for the Dinomaly model.

        Performs a single training iteration by computing feature reconstruction loss
        between encoder and decoder features. Uses progressive cosine similarity loss
        with the hardest mining to focus training on difficult examples.

        Args:
            batch (Batch): Input batch containing images and metadata.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            STEP_OUTPUT: Dictionary containing the computed loss value.

        Raises:
            ValueError: If model output doesn't contain required features during training.

        Note:
            The loss function uses progressive weight scheduling where the hardest
            mining percentage increases from 0 to 0.9 over 1000 steps, focusing
            on increasingly difficult examples as training progresses.
        """
        del args, kwargs  # These variables are not used.
        loss = self.model(batch.image, global_step=self.global_step)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Validation step for the Dinomaly model.

        Performs inference on the validation batch to compute anomaly scores
        and anomaly maps. The model operates in evaluation mode to generate
        predictions for anomaly detection evaluation.

        Args:
            batch (Batch): Input batch containing images and metadata.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            STEP_OUTPUT: Updated batch with pred_score (anomaly scores) and
                anomaly_map (pixel-level anomaly maps) predictions.

        Raises:
            Exception: If an error occurs during validation inference.

        Note:
            During validation, the model returns InferenceBatch with anomaly
            scores and maps computed from encoder-decoder feature comparisons.
        """
        del args, kwargs  # These variables are not used.

        predictions = self.model(batch.image)
        return batch.update(pred_score=predictions.pred_score, anomaly_map=predictions.anomaly_map)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure optimizer and learning rate scheduler for Dinomaly training.

        Sets up the training configuration with frozen DINOv2 encoder and trainable
        bottleneck and decoder components. Uses StableAdamW optimizer with warm
        cosine learning rate scheduling.

        The total number of training steps is determined dynamically from the trainer
        configuration, supporting both max_steps and max_epochs settings.

        Returns:
            OptimizerLRScheduler: Tuple containing optimizer and scheduler configurations.

        Raises:
            ValueError: If neither max_epochs nor max_steps is defined.

        Note:
            - DINOv2 encoder parameters are frozen to preserve pre-trained features
            - Only bottleneck MLP and decoder parameters are trained
            - Uses truncated normal initialization for Linear layers
            - Learning rate schedule: warmup (100 steps) + cosine decay
            - Base learning rate: 2e-3, final learning rate: 2e-4
            - Total steps determined from trainer's max_steps or max_epochs
        """
        # Determine total training steps dynamically from trainer configuration
        # Check if the trainer has valid max_epochs and max_steps set
        max_epochs = getattr(self.trainer, "max_epochs", -1)
        max_steps = getattr(self.trainer, "max_steps", -1)

        if max_epochs is None:
            max_epochs = -1
        if max_steps is None:
            max_steps = -1

        if max_epochs < 0 and max_steps < 0:
            msg = "A finite number of steps or epochs must be defined"
            raise ValueError(msg)

        if max_epochs < 0:
            # max_epochs not set, use max_steps directly
            total_steps = max_steps
        elif max_steps < 0:
            # max_steps not set, calculate from max_epochs
            total_steps = max_epochs * len(self.trainer.datamodule.train_dataloader())
        else:
            # Both are set, use the minimum (training stops at whichever comes first)
            total_steps = min(max_steps, max_epochs * len(self.trainer.datamodule.train_dataloader()))

        optimizer_config = TRAINING_CONFIG["optimizer"]
        assert isinstance(optimizer_config, dict)
        optimizer = StableAdamW([{"params": self.trainable_modules.parameters()}], **optimizer_config)

        # Create a scheduler config with dynamically determined total steps
        scheduler_config = TRAINING_CONFIG["scheduler"].copy()
        assert isinstance(scheduler_config, dict)
        scheduler_config["total_iters"] = total_steps

        lr_scheduler = WarmCosineScheduler(optimizer, **scheduler_config)

        return [optimizer], [lr_scheduler]

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Dinomaly is an unsupervised anomaly detection model that learns normal
        data patterns without requiring anomaly labels during training.

        Returns:
            LearningType: Always returns LearningType.ONE_CLASS for unsupervised learning.

        Note:
            This property may be subject to change if supervised training support
            is introduced in future versions.
        """
        return LearningType.ONE_CLASS

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return Dinomaly-specific trainer arguments.

        Provides configuration arguments optimized for Dinomaly training,
        excluding max_steps to allow users to set their own training duration.

        Returns:
            dict[str, Any]: Dictionary of trainer arguments with strategy
                configuration for optimal training performance. Does not include
                max_steps so it can be set by the engine or user.

        Note:
            The max_steps is intentionally excluded to allow user override.
        """
        trainer_config = TRAINING_CONFIG["trainer"].copy()
        assert isinstance(trainer_config, dict)
        # Remove max_steps to allow user override
        trainer_config.pop("max_steps", None)
        return trainer_config

    @staticmethod
    def _initialize_trainable_modules(trainable_modules: torch.nn.ModuleList) -> None:
        """Initialize trainable modules with truncated normal initialization.

        Args:
            trainable_modules: ModuleList containing modules to initialize
        """
        for m in trainable_modules.modules():
            if isinstance(m, torch.nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.constant_(m.weight, 1.0)
