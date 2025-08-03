# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch modules for the Reverse Distillation model implementation.

This module contains the core components used in the Reverse Distillation model
architecture, including the bottleneck layer and decoder network.

The components work together to learn a compact representation of normal images
through distillation and reconstruction:

- Bottleneck layer: Compresses features into a lower dimensional space

Example:
    >>> from anomalib.models.image.reverse_distillation.components import (
    ...     get_bottleneck_layer,
    ...     get_decoder
    ... )
    >>> bottleneck = get_bottleneck_layer()
    >>> decoder = get_decoder()

See Also:
    - :func:`anomalib.models.image.reverse_distillation.components.bottleneck`:
        Bottleneck layer implementation
"""

from .bottleneck import get_bottleneck_layer

__all__ = ["get_bottleneck_layer"]
