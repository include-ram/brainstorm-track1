"""
Channel projection utilities for reducing high-density electrode arrays.

This module provides methods to reduce the number of channels from high-density
arrays (like 1024 channels) to smaller representations suitable for models
trained on lower-density recordings.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA


class LearnedChannelProjection(nn.Module):
    """
    Learnable linear projection for channel reduction.

    Maps from a high number of input channels to fewer output channels
    using a learned linear transformation.

    Attributes:
        projection: Linear layer performing the channel reduction.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initialize the projection layer.

        Args:
            in_channels: Number of input channels (e.g., 1024).
            out_channels: Number of output channels (e.g., 64).
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.projection = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input channels to reduced dimension.

        Args:
            x: Input tensor of shape (batch, in_channels) or (batch, time, in_channels).

        Returns:
            Projected tensor of shape (batch, out_channels) or (batch, time, out_channels).
        """
        return self.projection(x)


class SpatialAverageProjection:
    """
    Reduce channels by averaging groups of spatially adjacent electrodes.

    Groups electrodes into spatial blocks and averages within each block.
    Useful when electrode layout is known and spatial relationships matter.
    """

    def __init__(
        self,
        grid_height: int = 31,
        grid_width: int = 32,
        target_height: int = 8,
        target_width: int = 8,
    ) -> None:
        """
        Initialize spatial averaging projection.

        Args:
            grid_height: Height of original electrode grid.
            grid_width: Width of original electrode grid.
            target_height: Height of target grid after reduction.
            target_width: Width of target grid after reduction.
        """
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.target_height = target_height
        self.target_width = target_width

        # Compute block sizes
        self.block_h = grid_height // target_height
        self.block_w = grid_width // target_width

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Apply spatial averaging to reduce channels.

        Args:
            x: Input array of shape (n_samples, n_channels) where
               n_channels = grid_height * grid_width.

        Returns:
            Reduced array of shape (n_samples, target_height * target_width).
        """
        n_samples = x.shape[0]

        # Reshape to grid
        x_grid = x.reshape(n_samples, self.grid_height, self.grid_width)

        # Average over blocks
        result = []
        for i in range(self.target_height):
            for j in range(self.target_width):
                h_start = i * self.block_h
                h_end = min((i + 1) * self.block_h, self.grid_height)
                w_start = j * self.block_w
                w_end = min((j + 1) * self.block_w, self.grid_width)

                block = x_grid[:, h_start:h_end, w_start:w_end]
                block_mean = block.reshape(n_samples, -1).mean(axis=1)
                result.append(block_mean)

        return np.stack(result, axis=1)


class PCAProjection:
    """
    Reduce channels using Principal Component Analysis.

    Projects high-dimensional channel data onto principal components,
    retaining the most informative dimensions.
    """

    def __init__(self, n_components: int = 64) -> None:
        """
        Initialize PCA projection.

        Args:
            n_components: Number of principal components to retain.
        """
        self.n_components = n_components
        self.pca: PCA | None = None
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PCAProjection":
        """
        Fit PCA on training data.

        Args:
            X: Training data of shape (n_samples, n_channels).

        Returns:
            Self for method chaining.
        """
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X)
        self.mean_ = self.pca.mean_
        self.components_ = self.pca.components_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply PCA transformation.

        Args:
            X: Input data of shape (n_samples, n_channels).

        Returns:
            Transformed data of shape (n_samples, n_components).
        """
        if self.pca is None:
            raise RuntimeError("PCA not fitted. Call fit() first.")
        return self.pca.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA and transform data in one step.

        Args:
            X: Training data of shape (n_samples, n_channels).

        Returns:
            Transformed data of shape (n_samples, n_components).
        """
        self.fit(X)
        return self.transform(X)

    def get_torch_projection(self) -> nn.Linear:
        """
        Convert fitted PCA to a PyTorch linear layer for inference.

        Returns:
            Linear layer that applies the PCA projection.

        Raises:
            RuntimeError: If PCA not fitted.
        """
        if self.pca is None or self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA not fitted. Call fit() first.")

        in_features = self.components_.shape[1]
        out_features = self.components_.shape[0]

        linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            # PCA transform is: (X - mean) @ components.T
            # We can represent this as linear: X @ W.T + b
            # where W = components and b = -mean @ components.T
            linear.weight.copy_(torch.tensor(self.components_, dtype=torch.float32))
            bias = -self.mean_ @ self.components_.T
            linear.bias.copy_(torch.tensor(bias, dtype=torch.float32))

        return linear
