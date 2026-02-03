"""
MLP model with sklearn-style fit/predict API.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class MLP:
    """MLP regressor with sklearn-style fit/predict API."""

    def __init__(self, hidden_dims=[128, 64], dropout=0.1, lr=1e-3,
                 n_epochs=100, batch_size=256):
        """
        Initialize MLP model.

        Args:
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            lr: Learning rate
            n_epochs: Number of training epochs
            batch_size: Batch size for training
        """
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.model = None

    def _build_model(self, input_dim):
        """Build the neural network architecture."""
        layers = []
        prev_dim = input_dim
        for h in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y, envs=None):
        """
        Train the model.

        Args:
            X: Features array of shape (n_samples, n_features)
            y: Target array of shape (n_samples,)
            envs: Environment labels (ignored for standard MLP)

        Returns:
            self
        """
        self.model = self._build_model(X.shape[1])

        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).view(-1, 1)
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.n_epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = F.mse_loss(self.model(X_batch), y_batch)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        """
        Make predictions.

        Args:
            X: Features array of shape (n_samples, n_features)

        Returns:
            Predictions array of shape (n_samples,)
        """
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            return self.model(X_t).numpy().ravel()
