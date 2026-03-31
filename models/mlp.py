"""
MLP model with sklearn-style fit/predict API.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class MLP:
    """MLP regressor with sklearn-style fit/predict API."""

    def __init__(self, hidden_dims=[128, 64], dropout=0.1, lr=1e-3,
                 n_epochs=100, batch_size=256, early_stopping_rounds=10):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stopping_rounds = early_stopping_rounds
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

    def fit(self, X, y, eval_set=None, envs=None):
        self.model = self._build_model(X.shape[1])

        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).view(-1, 1)
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        use_val = eval_set is not None
        if use_val:
            X_val_t = torch.tensor(eval_set[0][0], dtype=torch.float32)
            y_val_t = torch.tensor(eval_set[0][1], dtype=torch.float32).view(-1, 1)
            best_val_loss = float('inf')
            best_weights = None
            rounds_without_improvement = 0

        pbar = tqdm(range(self.n_epochs), desc="MLP", unit="epoch")
        for _ in pbar:
            self.model.train()
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = F.mse_loss(self.model(X_batch), y_batch)
                loss.backward()
                optimizer.step()

            if use_val:
                self.model.eval()
                with torch.no_grad():
                    val_loss = F.mse_loss(self.model(X_val_t), y_val_t).item()
                pbar.set_postfix(val_loss=f"{val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = copy.deepcopy(self.model.state_dict())
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1
                    if rounds_without_improvement >= self.early_stopping_rounds:
                        self.model.load_state_dict(best_weights)
                        break

        if use_val and best_weights is not None:
            self.model.load_state_dict(best_weights)

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
