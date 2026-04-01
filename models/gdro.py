"""
Group DRO model with sklearn-style fit/predict API.

Implements Group Distributionally Robust Optimization from:
Sagawa et al. "Distributionally Robust Neural Networks for Group Shifts" (2019)
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .torch_utils import FluxnetDataset, StratifiedBatchSampler


class GroupDRO:
    """
    Group DRO regressor with sklearn-style fit/predict API.

    Optimizes for worst-group performance by dynamically upweighting
    groups with higher loss during training.
    """

    def __init__(self, hidden_dims=[128, 64], dropout=0.1, lr=1e-3,
                 n_epochs=100, batch_size=256, group_weight_step=0.01,
                 early_stopping_rounds=10):
        """
        Initialize Group DRO model.

        Args:
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            lr: Learning rate
            n_epochs: Maximum number of training epochs
            batch_size: Batch size for training
            group_weight_step: Step size for group weight updates (eta in paper)
            early_stopping_rounds: Stop if val loss doesn't improve for this many epochs
        """
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.group_weight_step = group_weight_step
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
        """
        Train with Group DRO.

        Args:
            X: Features array of shape (n_samples, n_features)
            y: Target array of shape (n_samples,)
            eval_set: Optional list with one tuple [(X_val, y_val)] for early stopping
            envs: Environment labels (required for Group DRO)

        Returns:
            self
        """
        if envs is None:
            raise ValueError("GroupDRO requires environment labels (envs)")

        self.model = self._build_model(X.shape[1])

        # Map environment labels to integer indices
        unique_envs = np.unique(envs)
        env_to_idx = {e: i for i, e in enumerate(unique_envs)}
        env_indices = np.array([env_to_idx[e] for e in envs])
        n_groups = len(unique_envs)

        # Create dataset and stratified sampler
        dataset = FluxnetDataset(X, y, env_indices)
        sampler = StratifiedBatchSampler(env_indices, self.batch_size)
        loader = DataLoader(dataset, batch_sampler=sampler)

        # Initialize optimizer and group weights
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        group_weights = torch.ones(n_groups) / n_groups

        use_val = eval_set is not None
        if use_val:
            X_val_t = torch.tensor(eval_set[0][0], dtype=torch.float32)
            y_val_t = torch.tensor(eval_set[0][1], dtype=torch.float32).view(-1, 1)
            best_val_loss = float('inf')
            best_weights = None
            rounds_without_improvement = 0

        pbar = tqdm(range(self.n_epochs), desc="GroupDRO", unit="epoch")
        for _ in pbar:
            self.model.train()
            for X_batch, y_batch, env_batch in loader:
                optimizer.zero_grad()
                pred = self.model(X_batch)

                # Compute per-sample MSE loss
                sample_losses = F.mse_loss(pred, y_batch, reduction='none').squeeze()

                # Compute per-group average loss
                group_losses = torch.zeros(n_groups)
                group_counts = torch.zeros(n_groups)
                for g in range(n_groups):
                    mask = (env_batch == g)
                    if mask.sum() > 0:
                        group_losses[g] = sample_losses[mask].mean()
                        group_counts[g] = mask.sum()

                # Group DRO objective: weighted sum of group losses
                loss = (group_weights * group_losses).sum()
                loss.backward()
                optimizer.step()

                # Update group weights using exponentiated gradient ascent
                with torch.no_grad():
                    present_mask = group_counts > 0
                    if present_mask.any():
                        exponent = self.group_weight_step * group_losses[present_mask]
                        exponent = torch.clamp(exponent, max=10.0) # Prevent overflow
                        group_weights[present_mask] *= torch.exp(exponent)
                        group_weights = group_weights / group_weights.sum()

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
