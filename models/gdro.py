"""
Group DRO model with sklearn-style fit/predict API.

Implements Group Distributionally Robust Optimization from:
Sagawa et al. "Distributionally Robust Neural Networks for Group Shifts" (2019)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class FluxnetDataset(Dataset):
    """PyTorch Dataset that includes environment/group indices."""

    def __init__(self, X, y, env_indices):
        """
        Args:
            X: Features array of shape (n_samples, n_features)
            y: Target array of shape (n_samples,)
            env_indices: Integer environment indices of shape (n_samples,)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        self.envs = torch.tensor(env_indices, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.envs[idx]


class StratifiedBatchSampler:
    """
    Batch sampler that ensures each batch has samples from multiple groups.

    Shuffles within each group, then interleaves samples across groups
    to create batches with good group coverage.
    """

    def __init__(self, env_indices, batch_size, drop_last=False):
        """
        Args:
            env_indices: Array of integer environment indices
            batch_size: Number of samples per batch
            drop_last: Whether to drop the last incomplete batch
        """
        self.env_indices = np.array(env_indices)
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Group sample indices by environment
        self.group_indices = {}
        for i, g in enumerate(env_indices):
            self.group_indices.setdefault(g, []).append(i)

        self.n_groups = len(self.group_indices)

    def __iter__(self):
        # Shuffle indices within each group
        shuffled_groups = {}
        for g, indices in self.group_indices.items():
            shuffled_groups[g] = np.random.permutation(indices).tolist()

        # Interleave: cycle through groups, taking one sample at a time
        all_indices = []
        groups = list(shuffled_groups.keys())
        group_ptrs = {g: 0 for g in groups}

        # Continue until all groups are exhausted
        while True:
            added_any = False
            for g in groups:
                if group_ptrs[g] < len(shuffled_groups[g]):
                    all_indices.append(shuffled_groups[g][group_ptrs[g]])
                    group_ptrs[g] += 1
                    added_any = True
            if not added_any:
                break

        # Create batches
        n_samples = len(all_indices)
        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            if self.drop_last and (end - start) < self.batch_size:
                break
            yield all_indices[start:end]

    def __len__(self):
        n_samples = len(self.env_indices)
        if self.drop_last:
            return n_samples // self.batch_size
        return (n_samples + self.batch_size - 1) // self.batch_size


class GroupDRO:
    """
    Group DRO regressor with sklearn-style fit/predict API.

    Optimizes for worst-group performance by dynamically upweighting
    groups with higher loss during training.
    """

    def __init__(self, hidden_dims=[128, 64], dropout=0.1, lr=1e-3,
                 n_epochs=100, batch_size=256, group_weight_step=0.01):
        """
        Initialize Group DRO model.

        Args:
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            lr: Learning rate
            n_epochs: Number of training epochs
            batch_size: Batch size for training
            group_weight_step: Step size for group weight updates (eta in paper)
        """
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.group_weight_step = group_weight_step
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

    def fit(self, X, y, envs, eval_set=None):
        """
        Train with Group DRO.

        Args:
            X: Features array of shape (n_samples, n_features)
            y: Target array of shape (n_samples,)
            envs: Environment labels (required for Group DRO)

        Returns:
            self
        """
        if envs is None:
            raise ValueError("GroupDRO requires environment labels (envs)")

        self.model = self._build_model(X.shape[1])

        # Map environment labels to integer indices
        unique_envs = np.unique(envs)
        self.env_to_idx = {e: i for i, e in enumerate(unique_envs)}
        self.idx_to_env = {i: e for e, i in self.env_to_idx.items()}
        env_indices = np.array([self.env_to_idx[e] for e in envs])
        n_groups = len(unique_envs)

        # Create dataset and stratified sampler
        dataset = FluxnetDataset(X, y, env_indices)
        sampler = StratifiedBatchSampler(env_indices, self.batch_size)
        loader = DataLoader(dataset, batch_sampler=sampler)

        # Initialize optimizer and group weights
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        group_weights = torch.ones(n_groups) / n_groups

        self.model.train()
        for epoch in range(self.n_epochs):
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
                    # Only update weights for groups present in batch
                    present_mask = group_counts > 0
                    if present_mask.any():
                        group_weights[present_mask] *= torch.exp(
                            self.group_weight_step * group_losses[present_mask]
                        )
                        # Project back to probability simplex
                        group_weights = group_weights / group_weights.sum()

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
