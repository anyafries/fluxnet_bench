"""
Shared PyTorch utilities for neural network models.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


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
