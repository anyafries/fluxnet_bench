"""
Deep CORAL and MMD models with sklearn-style fit/predict API.

CORAL implements Deep CORAL (Correlation Alignment) for domain generalization from:
  Sun & Saenko, "Deep CORAL: Correlation Alignment for Deep Domain Adaptation" (ECCV 2016)

MMD implements Maximum Mean Discrepancy domain generalization.
  Adapted from https://github.com/mlfoundations/tableshift

In the domain generalization setting, penalties are applied pairwise (or vs. global)
across all training environments, encouraging domain-invariant representations.
"""

import copy
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .torch_utils import FluxnetDataset, StratifiedBatchSampler


class _AbstractDomainAlignment(ABC):
    """
    Base class for domain-alignment regularizers (CORAL, MMD).

    Subclasses implement `_compute_penalty(feats, env_batch, subset_envs)`
    which returns `(penalty_tensor, lambda_weight)`.
    """

    def __init__(self, hidden_dims=[128, 64], dropout=0.1, lr=1e-3,
                 n_epochs=500, batch_size=256, early_stopping_rounds=10):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stopping_rounds = early_stopping_rounds
        self.feature_extractor = None
        self.head = None

    def _build_model(self, input_dim):
        layers = []
        prev_dim = input_dim
        for h in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            prev_dim = h
        self.feature_extractor = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, 1)

    def _forward(self, X):
        feats = self.feature_extractor(X)
        return self.head(feats), feats

    @abstractmethod
    def _compute_penalty(self, feats, env_batch, subset_envs):
        ...

    def fit(self, X, y, eval_set=None, envs=None):
        """
        Train with domain alignment penalty.

        Args:
            X: Features array (n_samples, n_features)
            y: Targets array (n_samples,)
            eval_set: Optional [(X_val, y_val)] for early stopping
            envs: Environment labels (required)
        """
        if envs is None:
            raise ValueError(f"{type(self).__name__} requires environment labels (envs)")

        self._build_model(X.shape[1])

        unique_envs = np.unique(envs)
        env_to_idx = {e: i for i, e in enumerate(unique_envs)}
        env_indices = np.array([env_to_idx[e] for e in envs])
        dataset = FluxnetDataset(X, y, env_indices)
        sampler = StratifiedBatchSampler(env_indices, self.batch_size)
        loader = DataLoader(dataset, batch_sampler=sampler)

        params = list(self.feature_extractor.parameters()) + list(self.head.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)

        use_val = eval_set is not None
        if use_val:
            X_val_t = torch.tensor(eval_set[0][0], dtype=torch.float32)
            y_val_t = torch.tensor(eval_set[0][1], dtype=torch.float32).view(-1, 1)
            best_val_loss = float('inf')
            best_extractor_weights = None
            best_head_weights = None
            rounds_without_improvement = 0

        pbar = tqdm(range(self.n_epochs), desc=type(self).__name__, unit="epoch")
        for _ in pbar:
            self.feature_extractor.train()
            self.head.train()
            for X_batch, y_batch, env_batch in loader:
                optimizer.zero_grad()
                pred, feats = self._forward(X_batch)
                mse = F.mse_loss(pred, y_batch)

                unique_batch_envs = torch.unique(env_batch)
                perm = torch.randperm(len(unique_batch_envs))
                subset_envs = unique_batch_envs[perm[:self._n_pairs]]

                penalty, lam = self._compute_penalty(feats, env_batch, subset_envs)
                loss = mse + lam * penalty
                loss.backward()
                optimizer.step()

            if use_val:
                self.feature_extractor.eval()
                self.head.eval()
                with torch.no_grad():
                    val_pred = self.head(self.feature_extractor(X_val_t))
                    val_loss = F.mse_loss(val_pred, y_val_t).item()
                pbar.set_postfix(val_loss=f"{val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_extractor_weights = copy.deepcopy(self.feature_extractor.state_dict())
                    best_head_weights = copy.deepcopy(self.head.state_dict())
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1
                    if rounds_without_improvement >= self.early_stopping_rounds:
                        self.feature_extractor.load_state_dict(best_extractor_weights)
                        self.head.load_state_dict(best_head_weights)
                        break

        if use_val and best_extractor_weights is not None:
            self.feature_extractor.load_state_dict(best_extractor_weights)
            self.head.load_state_dict(best_head_weights)

        return self

    def predict(self, X):
        self.feature_extractor.eval()
        self.head.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            return self.head(self.feature_extractor(X_t)).numpy().ravel()


class CORAL(_AbstractDomainAlignment):
    """
    Deep CORAL regressor with sklearn-style fit/predict API.

    Minimizes MSE plus a CORAL penalty that aligns feature distribution
    means and covariances across training environments.
    """

    def __init__(self, hidden_dims=[128, 64], dropout=0.1, lr=1e-3,
                 n_epochs=500, batch_size=256, early_stopping_rounds=10,
                 coral_lambda=1.0, num_coral_pairs=10):
        super().__init__(hidden_dims=hidden_dims, dropout=dropout, lr=lr,
                         n_epochs=n_epochs, batch_size=batch_size,
                         early_stopping_rounds=early_stopping_rounds)
        self.coral_lambda = coral_lambda
        self.num_coral_pairs = num_coral_pairs
        self._n_pairs = num_coral_pairs

    def _compute_penalty(self, feats, env_batch, subset_envs):
        global_mean = feats.mean(0)
        cent_batch = feats - global_mean
        global_cov = (cent_batch.T @ cent_batch) / (len(feats) - 1)
        coral = torch.tensor(0.0, device=feats.device)
        for env_id in subset_envs:
            mask = env_batch == env_id
            if mask.sum() > 1:
                f = feats[mask]
                m = f.mean(0)
                c = (f - m).T @ (f - m) / (len(f) - 1)
                coral += (m - global_mean).pow(2).mean()
                coral += (c - global_cov).pow(2).mean()
        return coral / len(subset_envs), self.coral_lambda


class MMD(_AbstractDomainAlignment):
    """
    MMD regressor using multi-scale Gaussian kernel, sklearn-style fit/predict API.

    Minimizes MSE plus a pairwise MMD penalty between sampled environment pairs,
    encouraging domain-invariant feature representations.

    Adapted from https://github.com/mlfoundations/tableshift
    """

    def __init__(self, hidden_dims=[128, 64], dropout=0.1, lr=1e-3,
                 n_epochs=500, batch_size=1024, early_stopping_rounds=10,
                 mmd_gamma=1.0, num_mmd_pairs=5):
        super().__init__(hidden_dims=hidden_dims, dropout=dropout, lr=lr,
                         n_epochs=n_epochs, batch_size=batch_size,
                         early_stopping_rounds=early_stopping_rounds)
        self.mmd_gamma = mmd_gamma
        self.num_mmd_pairs = num_mmd_pairs
        self._n_pairs = num_mmd_pairs

    def _my_cdist(self, x1, x2):
        # Copied from https://github.com/mlfoundations/tableshift
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.T, x1, x2.T, alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def _gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        # Copied from https://github.com/mlfoundations/tableshift
        D = self._my_cdist(x, y)
        K = torch.zeros_like(D)
        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))
        return K

    def _compute_penalty(self, feats, env_batch, subset_envs):
        env_feats = [
            feats[env_batch == e]
            for e in subset_envs
            if (env_batch == e).sum() > 1
        ]
        penalty = torch.tensor(0.0, device=feats.device)
        n_pairs = 0
        for i in range(len(env_feats)):
            for j in range(i + 1, len(env_feats)):
                Kxx = self._gaussian_kernel(env_feats[i], env_feats[i]).mean()
                Kyy = self._gaussian_kernel(env_feats[j], env_feats[j]).mean()
                Kxy = self._gaussian_kernel(env_feats[i], env_feats[j]).mean()
                penalty += Kxx + Kyy - 2 * Kxy
                n_pairs += 1
        if n_pairs > 0:
            penalty /= n_pairs
        return penalty, self.mmd_gamma
