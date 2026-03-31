"""
Deep CORAL model with sklearn-style fit/predict API.

Implements Deep CORAL (Correlation Alignment) for domain generalization from:
Sun & Saenko, "Deep CORAL: Correlation Alignment for Deep Domain Adaptation" (ECCV 2016)

In the domain generalization setting, the CORAL penalty is applied pairwise across
all training environments (rather than source→target), encouraging the network to
learn domain-invariant representations.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .torch_utils import FluxnetDataset, StratifiedBatchSampler




class CORAL:
    """
    Deep CORAL regressor with sklearn-style fit/predict API.

    Minimizes MSE plus a CORAL penalty that aligns feature distribution
    means and covariances across training environments.
    """

    def __init__(self, hidden_dims=[128, 64], dropout=0.1, lr=1e-3,
                 n_epochs=500, batch_size=256, early_stopping_rounds=10,
                 coral_lambda=1.0, num_coral_pairs=10):
        """
        Args:
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            lr: Learning rate
            n_epochs: Maximum number of training epochs
            batch_size: Batch size for training
            early_stopping_rounds: Stop if val loss doesn't improve for this many epochs
            coral_lambda: Weight on the CORAL alignment penalty
        """
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stopping_rounds = early_stopping_rounds
        self.coral_lambda = coral_lambda
        self.num_coral_pairs = num_coral_pairs
        self.feature_extractor = None
        self.head = None

    def _build_model(self, input_dim):
        """Build feature extractor and regression head."""
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

    def fit(self, X, y, eval_set=None, envs=None):
        """
        Train with Deep CORAL.

        Args:
            X: Features array of shape (n_samples, n_features)
            y: Target array of shape (n_samples,)
            eval_set: Optional list with one tuple [(X_val, y_val)] for early stopping
            envs: Environment labels (required for CORAL penalty)

        Returns:
            self
        """
        if envs is None:
            raise ValueError("CORAL requires environment labels (envs)")

        self._build_model(X.shape[1])

        # Map environment labels to integer indices
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

        pbar = tqdm(range(self.n_epochs), desc="CORAL", unit="epoch")
        for _ in pbar:
            self.feature_extractor.train()
            self.head.train()
            for X_batch, y_batch, env_batch in loader:
                optimizer.zero_grad()
                pred, feats = self._forward(X_batch)
                mse = F.mse_loss(pred, y_batch)

                # Global stats for the whole batch
                global_mean = feats.mean(0)
                cent_batch = feats - global_mean
                global_cov = (cent_batch.T @ cent_batch) / (len(feats) - 1)

                # Align each environment to global stats
                coral = torch.tensor(0.0, device=feats.device)
                unique_batch_envs = torch.unique(env_batch)
                unique_batch_envs = unique_batch_envs[torch.randperm(len(unique_batch_envs))]
                subset_envs = unique_batch_envs[:self.num_coral_pairs]  
                for env_id in subset_envs:
                    mask = env_batch == env_id
                    if mask.sum() > 1:
                        feats_env = feats[mask]
                        m_env = feats_env.mean(0)
                        c_env = (feats_env - m_env).T @ (feats_env - m_env) / (len(feats_env) - 1)
                        coral += (m_env - global_mean).pow(2).mean()
                        coral += (c_env - global_cov).pow(2).mean()

                loss = mse + self.coral_lambda * (coral / len(unique_batch_envs))
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
        """
        Make predictions.

        Args:
            X: Features array of shape (n_samples, n_features)

        Returns:
            Predictions array of shape (n_samples,)
        """
        self.feature_extractor.eval()
        self.head.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            return self.head(self.feature_extractor(X_t)).numpy().ravel()
