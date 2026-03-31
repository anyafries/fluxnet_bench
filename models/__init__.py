"""
Model definitions for FLUXNET benchmark.

This module contains baseline models for the benchmark.
"""

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from .mlp import MLP
from .gdro import GroupDRO
from .coral import CORAL

from utils.utils import setup_logging

logger = setup_logging(__name__)


def get_model(model_name):
    """
    Factory function to get a model instance by name.

    Args:
        model_name (str): Name of the model ('xgb', 'lr', 'mlp', 'gdro')

    Returns:
        A model instance with fit() and predict() methods

    Raises:
        NotImplementedError: If the model name is not recognized
    """
    params = get_default_params(model_name)

    if model_name == 'xgb':
        return XGBRegressor(**params)
    elif model_name == 'lr':
        return LinearRegression()
    elif model_name == 'mlp':
        return MLP(**params)
    elif model_name == 'gdro':
        return GroupDRO(**params)
    elif model_name == 'coral':
        return CORAL(**params)
    else:
        raise NotImplementedError(
            f"Model `{model_name}` not implemented. "
            f"Available models: 'xgb', 'lr', 'mlp', 'gdro', 'coral'"
        )


def get_default_params(model_name):
    """
    Returns default parameters for the specified model.

    Args:
        model_name (str): The name of the model.

    Returns:
        dict: Default parameters for the model
    """
    params = {}
    if model_name == 'xgb':
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'early_stopping_rounds': 10,
        }
    elif model_name == 'mlp':
        params = {
            'hidden_dims': [128, 64],
            'dropout': 0.1,
            'lr': 1e-3,
            'n_epochs': 500,
            'batch_size': 256,
            'early_stopping_rounds': 10,
        }
    elif model_name == 'gdro':
        params = {
            'hidden_dims': [128, 64],
            'dropout': 0.1,
            'lr': 1e-3,
            'n_epochs': 100,
            'batch_size': 256,
            'group_weight_step': 0.01
        }
    elif model_name == 'coral':
        params = {
            'hidden_dims': [128, 64],
            'dropout': 0.1,
            'lr': 1e-3,
            'n_epochs': 500,
            'batch_size': 1024,
            'early_stopping_rounds': 10,
            'coral_lambda': 1.0,
            'num_coral_pairs': 10
        }
    return params
