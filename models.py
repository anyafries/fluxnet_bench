"""
Model definitions for FLUXNET benchmark.

This module contains baseline models for the benchmark.
"""

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from utils.utils import setup_logging

logger = setup_logging(__name__)


def get_model(model_name, params=None):
    """
    Factory function to get a model instance by name.

    Args:
        model_name (str): Name of the model ('xgb', 'lr')
        params (dict): Parameters to initialize the model

    Returns:
        A model instance with fit() and predict() methods

    Raises:
        NotImplementedError: If the model name is not recognized
    """
    if params is None:
        params = {}

    if model_name == 'xgb':
        return XGBRegressor(**params)
    elif model_name == 'lr':
        return LinearRegression()
    else:
        raise NotImplementedError(
            f"Model `{model_name}` not implemented. "
            f"Available models: 'xgb', 'lr'"
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
            'learning_rate': 0.1
        }
    return params
