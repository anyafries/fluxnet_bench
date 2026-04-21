"""
Model definitions for FLUXNET benchmark.

Available models:
  - 'lr'            : Linear Regression (sklearn)
  - 'ridge'         : Ridge Regression (sklearn)
  - 'xgb'           : XGBoost Regressor
  - 'mlp'           : Multi-layer Perceptron
  - 'gdro'          : Group DRO (worst-group loss minimization)
  - 'coral'         : CORAL domain adaptation
  - 'mmd'           : MMD domain adaptation
  - 'maxrm_mse'     : MaxRM Random Forest with MSE risk
  - 'maxrm_regret'  : MaxRM Random Forest with regret risk
"""

import itertools

from utils.utils import setup_logging

logger = setup_logging(__name__)


def get_model(model_name, params=None):
    """
    Factory function to get a model instance by name.

    Args:
        model_name (str): Name of the model. See module docstring for options.
        params (dict, optional): Parameters to initialize the model.
            If None, defaults will be used.

    Returns:
        A model instance with fit() and predict() methods

    Raises:
        NotImplementedError: If the model name is not recognized
    """
    # params = get_default_params(model_name)

    if model_name == 'xgb':
        from xgboost import XGBRegressor
        return XGBRegressor(**params)
    
    elif model_name == 'lr':
        from sklearn.linear_model import LinearRegression
        return LinearRegression()
    
    elif model_name == 'robust-lr':
        from sklearn.linear_model import HuberRegressor
        return HuberRegressor(**params)
    
    elif model_name == 'ridge':
        from sklearn.linear_model import Ridge
        return Ridge(**params)
    
    elif model_name == 'mlp':
        from .mlp import MLP
        return MLP(**params)
    
    elif model_name == 'gdro':
        from .gdro import GroupDRO
        return GroupDRO(**params)
    
    elif model_name == 'coral':
        from .coral import CORAL
        return CORAL(**params)
    
    elif model_name == 'mmd':
        from .coral import MMD
        return MMD(**params)
    
    elif model_name == 'maxrm-mse':
        from .maxrm_rf import MaxRM_RF
        return MaxRM_RF(risk='mse', **params)
    
    elif model_name == 'maxrm-regret':
        from .maxrm_rf import MaxRM_RF
        return MaxRM_RF(risk='regret', **params)
    
    else:
        raise NotImplementedError(
            f"Model `{model_name}` not implemented. "
            f"Available models: 'xgb', 'lr', 'ridge', 'mlp', 'gdro', 'coral', 'mmd', 'maxrm_mse', 'maxrm_regret'"
        )


# def get_default_params(model_name):
#     """
#     Returns default parameters for the specified model.

#     Args:
#         model_name (str): The name of the model.

#     Returns:
#         dict: Default parameters for the model
#     """
#     params = {}
#     if model_name == 'xgb':
#         params = {
#             'objective': 'reg:squarederror',
#             'n_estimators': 100,
#             'max_depth': 5,
#             'learning_rate': 0.1,
#             'early_stopping_rounds': 10,
#         }
#     elif model_name in ['maxrm_mse', 'maxrm_regret']:
#         params = {
#             'n_estimators': 100,
#             'min_samples_leaf': 30,
#         }
#     elif model_name == 'mlp':
#         params = {
#             'hidden_dims': [128, 64],
#             'dropout': 0.1,
#             'lr': 1e-3,
#             'n_epochs': 500,
#             'batch_size': 256,
#             'early_stopping_rounds': 10,
#         }
#     elif model_name == 'gdro':
#         params = {
#             'hidden_dims': [128, 64],
#             'dropout': 0.1,
#             'lr': 1e-3,
#             'n_epochs': 100,
#             'batch_size': 256,
#             'group_weight_step': 0.01,
#             'early_stopping_rounds': 10,
#         }
#     elif model_name == 'coral':
#         params = {
#             'hidden_dims': [128, 64],
#             'dropout': 0.1,
#             'lr': 1e-3,
#             'n_epochs': 500,
#             'batch_size': 1024,
#             'early_stopping_rounds': 10,
#             'coral_lambda': 1.0,
#             'num_coral_pairs': 10
#         }
#     elif model_name == 'mmd':
#         params = {
#             'hidden_dims': [128, 64],
#             'dropout': 0.1,
#             'lr': 1e-3,
#             'n_epochs': 500,
#             'batch_size': 1024,
#             'early_stopping_rounds': 10,
#             'mmd_gamma': 1.0,
#             'num_mmd_pairs': 5,
#         }
#     return params


def get_param_grid(model_name):
    """
    Returns a list of parameter dictionaries to test for a model.
    Aims for 2-4 variations per model to keep benchmarks manageable.
    """
    grid = []
    
    if model_name == 'lr':
        grid = [{}]

    elif model_name == 'ridge':
        options = {
            'alpha': [0.1, 1.0, 10.0]
        }
        keys, values = zip(*options.items())
        grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

    elif model_name == 'robust-lr':
        options = {
            'alpha': [0.1, 1.0, 10.0],
            'epsilon': [1.35, 1.5]
        }
        keys, values = zip(*options.items())
        grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

    elif model_name in ['maxrm_mse', 'maxrm_regret']:
        options = {
            'n_estimators': [100, 500],
            'min_samples_leaf': [10, 30],
        }
        keys, values = zip(*options.items())
        grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

    elif model_name == 'xgb':
        # Focus on depth and shrinkage
        options = {
            'n_estimators': [100, 500],
            'max_depth': [3, 6],
            'learning_rate': [0.05, 0.1],
            'objective': ['reg:squarederror'],
            'early_stopping_rounds': [10]
        }
        keys, values = zip(*options.items())
        grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

    elif model_name in ['mlp', 'gdro', 'coral', 'mmd']:
        # Base deep learning params
        base_options = {
            'hidden_dims': [[128, 64], [256, 128]],
            'lr': [1e-3, 1e-4],
            'n_epochs': [100],
            'batch_size': [1024, 2048]
        }
        
        # Add model-specific hyperparams
        if model_name == 'gdro':
            base_options['group_weight_step'] = [0.01, 0.1]
        elif model_name == 'coral':
            base_options['coral_lambda'] = [0.1, 1.0]
        elif model_name == 'mmd':
            base_options['mmd_gamma'] = [0.1, 1.0]
            
        keys, values = zip(*options.items() if 'options' in locals() else base_options.items())
        grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

    return grid