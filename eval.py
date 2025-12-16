"""
Evaluation metrics for FLUXNET benchmark.

This module contains functions to evaluate model predictions using
various metrics including MSE, RMSE, R², MAE, and NSE (Nash-Sutcliffe Efficiency).
"""

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def nse(ytrue, ypred):
    """
    Calculate Nash-Sutcliffe Efficiency (NSE).

    NSE is commonly used in hydrology and is equivalent to R² but
    provides a different interpretation for model performance.

    Args:
        ytrue (array-like): True values
        ypred (array-like): Predicted values

    Returns:
        float: NSE value (ranges from -inf to 1, where 1 is perfect)
    """
    numerator = np.sum((ytrue - ypred) ** 2)
    denominator = np.sum((ytrue - np.mean(ytrue)) ** 2)
    return 1 - (numerator / denominator)


def evaluate_fold(ytrue, ypred, env, verbose=True, digits=3):
    """
    Evaluate predictions for multiple sites.

    Computes multiple metrics including MSE, RMSE, R², MAE, relative error, and NSE.

    Args:
        ytrue (array-like): True values
        ypred (array-like): Predicted values
        verbose (bool): If True, log the results
        digits (int): Number of decimal places for logging

    Returns:
        dict: Dictionary containing all computed metrics
    """
    out = []
    for e in np.unique(env):
        mask = env == e
        ytrue_e = ytrue[mask]
        ypred_e = ypred[mask]
        
        mse = mean_squared_error(ytrue_e, ypred_e)
        results = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2_score': r2_score(ytrue_e, ypred_e),
            'relative_error': np.mean(np.abs(ytrue_e - ypred_e) / np.abs(ytrue_e)),
            'mae': np.mean(np.abs(ytrue_e - ypred_e)),
            'nse': nse(ytrue_e, ypred_e)
        }
        if verbose:
            logger.info(f"  Env: {e} over {len(ytrue_e)} predictions:")
            logger.info(f"* RESULTS over {len(ytrue)} predictions:")
            logger.info("\t " + ", ".join(
                f"{metric.upper()}={value:.{digits}f}"
                for metric, value in results.items()
            ))
        results['env'] = e
        out.append(results)
    return pd.DataFrame(out)