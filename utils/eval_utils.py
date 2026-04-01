"""
Evaluation metrics for FLUXNET benchmark.

This module contains functions to evaluate model predictions using
various metrics. For temporal aggregation, see aggregation.py.
"""

import json
import numpy as np
import pandas as pd
import os

from utils.aggregation import AGGREGATIONS
from utils.utils import setup_logging, get_metrics_path, get_params_path, save_csv, load_csv

logger = setup_logging(__name__)


# -----------------------------------------------------------------------
# ----------------------- Individual Metric Functions -------------------
# -----------------------------------------------------------------------
# Each function takes ytrue, ypred arrays and returns a scalar metric.
# Convention: ytrue = observations (x in QuickEval), ypred = predictions (y in QuickEval)

def rmse(ytrue, ypred):
    """Root Mean Squared Error."""
    return np.sqrt(np.nanmean((ypred - ytrue) ** 2))


def mse(ytrue, ypred):
    """Mean Squared Error."""
    return np.nanmean((ypred - ytrue) ** 2)


def mae(ytrue, ypred):
    """Mean Absolute Error (MAD in QuickEval)."""
    return np.nanmean(np.abs(ypred - ytrue))


def bias(ytrue, ypred):
    """Mean bias (obs - pred)."""
    return np.nanmean(ytrue - ypred)


def relative_bias(ytrue, ypred):
    """Relative bias: bias / mean(obs)."""
    mask = np.isfinite(ytrue) & np.isfinite(ypred)
    if mask.sum() == 0:
        return np.nan
    return np.nanmean(ytrue[mask] - ypred[mask]) / np.nanmean(ytrue[mask])


def nse(ytrue, ypred):
    """
    Nash-Sutcliffe Efficiency (NSE / MEF in QuickEval).

    NSE = 1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2)

    Returns:
        float: NSE value (ranges from -inf to 1, where 1 is perfect)
    """
    mask = np.isfinite(ytrue) & np.isfinite(ypred)
    ytrue_m = ytrue[mask]
    ypred_m = ypred[mask]

    ss_res = np.sum((ytrue_m - ypred_m) ** 2)
    ss_tot = np.sum((ytrue_m - np.mean(ytrue_m)) ** 2)

    if ss_tot == 0:
        return np.nan
    return 1 - (ss_res / ss_tot)


def r2_score(ytrue, ypred):
    """Coefficient of determination (R²)."""
    mask = np.isfinite(ytrue) & np.isfinite(ypred)
    if mask.sum() < 2:
        return np.nan
    if (np.std(ytrue[mask]) == 0) or (np.std(ypred[mask]) == 0):
        return np.nan
    corr = np.corrcoef(ytrue[mask], ypred[mask])[0, 1]
    return corr ** 2


# Default metrics to compute
DEFAULT_METRICS = {
    'mse': mse,
    'rmse': rmse,
    'mae': mae,
    'nse': nse,
    'r2_score': r2_score,
    'bias': bias,
}

# -----------------------------------------------------------------------
# -------------------------- Metrics I/O --------------------------------
# -----------------------------------------------------------------------

def compute_and_save_metrics(predictions_df, setting, target, model_name):
    """Save metrics DataFrame to CSV."""
    metrics_path = get_metrics_path(setting, target, model_name)
    metrics_df = compute_metrics(predictions_df, model_name, setting, target)
    save_csv(metrics_df, metrics_path)
    return metrics_df


def load_metrics(setting, target, model_name):
    """Load metrics file for a given experiment."""
    metrics_path = get_metrics_path(setting, target, model_name)
    return load_csv(metrics_path)


def save_best_params(best_params, setting, target, model_name):
    """Save the best hyperparameter dictionary to a JSON file."""
    path = get_params_path(setting, target, model_name)
    # Ensure the directory exists (in case it's the first file being saved)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(best_params, f, indent=4)
    logger.info(f"Saved best parameters to {path}")


# -----------------------------------------------------------------------
# ----------------------- Metric Computation ----------------------------
# -----------------------------------------------------------------------

def compute_metrics_for_group(ytrue, ypred, metrics=None):
    """
    Compute all metrics for a single group (e.g., one site).

    Args:
        ytrue: Array of true values
        ypred: Array of predicted values
        metrics: Dict of {name: function} for metrics to compute

    Returns:
        dict: Computed metric values
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    ytrue = np.asarray(ytrue)
    ypred = np.asarray(ypred)

    if all(np.isnan(ytrue)) or all(np.isnan(ypred)):
        return {name: np.nan for name in metrics.keys()}

    results = {}
    for name, func in metrics.items():
        try:
            results[name] = func(ytrue, ypred)
        except Exception:
            results[name] = np.nan

    return results


def compute_diagnostics(df, scale='daily', metrics=None):
    """
    Compute metrics at a given temporal scale.

    This is analogous to QuickEval's compute_diagnostics.
    Data is first aggregated to the specified scale, then metrics
    are computed per environment ('env').

    Args:
        df: DataFrame with y_true, y_pred, env, time columns
        scale: Temporal scale ('daily', 'weekly', 'monthly', 'seasonal', 'anom', 'iav', 'spatial')
        metrics: Dict of {name: function} for metrics to compute

    Returns:
        pd.DataFrame: One row per group with all metric columns
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    # Aggregate data to the specified scale
    if scale in AGGREGATIONS:
        agg_df = AGGREGATIONS[scale](df)
    else:
        raise ValueError(f"Unknown scale: {scale}. Available: {list(AGGREGATIONS.keys())}")

    # For other scales, compute metrics per group
    results = []
    for group_val in agg_df['env'].unique():
        mask = agg_df['env'] == group_val
        group_df = agg_df[mask]

        group_metrics = compute_metrics_for_group(
            group_df['y_true'].values,
            group_df['y_pred'].values,
            metrics
        )
        group_metrics['env'] = group_val
        group_metrics['n_samples'] = len(group_df)
        results.append(group_metrics)

    return pd.DataFrame(results)


def compute_metrics(predictions_df, model_name, setting, target, scales=None):
    """
    Compute metrics at all temporal scales for an experiment.

    Args:
        predictions_df: DataFrame with y_true, y_pred, env, time columns
        model_name: Model name for metadata
        setting: Setting name for metadata
        target: Target variable for metadata
        scales: List of scales to compute (default: all available)

    Returns:
        pd.DataFrame with metrics for all scales combined
    """
    if predictions_df is None:
        logger.warning("No predictions DataFrame provided, cannot compute metrics.")
        return None
    
    if scales is None:
        scales = list(AGGREGATIONS.keys())

    all_results = []
    for scale in scales:
        try:
            results_df = compute_diagnostics(predictions_df, scale=scale)
            results_df['model'] = model_name
            results_df['setting'] = setting
            results_df['target'] = target
            results_df['scale'] = scale
            all_results.append(results_df)
            n_samples = results_df['n_samples']
            min_s, mean_s, max_s = n_samples.min(), n_samples.mean(), n_samples.max()
            summary_log = f"Computed {scale} metrics: {len(results_df)} groups"
            logger.info(summary_log+\
                        f"{"\t" if len(summary_log) < 33 else ""}" +\
                        f"\t(min samples: {min_s}, mean: {mean_s:.1f}, max: {max_s})")
        except Exception as e:
            logger.warning(f"Could not compute {scale} metrics: {e}")

    if not all_results:
        return None

    combined = pd.concat(all_results, ignore_index=True)

    # Reorder columns for readability
    leading_cols = ['target', 'setting', 'model', 'scale', 'env', 'n_samples']
    other_cols = [c for c in combined.columns if c not in leading_cols]
    combined = combined[[c for c in leading_cols if c in combined.columns] + other_cols]

    return combined