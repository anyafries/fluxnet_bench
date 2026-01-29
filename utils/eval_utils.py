"""
Evaluation metrics for FLUXNET benchmark.

This module contains functions to evaluate model predictions using
various metrics. For temporal aggregation, see aggregation.py.
"""

import numpy as np
import pandas as pd

from utils.aggregation import AGGREGATIONS
from utils.utils import setup_logging, get_metrics_path, save_csv, load_csv

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


def relative_error(ytrue, ypred):
    """Mean Relative Absolute Deviation (MrAD in QuickEval)."""
    return np.nanmean(np.abs((ypred - ytrue) / ytrue))


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
    corr = np.corrcoef(ytrue[mask], ypred[mask])[0, 1]
    return corr ** 2


def pearson_corr(ytrue, ypred):
    """Pearson correlation coefficient."""
    mask = np.isfinite(ytrue) & np.isfinite(ypred)
    if mask.sum() < 2:
        return np.nan
    return np.corrcoef(ytrue[mask], ypred[mask])[0, 1]


# Default metrics to compute (maps name -> function)
DEFAULT_METRICS = {
    'mse': mse,
    'rmse': rmse,
    'mae': mae,
    'nse': nse,
    'r2_score': r2_score,
    'bias': bias,
    'relative_error': relative_error,
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

    # For spatial scale, metrics are computed across sites (one value total)
    # TODO: fix, this is just wrong??
    #       * n_samples is 2 instead of 25 sites, why?
    #       * should we compute metrics across all sites, or per site?
    #       * env is NaN for spatial scale, why? -> what should it be?
    if scale == 'spatial':
        result = compute_metrics_for_group(
            agg_df['y_true'].values,
            agg_df['y_pred'].values,
            metrics
        )
        result['n_samples'] = len(agg_df)
        return pd.DataFrame([result])

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


def compute_all_diagnostics(df, scales=None, metrics=None):
    """
    Compute metrics at multiple temporal scales.

    Args:
        df: DataFrame with y_true, y_pred, env, time columns
        scales: List of scales to compute (default: all available)
        metrics: Dict of {name: function} for metrics

    Returns:
        dict: {scale: DataFrame of metrics}
    """
    if scales is None:
        scales = list(AGGREGATIONS.keys())

    results = {}
    for scale in scales:
        try:
            results[scale] = compute_diagnostics(df, scale, metrics)
        except Exception as e:
            logger.warning(f"Could not compute {scale} diagnostics: {e}")

    return results


def compute_metrics(predictions_df, model_name, setting, target, scales=None):
    """
    Compute metrics at all temporal scales for an experiment.

    This combines compute_diagnostics across all scales and adds metadata.
    Used by train_model.py to compute metrics after training, and results
    are saved to results/metrics/{exp_name}.csv.

    Args:
        predictions_df: DataFrame with y_true, y_pred, env, time columns
        model_name: Model name for metadata
        setting: Setting name for metadata
        target: Target variable for metadata
        scales: List of scales to compute (default: all available)

    Returns:
        pd.DataFrame with metrics for all scales combined
    """
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
            logger.info(f"Computed {scale} metrics: {len(results_df)} groups")
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


# -----------------------------------------------------------------------
# ----------------------- Simple Evaluation Function --------------------
# -----------------------------------------------------------------------

def evaluate_fold(ytrue, ypred, envs, verbose=True, digits=3):
    """
    Evaluate predictions for multiple environments.

    This is a simplified interface that computes daily metrics per environment.
    For full multi-scale evaluation, use compute_all_diagnostics.

    Args:
        ytrue (array-like): True values
        ypred (array-like): Predicted values
        envs (array-like): Environment labels
        verbose (bool): If True, log the results
        digits (int): Number of decimal places for logging

    Returns:
        pd.DataFrame: DataFrame with one row per env and metric columns
    """
    ytrue = np.asarray(ytrue)
    ypred = np.asarray(ypred)
    envs = np.asarray(envs)

    results = []
    for env in np.unique(envs):
        mask = envs == env
        env_metrics = compute_metrics_for_group(ytrue[mask], ypred[mask])
        env_metrics['env'] = env
        env_metrics['n_samples'] = mask.sum()
        results.append(env_metrics)

    results_df = pd.DataFrame(results)

    if verbose:
        # Log summary statistics across all environments
        for metric in DEFAULT_METRICS.keys():
            values = results_df[metric].dropna()
            if len(values) > 0:
                logger.info(
                    f"  {metric.upper()}: "
                    f"median={values.median():.{digits}f}, "
                    f"mean={values.mean():.{digits}f}, "
                    f"std={values.std():.{digits}f}"
                )

    return results_df
