"""Temporal aggregation functions for FLUXNET benchmark.

Pandas-based reimplementation of QuickEval's aggregation_util.py.
Input DataFrames must have columns: y_true, y_pred, env, time (datetime).

Supports:
- Temporal resampling (daily → weekly/monthly/yearly)
- Mean Seasonal Cycle (MSC) with leap year handling
- Anomalies from MSC
- Inter-annual variability (IAV)
- Spatial (site) means
- Outlier detection (IQR or z-score based)
- Moving window MSC computation
"""

import numpy as np
import pandas as pd

from utils.utils import setup_logging

logger = setup_logging(__name__)


# =============================================================================
# Helper Functions
# =============================================================================

def _apply_mask(df, mask):
    """Attach boolean mask without removing the raw values."""
    df = df.copy()
    if mask is None:
        df['valid'] = True
    else:
        df['valid'] = mask
    return df


def _ensure_daily(df, mask, min_contribution_hour_to_day=12):
    """Helper to convert hourly data to daily before coarser aggregations."""
    df_daily = aggregate_daily(df, mask=mask, min_contribution=min_contribution_hour_to_day)
    # Create a new mask based on valid aggregated daily values
    assert sum(df_daily['y_pred'].isna()) == 0, \
        "Unexpected NaNs in y_pred after daily aggregation"
    new_mask = df_daily['y_true'].notna() & df_daily['y_pred'].notna()
    return df_daily, new_mask


def _agg_with_threshold(df, groupby_cols, min_contribution, method='mean',
                        early_masking=False):
    """
    Aggregate y_true/y_pred with validity threshold.
    """
    agg_func = method if method in ['mean', 'median'] else 'mean'
    
    if min_contribution < 1:
        if early_masking:
            raise NotImplementedError("Early masking with weighted threshold is not implemented yet.")  
        
        # Prepare columns to compute the weighted threshold: frac = sum(|data| * valid) / sum(|data|)
        df['_abs_true'] = np.abs(df['y_true'])
        df['_abs_pred'] = np.abs(df['y_pred'])
        df['_valid_abs_true'] = df['_abs_true'].where(df['valid'], 0)
        df['_valid_abs_pred'] = df['_abs_pred'].where(df['valid'], 0)
        
        result = df.groupby(groupby_cols).agg(
            y_true=('y_true', agg_func),
            y_pred=('y_pred', agg_func),
            _abs_sum_true=('_abs_true', 'sum'),
            _abs_sum_pred=('_abs_pred', 'sum'),
            _valid_abs_sum_true=('_valid_abs_true', 'sum'),
            _valid_abs_sum_pred=('_valid_abs_pred', 'sum'),
        )
        
        # Compute fraction of valid data (weighted by abs value)
        frac_true = result['_valid_abs_sum_true'] / result['_abs_sum_true'].replace(0, np.nan)
        frac_pred = result['_valid_abs_sum_pred'] / result['_abs_sum_pred'].replace(0, np.nan)

        # Apply threshold filtering
        result.loc[frac_true < min_contribution, 'y_true'] = np.nan
        
    else:
        if early_masking:
            # Count-based threshold
            df['_y_true_masked'] = df['y_true'].where(df['valid'])
            df['_y_pred_masked'] = df['y_pred'].where(df['valid'])
            true_col, pred_col = '_y_true_masked', '_y_pred_masked'
        else:
            true_col, pred_col = 'y_true', 'y_pred'


        result = df.groupby(groupby_cols).agg(
            y_true=(true_col, agg_func),
            y_pred=(pred_col, agg_func),
            _n_valid=('valid', 'sum')
        )
        
        # Apply threshold filtering
        result.loc[result['_n_valid'] < min_contribution, 'y_true'] = np.nan
        
    # Return just the target columns cleanly
    return result[['y_true', 'y_pred']].reset_index()


# =============================================================================
# Temporal Resampling: Daily → Coarser
# =============================================================================

def aggregate_hourly(df, mask=None):
    """Return hourly data as-is, or apply mask if provided."""
    df = _apply_mask(df, mask)
    return df[df['valid']].drop(columns=['valid']).copy()


def aggregate_daily(df, mask=None, min_contribution=12):
    """Aggregate hourly data to daily means, or return as-is if already daily."""
    df = _apply_mask(df, mask)
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    
    # Check if already daily (max 1 entry per env per date)
    if df.empty or df.groupby(['env', df['time'].dt.floor('D')]).size().max() <= 1:
        return df[df['valid']].drop(columns=['valid']).copy()
        
    df['_date'] = df['time'].dt.floor('D')
    result = _agg_with_threshold(df, ['env', '_date'], min_contribution)
    return result.rename(columns={'_date': 'time'})


def aggregate_weekly(df, mask=None, min_contribution=4, 
                     min_contribution_hour_to_day=12):
    """
    Aggregate daily data to weekly means.

    Args:
        df: DataFrame with y_true, y_pred, env, time columns
        mask: Optional boolean mask for valid data
        min_contribution: Minimum fraction or count of valid days
        min_contribution_hour_to_day: Minimum fraction or count of valid hours

    Returns:
        DataFrame with weekly aggregated values
    """
    df, mask = _ensure_daily(df, mask, min_contribution_hour_to_day)
    df = _apply_mask(df, mask)
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df['_week'] = df['time'].dt.to_period('W')

    result = _agg_with_threshold(df, ['env', '_week'], min_contribution)
    result['time'] = result['_week'].dt.start_time + pd.Timedelta(days=3)  # Mid-week
    return result.drop(columns=['_week'])


def aggregate_monthly(df, mask=None, min_contribution=15, 
                      min_contribution_hour_to_day=12):
    """
    Aggregate daily data to monthly means.

    Args:
        df: DataFrame with y_true, y_pred, env, time columns
        mask: Optional boolean mask for valid data
        min_contribution: Minimum fraction or count of valid days
        min_contribution_hour_to_day: Minimum fraction or count of valid hours

    Returns:
        DataFrame with monthly aggregated values
    """
    df, mask = _ensure_daily(df, mask, min_contribution_hour_to_day)
    df = _apply_mask(df, mask)
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df['_month'] = df['time'].dt.to_period('M')

    result = _agg_with_threshold(df, ['env', '_month'], min_contribution)
    result['time'] = result['_month'].dt.start_time + pd.Timedelta(days=14)  # Mid-month
    return result.drop(columns=['_month'])


def aggregate_yearly(df, mask=None, min_contribution=183, 
                      min_contribution_hour_to_day=12):
    """
    Aggregate daily data to yearly means.

    Args:
        df: DataFrame with y_true, y_pred, env, time columns
        mask: Optional boolean mask for valid data
        min_contribution: Minimum fraction or count of valid days
        min_contribution_hour_to_day: Minimum fraction or count of valid hours

    Returns:
        DataFrame with yearly aggregated values
    """
    df, mask = _ensure_daily(df, mask, min_contribution_hour_to_day)
    df = _apply_mask(df, mask)
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df['_year'] = df['time'].dt.year

    result = _agg_with_threshold(df, ['env', '_year'], min_contribution)
    # Set time to mid-year
    result['time'] = pd.to_datetime(result['_year'].astype(str) + '-07-01')
    return result.drop(columns=['_year'])


# =============================================================================
# Mean Seasonal Cycle (MSC)
# =============================================================================

def compute_msc(df, mask=None, min_contribution=2, 
                min_contribution_hour_to_day=12, method='mean',
                return_outlier_mask=False):
    """
    Compute mean seasonal cycle (MSC) per site.

    Handles leap years by treating DOY 366 separately from DOY 1-365.

    Args:
        df: DataFrame with y_true, y_pred, env, time columns
        mask: Optional boolean Series for valid data
        min_contribution: Min years per DOY or fraction
        min_contribution_hour_to_day: Minimum fraction or count of valid hours
        method: 'mean' or 'median'
        return_long: If True, expand MSC to original time series length
        return_outlier_mask: If True, also return outlier detection results
        z_outlier: Threshold for outliers (default: 3 for mean, 1.5 for median)
        test_direction: -1 (low only), 0 (both), 1 (high only)

    Returns:
        If return_long=False: DataFrame with one row per (env, doy)
        If return_long=True: DataFrame with MSC values at original timestamps
        If return_outlier_mask=True: tuple of (msc, outlier_mask, lower_thresh, upper_thresh)
    """
    df, mask = _ensure_daily(df, mask, min_contribution_hour_to_day)
    df = _apply_mask(df, mask)
    df = df.copy()
    
    df['time'] = pd.to_datetime(df['time'])
    df['doy'] = df['time'].dt.dayofyear
    df['year'] = df['time'].dt.year

    if return_outlier_mask:
        raise DeprecationWarning("Outlier detection is not yet implemented in this function.")

    agg_func = method if method in ['mean', 'median'] else 'mean'

    # Compute MSC per (env, doy)
    msc = _agg_with_threshold(df, ['env', 'doy'], min_contribution, 
                              method=agg_func, early_masking=True)

    # if return_long:
    #     # Expand MSC to original time series length
    #     msc_long = df[['env', 'time', 'doy']].merge(
    #         msc[['env', 'doy', 'y_true', 'y_pred']],
    #         on=['env', 'doy'],
    #         how='left'
    #     )
    #     msc_long = msc_long[['env', 'time', 'y_true', 'y_pred']]

    #     return msc_long

    return msc


# =============================================================================
# Derived Aggregations
# =============================================================================

def aggregate_seasonal(df, mask=None, min_contribution=2, 
                       min_contribution_hour_to_day=12,
                       method='mean'):
    """
    Compute mean seasonal cycle (short form).

    Returns one value per (env, doy) - the multi-year average for each day-of-year.

    Args:
        df: DataFrame with y_true, y_pred, env, time columns
        mask: Optional boolean mask for valid data
        min_contribution: Minimum years with valid data per DOY
        min_contribution_hour_to_day: Minimum fraction or count of valid hours
        method: 'mean' or 'median'

    Returns:
        DataFrame with one row per (env, doy)
    """
    df, mask = _ensure_daily(df, mask, min_contribution_hour_to_day)
    return compute_msc(df, mask=mask, min_contribution=min_contribution,
                       method=method)


def aggregate_anomaly(df, mask=None, min_contribution=2, 
                      min_contribution_hour_to_day=12,
                      method='mean'):
    """
    Compute anomalies from mean seasonal cycle.

    For each sample, subtracts the site's MSC value for that day-of-year.

    Args:
        df: DataFrame with y_true, y_pred, env, time columns
        mask: Optional boolean mask for valid data
        min_contribution: Minimum years for MSC computation
        min_contribution_hour_to_day: Minimum fraction or count of valid hours
        method: 'mean' or 'median' for MSC

    Returns:
        DataFrame with anomaly values (original - MSC)
    """
    df, mask = _ensure_daily(df, mask, min_contribution_hour_to_day)
    df = _apply_mask(df, mask)
    df = df.copy()

    # if there are not unique (doy, year) pairs, we need to aggregate to daily first
    df['time'] = pd.to_datetime(df['time'])
    df['doy'] = df['time'].dt.dayofyear
    df['year'] = df['time'].dt.year

    df['time'] = pd.to_datetime(df['time'])
    df['doy'] = df['time'].dt.dayofyear

    # Compute MSC
    msc = compute_msc(df, mask=None, min_contribution=min_contribution,
                      method=method)
    msc = msc.rename(columns={'y_true': 'msc_true', 'y_pred': 'msc_pred'})

    # Merge and compute anomalies
    result = df.merge(msc[['env', 'doy', 'msc_true', 'msc_pred']], 
                      on=['env', 'doy'], how='left')
    result['y_true'] = result['y_true'] - result['msc_true']
    result['y_pred'] = result['y_pred'] - result['msc_pred']

    return result[['env', 'time', 'y_true', 'y_pred']]


def aggregate_iav(df, mask=None, min_contribution=183,
                  min_contribution_hour_to_day=12):
    """
    Compute inter-annual variability (IAV).

    Computes yearly means per site, then subtracts the site's multi-year mean.

    Args:
        df: DataFrame with y_true, y_pred, env, time columns
        mask: Optional boolean mask for valid data
        min_contribution: Minimum fraction/count of valid days per year
        min_contribution_hour_to_day: Minimum fraction or count of valid hours

    Returns:
        DataFrame with IAV values (yearly mean - site mean)
    """
    df, mask = _ensure_daily(df, mask, min_contribution_hour_to_day)

    # First aggregate to yearly
    yearly = aggregate_yearly(df, mask=mask, min_contribution=min_contribution)

    # Compute site means
    site_means = yearly.groupby('env')[['y_true', 'y_pred']].transform('mean')

    # Compute IAV as deviation from site mean
    yearly['y_true'] = yearly['y_true'] - site_means['y_true']
    yearly['y_pred'] = yearly['y_pred'] - site_means['y_pred']

    return yearly


# =============================================================================
# Registry
# =============================================================================

AGGREGATIONS = {
    'hourly': aggregate_hourly,
    'daily': aggregate_daily,
    'weekly': aggregate_weekly,
    'monthly': aggregate_monthly,
    'seasonal': aggregate_seasonal,
    'anom': aggregate_anomaly,
    'iav': aggregate_iav
}
