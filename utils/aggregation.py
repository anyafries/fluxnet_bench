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
    """Apply boolean mask to y_true and y_pred columns."""
    if mask is None:
        return df
    df = df.copy()
    df.loc[~mask, ['y_true', 'y_pred']] = np.nan
    return df


def _agg_with_threshold(df, groupby_cols, min_contribution, method='mean'):
    """
    Aggregate y_true/y_pred with validity threshold.

    Args:
        df: DataFrame with y_true, y_pred columns
        groupby_cols: Columns to group by
        min_contribution: If >= 1, minimum count of valid samples.
                         If < 1, minimum fraction of valid samples.
        method: 'mean' or 'median'

    Returns:
        Aggregated DataFrame with y_true, y_pred columns
    """
    grouped = df.groupby(groupby_cols)
    agg_func = method if method in ['mean', 'median'] else 'mean'

    if min_contribution < 1:
        # Weighted threshold: fraction based on abs values
        # frac = sum(|data| * valid) / sum(|data|)
        result = grouped.agg(
            y_true=(('y_true', agg_func)),
            y_pred=(('y_pred', agg_func)),
            _abs_sum_true=('y_true', lambda x: np.abs(x).sum()),
            _abs_sum_pred=('y_pred', lambda x: np.abs(x).sum()),
            _valid_abs_sum_true=('y_true', lambda x: np.abs(x.dropna()).sum()),
            _valid_abs_sum_pred=('y_pred', lambda x: np.abs(x.dropna()).sum()),
        )
        # Compute fraction of valid data (weighted by abs value)
        frac_true = result['_valid_abs_sum_true'] / result['_abs_sum_true'].replace(0, np.nan)
        frac_pred = result['_valid_abs_sum_pred'] / result['_abs_sum_pred'].replace(0, np.nan)

        # Apply threshold
        result.loc[frac_true < min_contribution, 'y_true'] = np.nan
        result.loc[frac_pred < min_contribution, 'y_pred'] = np.nan

        result = result[['y_true', 'y_pred']]
    else:
        # Count-based threshold
        result = grouped.agg(
            y_true=('y_true', agg_func),
            y_pred=('y_pred', agg_func),
            _n_valid_true=('y_true', 'count'),
            _n_valid_pred=('y_pred', 'count'),
        )
        result.loc[result['_n_valid_true'] < min_contribution, 'y_true'] = np.nan
        result.loc[result['_n_valid_pred'] < min_contribution, 'y_pred'] = np.nan
        result = result[['y_true', 'y_pred']]

    return result.reset_index()


# =============================================================================
# Temporal Resampling: Daily → Coarser
# =============================================================================

def aggregate_daily(df, mask=None):
    """No aggregation - return as-is (already daily)."""
    df = _apply_mask(df, mask)
    return df.copy()


def aggregate_weekly(df, mask=None, min_contribution=0.5):
    """
    Aggregate daily data to weekly means.

    Args:
        df: DataFrame with y_true, y_pred, env, time columns
        mask: Optional boolean mask for valid data
        min_contribution: Minimum fraction (< 1) or count (>= 1) of valid days

    Returns:
        DataFrame with weekly aggregated values
    """
    df = _apply_mask(df, mask)
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df['_week'] = df['time'].dt.to_period('W')

    result = _agg_with_threshold(df, ['env', '_week'], min_contribution)
    result['time'] = result['_week'].dt.start_time + pd.Timedelta(days=3)  # Mid-week
    return result.drop(columns=['_week'])


def aggregate_monthly(df, mask=None, min_contribution=0.5):
    """
    Aggregate daily data to monthly means.

    Args:
        df: DataFrame with y_true, y_pred, env, time columns
        mask: Optional boolean mask for valid data
        min_contribution: Minimum fraction (< 1) or count (>= 1) of valid days

    Returns:
        DataFrame with monthly aggregated values
    """
    df = _apply_mask(df, mask)
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df['_month'] = df['time'].dt.to_period('M')

    result = _agg_with_threshold(df, ['env', '_month'], min_contribution)
    result['time'] = result['_month'].dt.start_time + pd.Timedelta(days=14)  # Mid-month
    return result.drop(columns=['_month'])


def aggregate_yearly(df, mask=None, min_contribution=0.5):
    """
    Aggregate daily data to yearly means.

    Args:
        df: DataFrame with y_true, y_pred, env, time columns
        mask: Optional boolean mask for valid data
        min_contribution: Minimum fraction (< 1) or count (>= 1) of valid days

    Returns:
        DataFrame with yearly aggregated values
    """
    df = _apply_mask(df, mask)
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df['_year'] = df['time'].dt.year

    result = _agg_with_threshold(df, ['env', '_year'], min_contribution)
    # Set time to mid-year
    result['time'] = pd.to_datetime(result['_year'].astype(str) + '-07-01')
    return result.drop(columns=['_year']).rename(columns={'_year': 'year'})


# =============================================================================
# Mean Seasonal Cycle (MSC)
# =============================================================================

def compute_msc(df, mask=None, min_contribution=2, method='mean',
                return_long=False, return_outlier_mask=False,
                z_outlier=None, test_direction=0):
    """
    Compute mean seasonal cycle (MSC) per site.

    Handles leap years by treating DOY 366 separately from DOY 1-365.

    Args:
        df: DataFrame with y_true, y_pred, env, time columns
        mask: Optional boolean Series for valid data
        min_contribution: Min years per DOY (>= 1) or fraction (< 1)
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
    df = _apply_mask(df, mask)
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df['doy'] = df['time'].dt.dayofyear
    df['year'] = df['time'].dt.year

    agg_func = method if method in ['mean', 'median'] else 'mean'

    # Compute MSC per (env, doy)
    msc = _agg_with_threshold(df, ['env', 'doy'], min_contribution, method=agg_func)

    # Handle outlier detection if requested
    outlier_result = None
    if return_outlier_mask:
        if z_outlier is None:
            z_outlier = 1.5 if method == 'median' else 3.0

        # Compute spread per (env, doy)
        grouped = df.groupby(['env', 'doy'])

        if method == 'median':
            # IQR-based: outliers are outside [Q1 - z*IQR, Q3 + z*IQR]
            q25 = grouped[['y_true', 'y_pred']].quantile(0.25).reset_index()
            q75 = grouped[['y_true', 'y_pred']].quantile(0.75).reset_index()
            q25 = q25.rename(columns={'y_true': 'q25_true', 'y_pred': 'q25_pred'})
            q75 = q75.rename(columns={'y_true': 'q75_true', 'y_pred': 'q75_pred'})

            spread_df = q25.merge(q75, on=['env', 'doy'])
            spread_df['iqr_true'] = spread_df['q75_true'] - spread_df['q25_true']
            spread_df['iqr_pred'] = spread_df['q75_pred'] - spread_df['q25_pred']
            spread_df['lower_true'] = spread_df['q25_true'] - z_outlier * spread_df['iqr_true']
            spread_df['upper_true'] = spread_df['q75_true'] + z_outlier * spread_df['iqr_true']
            spread_df['lower_pred'] = spread_df['q25_pred'] - z_outlier * spread_df['iqr_pred']
            spread_df['upper_pred'] = spread_df['q75_pred'] + z_outlier * spread_df['iqr_pred']
        else:
            # Z-score based: outliers are outside [mean - z*std, mean + z*std]
            std_df = grouped[['y_true', 'y_pred']].std().reset_index()
            std_df = std_df.rename(columns={'y_true': 'std_true', 'y_pred': 'std_pred'})
            spread_df = msc.merge(std_df, on=['env', 'doy'])
            spread_df['lower_true'] = spread_df['y_true'] - z_outlier * spread_df['std_true']
            spread_df['upper_true'] = spread_df['y_true'] + z_outlier * spread_df['std_true']
            spread_df['lower_pred'] = spread_df['y_pred'] - z_outlier * spread_df['std_pred']
            spread_df['upper_pred'] = spread_df['y_pred'] + z_outlier * spread_df['std_pred']

        # Create expanded thresholds for original data
        df_with_thresh = df.merge(
            spread_df[['env', 'doy', 'lower_true', 'upper_true', 'lower_pred', 'upper_pred']],
            on=['env', 'doy'],
            how='left'
        )

        # Create outlier mask
        if test_direction == 0:
            valid_true = (df_with_thresh['y_true'] >= df_with_thresh['lower_true']) & \
                        (df_with_thresh['y_true'] <= df_with_thresh['upper_true'])
            valid_pred = (df_with_thresh['y_pred'] >= df_with_thresh['lower_pred']) & \
                        (df_with_thresh['y_pred'] <= df_with_thresh['upper_pred'])
        elif test_direction == -1:
            valid_true = df_with_thresh['y_true'] >= df_with_thresh['lower_true']
            valid_pred = df_with_thresh['y_pred'] >= df_with_thresh['lower_pred']
        else:  # test_direction == 1
            valid_true = df_with_thresh['y_true'] <= df_with_thresh['upper_true']
            valid_pred = df_with_thresh['y_pred'] <= df_with_thresh['upper_pred']

        outlier_mask = valid_true & valid_pred
        outlier_result = (
            outlier_mask,
            df_with_thresh[['lower_true', 'lower_pred']],
            df_with_thresh[['upper_true', 'upper_pred']]
        )

    if return_long:
        # Expand MSC to original time series length
        msc_long = df[['env', 'time', 'doy']].merge(
            msc[['env', 'doy', 'y_true', 'y_pred']],
            on=['env', 'doy'],
            how='left'
        )
        msc_long = msc_long[['env', 'time', 'y_true', 'y_pred']]

        if return_outlier_mask:
            return msc_long, outlier_result[0], outlier_result[1], outlier_result[2]
        return msc_long

    if return_outlier_mask:
        return msc, outlier_result[0], outlier_result[1], outlier_result[2]
    return msc


def compute_msc_moving_window(df, mask=None, nyears_window=5, min_contribution=2,
                               method='mean', return_outlier_mask=False,
                               z_outlier=None, test_direction=0):
    """
    Compute MSC with a moving window of years.

    For each year, computes MSC using data from surrounding years within the window.

    Args:
        df: DataFrame with y_true, y_pred, env, time columns
        mask: Optional boolean mask for valid data
        nyears_window: Number of years in the window
        min_contribution: Min years per DOY for MSC
        method: 'mean' or 'median'
        return_outlier_mask: If True, also return outlier mask
        z_outlier: Threshold for outlier detection
        test_direction: Direction for outlier testing

    Returns:
        DataFrame with MSC values expanded to original timestamps
    """
    if nyears_window == 0:
        return compute_msc(df, mask=mask, min_contribution=min_contribution,
                          method=method, return_long=True,
                          return_outlier_mask=return_outlier_mask,
                          z_outlier=z_outlier, test_direction=test_direction)

    df = _apply_mask(df, mask)
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df['year'] = df['time'].dt.year

    years = sorted(df['year'].unique())
    half_window = nyears_window // 2

    results = []
    outlier_masks = []

    for year in years:
        # Determine window bounds
        if year - half_window < min(years):
            year_start = min(years)
            year_end = min(years) + nyears_window
        elif year + half_window + 1 > max(years):
            year_start = max(years) - nyears_window
            year_end = max(years)
        else:
            year_start = year - half_window
            year_end = year + half_window + 1

        # Select data within window
        window_mask = (df['year'] >= year_start) & (df['year'] <= year_end)
        window_df = df[window_mask]

        # Compute MSC for this window
        if return_outlier_mask:
            msc, out_mask, _, _ = compute_msc(
                window_df, mask=None, min_contribution=min_contribution,
                method=method, return_long=True, return_outlier_mask=True,
                z_outlier=z_outlier, test_direction=test_direction
            )
            # Keep only current year's data
            year_mask = pd.to_datetime(msc['time']).dt.year == year
            results.append(msc[year_mask])
            outlier_masks.append(out_mask[year_mask])
        else:
            msc = compute_msc(
                window_df, mask=None, min_contribution=min_contribution,
                method=method, return_long=True
            )
            # Keep only current year's data
            year_mask = pd.to_datetime(msc['time']).dt.year == year
            results.append(msc[year_mask])

    result_df = pd.concat(results, ignore_index=True)

    if return_outlier_mask:
        outlier_mask = pd.concat(outlier_masks, ignore_index=True)
        return result_df, outlier_mask

    return result_df


# =============================================================================
# Derived Aggregations
# =============================================================================

def aggregate_seasonal(df, mask=None, min_contribution=2, method='mean'):
    """
    Compute mean seasonal cycle (short form).

    Returns one value per (env, doy) - the multi-year average for each day-of-year.

    Args:
        df: DataFrame with y_true, y_pred, env, time columns
        mask: Optional boolean mask for valid data
        min_contribution: Minimum years with valid data per DOY
        method: 'mean' or 'median'

    Returns:
        DataFrame with one row per (env, doy)
    """
    return compute_msc(df, mask=mask, min_contribution=min_contribution,
                       method=method, return_long=False)


def aggregate_anomaly(df, mask=None, min_contribution=2, method='mean'):
    """
    Compute anomalies from mean seasonal cycle.

    For each sample, subtracts the site's MSC value for that day-of-year.

    Args:
        df: DataFrame with y_true, y_pred, env, time columns
        mask: Optional boolean mask for valid data
        min_contribution: Minimum years for MSC computation
        method: 'mean' or 'median' for MSC

    Returns:
        DataFrame with anomaly values (original - MSC)
    """
    df = _apply_mask(df, mask)
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df['doy'] = df['time'].dt.dayofyear

    # Compute MSC (short form)
    msc = compute_msc(df, mask=None, min_contribution=min_contribution,
                      method=method, return_long=False)
    msc = msc.rename(columns={'y_true': 'msc_true', 'y_pred': 'msc_pred'})

    # Merge and compute anomalies
    result = df.merge(msc[['env', 'doy', 'msc_true', 'msc_pred']], on=['env', 'doy'], how='left')
    result['y_true'] = result['y_true'] - result['msc_true']
    result['y_pred'] = result['y_pred'] - result['msc_pred']

    return result[['env', 'time', 'y_true', 'y_pred']]


def aggregate_iav(df, mask=None, min_contribution=0.5):
    """
    Compute inter-annual variability (IAV).

    Computes yearly means per site, then subtracts the site's multi-year mean.

    Args:
        df: DataFrame with y_true, y_pred, env, time columns
        mask: Optional boolean mask for valid data
        min_contribution: Minimum fraction/count of valid days per year

    Returns:
        DataFrame with IAV values (yearly mean - site mean)
    """
    # First aggregate to yearly
    yearly = aggregate_yearly(df, mask=mask, min_contribution=min_contribution)

    # Compute site means
    site_means = yearly.groupby('env')[['y_true', 'y_pred']].transform('mean')

    # Compute IAV as deviation from site mean
    yearly['y_true'] = yearly['y_true'] - site_means['y_true']
    yearly['y_pred'] = yearly['y_pred'] - site_means['y_pred']

    return yearly


def aggregate_spatial(df, mask=None, min_contribution=0.5):
    """
    Compute site means for spatial variability.

    First aggregates to yearly, then computes multi-year mean per site.
    This matches QuickEval's site_mean function.

    Args:
        df: DataFrame with y_true, y_pred, env, time columns
        mask: Optional boolean mask for valid data
        min_contribution: Minimum fraction/count of valid days per year

    Returns:
        DataFrame with one row per site (env)
    """
    # First aggregate to yearly
    yearly = aggregate_yearly(df, mask=mask, min_contribution=min_contribution)

    # Compute multi-year mean per site
    site_means = yearly.groupby('env').agg({
        'y_true': 'mean',
        'y_pred': 'mean'
    }).reset_index()

    return site_means


# =============================================================================
# Registry
# =============================================================================

AGGREGATIONS = {
    'daily': aggregate_daily,
    'weekly': aggregate_weekly,
    'monthly': aggregate_monthly,
    'seasonal': aggregate_seasonal,
    'anom': aggregate_anomaly,
    'iav': aggregate_iav,
    'spatial': aggregate_spatial,
}
