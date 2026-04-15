"""Plotting functions for FLUXNET benchmark results."""


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import pandas as pd
import seaborn as sns

from utils.utils import setup_logging

logger = setup_logging(__name__)

PLOTS_DIR = 'results/plots'
SCALES = ['hourly', 'daily', 'weekly', 'monthly', 'seasonal', 'anom', 'iav']

# Model ordering: lr first, xgb second, then alphabetically
MODEL_ORDER = ['lr', 'xgb']

# Setting ordering: time-split, spatial-easy, spatial-hard
SETTINGS_ORDER = ['time-split', 'spatial-easy', 'spatial-hard']

# Metrics where higher is better (affects sorting direction and labels)
HIGHER_IS_BETTER = {'nse', 'r2_score', 'pearson_corr'}


def get_ordered_models(models):
    """Order models: lr, xgb, then alphabetically."""
    models = list(models)
    ordered = [m for m in MODEL_ORDER if m in models]
    remaining = sorted([m for m in models if m not in MODEL_ORDER])
    return ordered + remaining


def get_ordered_settings(settings):
    """Order settings: time-split, spatial-easy, spatial-hard, then alphabetically."""
    settings = list(settings)
    ordered = [s for s in SETTINGS_ORDER if s in settings]
    remaining = sorted([s for s in settings if s not in SETTINGS_ORDER])
    return ordered + remaining


def is_higher_better(metric):
    """Check if higher values are better for this metric."""
    return metric.lower() in HIGHER_IS_BETTER


def plot_metric_by_setting(results, target, metric, scale, ax, agg='median', legend=False):
    """
    Plot metric across settings for one scale (single subplot).

    Args:
        results: DataFrame with columns target, setting, model, scale, env, metric
        target: Target variable to filter (e.g., 'GPP')
        metric: Metric column name (e.g., 'rmse')
        scale: Temporal scale to filter (e.g., 'daily')
        agg: Aggregation function to apply (default: 'median')
        ax: Matplotlib axes to plot on
    """
    subset = results[(results['target'] == target) & (results['scale'] == scale)]
    if subset.empty:
        ax.set_title(f"{scale} (no data)")
        return

    data = (
        subset
        .groupby(['setting', 'model'])[metric]
        .agg(agg)
        .reset_index()
    )

    hue_order = get_ordered_models(data['model'].unique())
    data['setting'] = pd.Categorical(data['setting'], categories=SETTINGS_ORDER, ordered=True)
    data = data.sort_values('setting')
    for i, plot_func in enumerate([sns.lineplot, sns.scatterplot]):
        plot_func(data=data, x='setting', y=metric, ax=ax, hue='model',
                  hue_order=hue_order, legend=legend&(i==1))

    ax.set_xticks(range(len(SETTINGS_ORDER)))
    ax.set_xticklabels(SETTINGS_ORDER)
    ax.set_title(scale)
    ax.set_xlabel('')


def plot_metric_grid(results, target, metric='rmse', agg='median', outdir=PLOTS_DIR):
    """
    Create 3x2 grid showing metric across settings for all scales.

    Args:
        results: DataFrame with results
        target: Target variable (e.g., 'GPP')
        metric: Metric to plot (default: 'rmse')
        agg: Aggregation function to apply (default: 'median')
        outdir: Output directory for saved plot
    """
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(4, 2, figsize=(6, 8))
    axes = axes.flatten()

    for i, scale in enumerate(SCALES):
        plot_metric_by_setting(results, target, metric, scale, axes[i],
                               agg=agg, legend=(i == 0))
    axes[0].legend(title='')

    fig.suptitle(f"{target}")
    plt.tight_layout()

    outfile = os.path.join(outdir, f"{agg}_{target}_{metric}_by_scale.png")
    plt.savefig(outfile, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {outfile}")


def plot_cdf(results, target, metric, scale, setting, ax):
    """
    Plot CDF of metric for one target/scale/setting (single subplot).

    Args:
        results: DataFrame with results
        target: Target variable to filter
        metric: Metric column name
        scale: Temporal scale to filter
        setting: Setting to filter
        ax: Matplotlib axes to plot on
    """
    subset = results[
        (results['target'] == target) &
        (results['setting'] == setting) &
        (results['scale'] == scale)
    ]

    if subset.empty:
        ax.set_title(f"{setting} (no data)")
        return

    higher_better = is_higher_better(metric)
    models = get_ordered_models(subset['model'].unique())

    for model_name in models:
        model_data = subset[subset['model'] == model_name]
        values = model_data[metric].dropna().values
        if len(values) == 0:
            continue

        if higher_better:
            # Sort descending for higher-is-better metrics
            sorted_values = np.sort(values)[::-1]
        else:
            # Sort ascending for lower-is-better metrics
            sorted_values = np.sort(values)

        ax.plot(sorted_values, np.linspace(0, 1, len(sorted_values)), label=model_name)

    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlabel(f'{metric.upper()} (x)')

    env = "sites-years" if setting == 'time-split' else "sites"
    if higher_better:
        ax.set_ylabel(f'% of {env} with {metric.upper()} >= x')
    else:
        ax.set_ylabel(f'% of {env} with {metric.upper()} <= x')

    if metric.lower() == 'nse':
        ax.set_xlim(-0.5, 1.0)
    else:
        ax.set_xlim(left=0)

    ax.set_title(setting)
    ax.legend()


def plot_quantile(results, target, metric, scale, setting, ax, y_limit=None):
    """
    Plot Quantile function (Sorted Performance Curve) of metric.
    
    Args:
        results: DataFrame with results
        target: Target variable to filter
        metric: Metric column name
        scale: Temporal scale to filter
        setting: Setting to filter
        ax: Matplotlib axes to plot on
        y_limit: Optional float to cap the Y-axis (useful for exploding errors)
    """
    subset = results[
        (results['target'] == target) &
        (results['setting'] == setting) &
        (results['scale'] == scale)
    ]

    if subset.empty:
        ax.set_title(f"{setting} (no data)")
        return

    higher_better = is_higher_better(metric)
    models = get_ordered_models(subset['model'].unique())

    for model_name in models:
        model_data = subset[subset['model'] == model_name]
        values = model_data[metric].dropna().values

        if len(values) == 0:
            continue

        if higher_better:
            # Sort descending: best (high) to worst (low)
            sorted_values = np.sort(values)[::-1]
        else:
            # Sort ascending: best (low) to worst (high)
            sorted_values = np.sort(values)

        percentiles = np.linspace(0, 1, len(sorted_values))
        ax.plot(percentiles, sorted_values, label=model_name)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    env = "sites" if setting.startswith('spatial') else "site-years"
    if higher_better:
        ax.set_xlabel(f'% of {env} with {metric.upper()} >= y')
    else:
        ax.set_xlabel(f'% of {env} with {metric.upper()} <= y')

    if metric.lower() == 'nse':
        ax.set_ylim(-0.5, 1.0)
        ax.axhline(0, color='k', linestyle=':', linewidth=1)

    ax.set_ylabel(f'{metric.upper()} (y)')
    ax.set_title(setting)

    if y_limit:
        ax.set_ylim(top=y_limit)

    ax.legend()


def plot_cdf_grid(results, target, metric='rmse', scale='daily', outdir=PLOTS_DIR):
    """
    Create subplots showing CDF for each available setting.

    Args:
        results: DataFrame with results
        target: Target variable (e.g., 'GPP')
        metric: Metric to plot (default: 'rmse')
        scale: Temporal scale (default: 'daily')
        outdir: Output directory for saved plot
    """
    os.makedirs(outdir, exist_ok=True)

    # Get available settings for this target/scale
    subset = results[(results['target'] == target) & (results['scale'] == scale)]
    settings = get_ordered_settings(subset['setting'].unique())

    if len(settings) == 0:
        logger.warning(f"No data for {target} at {scale} scale")
        return

    n_settings = len(settings)
    fig, axes = plt.subplots(1, n_settings, figsize=(4 * n_settings, 4), sharey=True)

    if n_settings == 1:
        axes = [axes]

    for i, setting in enumerate(settings):
        plot_cdf(results, target, metric, scale, setting, axes[i])

    fig.suptitle(f"{target} ({scale})")
    plt.tight_layout()

    outfile = os.path.join(outdir, f"{target}_{metric}_{scale}_cdf.png")
    plt.savefig(outfile, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {outfile}")


# TODO: add logger
# TODO: add option for higher-is-better metrics
# TODO: saving should be cleaner and match other functions
# TODO: clean up to match style of other functions
# TODO: would actually be nice if this whole thing was transposed 
# TODO: make more modular/general -> only for one scale for example
# TODO: handle ties in medal rankings
# https://pandas.pydata.org/docs/user_guide/style.html
def create_leaderboard(df, target, metric, filename, aggfunc='median'):
    # --- 1. Data Preparation ---
    # Filter by target
    subset = df[(df['target'] == target) & (df['scale'] != 'spatial')]
    
    # Pivot: Index=(Setting, Scale), Cols=Model, Values=Metric
    settings = get_ordered_settings(subset['setting'].unique())
    scales = subset['scale'].unique()
    
    pivot_df = subset.pivot_table(
        index=['setting', 'scale'], 
        columns='model', 
        values=metric, 
        aggfunc=aggfunc
    )
    
    # Reindex to enforce correct order and use a MultiIndex for rows
    idx = pd.MultiIndex.from_product([settings, scales], names=[None, None])
    pivot_df = pivot_df.reindex(index=idx)
    pivot_df.columns.name = None # Clear the 'model' column heading label

    # --- Calculate Medals and points ---
    # Rank models across each row (1 is lowest/best)
    ranks = pivot_df.rank(axis=1, method='first')
    gold = (ranks == 1).sum()
    silver = (ranks == 2).sum()
    bronze = (ranks == 3).sum()
    
    points = (gold * 3) + (silver * 2) + (bronze * 1)
    
    # Create the summary rows and prepend them
    summary_data = {
        col: [f"{points[col]} pts", f"🥇{gold[col]}    🥈{silver[col]}    🥉{bronze[col]}"] 
        for col in pivot_df.columns
    }
    summary_df = pd.DataFrame(
        summary_data, 
        index=pd.MultiIndex.from_tuples([('Summary', 'Points'), ('Summary', 'Medals')])
    )
    pivot_df = pd.concat([summary_df, pivot_df])
    
    # --- 2. Color Logic Calculation ---
    def highlight_medals(row):
        if row.name[0] == 'Summary':
            return ['font-weight: bold; background-color: #f8f9fa'] * len(row)
        
        styles = [''] * len(row)
        # Drop NaNs and sort to find the top 3 models per row
        valid_vals = row.dropna().sort_values(ascending=True)
        colors = {
            0: 'background-color: #FFD700', # Gold
            1: 'background-color: #C0C0C0', # Silver
            2: 'background-color: #CD7F32'  # Bronze
        }
        for rank in range(min(3, len(valid_vals))):
            loc = row.index.get_loc(valid_vals.index[rank])
            styles[loc] = colors[rank]
        return styles

    # --- 3. Apply Styles and Export to HTML ---
    table = {
        'selector': '', 
        'props': [('border-collapse', 'collapse'), ('font-family', 'sans-serif')]
    }
    cells = {
        'selector': 'th, td', 
        'props': [('text-align', 'center'), ('padding', '8px')]
    }
    body_cells = {
        'selector': 'td',
        'props': [('border', '0.5px solid gray')]
    }
    # Styling for the Setting row index
    heading1 = {
        'selector': 'th.row_heading.level0', 
        'props': [('font-weight', 'bold'), ('border-bottom', '1px solid black'),
                  ('text-align', 'center'), ('border-left', '1px solid black'), 
                  ('border-top', '1px solid black')]
    }
    # Styling for the Scale row index
    heading2 = {
        'selector': 'th.row_heading.level1', 
        'props': [('font-weight', 'normal'), ('text-align', 'right'),
                  ('border-right', '1px solid black')]
    }
    # Styling for the Model column headers
    heading_cols = {
        'selector': 'th.col_heading',
        'props': [('font-weight', 'bold'), ('border-top', '1px solid black'), 
                  ('border-bottom', '1px solid black')]
    }

    styler = (
        pivot_df.style
        .format(precision=2, na_rep="-")
        .set_table_styles([table, cells, body_cells, heading1, heading2, heading_cols])
        .apply(highlight_medals, axis=1) # Changed to axis=1 to evaluate medals across rows
    )

    # Add solid lines between setting groups (horizontal borders)
    for i in range(len(pivot_df) - 1):
        if pivot_df.index[i][0] != pivot_df.index[i+1][0]:
            # thickness = "2px" if pivot_df.index[i][0] == "Summary" else "1px"
            thickness = "1px"
            styler.set_table_styles({
                pivot_df.index[i]: [
                    {"selector": "th", "props": f"border-bottom: {thickness} solid black"},
                    {"selector": "td", "props": f"border-bottom: {thickness} solid black"},
                ]
            }, overwrite=False, axis=1)

    # Add left and right outer borders to the data columns
    styler.set_table_styles({
        pivot_df.columns[0]: [
            {"selector": "th, td", "props": "border-left: 1px solid black"}
        ],
        pivot_df.columns[-1]: [
            {"selector": "th, td", "props": "border-right: 1px solid black"}
        ]
    }, overwrite=False, axis=0)

    # Add top and bottom outer borders to the dataframe
    styler.set_table_styles({
        pivot_df.index[0]: [
            {"selector": "th, td", "props": "border-top: 1px solid black"}
        ],
        pivot_df.index[-1]: [
            {"selector": "th, td", "props": "border-bottom: 1px solid black"}
        ]
    }, overwrite=False, axis=1)

    # Save to HTML file
    html_output = '<meta charset="UTF-8">\n' + styler.to_html()
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_output)
    print(f"Transposed styled HTML table saved to {filename}")