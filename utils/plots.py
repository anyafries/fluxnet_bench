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
SCALES = ['daily', 'weekly', 'monthly', 'seasonal', 'anom', 'iav']

# Model ordering: lr first, xgb second, then alphabetically
MODEL_ORDER = ['lr', 'xgb']

# Metrics where higher is better (affects sorting direction and labels)
HIGHER_IS_BETTER = {'nse', 'r2_score', 'pearson_corr'}


def get_ordered_models(models):
    """Order models: lr, xgb, then alphabetically."""
    models = list(models)
    ordered = [m for m in MODEL_ORDER if m in models]
    remaining = sorted([m for m in models if m not in MODEL_ORDER])
    return ordered + remaining


def is_higher_better(metric):
    """Check if higher values are better for this metric."""
    return metric.lower() in HIGHER_IS_BETTER


def plot_metric_by_setting(results, target, metric, scale, ax, legend=False):
    """
    Plot metric across settings for one scale (single subplot).

    Args:
        results: DataFrame with columns target, setting, model, scale, env, metric
        target: Target variable to filter (e.g., 'GPP')
        metric: Metric column name (e.g., 'rmse')
        scale: Temporal scale to filter (e.g., 'daily')
        ax: Matplotlib axes to plot on
    """
    subset = results[(results['target'] == target) & (results['scale'] == scale)]
    if subset.empty:
        ax.set_title(f"{scale} (no data)")
        return

    data = (
        subset
        .groupby(['setting', 'model'])[metric]
        .median()
        .reset_index()
    )

    hue_order = get_ordered_models(data['model'].unique())
    for i, plot_func in enumerate([sns.lineplot, sns.scatterplot]):
        plot_func(data=data, x='setting', y=metric, ax=ax, hue='model',
                  hue_order=hue_order, legend=legend&(i==1))

    ax.set_title(scale)
    ax.set_xlabel('')


def plot_metric_grid(results, target, metric='rmse', outdir=PLOTS_DIR):
    """
    Create 3x2 grid showing metric across settings for all scales.

    Args:
        results: DataFrame with results
        target: Target variable (e.g., 'GPP')
        metric: Metric to plot (default: 'rmse')
        outdir: Output directory for saved plot
    """
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(6, 8))
    axes = axes.flatten()

    for i, scale in enumerate(SCALES):
        plot_metric_by_setting(results, target, metric, scale, axes[i],
                               legend=(i == 0))
    axes[0].legend(title='')

    fig.suptitle(f"{target}")
    plt.tight_layout()

    outfile = os.path.join(outdir, f"{target}_{metric}_by_scale.png")
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

    if higher_better:
        ax.set_ylabel(f'% envs with {metric.upper()} >= x')
    else:
        ax.set_ylabel(f'% envs with {metric.upper()} <= x')

    if metric.lower() == 'nse':
        ax.set_xlim(-0.5, 1.0)

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

    if higher_better:
        ax.set_xlabel(f'% of sites with {metric.upper()} >= y')
    else:
        ax.set_xlabel(f'% of sites with {metric.upper()} <= y')

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
    settings = sorted(subset['setting'].unique())

    if len(settings) == 0:
        logger.warning(f"No data for {target} at {scale} scale")
        return

    n_settings = len(settings)
    fig, axes = plt.subplots(1, n_settings, figsize=(4 * n_settings, 4), sharey=True)

    if n_settings == 1:
        axes = [axes]

    for i, setting in enumerate(settings):
        plot_quantile(results, target, metric, scale, setting, axes[i])

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
# https://pandas.pydata.org/docs/user_guide/style.html
def create_leaderboard(df, target, metric, filename, aggfunc='median'):
    # --- 1. Data Preparation ---
    # Filter by target
    subset = df[df['target'] == target]
    
    # Pivot: Index=Model, Cols=(Setting, Scale), Values=Metric
    settings = subset['setting'].unique()
    scales = subset['scale'].unique()
    cols = pd.MultiIndex.from_product([settings, scales], names=['', ''])
    pivot_df = subset.pivot_table(
        index='model', 
        columns=['setting', 'scale'], 
        values=metric, 
        aggfunc=aggfunc
    ).reindex(columns=cols).reset_index()
    pivot_df.columns = pd.MultiIndex.from_tuples(
        [('', '')] + [c for c in pivot_df.columns[1:]]
    )
    
    # --- 2. Color Logic Calculation ---
    def highlight_medals(column):
        if column.name == ('', ''):
            return [''] * len(column)
        styles = [''] * len(column)
        valid_vals = column.dropna().sort_values(ascending=True)
        colors = {
            0: 'background-color: #FFD700', # Gold
            1: 'background-color: #C0C0C0', # Silver
            2: 'background-color: #CD7F32'  # Bronze
        }
        for rank in range(min(3, len(valid_vals))):
            loc = column.index.get_loc(valid_vals.index[rank])
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
    heading1 = {
        'selector': 'th.col_heading.level0', 
        'props': [('font-weight', 'bold'), ('border-right', '1px solid black'),
                  ('border-top', '1px solid black'), ('border-top', '1px solid black')]
    }
    heading2 = {
        'selector': 'th.col_heading.level1', 
        'props': [('font-weight', 'normal')]
    }
    styler = (
        pivot_df.style
        .format(precision=2, na_rep="-")
        .hide(axis="index")
        .set_table_styles([table, cells, body_cells, heading1, heading2])
        .apply(highlight_medals, axis=0)
    )

    # Add solid lines between setting groups
    group_size = len(scales)
    border_indices = [(i * group_size) for i in range(0, len(settings) + 1)] 
    for col in [pivot_df.columns[i] for i in border_indices]:
        styler = styler.set_table_styles({
            col: [
                {"selector": "th", "props": "border-right: 1px solid black"},
                {"selector": "td", "props": "border-right: 1px solid black"},
            ]
        }, overwrite=False, axis=0)
    styler.set_table_styles({
        pivot_df.columns[0]: [
            {"selector": "th", "props": "border-left: 1px solid black"},
            {"selector": "td", "props": "border-left: 1px solid black"}
            ]
    }, overwrite=False, axis=0)
    styler.set_table_styles({
        0: [{"selector": "td", "props": "border-top: 1px solid black"}],
        len(pivot_df)-1: [{"selector": "td", "props": "border-bottom: 1px solid black"}]
    }, overwrite=False, axis=1)

    # Save to HTML file
    styler.to_html(filename)
    print(f"Styled HTML table saved to {filename}")