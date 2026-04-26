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
MODEL_ORDER = ['xgb', 'lightgbm', 'mlp', 
               'gdro', 'coral', 'mmd', 
            #    'maxrm_mse', 'maxrm_regret', 
               'lr', 'robust-lr', 'ridge',  'constant']
color_palette = sns.color_palette("tab10", n_colors=len(MODEL_ORDER))
MODEL_COLORS = {model: color_palette[i] for i, model in enumerate(MODEL_ORDER)}

# Setting ordering: time-split, spatial-easy, spatial-hard
SETTINGS_ORDER = ['time-split', 'spatial-easy', 'spatial-hard', 
                  'LST', 'TA', 'VPD', 
                  'PFT_CRO', 'PFT_ENF', 'PFT_GRA', 'PFT_WET', 
                  'forest', 'grass-savanna', 'schrub-savanna',
                  'europe', 'rest-of-world',
                  ] + [f'hard-{i}' for i in range(1, 6)] + ['time-space']

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


def plot_metric_by_setting(results, target, metric, scale, ax, agg='median', 
                           legend=False, ymax=None):
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

    # hue_order = get_ordered_models(data['model'].unique())
    categories = SETTINGS_ORDER + [s for s in np.sort(data['setting'].unique()) if s not in SETTINGS_ORDER]
    data['setting'] = pd.Categorical(data['setting'], categories=categories, ordered=True)
    data = data.sort_values('setting')
    for i, plot_func in enumerate([sns.lineplot, sns.scatterplot]):
        plot_func(data=data, x='setting', y=metric, ax=ax, hue='model',
                  palette=MODEL_COLORS, legend=legend&(i==1))

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=90, ha='center')
    ax.set_title(scale)
    ax.set_xlabel('')
    ax.set_ylim(bottom=0)
    if ymax is not None:
        ax.set_ylim(top=ymax)


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

    fig, axes = plt.subplots(4, 2, figsize=(8, 8), sharex=True)
    axes = axes.flatten()

    for i, scale in enumerate(SCALES):
        plot_metric_by_setting(results, target, metric, scale, axes[i],
                               agg=agg, legend=(i == len(SCALES) - 1), 
                               ymax=1 if metric.lower() == 'nse' else None)
    axes[len(SCALES) - 1].legend(title='')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f"{target}")
    plt.tight_layout()

    if callable(agg):
        agg = 'quantile'
    outfile = os.path.join(outdir, f"{agg}_{target}_{metric}_by_scale.png")
    plt.savefig(outfile, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {outfile}")


def plot_cdf(results, target, metric, scale, setting, ax, xmax=None):
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

        ax.plot(sorted_values, np.linspace(0, 1, len(sorted_values)), 
                label=model_name, color=MODEL_COLORS.get(model_name, 'gray'))

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
        ax.set_xlim(0, xmax)

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
        ax.plot(percentiles, sorted_values, label=model_name,
                color=MODEL_COLORS.get(model_name, 'gray'))

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
    fig, axes = plt.subplots(1, n_settings, figsize=(4 * n_settings, 4), 
                             sharey=True, sharex=True)

    if n_settings == 1:
        axes = [axes]

    xmax = results[
        (results['target'] == target) &
        (results['scale'] == scale)
    ][metric].max()

    for i, setting in enumerate(settings):
        plot_cdf(results, target, metric, scale, setting, axes[i], xmax=xmax)

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
import pandas as pd

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
    # Use method='min' for standard competition ranking (1, 1, 3) 
    # Use method='dense' if you want consecutive ranking (1, 1, 2)
    ranks = pivot_df.rank(axis=1, method='min')
    
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
        # Skip styling for the summary rows
        if row.name[0] == 'Summary':
            return ['font-weight: bold; background-color: #f8f9fa'] * len(row)
        
        styles = [''] * len(row)
        
        # Rank the row using the exact same tie-breaking logic as the points
        row_ranks = row.rank(method='min')
        
        colors = {
            1: 'background-color: #FFD700', # Gold
            2: 'background-color: #C0C0C0', # Silver
            3: 'background-color: #CD7F32'  # Bronze
        }
        
        # Apply colors based on the computed rank
        for i, rank in enumerate(row_ranks):
            if rank in colors:
                styles[i] = colors[rank]
                
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
        .apply(highlight_medals, axis=1)
    )

    # Add solid lines between setting groups (horizontal borders)
    for i in range(len(pivot_df) - 1):
        if pivot_df.index[i][0] != pivot_df.index[i+1][0]:
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


def get_hex_color(val, min_val, max_val, lower_is_better=True):
    """Interpolates a value into a Red-White-Green Hex color string."""
    if pd.isna(val) or max_val == min_val:
        return "FFFFFF" # Default white for NaNs or completely tied columns
        
    # Normalize between 0 (worst) and 1 (best)
    if lower_is_better:
        ratio = (max_val - val) / (max_val - min_val)
    else:
        ratio = (val - min_val) / (max_val - min_val)
        
    # Standard Excel-like Red, White, Green
    # Red: (248, 105, 107), White: (255, 255, 255), Green: (99, 190, 123)
    if ratio < 0.5:
        # Interpolate Red to White
        t = ratio / 0.5
        r = int(248 + t * (255 - 248))
        g = int(105 + t * (255 - 105))
        b = int(107 + t * (255 - 107))
    else:
        # Interpolate White to Green
        t = (ratio - 0.5) / 0.5
        r = int(255 + t * (99 - 255))
        g = int(255 + t * (190 - 255))
        b = int(255 + t * (123 - 255))
        
    return f"{r:02X}{g:02X}{b:02X}"


def format_sig_figs(val, n=2):
    """Formats a number to n significant figures."""
    if pd.isna(val) or val == 0:
        return str(val)
    # This handles scientific notation and standard float formatting automatically
    return f"{val:.{n}g}"


def get_hex_relative_color(val, best_val, rel_threshold=1.0, lower_is_better=True):
    """
    Colors values based on their distance from the best value in the column.
    Green = Best
    White = Best + (Best * rel_threshold) [for lower_is_better]
    """
    if pd.isna(val) or pd.isna(best_val):
        return "FFFFFF"

    # Define the 'Limit' where the color fades to white
    if lower_is_better:
        # e.g., Best is 0.1, threshold is 1.0 (double). Limit is 0.2.
        limit = best_val * (1 + rel_threshold)
        if val <= best_val: ratio = 1.0
        elif val >= limit: ratio = 0.0
        else:
            # Linear interpolation between best and limit
            ratio = (limit - val) / (limit - best_val)
    else:
        # e.g., Best is 0.8, threshold is 0.5 (half). Limit is 0.4.
        limit = best_val * (1 - rel_threshold)
        if val >= best_val: ratio = 1.0
        elif val <= limit: ratio = 0.0
        else:
            ratio = (val - limit) / (best_val - limit)

    # Gradient: Green (99, 190, 123) to White (255, 255, 255)
    r = int(255 - (ratio * (255 - 99)))
    g = int(255 - (ratio * (255 - 190)))
    b = int(255 - (ratio * (255 - 123)))
    
    return f"{r:02X}{g:02X}{b:02X}"



def create_latex_leaderboard(df, target, metric, filename, aggfunc='median', 
                             lower_is_better=True,
                             scale_order=['hourly', 'daily', 'weekly', 'monthly', 'seasonal', 'anom', 'iav'],
                             model_order=None,
                             rel_threshold=0.2, display_mode='rank'):
    # --- 1. Data Preparation (Transposed) ---
    subset = df[(df['target'] == target) & (df['scale'] != 'spatial')]
    
    pivot_df = subset.pivot_table(
        index='model', 
        columns=['setting', 'scale'], 
        values=metric, 
        aggfunc=aggfunc
    )

    # --- Enforce Custom Scale Order ---
    if scale_order is not None:
        # Extract the settings in their current order to preserve them
        settings = pivot_df.columns.get_level_values(0).unique()
        
        # Rebuild the list of columns enforcing the scale_order per setting
        ordered_cols = []
        for s in settings:
            for sc in scale_order:
                if (s, sc) in pivot_df.columns:
                    ordered_cols.append((s, sc))
                    
            # Catch any stray scales present in the data but missing from your scale_order
            for col in pivot_df.columns:
                if col[0] == s and col not in ordered_cols:
                    ordered_cols.append(col)
                    
        # Apply the new column order to the DataFrame
        pivot_df = pivot_df[ordered_cols]

    # --- Enforce Custom Model Order ---
    if model_order is not None:
        # Reindex the DataFrame to have rows in the specified model order
        pivot_df = pivot_df.reindex(model_order)
    
    # --- Calculate Medals and points ---
    ranks = pivot_df.rank(axis=0, method='min', ascending=lower_is_better)
    
    gold = (ranks == 1).sum(axis=1)
    silver = (ranks == 2).sum(axis=1)
    bronze = (ranks == 3).sum(axis=1)
    
    points = (gold * 3) + (silver * 2) + (bronze * 1)
    
    # --- 2. Build the Cell Strings (Colors + Rank numbers) ---
    # latex_df = pd.DataFrame(index=pivot_df.index, columns=pivot_df.columns)
    
    # for col in pivot_df.columns:
    #     col_data = pivot_df[col]
    #     min_v = col_data.min()
    #     max_v = col_data.max()
        
    #     for row_idx in pivot_df.index:
    #         val = col_data[row_idx]
    #         if pd.isna(val):
    #             latex_df.loc[row_idx, col] = "-"
    #             continue
                
    #         hex_color = get_hex_color(val, min_v, max_v, lower_is_better=lower_is_better)
            
    #         rank_val = ranks.loc[row_idx, col]
    #         text = str(int(rank_val)) if rank_val <= 3 else ""
            
    #         latex_df.loc[row_idx, col] = f"\\cellcolor[HTML]{{{hex_color}}} {text}"
    latex_df = pd.DataFrame(index=pivot_df.index, columns=pivot_df.columns)
    
    for col in pivot_df.columns:
        col_data = pivot_df[col]
        # Reference point for the scale: The best model in this specific task
        best_in_col = col_data.min() if lower_is_better else col_data.max()
        
        for row_idx in pivot_df.index:
            val = col_data[row_idx]
            if pd.isna(val):
                latex_df.loc[row_idx, col] = "-"
                continue

            if display_mode == 'value':
                cell_text = format_sig_figs(val, n=2)
            elif display_mode == 'rank':
                r_val = ranks.loc[row_idx, col]
                cell_text = str(int(r_val)) if r_val <= 3 else ""
            else:
                cell_text = ""
                
            hex_color = get_hex_relative_color(val, best_in_col, rel_threshold, lower_is_better)
            latex_df.loc[row_idx, col] = f"\\cellcolor[HTML]{{{hex_color}}} {cell_text}"

    # --- 3. Append Summary Columns and Sort ---
    latex_df[('Summary', 'Points')] = points.astype(int)
    # latex_df[('Summary', 'Medals')] = [f"G:{int(g)} S:{int(s)} B:{int(b)}" for g, s, b in zip(gold, silver, bronze)]
    
    if model_order is None:
        latex_df = latex_df.sort_values(by=('Summary', 'Points'), ascending=False)
    latex_df.index.name = None 

    # --- 4. Format Column Headers & Build Gaps ---
    new_cols = []
    col_format = "l" # First column is for the model names
    
    # Build the header names and the LaTeX column format layout simultaneously
    prev_setting = latex_df.columns[0][0]
    
    for setting, scale in latex_df.columns:
        # Add visual gaps between settings using @{\hspace{...}}
        if setting != prev_setting:
            col_format += "@{\\hspace{1.5em}}" 
            prev_setting = setting
        col_format += "c"
        
        # Apply the rotation wrapper
        if setting == 'Summary':
            new_cols.append((setting, scale))
        else:
            new_cols.append((setting, f"\\rotatebox{{90}}{{{scale}}}"))
            
    latex_df.columns = pd.MultiIndex.from_tuples(new_cols)

    # --- 5. Export to LaTeX ---
    latex_str = latex_df.to_latex(
        escape=False, 
        na_rep="-", 
        column_format=col_format,
        multicolumn_format="c"
    )
    
    # Wrap in a group to locally shrink the column padding (makes the colored boxes narrower)
    latex_str = "{\\setlength{\\tabcolsep}{3pt}\n" + latex_str + "}\n"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(latex_str)
        
    print(f"Publication-ready LaTeX leaderboard saved to {filename}")

    # model_order = latex_df.index.tolist()
    # return model_order

    return latex_str