"""
Script to plot results from multiple experiments for the paper.
"""

import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys

from paper_plot_utils import (
    create_latex_leaderboard,
    get_relative_errors_by_flux,
    plot_saturation_by_flux,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import load_all_metrics
from utils.plots import plot_cdf
from utils.utils import setup_logging

logger = setup_logging(__name__)


SETTINGS = ['time-split', 'spatial-easy40', 'TA40']
SETTING_NAMES = {
    'time-split': 'temporal',
    'spatial-easy40': 'spatial',
    'TA40': 'temperature',
}
TARGETS = ['ET', 'GPP', 'NEE'] 
MODELS = ['lr', 'xgb', 'mlp', 'mmd', 'coral', 'gdro', 'constant']
MODEL_ORDER = ['xgb', 'lightgbm', 'mlp', 'gdro', 'coral', 'mmd', 
               'lr', 'robust-lr', 'ridge',  'constant']
color_palette = sns.color_palette("tab10", n_colors=len(MODEL_ORDER))
MODEL_COLORS = {model: color_palette[i] for i, model in enumerate(MODEL_ORDER)}

RAW_SCALES = ['hourly', 'weekly', 'seasonal', 'anom', 'iav', 'spatial']
SCALES = ['hourly', 'weekly', 'seasonal', 'anom', 'iav', 'site-mean'] 
SCALE_COLORS = {
    "hourly":   "#4a2377",
    "weekly":   "#8cc5e3",
    "seasonal": "#f55f99",
    "anom":     "#0d7d87",
    "iav":      "#f47a00",
    "site-mean":  "#1b9e77",
}

PLOTS_DIR = f'paper_experiments/plots'
STYLE_FILE_PATH = 'utils/neurips.mplstyle'

if __name__ == "__main__":
    results = load_all_metrics(
        settings=SETTINGS,
        targets=TARGETS,
        models=MODELS,
        scales=RAW_SCALES,
        val_strategy='mean',
        rerun=False,

    )
    print(results.head())
    results['scale'] = results['scale'].replace({'spatial': 'site-mean'})

    plt.style.use(STYLE_FILE_PATH)

    # flux_saturation_results = []
    for target in ['GPP', 'ET', 'NEE']:
        if target == 'ET':
            # multiply RMSE by 100 
            results_tex = results.copy()
            results_tex['rmse'] = results_tex['rmse'] * 100
        else: 
            results_tex = results.copy()

        # Create LaTeX leaderboard for each target 
        for aggname, aggfunc in [
            ('median', 'median'), ('90q', lambda x: x.quantile(0.9))
        ]:
            tables = ['supp']
            if target == 'ET' and aggname == '90q':  
                tables.append('main')
            for table in tables:
                if table == 'main':
                    filename = f'{PLOTS_DIR}/table_{aggname}_{target}.tex'
                else: 
                    filename = f'{PLOTS_DIR}/table_{aggname}_{target}_supp.tex'
                create_latex_leaderboard(
                    results_tex, target, metric='rmse', 
                    lower_is_better=True,
                    scale_order=SCALES,
                    settings_order=SETTINGS,
                    settings_names=SETTING_NAMES,
                    aggfunc=aggfunc,
                    filename=filename,
                    display_mode='value',
                    main_table=table=='main',
                )

        # CDF for hourly, weekly, seasonal for TA40 ET
        results_t = results[results['target'] == target].copy()
        if target == 'ET':
            results_t['rmse'] = results_t['rmse'] * 100
        for scale in ['hourly', 'weekly', 'seasonal', 'site-mean']:
            fig, ax = plt.subplots(figsize=(1.5, 1.6))
            ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.5)
            ax.axhline(0.9, color='gray', linestyle='--', linewidth=0.5)
            plot_cdf(results_t, scale=scale, target=target, 
                    setting='TA40',  metric='rmse', ax=ax,
                    linestyle='-', linewidth=1)
            if scale == "weekly" and target == "ET":
                ax.set_xlim(1, 10)
            ax.set_title('')
            ax.set_xlabel('RMSE')
            ax.set_ylabel('Cumulative Probability')
            leg = ax.legend(
                title="", 
                frameon=True, 
                handlelength=0.6,
                handleheight=0.4,
                handletextpad=0.3,
                labelspacing=0.2,
                borderpad=0.2,
            )
            leg.get_frame().set_linewidth(0.5)
            leg.get_frame().set_edgecolor('lightgray')
            plt.savefig(f'{PLOTS_DIR}/cdf_{target}_{scale}.png', bbox_inches='tight', dpi=300)
            print(f"CDF plot for {target} at {scale} scale saved to {PLOTS_DIR}/cdf_{target}_{scale}.png")