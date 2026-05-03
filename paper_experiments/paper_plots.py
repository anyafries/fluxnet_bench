"""
Script to load and compare results from multiple experiments.

This script loads pre-computed metrics if available, or computes them from
predictions. Results are saved to results/metrics/ for subsequent runs.

Usage:
    python eval.py --setting spatial-easy --target GPP
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
PLOTS_DIR = f'/r/scratch/users/anfries/fluxnet_data/results/plots/mean'


if __name__ == "__main__":
    results = load_all_metrics(
        settings=SETTINGS,
        targets=TARGETS,
        models=MODELS,
        scales=RAW_SCALES,
        val_strategy='mean',
        rerun=False,

    )
    results['scale'] = results['scale'].replace({'spatial': 'site-mean'})

    plt.style.use('/u/anfries/fluxnet_bench/utils/neurips.mplstyle')

    flux_saturation_results = []
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
            create_latex_leaderboard(
                results_tex, target, metric='rmse', 
                lower_is_better=True,
                scale_order=SCALES,
                settings_order=SETTINGS,
                settings_names=SETTING_NAMES,
                aggfunc=aggfunc,
                filename=f'{PLOTS_DIR}/table_{aggname}_{target}.tex',
                display_mode='value'
            )

    #     # Prepare errors for plotting the comparison of saturation of fluxes
    #     rel_err_median = get_relative_errors_by_flux(
    #         results_tex,
    #         target=target,
    #         metric="rmse",
    #         aggfunc=lambda x: x.quantile(0.9),
    #         settings_order=["time-split", "spatial-easy40", "TA40"],
    #         lower_is_better=True
    #     )
    #     rel_err_median['target'] = target
    #     flux_saturation_results.append(rel_err_median)

    # CDF for hourly, weekly, seasonal for TA40 ET
    results_et = results[results['target'] == 'ET']
    results_et['rmse'] = results_et['rmse'] * 100
    for scale in ['hourly', 'weekly', 'seasonal', 'spatial']:
        fig, ax = plt.subplots(figsize=(1.5, 1.6))
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.5)
        plot_cdf(results_et, scale=scale, target='ET', 
                 setting='TA40',  metric='rmse', ax=ax,
                 linestyle='-', linewidth=1)
        if scale == "weekly":
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
        plt.savefig(f'{PLOTS_DIR}/cdf_et_{scale}.png', bbox_inches='tight', dpi=300)
        print(f"CDF plot for ET at {scale} scale saved to {PLOTS_DIR}/cdf_et_{scale}.png")


    # # Final plot comparing saturation of fluxes across scales and settings
    # flux_saturation_results_df = pd.concat(flux_saturation_results, ignore_index=True)
    # fig = plt.figure(figsize=(3.5, 2))
    # plot_saturation_by_flux(
    #     fig,
    #     flux_saturation_results_df, 
    #     filename=f'{PLOTS_DIR}/relative_errors_fluxes.png',
    #     settings=SETTINGS,
    #     scales=SCALES,
    #     scale_colors=SCALE_COLORS,
    #     setting_names=SETTING_NAMES,
    # )

        
        

