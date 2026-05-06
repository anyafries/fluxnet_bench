import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from statsmodels.graphics.tsaplots import plot_acf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader import load_data

plt.style.use('utils/neurips.mplstyle')
df = load_data('data')
target = 'ET'

for site in ['DE-Hai', "AU-Cum", "AU-GWW", "CZ-RAJ"]:
    fig, axes = plt.subplots(3, 3, figsize=(5, 3.5), sharey=True)
    df_site = df[df['site_id'] == site].copy()
    df_site['timestamp'] = pd.to_datetime(df_site['time'], format='%Y-%m-%d %H:%M:%S')
    scales = ['hourly', 'daily', 'weekly']
    scale_lags = {
        'hourly': 38,  # 3 days
        'daily': 42,   # 6 weeks
        'weekly': 52   # 1 year
    }
    for j, target in enumerate(['ET', 'GPP', 'NEE']):
        for i, scale in enumerate(scales):
            ax = axes[j, i]
            if scale == 'hourly':
                df_agg = df_site.copy()
            elif scale == 'daily':
                df_agg = df_site.resample('D', on='timestamp')[target].mean().reset_index()
            elif scale == 'weekly':
                df_agg = df_site.resample('W', on='timestamp')[target].mean().reset_index()
            else:
                raise ValueError(f"Unsupported scale: {scale}")
            # Interpolate missing values
            df_agg[target] = df_agg[target].interpolate()  
            
            plot_acf(df_agg[target], lags=scale_lags[scale], fft=True, ax=ax,
                    vlines_kwargs={'linewidth': 0.8}, markersize=2)
            ax.set_xlabel(f'')
            ax.set_title(f'')
    
    # add a bold title for each column
    for i, scale in enumerate(scales):
        axes[0, i].set_title(f'{scale.capitalize()}', fontsize=10)
        lag_name = 'hours' if scale == 'hourly' else ('days' if scale == 'daily' else 'weeks')
        axes[2, i].set_xlabel(f'Lag (in {lag_name})', fontsize=10)
    # add a bold title for each row
    for j, target in enumerate(['ET', 'GPP', 'NEE']):
        axes[j, 0].set_ylabel(f'{target}', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'paper_experiments/plots/acf_{site}.png', bbox_inches='tight', dpi=300)
    print(f"Saved ACF plot for {site}")
    plt.close()