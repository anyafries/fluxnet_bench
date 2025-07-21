import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)

def nse(ytrue, ypred):
    numerator = np.sum((ytrue - ypred) ** 2)
    denominator = np.sum((ytrue - np.mean(ytrue)) ** 2)
    return 1 - (numerator / denominator)


def evaluate_fold(ytrue, ypred, verbose=True, digits=3):
    mse = mean_squared_error(ytrue, ypred)
    results = {
        'mse': mse, 
        'rmse': np.sqrt(mse), 
        'r2_score': r2_score(ytrue, ypred), 
        'relative_error': np.mean(np.abs(ytrue - ypred) / np.abs(ytrue)), 
        'mae': np.mean(np.abs(ytrue - ypred)),
        'nse': nse(ytrue, ypred)
    }
    if verbose:
        logger.info(f"* RESULTS over {len(ytrue)} predictions:")
        logger.info("\t "+", ".join(f"{metric.upper()}={value:.{digits}f}" for metric, value in results.items()))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='data_cleaned/')
    parser.add_argument("--override", type=bool, default=False)
    parser.add_argument("--agg", type=str, default='seasonal')
    parser.add_argument("--setting", type=str, default='insite')
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--cv", action="store_true")
    
    path = parser.parse_args().path
    override = parser.parse_args().override
    agg = parser.parse_args().agg
    setting = parser.parse_args().setting
    start = parser.parse_args().start
    stop = parser.parse_args().stop
    cv = parser.parse_args().cv

    # Find any files of the form results/{agg}_{setting}*
    results_files = [f for f in os.listdir('results') 
                     if f.startswith(f"{agg}_{setting}")] # need to add cv here
    
    # divide the results files into models
    groups = {}
    for filename in results_files:
        parts = filename.split("_")
        if len(parts) > 2:  # Ensure there are at least three parts
            model_name = parts[2]  # Extract the third part (MODELNAME)
            if model_name not in groups:
                groups[model_name] = [filename]
            else:
                groups[model_name].append(filename)
    
    # Load data
    all_results = []
    for model_name, results_files in groups.items():
        model_data = []
        for f in results_files:
            model_data.append(pd.read_csv("results/"+f))
        model_data = pd.concat(model_data, ignore_index=True)
        model_data['model'] = model_name
        all_results.append(model_data)

    # Only select the first entry for each group (in case of multiple runs)
    all_results = pd.concat(all_results, ignore_index=True)
    all_results = all_results.sort_values(by='group')
    all_results = all_results.groupby(['model', 'group']).first().reset_index()
    print(all_results)

    # TODO: is this good enough?

    # Plot the results
    sns.boxplot(x='model', y='rmse', data=all_results)
    plt.ylim(0, 0.5)
    plt.savefig('results/plots_tmp/boxplot_rmse.png')
    plt.close()

    sns.boxplot(x='model', y='nse', data=all_results, showfliers=False)
    # plt.ylim(-2, 1)
    plt.savefig('results/plots_tmp/boxplot_nse.png')
    plt.close()

    sns.boxplot(x='model', y='r2_score', data=all_results, showfliers=False)
    plt.savefig('results/plots_tmp/boxplot_r2.png')
    plt.close()

    sns.histplot(data=all_results, x='rmse', hue='model', kde=True, bins=20,
                 alpha=0.4)
    plt.savefig('results/plots_tmp/histo_rmse.png')
    plt.close()