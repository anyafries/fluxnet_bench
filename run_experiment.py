import argparse
import logging
import numpy as np
import os
import pandas as pd

from dataloader import generate_fold_info, get_fold_df
from eval import evaluate_fold
from models import get_model, get_default_params

# USEFUL LINKS:
# https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-and-model-selection
# https://scikit-learn.org/stable/modules/grid_search.html#grid-search

# Set up root logger (only once)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='data/', 
                        help="Path to the data directory")
    parser.add_argument("--override", action='store_true', 
                        help="Override existing results")
    parser.add_argument("--agg", type=str, 
                        choices=['daily', 'daily-2017', 'daily-100-2017'], 
                        default='daily', help="Data aggregation level")
    parser.add_argument("--setting", type=str,
                        choices=['time-split', 'spatial-easy', 'spatial-hard'],
                        default='time-split', help="Experiment setting")
    parser.add_argument("--target", type=str, choices=['GPP', 'NEE', 'Qle'],
                        default='GPP', help="Target variable to predict")
    parser.add_argument("--model_name", type=str, choices=['xgb', 'lr'],
                        default='xgb', help="Model to use for the experiment")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Custom name for the experiment")
    parser.add_argument("--params", type=str, default=None,
                        help="Path to parameter file for the model")
    
    args = parser.parse_args()
    path = args.path
    override = args.override
    agg = args.agg
    setting = args.setting
    target = args.target
    model_name = args.model_name
    exp_name = args.experiment_name
    params = args.params

    if exp_name is None:
        exp_name = f"{agg}_{setting}_{target}_{model_name}"
    outfile = f"results/{exp_name}.csv"

    if not os.path.exists('results'):
        os.makedirs('results', exist_ok=True)
    if os.path.exists(outfile) and (not override):
        logging.info(f"Results for experiment {exp_name} already exist at {outfile}. Use --override to overwrite.")
        exit(0)

    # Get model parameters
    if params is not None:
        # TODO: check this code
        params = pd.read_csv(params)
        params = params.to_dict(orient='records')[0]
    else: 
        params = get_default_params(model_name)

    # Load data
    data_path = os.path.join(path, f"{agg}.csv")
    logging.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, index_col=0).reset_index(drop=True).\
        dropna(subset=[target])
    
    # Set-up groups
    groups = generate_fold_info(df, setting)
    results = []    

    # Min samples per environment (site)
    # TODO: remove later, should be handled by the cleaned data
    min_samples_per_env = None
    if setting in ["insite", "insite-random", "time-split"]:
        min_samples_per_env = 100
    elif setting == "insite-monthly":
        min_samples_per_env = 20
    elif setting in ["spatial", "spatial-easy", "spatial-hard"]:
        min_samples_per_env = 100

    # Run experiment
    for group_id, group in enumerate(groups):
        logging.info(f"Running group: {group}...")
        xtrain, ytrain, xtest, ytest, train_ids, test_ids = get_fold_df(
            df,
            setting,
            group,
            remove_missing=True,
            target=target,
            min_samples=min_samples_per_env
        )
        if xtrain is None: continue

        # Get model
        model = get_model(model_name, params=params)
        model.fit(xtrain, ytrain)

        # TODO: handle missing values properly in data already
        feature_mask = ~np.isnan(xtest).any(axis=1)
        xtest_filtered = xtest[feature_mask]
        ytest_filtered = ytest[feature_mask]
        test_ids_filtered = test_ids[feature_mask]

        # Evaluate model
        ypred = model.predict(xtest_filtered)
        res = evaluate_fold(ytest_filtered, ypred, test_ids_filtered,
                            verbose=True, digits=3)
        res['group'] = str(group)
        res['group_id'] = group_id
        results.append(res) 

    # Save results
    logging.info(f"Saving results to {outfile}")
    results_df = pd.concat(results, ignore_index=True)
    results_df.to_csv(outfile, index=False)
