import argparse
import os
import pandas as pd

from dataloader import get_data_split, save_predictions
from models import get_model, get_default_params
from utils.eval_utils import compute_and_save_metrics
from utils.utils import setup_logging, get_exp_name, get_metrics_path

logger = setup_logging(__name__)

ALL_SETTINGS = ['time-split', 'spatial-easy', 'spatial-hard']
ALL_TARGETS = ['GPP', 'NEE', 'Qle']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='data/',
                        help="Path to the data directory")
    parser.add_argument("--override", action='store_true',
                        help="Override existing results")
    parser.add_argument("--setting", type=str,
                        choices=['time-split', 'spatial-easy', 'spatial-hard', 'all'],
                        default='time-split', help="Experiment setting")
    parser.add_argument("--target", type=str, choices=['GPP', 'NEE', 'Qle', 'all'],
                        default='all', help="Target variable to predict")
    parser.add_argument("--model_name", type=str, choices=['xgb', 'lr'],
                        default='lr', help="Model to use for the experiment")

    args = parser.parse_args()
    model_name = args.model_name

    # Determine which settings and targets to run
    settings = ALL_SETTINGS if args.setting == 'all' else [args.setting]
    targets = ALL_TARGETS if args.target == 'all' else [args.target]

    # Get model parameters
    params = get_default_params(model_name)

    # Run experiments
    for setting in settings:
        for target in targets:
            exp_name = get_exp_name(setting, target, model_name)
            metrics_path = get_metrics_path(setting, target, model_name)

            if os.path.exists(metrics_path) and (not args.override):
                logger.info(f"Results for {exp_name} already exist. Use --override to overwrite.")
                continue

            # Load data
            data_path = os.path.join(args.path, "daily.csv")
            logger.info(f"Loading data from {data_path}...")
            df = pd.read_csv(data_path).dropna(subset=[target])

            # TODO[LATER]: this function will change depending on how we store the data, and the preprocessing we do before
            #       -> NB: before replacing, make sure to do all the data checks this function does
            xtrain, ytrain, envs_train, xtest, ytest, envs_test = get_data_split(
                df,
                setting,
                remove_missing=True,
                target=target,
            )

            # Get model
            model = get_model(model_name, params=params)
            model.fit(xtrain, ytrain)

            # Evaluate model
            ypred = model.predict(xtest)

            # Save results and predictions
            preds_df = save_predictions(df, ypred, setting, target, model_name)
            metrics_df = compute_and_save_metrics(preds_df, setting, 
                                                  target, model_name)
