import argparse
import os
import pandas as pd

from sklearn.metrics import mean_squared_error

from dataloader import get_data_split, save_predictions
from models import get_model, get_param_grid
from tests.test_models import test_model
from utils.eval_utils import compute_and_save_metrics, save_best_params
from utils.utils import setup_logging, get_exp_name, get_metrics_path

logger = setup_logging(__name__)

ALL_SETTINGS = ['time-split', 'spatial-easy', 'spatial-hard']
ALL_TARGETS = ['GPP', 'NEE', 'Qle']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='data/',
                        help="Path to the data directory")
    parser.add_argument("--rerun", action='store_true',
                        help="Rerun existing results")
    parser.add_argument("--setting", type=str,
                        choices=['time-split', 'spatial-easy', 'spatial-hard', 'all'],
                        default='all', help="Experiment setting")
    parser.add_argument("--target", type=str, 
                        choices=['GPP', 'NEE', 'Qle', 'all'],
                        default='all', help="Target variable to predict")
    parser.add_argument("--model_name", type=str, 
                        choices=['xgb', 'lr', 'mlp', 'gdro', 'coral', 'mmd'],
                        default='lr', help="Model to use for the experiment")

    args = parser.parse_args()
    model_name = args.model_name

    # Determine which settings and targets to run
    settings = ALL_SETTINGS if args.setting == 'all' else [args.setting]
    targets = ALL_TARGETS if args.target == 'all' else [args.target]

    # Load data
    data_path = os.path.join(args.path, "daily_with_mask.csv")
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # get param grid for the model
    param_grid = get_param_grid(model_name)

    # Run experiments
    print(zip(settings, targets))
    for setting in settings:
        for target in targets:
            logger.info(f"Running experiment: setting={setting}, target={target}, model={model_name}")
            exp_name = get_exp_name(setting, target, model_name)
            metrics_path = get_metrics_path(setting, target, model_name)
            
            if os.path.exists(metrics_path) and (not args.rerun):
                logger.info(f"Results for {exp_name} already exist. Use --rerun to rerun.")
                continue

            best_score = float('inf')
            best_params = None
            best_model = None

            # TODO[LATER]: this function will change depending on how we store the data, and the preprocessing we do before
            #       -> NB: before replacing, make sure to do all the data checks this function does
            # train, val, xtest = get_data_split(
            # Load data once per setting/target
            train, val, test = get_data_split(
                df,
                setting,
                target=target,
                # only remove missing features for models that can't handle them
                remove_missing_features=model_name in ['lr', 'mlp', 'gdro', 'coral', 'mmd'],  
                remove_missing_target=True,
            )
            xtrain, ytrain, envs_train = train
            xval, yval, envs_val = val
            xtest = test[0]

            logger.info(f"Starting Grid Search for {model_name} on {setting}-{target}...")

            # Hyperparameter tuning loop
            for i, params in enumerate(param_grid):
                exp_name = get_exp_name(setting, target, model_name)
                metrics_path = get_metrics_path(setting, target, model_name)

                
                # Get model and train
                model = get_model(model_name, params)
                use_eval_set, use_envs = test_model(model)

                # Build fit arguments dynamically
                fit_args = {'X': xtrain, 'y': ytrain}
                if use_eval_set:
                    fit_args['eval_set'] = [(xval, yval)]
                if use_envs:
                    fit_args['envs'] = envs_train.values
                if model_name == 'xgb':
                    fit_args['verbose'] = False

                # Train model
                model.fit(**fit_args)

                # Evaluate on validation set
                val_preds = model.predict(xval)
                val_score = mean_squared_error(yval, val_preds) # TODO!!!!!
                logger.info(f"Params {i+1}/{len(param_grid)} complete. Val MSE: {val_score:.4f}") 

                # Track the best model
                if val_score < best_score:
                    best_score = val_score
                    best_params = params
                    best_model = model

            # Final evaluation on test set with best model
            logger.info(f"Model selection complete. Best Val MSE: {best_score:.4f}.")
            save_best_params(best_params, setting, target, model_name)
            ypred = best_model.predict(xtest)

            # Save best results and predictions
            logger.info(f"Saving predictions and metrics for {exp_name}...")
            preds_df = save_predictions(test, ypred, setting, target, model_name)
            metrics_df = compute_and_save_metrics(preds_df, setting, target, model_name)
