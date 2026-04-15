import argparse
import os
import pandas as pd

from sklearn.metrics import root_mean_squared_error

from dataloader import load_data, get_data_split, save_predictions
from models import get_model, get_param_grid
from tests.test_models import test_model
from utils.eval_utils import compute_and_save_metrics, save_best_params
from utils.utils import setup_logging, get_metrics_path

logger = setup_logging(__name__)

ALL_SETTINGS = [
    'time-split', 'spatial-easy', 'spatial-hard'
]
ALL_TARGETS = ['GPP', 'NEE', 'ET']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='data/',
                        help="Path to the data directory")
    parser.add_argument("--rerun", action='store_true',
                        help="Rerun existing results")
    parser.add_argument("--setting", type=str,
                        # choices=['time-split', 'spatial-easy', 'spatial-hard', 'all'],
                        default='all', help="Experiment setting")
    parser.add_argument("--target", type=str, 
                        choices=['GPP', 'NEE', 'ET', 'all'],
                        default='all', help="Target variable to predict")
    parser.add_argument("--model_name", type=str, 
                        default='lr', help="Model to use for the experiment")

    args = parser.parse_args()
    model_name = args.model_name

    # Determine which settings and targets to run
    settings = ALL_SETTINGS if args.setting == 'all' else [args.setting]
    targets = ALL_TARGETS if args.target == 'all' else [args.target]

    # Load data
    data_path = os.path.join(args.path, "daily_with_mask.csv")
    logger.info(f"Loading data from {args.path}...")
    df = load_data(args.path)

    # get param grid for the model
    param_grid = get_param_grid(model_name)

    # Run experiments
    for setting in settings:
        for target in targets:
            logger.info(f"Running experiment: setting={setting}, target={target}, model={model_name}")
            if not args.rerun and all(
                os.path.exists(get_metrics_path(setting, target, model_name, s))
                for s in ['mean', 'max', 'discrepancy']
            ):
                logger.info(f"Results for {setting}/{target}/{model_name} already exist. Use --rerun to rerun.")
                continue

            best = {s: {'score': float('inf'), 'params': None, 'model': None}
                    for s in ['mean', 'max', 'discrepancy']}

            # TODO[LATER]: this function will change depending on how we store the data, and the preprocessing we do before
            #       -> NB: before replacing, make sure to do all the data checks this function does
            # train, val, xtest = get_data_split(
            # Load data once per setting/target
            train, val, test = get_data_split(
                df,
                setting,
                target=target,
                remove_missing_target=True,
                path=args.path,
            )
            xtrain, ytrain, envs_train = train
            xval, yval, envs_val = val
            xtest = test[0]

            logger.info(f"Starting Grid Search for {model_name} on {setting}-{target}...")

            # Hyperparameter tuning loop
            for i, params in enumerate(param_grid):
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
                val_df = pd.DataFrame({'y_true': yval, 'y_pred': val_preds, 'env': envs_val.values})
                site_rmse = val_df.groupby('env')[['y_true', 'y_pred']].apply(
                    lambda g: root_mean_squared_error(g['y_true'], g['y_pred'])
                )
                scores = {
                    'mean': site_rmse.mean(),
                    'max': site_rmse.max(),
                    'discrepancy': site_rmse.max() - site_rmse.min(),
                }
                for s, score in scores.items():
                    if score < best[s]['score']:
                        best[s].update({'score': score, 'params': params, 'model': model})
                logger.info(
                    f"Params {i+1}/{len(param_grid)} complete. "
                    f"mean={scores['mean']:.4f}, max={scores['max']:.4f}, "
                    f"discrepancy={scores['discrepancy']:.4f}"
                )

            # Final evaluation on test set with best model per strategy
            logger.info("Model selection complete.")
            logger.info(f"Saving predictions and metrics for {setting}/{target}/{model_name}...")
            for strategy, b in best.items():
                logger.info(f"  [{strategy}] Best val score: {b['score']:.4f}, params: {b['params']}")
                save_best_params(b['params'], setting, target, model_name, val_strategy=strategy)
                ypred = b['model'].predict(xtest)
                preds_df = save_predictions(test, ypred, setting, target, model_name, val_strategy=strategy)
                compute_and_save_metrics(preds_df, setting, target, model_name, val_strategy=strategy)
