import argparse
import itertools
import logging
import os
import pandas as pd

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from dataloader import generate_fold_info, get_fold_df
from eval import evaluate_fold

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

def get_model(model, params={}):
    """
    Returns a model instance based on the model name and parameters.
    
    Args:
        model (str): The name of the model to instantiate.
        params (dict): Parameters to initialize the model.
    """
    if model == 'xgb':
        return XGBRegressor(**params)
    elif model == 'lr':
        return LinearRegression()
    else:
        raise NotImplementedError(f"Model `{model}` not implemented.")
    

def get_default_params(model):
    """
    Returns default parameters for the specified model.
    
    Args:
        model (str): The name of the model.
    """
    params = {}
    if model_name == 'xgb':
        params = {
            'objective': 'reg:squarederror', 
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1
        }
    # elif model_name == 'irm':
    #     params = {'n_iterations': 1000,
    #               'lr': 0.01}
    return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='data_cleaned/', 
                        help="Path to the data directory")
    parser.add_argument("--override", action='store_true', 
                        help="Override existing results")
    parser.add_argument("--agg", type=str, choices=['seasonal', 'daily', 'raw'], 
                        default='seasonal', help="Data aggregation level")
    parser.add_argument("--setting", type=str, 
                        choices=['insite', 'insite-random', 'logo', 'loso'], 
                        default='insite', help="Experiment setting")
    parser.add_argument("--start", type=int, default=0, 
                        help="Start index for the experiment")
    parser.add_argument("--stop", type=int, default=None, 
                        help="Stop index for the experiment")
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
    start = args.start
    stop = args.stop
    model_name = args.model_name
    exp_name = args.experiment_name
    params = args.params

    if exp_name is None:
        exp_name = f"{agg}_{setting}_{model_name}_start{start}_stop{stop}_cv{False}"

    # Get model parameters
    if params is not None:
        # TODO: check this code
        params = pd.read_csv(params)
        params = params.to_dict(orient='records')[0]
    else: 
        params = get_default_params(model_name)

    # Load data
    data_path = path+agg+".csv"
    logging.info("Loading data...")
    df = pd.read_csv(data_path, index_col=0).reset_index(drop=True)
    
    # Set-up groups
    groups = generate_fold_info(df, setting, start, stop)
    results = []    

    # Run experiment
    for group in groups:
        logging.info(f"Running group: {group}...")
        xtrain, ytrain, xtest, ytest, train_ids = get_fold_df(
            df, setting, group, remove_missing=True)
        if xtrain is None: continue

        # Get model
        model = get_model(model_name, params=params)
        model.fit(xtrain, ytrain)

        # Evaluate model
        ypred = model.predict(xtest)
        res = evaluate_fold(ytest, ypred, verbose=True, digits=3)
        res['group'] = group
        results.append(res) 

    # Save results
    logging.info(f"Saving results to results_{exp_name}.csv")
    results_df = pd.DataFrame(results)
    # if results folder does not exist, create it
    if not os.path.exists('results'):
        os.makedirs('results')
    results_df.to_csv(f"results/{exp_name}.csv", index=False)
