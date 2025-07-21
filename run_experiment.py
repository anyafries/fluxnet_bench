import argparse
import itertools
import logging
import os
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, GroupKFold, TimeSeriesSplit, cross_val_score, GridSearchCV
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
    

def get_default_params(model, cv=False):
    """
    Returns default parameters for the specified model.
    
    Args:
        model (str): The name of the model.
        cv (bool): Whether to use cross-validation.
    """
    params = {}
    if model_name == 'xgb':
        if cv: 
            params = {'objective': ['reg:squarederror'],
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2]}
        else:
            params = {'objective': 'reg:squarederror',
                    'n_estimators': 100,
                    'max_depth': 5,
                    'learning_rate': 0.1}

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
    parser.add_argument("--cv", action='store_true', 
                        help="Use cross-validation for the experiment")
    
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
    cv = args.cv

    if exp_name is None:
        exp_name = f"{agg}_{setting}_{model_name}_start{start}_stop{stop}_cv{cv}"

    # Get model parameters
    if params is not None:
        # TODO: check this code
        params = pd.read_csv(params)
        params = params.to_dict(orient='records')[0]
    else: 
        params = get_default_params(model_name, cv=cv)
    # if cv:
    #     keys = params.keys()
    #     values = params.values()
    #     all_param_combinations = [dict(zip(keys, combination)) 
    #                               for combination in itertools.product(*values)]


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
            df, setting, group, cv=cv, remove_missing=True)
        if xtrain is None: continue

        # Get model
        if not cv:
            model = get_model(model_name, params=params)
            model.fit(xtrain, ytrain)
        else: 
            n_splits = 5
            if xtrain.shape[0] < 2*n_splits:
                logging.warning(f"{group} has <{2*n_splits} entries, skipping.")
                continue
            if setting == 'insite': 
                # https://stats.stackexchange.com/a/268847
                folds = TimeSeriesSplit(n_splits=n_splits)
            else: 
                # TODO: use GroupKFold
                # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
                # folds = KFold(n_splits=5)
                folds = GroupKFold(n_splits=n_splits)

            clf = GridSearchCV(
                estimator=get_model(model_name),
                param_grid=params,
                scoring='neg_root_mean_squared_error',
                cv=folds 
            )
            if setting == 'insite': 
                clf.fit(xtrain, ytrain)
            else:
                clf.fit(xtrain, ytrain, groups=train_ids)
            logging.info(f"* Best parameters: {clf.best_params_}")
            model = clf.best_estimator_

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
