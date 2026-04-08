"""
MaxRM Random Forest models for worst-case risk minimization.

Implements two risk criteria on top of adaXT's RandomForest:
  - 'mse':    MaxRM-MSE — minimizes worst-group mean squared error
  - 'regret': MaxRM-Regret — minimizes worst-group regret (loss relative to
              per-environment ERM oracle)

Reference implementation: https://github.com/francescofreni/nldg/tree/main
"""

import numpy as np

from adaXT.random_forest import RandomForest
from sklearn.preprocessing import LabelEncoder

def modify_predictions(
    model,
    train_ids_int,
    risk,
    sols_erm=None,
    sols_erm_trees=None,
    n_jobs=10,
    verbose=True,
):
    """
    Try modify_predictions_trees with several solvers, then fall back to
    opt_method='extragradient'. Returns True if successful, False otherwise.
    """
    kwargs = {"method": risk, "n_jobs": n_jobs}
    if risk == "regret":
        kwargs["sols_erm"] = sols_erm
        kwargs["sols_erm_trees"] = sols_erm_trees

    solvers = [None, "ECOS", "SCS"]

    for solver in solvers:
        if verbose:
            solver_name = "default solver" if solver is None else solver
        try:
            if verbose:
                print(f"* Trying {solver_name}...")
            model.modify_predictions_trees(
                train_ids_int, **kwargs, solver=solver
            )
            return True
        except Exception as e:
            if verbose:
                print(f"* {solver_name} failed.")
                print(str(e))

    if verbose:
        print(
            f"* Fallback: all solvers failed. "
            "Retrying with opt_method='extragradient'."
        )
    try:
        model.modify_predictions_trees(
            train_ids_int, **kwargs, opt_method="extragradient"
        )
        return True
    except Exception:
        if verbose:
            print(f"* ERROR in modify_predictions_trees after all fallbacks")
        return False


class MaxRM_RF(RandomForest):
    """
    MaxRM Random Forest: adaXT RandomForest with worst-group risk minimization.

    After a standard RF fit, calls modify_predictions_trees() to re-weight
    leaf predictions so that the worst-group risk (MSE or regret) is minimized.

    Args:
        risk (str): 'mse' for MaxRM-MSE, 'regret' for MaxRM-Regret.

    Reference: https://github.com/francescofreni/nldg/tree/main
    """

    def __init__(self, n_estimators=100, seed=42,
                 min_samples_leaf=30, n_jobs=10, risk='mse'):
        params = {
            'n_estimators': n_estimators,
            'min_samples_leaf': min_samples_leaf,
            'n_jobs': n_jobs,
            'forest_type': 'Regression',
            'seed': seed,
        }
        super().__init__(**params)
        self.risk = risk
        self.n_estimators = n_estimators
        self._init_params = params

    def fit(self, X, y, envs):
        """
        Fit the forest, then adjust leaf predictions to minimize worst-group risk.

        Args:
            X: Feature matrix.
            y: Target vector.
            envs: Group/environment labels (integer-castable array).
        """
        if len(envs[0]) > 1:
            # Concat strings: [('a', 'b')] -> ["ab"]
            envs = ["".join(map(str, e)) for e in envs]

        le = LabelEncoder()
        train_ids_int = le.fit_transform(envs)

        print(f"Fitting MaxRM_RF with risk={self.risk}...")
        super().fit(X, y)

        if self.risk == 'regret':
            print("Computing ERM predictions for regret calculation...")
            sols_erm = np.zeros(len(train_ids_int))
            sols_erm_trees = np.zeros(
                (self.n_estimators, len(train_ids_int))
            )
            for env in np.unique(train_ids_int):
                mask = train_ids_int == env
                xtrain_env = X[mask]
                ytrain_env = y[mask]
                rf_env = RandomForest(**self._init_params)
                rf_env.fit(xtrain_env, ytrain_env)
                fitted_env = rf_env.predict(xtrain_env)
                sols_erm[mask] = fitted_env
                for i in range(self.n_estimators):
                    fitted_env_tree = rf_env.trees[i].predict(xtrain_env)
                    sols_erm_trees[i, mask] = fitted_env_tree

        print("Modifying predictions to minimize worst-group risk...")
        success = modify_predictions(
            model=self,
            train_ids_int=train_ids_int,
            risk=self.risk,
            sols_erm=sols_erm if self.risk == 'regret' else None,
            sols_erm_trees=sols_erm_trees if self.risk == 'regret' else None,
            n_jobs=self._init_params['n_jobs'],
            verbose=False,
        )
        if not success:
            print(f"WARNING: modify_predictions failed for MaxRM_RF with risk={self.risk}.")

    def predict(self, X):
        return super().predict(X)
