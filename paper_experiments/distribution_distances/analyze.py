import sys
import numpy as np
import os
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from scipy.stats import binomtest


from analysis_utils import (
    split_idx_2way, 
    split_idx_3way, 
    fit_domain_clf, 
    density_ratio, 
    weighted_rmse, 
    shared_support_weights
)
from analysis_plots import plot_feature_histograms, generate_conditional_plots

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dataloader import load_data, get_data_split

# TODO
# Currently: binomial p-value is for (raw) accuracy, decide how to report
# it, because we show the balanced accuracy
# The permutation test is for balanced accuracy

# ── Configuration ──────────────────────────────────────────────────────────
DATA_PATH    = "data/"
RANDOM_STATE = 42
SETTINGS     = ["time-split", "spatial-easy40", "TA40"]
TARGET       = "NEE"
N_ESTIMATORS = 100
MAX_DEPTH    = 6
CLIP_Q       = 0.99
SUPPORT_EPS  = 0.1
MIN_UNIQUE   = 5
N_PERM       = 1000
N_BOOT       = 1000
N_FULL_BOOT  = 10

GET_MARGINAL_METRICS = False  
GET_CONDITIONAL_METRICS = True
COMPUTE_STAT_SIG = True 
FULL_BOOTSTRAP = True

COMBINED_PLOT = True
MAKE_MARGINAL_PLOTS = False
MAKE_CONDITIONAL_PLOTS = False

# Define which variables you want to generate plots for
variables_to_plot = [
    'VPD', 
    # 'NDWI_SWIR2'
    # 'TA', 'SW_IN', 'SW_IN_POT_hourly', 'SW_IN_POT_daily_mean', 
    # 'dSW_IN_POT', 'dSW_IN_POT_DAY', 'LST_Day', 'LST_Night', 'EVI', 'NIRv', 
    
]

# ── Data ────────────────────────────────────────────────────────────────────
df  = load_data(DATA_PATH)
rng = np.random.default_rng(RANDOM_STATE)


# ── Helpers ─────────────────────────────────────────────────────────────────
def prep_data(setting):
    print(f"Preparing data for setting: {setting}")
    train, val, test, x_cols, y_cols = get_data_split(
        df, setting, target=TARGET,
        remove_missing_target=True, return_colnames=True, path=DATA_PATH,
    )
    xtrain, ytrain = train[0], np.asarray(train[1]).ravel()
    xtest,  ytest  = test[0],  np.asarray(test[1]).ravel()
    keep   = np.array([len(np.unique(xtrain[:, k])) > MIN_UNIQUE for k in range(xtrain.shape[1])])
    xtrain, xtest  = xtrain[:, keep], xtest[:, keep]
    mtrain = np.isfinite(xtrain).all(axis=1) & np.isfinite(ytrain)
    mtest  = np.isfinite(xtest).all(axis=1)  & np.isfinite(ytest)
    return xtrain[mtrain], ytrain[mtrain], xtest[mtest], ytest[mtest]


def bootstrap_rmse_drop(y_fixed, pred_fixed, y_boot, pred_boot, w_boot, n_iterations=1_000, random_state=42):
    """
    Non-parametric bootstrap significance test for the RMSE drop.
    H0: performance drop <= 0 (no conditional shift degradation).
    The drop is always: (Weighted Evaluation RMSE) - (Fixed Baseline RMSE).
    """
    fixed_rmse = np.sqrt(np.mean((y_fixed - pred_fixed) ** 2))

    boot_rng   = np.random.default_rng(random_state)
    boot_drops = []
    for _ in range(n_iterations):
        idx         = boot_rng.choice(len(y_boot), size=len(y_boot), replace=True)
        boot_w_rmse = weighted_rmse(y_boot[idx], pred_boot[idx], w_boot[idx])
        boot_drop   = boot_w_rmse - fixed_rmse
        boot_drops.append(boot_drop)

    boot_drops         = np.asarray(boot_drops)
    p_value            = np.mean(boot_drops <= 0)
    ci_lower, ci_upper = np.percentile(boot_drops, [2.5, 97.5])
    actual_w_rmse      = weighted_rmse(y_boot, pred_boot, w_boot)
    actual_drop        = actual_w_rmse - fixed_rmse
    std_drop           = np.std(boot_drops)
    
    return std_drop, ci_lower, ci_upper, p_value


def eval_setting(setting, fit_on="train", random_state=42, verbose=True, return_eval_arrays=False):
    rng = np.random.default_rng(random_state)
    xtrain, ytrain, xtest, ytest = prep_data(setting)

    # 1. Dynamically split data to maximize usage based on fit_on direction
    if fit_on == "train":
        idx_train_dom, idx_fit, idx_train_eval = split_idx_3way(len(xtrain), rng)
        idx_test_dom,  idx_test_eval  = split_idx_2way(len(xtest),  rng)
        
        xtrain_dom, ytrain_dom = xtrain[idx_train_dom], ytrain[idx_train_dom]
        xtest_dom,  ytest_dom  = xtest[idx_test_dom],  ytest[idx_test_dom]
        
        xfit, yfit = xtrain[idx_fit], ytrain[idx_fit]
        
        xtrain_eval, ytrain_eval = xtrain[idx_train_eval], ytrain[idx_train_eval]
        xtest_eval,  ytest_eval  = xtest[idx_test_eval],  ytest[idx_test_eval]
        
    elif fit_on == "test":
        idx_train_dom, idx_train_eval = split_idx_2way(len(xtrain), rng)
        idx_test_dom, idx_fit, idx_test_eval = split_idx_3way(len(xtest), rng)
        
        xtrain_dom, ytrain_dom = xtrain[idx_train_dom], ytrain[idx_train_dom]
        xtest_dom,  ytest_dom  = xtest[idx_test_dom],  ytest[idx_test_dom]
        
        xfit, yfit = xtest[idx_fit], ytest[idx_fit]
        
        xtrain_eval, ytrain_eval = xtrain[idx_train_eval], ytrain[idx_train_eval]
        xtest_eval,  ytest_eval  = xtest[idx_test_eval],  ytest[idx_test_eval]
    else:
        raise ValueError("fit_on must be 'train' or 'test'")

    # 2. Domain Classifier
    clf, auc, bal_acc = fit_domain_clf(xtrain_dom, xtest_dom, N_ESTIMATORS, MAX_DEPTH, random_state)
    n_dom_tr, n_dom_te = len(xtrain_dom), len(xtest_dom)

    # 3. Regressor Fit
    reg = HistGradientBoostingRegressor(max_iter=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=random_state)
    reg.fit(xfit, yfit)
    
    pred_train = reg.predict(xtrain_eval)
    pred_test = reg.predict(xtest_eval)

    # 4. Support Filtering
    s_train = clf.predict_proba(xtrain_eval)[:, 1]
    s_test = clf.predict_proba(xtest_eval)[:, 1]

    mask_train = s_train > SUPPORT_EPS
    mask_test = s_test < (1 - SUPPORT_EPS)
    print(f"  [{setting}] Support filtering: keeping {mask_train.mean():.2%} of train eval, {mask_test.mean():.2%} of test eval retained")

    # mask_train = (s_train > SUPPORT_EPS) & (s_train < (1 - SUPPORT_EPS))
    # mask_test  = (s_test  > SUPPORT_EPS) & (s_test  < (1 - SUPPORT_EPS))
    # print(f"  [{setting}] Strict support filtering: keeping {mask_train.mean():.2%} of train eval, {mask_test.mean():.2%} of test eval retained")

    xtrain_eval, ytrain_eval, pred_train = xtrain_eval[mask_train], ytrain_eval[mask_train], pred_train[mask_train]
    xtest_eval, ytest_eval, pred_test = xtest_eval[mask_test], ytest_eval[mask_test], pred_test[mask_test]

    # 5. Compute BOTH weights and all 4 RMSEs
    w_t2t, ess_t2t = density_ratio(clf, xtest_eval, "test_to_train", n_dom_tr, n_dom_te, CLIP_Q)
    w_tr2te, ess_tr2te = density_ratio(clf, xtrain_eval, "train_to_test", n_dom_tr, n_dom_te, CLIP_Q)
    print(f"  [{setting}] Effective Sample Size (Train->Test): {ess_tr2te:.2%}, (Test->Train): {ess_t2t:.2%}")

    w_train_shared, w_test_shared = shared_support_weights(
        clf, xtrain_eval, xtest_eval, n_dom_tr, n_dom_te, CLIP_Q
    )

    rmse_train = np.sqrt(np.mean((ytrain_eval - pred_train) ** 2))
    rmse_test = np.sqrt(np.mean((ytest_eval - pred_test) ** 2))
    
    w_rmse_test = weighted_rmse(ytest_eval, pred_test, w_t2t)
    w_rmse_train = weighted_rmse(ytrain_eval, pred_train, w_tr2te)
    shared_rmse_train = weighted_rmse(ytrain_eval, pred_train, w_train_shared)
    shared_rmse_test = weighted_rmse(ytest_eval, pred_test, w_test_shared)

    out = {
        "auc": auc, "bal_acc": bal_acc,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "w_rmse_test": w_rmse_test,   # Test weighted to look like Train
        "w_rmse_train": w_rmse_train, # Train weighted to look like Test
        "shared_rmse_train": shared_rmse_train,
        "shared_rmse_test": shared_rmse_test,
    }
    if return_eval_arrays:
        out.update({
            "ytrain_eval": ytrain_eval,
            "pred_train": pred_train,
            "ytest_eval": ytest_eval,
            "pred_test": pred_test,
            "w_t2t": w_t2t,
            "w_tr2te": w_tr2te,
        })
    return out


if GET_MARGINAL_METRICS:
    # ── Section 1: Domain Classifier AUC (P(X) shift) ───────────────────────────
    print("=" * 60)
    print("P(X) SHIFT — Domain Classifier")
    _stat_header = f"\t{'Binom P':>10}\t{'Perm P':>10}" if COMPUTE_STAT_SIG else ""
    print(f"\n{'Setting':<15}\t{'Bal. Acc.':>9}\t{'AUC':>6}{_stat_header}")
    for setting in SETTINGS:
        print(f"\nProcessing setting: {setting}")
        xtrain, ytrain, xtest, ytest = prep_data(setting)

        X = np.vstack([xtrain, xtest])
        y = np.array([0] * len(xtrain) + [1] * len(xtest))
        tr_idx = np.random.choice(len(X), int(0.5 * len(X)), replace=False)
        te_idx = np.setdiff1d(np.arange(len(X)), tr_idx)

        print(f"  [{setting}] fitting domain classifier...")
        clf = HistGradientBoostingClassifier(
            max_iter=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE
        )
        clf.fit(X[tr_idx], y[tr_idx])
        y_te   = y[te_idx]
        y_pred = clf.predict(X[te_idx])
        acc    = balanced_accuracy_score(y_te, y_pred)
        auc    = roc_auc_score(y_te, clf.predict_proba(X[te_idx])[:, 1])

        if COMPUTE_STAT_SIG:
            # Binomial test: is the raw correct-prediction count above the majority-class baseline?
            print(f"  [{setting}] running binomial test...")
            p_null    = max(np.mean(y_te == 1), np.mean(y_te == 0))
            successes = int(np.sum(y_te == y_pred))
            binom_p   = binomtest(k=successes, n=len(y_te), p=p_null, alternative='greater').pvalue

            # Permutation test: shuffle true labels against fixed predictions (no retraining)
            print(f"  [{setting}] running permutation test ({N_PERM} iterations)...")
            perm_rng  = np.random.default_rng(RANDOM_STATE)
            perm_accs = np.array([
                balanced_accuracy_score(perm_rng.permutation(y_te), y_pred)
                for _ in range(N_PERM)
            ])
            perm_p    = np.mean(perm_accs >= acc)
    
            if not FULL_BOOTSTRAP:
                # Bootstrap for confidence interval of accuracy (no retraining, just resample test set)
                print(f"  [{setting}] running bootstrap for accuracy CI ({N_BOOT} iterations)...")
                boot_rng  = np.random.default_rng(RANDOM_STATE)
                boot_accs = []
                for _ in range(N_BOOT):
                    idx = boot_rng.choice(len(y_te), size=len(y_te), replace=True)
                    boot_accs.append(balanced_accuracy_score(y_te[idx], y_pred[idx]))
                std_acc = np.std(boot_accs)
                ci_low_acc, ci_high_acc = np.percentile(boot_accs, [2.5, 97.5])
            else:
                print(f"  [{setting}] running FULL pipeline bootstrap for accuracy CI ({N_FULL_BOOT} iterations)...")
                boot_rng = np.random.default_rng(RANDOM_STATE + 1)
                boot_accs = []
                for i in range(N_FULL_BOOT):
                    # 1. New 50/50 split
                    b_tr_idx = boot_rng.choice(len(X), int(0.5 * len(X)), replace=False)
                    b_te_idx = np.setdiff1d(np.arange(len(X)), b_tr_idx)
                    
                    # 2. Re-initialize and retrain the model (different seed each time)
                    b_clf = HistGradientBoostingClassifier(
                        max_iter=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE + i
                    )
                    b_clf.fit(X[b_tr_idx], y[b_tr_idx])
                    
                    # 3. Predict and score
                    b_pred = b_clf.predict(X[b_te_idx])
                    boot_accs.append(balanced_accuracy_score(y[b_te_idx], b_pred))
                
                std_acc = np.std(boot_accs)
                ci_low_acc, ci_high_acc = np.percentile(boot_accs, [2.5, 97.5])

            acc_str = f"{acc:.4f} ± {std_acc:.4f}"
            _stat_row = f"\t{binom_p:10.4e}\t{perm_p:10.4e}"
        else:
            acc_str = f"{acc:.4f}"
            _stat_row = ""

        print(f"{setting:<15}\t{acc_str:>18}\t{auc:6.4f}{_stat_row}")


# if GET_CONDITIONAL_METRICS:
#     # ── Section 2: RMSE Increase — density-ratio reweighting (P(Y|X) shift) ─────────
#     print("\n" + "=" * 100)
#     print("P(Y|X) SHIFT — RMSE Increase Evaluated Under Both Marginals")

#     for fit_on in ["train", "test"]:
#         print(f"\n" + "-" * 105)
#         print(f"MODE: Model fit on {fit_on.upper()}")
#         print("-" * 105)

#         s2_results = {setting: eval_setting(setting, fit_on=fit_on, random_state=RANDOM_STATE, verbose=False) for setting in SETTINGS}
        
#         # Note: I've disabled the bootstrap printing here for simplicity, 
#         # but the data is easily accessible if you want to add it back!
        
#         print(f"{'Setting':<15} | {'Base (Train)':>12} | {'Base (Test)':>12} | {'Increase (Train Marg)':>18} | {'Increase (Test Marg)':>18} | {'AUC':>6}")
#         print("-" * 105)
        
#         for setting in SETTINGS:
#             r = s2_results[setting]
            
#             if fit_on == "train":
#                 # Baseline is Train. OOD is Test.
#                 # Shift under Train Marginal
#                 drop_train_marg = r["w_rmse_test"] - r["rmse_train"]
#                 pct_drop_train_marg = drop_train_marg / r["rmse_train"] * 100 if r["rmse_train"] > 0 else np.inf
                
#                 # Shift under Test Marginal
#                 drop_test_marg = r["rmse_test"] - r["w_rmse_train"]
#                 pct_drop_test_marg = drop_test_marg / r["w_rmse_train"] * 100 if r["w_rmse_train"] > 0 else np.inf
                
#             else: # fit_on == "test"
#                 # Baseline is Test. OOD is Train.
#                 # Shift under Test Marginal
#                 drop_test_marg = r["w_rmse_train"] - r["rmse_test"]
#                 pct_drop_test_marg = drop_test_marg / r["rmse_test"] * 100 if r["rmse_test"] > 0 else np.inf
                
#                 # Shift under Train Marginal
#                 drop_train_marg = r["rmse_train"] - r["w_rmse_test"]
#                 pct_drop_train_marg = drop_train_marg / r["w_rmse_test"] * 100 if r["w_rmse_test"] > 0 else np.inf
            
#             # Note: A POSITIVE drop always means the Out-of-Domain performance was WORSE 
#             # than the In-Domain performance (i.e., conditional shift degrades the model).
            
#             drop_str_tr = f"{drop_train_marg:6.4f} ({pct_drop_train_marg:5.1f}%)"
#             drop_str_te = f"{drop_test_marg:6.4f} ({pct_drop_test_marg:5.1f}%)"
            
#             print(f"{setting:<15} | {r['rmse_train']:12.4f} | {r['rmse_test']:12.4f} | {drop_str_tr:>18} | {drop_str_te:>18} | {r['auc']:6.4f}")


if GET_CONDITIONAL_METRICS:
    print("\n" + "=" * 120)
    print("P(Y|X) SHIFT — repeated random splitting plus refitting")

    n_repeats = N_FULL_BOOT  # or replace with a dedicated constant, e.g. N_COND_REPEATS

    for fit_on in ["train", "test"]:
        print(f"\n" + "-" * 130)
        print(f"MODE: Model fit on {fit_on.upper()} ({n_repeats} repeated runs)")
        print("-" * 130)

        print(
            f"{'Setting':<15} | {'RMSE Train':>18} | {'RMSE Test':>18} | "
            f"{'Increase (Train Marg)':>22} | {'Increase (Test Marg)':>22} | "
            f"{'Increase (Shared)':>22} | {'AUC':>18} | {'Bal. Acc.':>18}"
        )
        print("-" * 155)

        for setting in SETTINGS:
            runs = [
                eval_setting(
                    setting,
                    fit_on=fit_on,
                    random_state=RANDOM_STATE + i,
                    verbose=False,
                    return_eval_arrays=False,
                )
                for i in range(n_repeats)
            ]

            rmse_train_vals = np.array([r["rmse_train"] for r in runs])
            rmse_test_vals  = np.array([r["rmse_test"] for r in runs])
            auc_vals        = np.array([r["auc"] for r in runs])
            bal_acc_vals    = np.array([r["bal_acc"] for r in runs])

            if fit_on == "train":
                drop_train_marg_vals = np.array([
                    r["w_rmse_test"] - r["rmse_train"] for r in runs
                ])
                pct_drop_train_marg_vals = np.array([
                    (r["w_rmse_test"] - r["rmse_train"]) / r["rmse_train"] * 100
                    if r["rmse_train"] > 0 else np.inf
                    for r in runs
                ])

                drop_test_marg_vals = np.array([
                    r["rmse_test"] - r["w_rmse_train"] for r in runs
                ])
                pct_drop_test_marg_vals = np.array([
                    (r["rmse_test"] - r["w_rmse_train"]) / r["w_rmse_train"] * 100
                    if r["w_rmse_train"] > 0 else np.inf
                    for r in runs
                ])
                shared_drop_vals = np.array([
                    r["shared_rmse_test"] - r["shared_rmse_train"] for r in runs
                ])
                pct_shared_drop_vals = np.array([
                    (r["shared_rmse_test"] - r["shared_rmse_train"]) / r["shared_rmse_train"] * 100
                    if r["shared_rmse_train"] > 0 else np.inf
                    for r in runs
                ])

            else:  # fit_on == "test"
                drop_test_marg_vals = np.array([
                    r["w_rmse_train"] - r["rmse_test"] for r in runs
                ])
                pct_drop_test_marg_vals = np.array([
                    (r["w_rmse_train"] - r["rmse_test"]) / r["rmse_test"] * 100
                    if r["rmse_test"] > 0 else np.inf
                    for r in runs
                ])

                drop_train_marg_vals = np.array([
                    r["rmse_train"] - r["w_rmse_test"] for r in runs
                ])
                pct_drop_train_marg_vals = np.array([
                    (r["rmse_train"] - r["w_rmse_test"]) / r["w_rmse_test"] * 100
                    if r["w_rmse_test"] > 0 else np.inf
                    for r in runs
                ])

                shared_drop_vals = np.array([
                    r["shared_rmse_train"] - r["shared_rmse_test"] for r in runs
                ])
                pct_shared_drop_vals = np.array([
                    (r["shared_rmse_train"] - r["shared_rmse_test"]) / r["shared_rmse_test"] * 100
                    if r["shared_rmse_test"] > 0 else np.inf
                    for r in runs
                ])
                
            rmse_train_str = f"{rmse_train_vals.mean():.4f} ± {rmse_train_vals.std():.4f}"
            rmse_test_str  = f"{rmse_test_vals.mean():.4f} ± {rmse_test_vals.std():.4f}"
            drop_tr_str    = f"{pct_drop_train_marg_vals.mean():.2f}% ± {pct_drop_train_marg_vals.std():.2f}%"
            drop_te_str    = f"{pct_drop_test_marg_vals.mean():.2f}% ± {pct_drop_test_marg_vals.std():.2f}%"
            auc_str        = f"{auc_vals.mean():.4f} ± {auc_vals.std():.4f}"
            bal_acc_str    = f"{bal_acc_vals.mean():.4f} ± {bal_acc_vals.std():.4f}"
            shared_str = f"{pct_shared_drop_vals.mean():.2f}% ± {pct_shared_drop_vals.std():.2f}%"
            print(
                f"{setting:<15} | {rmse_train_str:>18} | {rmse_test_str:>18} | "
                f"{drop_tr_str:>22} | {drop_te_str:>22} | {shared_str:>22} | "
                f"{auc_str:>18} | {bal_acc_str:>18}"
            )


# ── Section 3: Plotting ─────────────────────────────────────────────────────
if MAKE_MARGINAL_PLOTS:
    print("\n" + "=" * 60)
    print("GENERATING HISTOGRAMS")
    # Generate the plots
    plot_feature_histograms(
        df=df, 
        data_path=DATA_PATH, 
        settings=SETTINGS, 
        variables=variables_to_plot,
        combined_plot=COMBINED_PLOT,
        style_path='../neurips.mplstyle' # Adjust this path if needed
    )
    print("Done generating histograms.")


if MAKE_CONDITIONAL_PLOTS:
    for mode in ['raw', 'model']:
        print("\n" + "=" * 60)
        print(f"GENERATING CONDITIONAL PLOTS (mode: {mode})")
        generate_conditional_plots(
            df=df, 
            data_path=DATA_PATH, 
            settings=SETTINGS, 
            variables=variables_to_plot,
            style_path='../neurips.mplstyle',
            combined_plot=COMBINED_PLOT,
            support_eps=SUPPORT_EPS,
            target=TARGET,
            mode=mode
        )
        print("Done generating conditional plots.")