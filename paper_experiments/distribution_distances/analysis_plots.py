import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import HistGradientBoostingRegressor

import sys

from analysis_utils import (
    split_idx_2way, 
    split_idx_3way, 
    fit_domain_clf, 
    density_ratio
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dataloader import get_data_split

colors = {'train': '#6B8E23', 'test': '#FF5200', 'val': '#7B68EE'}

######## Feature Distribution Histograms #######

def plot_feature_histograms(df, data_path, settings, variables, 
                            combined_plot=False,
                            style_path='../neurips.mplstyle'):
    """Generates individual density histograms for given variables and settings."""
    
    # 1. Apply the custom NeurIPS Matplotlib style 
    if os.path.exists(style_path):
        plt.style.use(style_path)
    else:
        print(f"Warning: Style file not found at {style_path}. Using default style.")

    # 2. Setup output directory
    out_dir = "plots/histograms"
    os.makedirs(out_dir, exist_ok=True)

    if combined_plot:
        assert len(variables) == 1, "Combined plot mode only supports one variable at a time."

    # 3. Define colors (purple is stored for future use, val is dropped from plotting)
    for i, setting in enumerate(settings):
        # Load the split for the current setting
        train_split, val_split, test_split, x_cols, _ = get_data_split(
            df, setting, target='ET', remove_missing_target=True,
            return_colnames=True, path=data_path
        )

        # Combine train and val into a single 'train' entity
        xtrain_combined = np.vstack([train_split[0], val_split[0]])
        xtest = test_split[0]

        for var in variables:
            if var not in x_cols:
                print(f"Variable '{var}' not found in {setting} columns. Skipping.")
                continue

            var_idx = x_cols.index(var)

            # Extract specific column data
            x_train_var = xtrain_combined[:, var_idx]
            x_test_var = xtest[:, var_idx]

            # Create tidy DataFrame for Seaborn
            df_plot = pd.DataFrame({
                var: np.concatenate([x_train_var, x_test_var]),
                'split': ['train'] * len(x_train_var) + ['test'] * len(x_test_var)
            })

            # 4. Generate the individual plot
            figsize = (1.5, 1.5) if not combined_plot else (1.5, 1)
            fig, ax = plt.subplots(figsize=figsize)
            
            sns.histplot(
                data=df_plot, 
                x=var, 
                hue='split', 
                ax=ax, 
                kde=True,
                linewidth=0, 
                stat='density', 
                common_norm=False,
                palette=colors, 
                alpha=0.5,
                legend=False
            )
            if not combined_plot:
                ax.set_xlabel(var)
                ax.set_ylabel('Density')
            else: 
                ax.set_xlabel('')
                ax.set_ylabel('Density', color='none' if i>0 else 'k')
                if i > 0:
                    ax.tick_params(axis='y', labelcolor='none')

            # 5. Save the plot
            plt.tight_layout()
            save_path = os.path.join(out_dir, f"{var}_{setting}.pdf")
            plt.savefig(save_path)
            plt.close(fig) 
            print(f"  Saved plot: {save_path}")


####### Conditional Slice Plots #######

# ── 1. Helpers ─────────────────────────────────────────────────────────────

def binned_mean_with_se(x, y, edges, weights=None):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    weights = np.ones_like(y, dtype=float) if weights is None else np.asarray(weights).ravel()

    x_mid, y_mean, y_se = [], [], []
    for b in range(len(edges) - 1):
        mask = (x >= edges[b]) & (x < edges[b + 1])
        if not np.any(mask):
            continue
            
        w = weights[mask]
        y_bin = y[mask]
        x_bin = x[mask]
        
        sum_w = np.sum(w)
        if sum_w == 0:
            continue
            
        x_m = np.sum(w * x_bin) / sum_w
        y_m = np.sum(w * y_bin) / sum_w
        
        sum_w_sq = np.sum(w ** 2)
        n_eff = (sum_w ** 2) / sum_w_sq if sum_w_sq > 0 else 0
        
        if n_eff > 1:
            var_w = np.sum(w * (y_bin - y_m) ** 2) / sum_w
            se = np.sqrt(var_w / n_eff)
        else:
            se = 0.0 
            
        x_mid.append(x_m)
        y_mean.append(y_m)
        y_se.append(se)
        
    return np.array(x_mid), np.array(y_mean), np.array(y_se)


# ── 2. Plotting Logic ──────────────────────────────────────────────────────
def conditional_slice_plot(xtrain, ytrain, xtest, ytest, feature_idx, 
                           feature_name=None, n_bins=10, ax=None, 
                           show_legend=False, target=None, support_eps=0.01,
                           mode='model'):
    """
    mode: 'model' fits a regressor and plots predicted f(x).
          'raw' plots the raw target y.
    """
    rng = np.random.default_rng(42)

    
    if mode == 'model':
        # 1. Split data 3 ways: Domain Clf (dom), Model Fit (fit), Evaluation (eval)
        idx_train_dom, idx_train_fit, idx_train_eval = split_idx_3way(len(xtrain), rng)
        idx_test_dom, idx_test_fit, idx_test_eval = split_idx_3way(len(xtest), rng)
    else:
        # If we're just plotting raw y, we only need a 2-way split for the domain classifier and evaluation
        idx_train_dom, idx_train_eval = split_idx_2way(len(xtrain), rng)
        idx_test_dom, idx_test_eval = split_idx_2way(len(xtest), rng)

    xtrain_dom, xtest_dom = xtrain[idx_train_dom], xtest[idx_test_dom]
    xtrain_eval, ytrain_eval = xtrain[idx_train_eval], ytrain[idx_train_eval]
    xtest_eval, ytest_eval = xtest[idx_test_eval], ytest[idx_test_eval]
    if mode == 'model':
        xtrain_fit, ytrain_fit = xtrain[idx_train_fit], ytrain[idx_train_fit]
        xtest_fit, ytest_fit = xtest[idx_test_fit], ytest[idx_test_fit]
        # check if there are nan in the fit splits
        if np.isnan(xtrain_fit).any() or np.isnan(ytrain_fit).any():
            print("Warning: NaN values found in training fit split. This may cause issues with model fitting.")
        if np.isnan(xtest_fit).any() or np.isnan(ytest_fit).any():
            print("Warning: NaN values found in test fit split. This may cause issues with model fitting.")


    # 2. Extract values to bin based on the mode
    if mode == 'model':
        # Fit regressors to get f_train and f_test
        reg_train = HistGradientBoostingRegressor(max_iter=100, max_depth=6, random_state=42)
        reg_train.fit(xtrain_fit, ytrain_fit)
        
        reg_test = HistGradientBoostingRegressor(max_iter=100, max_depth=6, random_state=42)
        reg_test.fit(xtest_fit, ytest_fit)

        # Predict f(X) on the eval splits
        val_train = reg_train.predict(xtrain_eval)
        val_test = reg_test.predict(xtest_eval)
        ylabel = 'Avg Prediction f(X)'
    elif mode == 'raw':
        # Bypass modeling, use raw target values from the evaluation split
        val_train = ytrain_eval
        val_test = ytest_eval
        ylabel = f'Avg {target}' if target is not None else 'Avg Target y'
    else:
        raise ValueError("mode must be either 'model' or 'raw'")

    # 3. Isolate the REMAINING features for importance weighting
    rem_idx = [i for i in range(xtrain.shape[1]) if i != feature_idx]
    
    xtrain_dom_rem = xtrain_dom[:, rem_idx]
    xtest_dom_rem = xtest_dom[:, rem_idx]
    xtrain_eval_rem = xtrain_eval[:, rem_idx]
    xtest_eval_rem = xtest_eval[:, rem_idx]

    # 4. Fit domain classifier on REMAINING features
    dom_clf_rem, _, _ = fit_domain_clf(xtrain_dom_rem, xtest_dom_rem)

    # Calculate density ratios based on remaining features
    w_test, _ = density_ratio(
        dom_clf_rem, xtest_eval_rem, 
        source='test_to_train',
        n_train_dom=len(xtrain_dom), n_test_dom=len(xtest_dom)
    )

    # Enforce overlapping support based on the background distribution
    # gives the probability of being in the test set based on the remaining features
    s_train = dom_clf_rem.predict_proba(xtrain_eval_rem)[:, 1]
    s_test = dom_clf_rem.predict_proba(xtest_eval_rem)[:, 1]

    mask_train = s_train > support_eps
    mask_test = s_test < (1 - support_eps)

    # Apply masks to the features, the weights, and the values (Y or f(X))
    xhold = xtrain_eval[mask_train, feature_idx]
    val_train_masked = val_train[mask_train]
    
    xtest_feat = xtest_eval[mask_test, feature_idx]
    val_test_masked = val_test[mask_test]
    w_test_masked = w_test[mask_test]

    # 5. Bin the target feature and calculate the average
    p05 = np.max([np.percentile(xhold, 5), np.percentile(xtest_feat, 5)])
    p95 = np.min([np.percentile(xhold, 95), np.percentile(xtest_feat, 95)])
    
    edges = np.linspace(p05, p95, n_bins + 1)
    edges[0] -= 1e-12 
    edges[-1] += 1e-12

    x_train_bin, v_mean_train, se_train = binned_mean_with_se(xhold, val_train_masked, edges)
    x_test_bin, v_mean_test, se_test = binned_mean_with_se(xtest_feat, val_test_masked, edges)
    x_test_w_bin, v_mean_test_w, se_test_w = binned_mean_with_se(
        xtest_feat, val_test_masked, edges, weights=w_test_masked)

    # 6. Plotting
    markersize, capsize, linewidth = 0.5, 2, 0.75
    elinewidth = 0.5         
    markeredgewidth = 0.5    
    
    label_prefix = 'f(x)' if mode == 'model' else 'y'
    
    ax.errorbar(x_train_bin, v_mean_train, yerr=1.96*se_train, marker='o',
                markersize=markersize, label=f'Train {label_prefix}', capsize=capsize,
                elinewidth=elinewidth, markeredgewidth=markeredgewidth, 
                color=colors['train'], linestyle='-', linewidth=1)
    
    ax.errorbar(x_test_bin, v_mean_test, yerr=1.96*se_test, marker='s', 
                markersize=markersize, label=f'Test {label_prefix}', capsize=capsize, 
                elinewidth=elinewidth, markeredgewidth=markeredgewidth,
                color="#CACACA", linestyle='--', linewidth=linewidth)
    
    ax.errorbar(x_test_w_bin, v_mean_test_w, yerr=1.96*se_test_w, marker='D', 
                markersize=markersize, label=f'w-Test {label_prefix}', capsize=capsize, 
                elinewidth=elinewidth, markeredgewidth=markeredgewidth,
                color=colors['test'], linestyle='-', linewidth=linewidth)

    ax.set_xlabel(feature_name if feature_name is not None else f'Feature {feature_idx}')
    ax.set_ylabel(ylabel)
        
    if show_legend:
        ax.legend(frameon=False)


# ── 3. Main Plotting Loop ──────────────────────────────────────────────────
def generate_conditional_plots(df, data_path, settings, variables, 
                               combined_plot=False,
                               style_path='../neurips.mplstyle',
                               support_eps=0.01, target='ET', mode='model'):
    
    if os.path.exists(style_path):
        plt.style.use(style_path)
    
    out_dir = f"plots/conditional_{mode}"
    os.makedirs(out_dir, exist_ok=True)

    if combined_plot:
        assert len(variables) == 1, "Combined plot mode only supports one variable at a time."

    for i, setting in enumerate(settings):
        train_split, val_split, test_split, x_cols, _ = get_data_split(
            df, setting, target=target, remove_missing_target=True,
            return_colnames=True, path=data_path
        )

        # Combine Train and Val
        xtrain_combined = np.vstack([train_split[0], val_split[0]])
        ytrain_combined = np.concatenate([np.asarray(train_split[1]).ravel(), np.asarray(val_split[1]).ravel()])
        xtest = test_split[0]
        ytest = np.asarray(test_split[1]).ravel()
        mtrain = np.isfinite(xtrain_combined).all(axis=1) & np.isfinite(ytrain_combined)
        mtest  = np.isfinite(xtest).all(axis=1) & np.isfinite(ytest)
        xtrain_combined, ytrain_combined = xtrain_combined[mtrain], ytrain_combined[mtrain]
        xtest, ytest = xtest[mtest], ytest[mtest]

        
        for var in variables:
            if var not in x_cols:
                continue

            var_idx = x_cols.index(var)

            figsize = (1.5, 1.5) if not combined_plot else (1.5, 1)
            fig, ax = plt.subplots(figsize=figsize)
            
            conditional_slice_plot(
                xtrain=xtrain_combined, 
                ytrain=ytrain_combined, 
                xtest=xtest, 
                ytest=ytest, 
                feature_idx=var_idx, 
                feature_name=var, 
                n_bins=10, 
                ax=ax,
                show_legend=False,
                target=target,
                support_eps=support_eps,
                mode=mode
            )

            if combined_plot:
                ax.set_xlabel('')
                ax.set_ylabel(f'Mean {target}', color='none' if i>0 else 'k')
                if i > 0:
                    ax.tick_params(axis='y', labelcolor='none')
                
            plt.tight_layout()
            save_path = os.path.join(out_dir, f"{var}_{setting}.pdf")
            plt.savefig(save_path)
            plt.close(fig)
            print(f"  Saved conditional {mode} plot: {save_path}")