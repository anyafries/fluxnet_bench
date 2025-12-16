import argparse
import matplotlib.pyplot as pl
import numpy as np
import os
import pandas as pd
import sys
import traceback
import warnings
import xarray as xr
import xgboost as xgb

from matplotlib.patches import Patch
from scipy import stats as st
from sklearn.linear_model import LinearRegression

from aggregation_util import compute_diagnostics

# Import spatial groups from our dataloader
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from dataloader import G1, G2, G3, G4, G5, G6, G7, G8, G9, G10

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run QuickEval benchmark')
parser.add_argument('--target', type=str, default='GPP', choices=['GPP', 'NEE', 'LE'], help='Target variable to predict')
parser.add_argument('--subset2017', action='store_true', help='Use only 2017 data')
parser.add_argument('--subset100', action='store_true', help='Use only 100 sites')
parser.add_argument('--eval-on-train', action='store_true', help='Evaluate on training data')
parser.add_argument('--our-setting', action='store_true', help='Use our setting (mask=True, min 100 samples)')
parser.add_argument('--metric', type=str, default='MSE', choices=['MSE', 'NSE'], help='Metric to plot')
parser.add_argument('--override', action='store_true', help='Override existing results')
args = parser.parse_args()

target = args.target
SUBSET2017 = args.subset2017
SUBSET100 = args.subset100
EVAL_ON_TRAIN = args.eval_on_train
OUR_SETTING = args.our_setting
metric = args.metric
metric_range = (-0.5, 1) if metric == "NSE" else None

results_dir = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(results_dir, exist_ok=True)
# Create filename based on settings
subset_str = ""
if SUBSET2017:
    subset_str = "-2017"
if SUBSET100:
    subset_str = "-100-2017"
eval_str = "_train" if EVAL_ON_TRAIN else ""

results_file = os.path.join(
    results_dir, f'quickeval_daily{subset_str}_{target}_lr{eval_str}.csv')

if os.path.exists(results_file) and (not args.override):
    print(f"Results already exist at {results_file}. Use --override to overwrite.")
    sys.exit(0)

# Custom warning handler to see the full stack trace
def warning_with_traceback(message, category, filename, lineno, file=None, line=None):
    traceback.print_stack()
    print(f"\n{category.__name__}: {message}\n")

warnings.showwarning = warning_with_traceback
warnings.filterwarnings('always', message='Degrees of freedom')

def violins(data, pos=0, bw_method=None, resolution=50, spread=1,
                    max_n_points=1000, side='both', fill_min=None, fill_max=None):
    if data.size>max_n_points:
        data = np.random.choice(data,max_n_points,replace=False)
    kde    = st.gaussian_kde(data,bw_method=bw_method)
    pointx = data
    pointy = kde.pdf(pointx)
    pointy = pointy/(2*pointy.max())
    if fill_min is None:
        fill_min = data.min()
    if fill_max is None:
        fill_max = data.max()
    fillx  = np.linspace(fill_min, fill_max, resolution)
    filly  = kde.pdf(fillx)
    filly  = filly/(2*filly.max())
    if side == 'both':
        pointy = pos+np.where(np.random.rand(pointx.shape[0])>0.5,-1,1)*np.random.rand(pointx.shape[0])*pointy*spread
        filly  = (pos-filly*spread,pos+filly*spread)
    else:
        pointy = pos+side*np.random.rand(pointx.shape[0])*pointy*spread
        filly  = (np.zeros(filly.size)+pos, (pos+side*filly*spread))
    return(pointx, pointy, fillx, filly)


training_kwargs = dict(
    num_boost_round       = 1000,
    early_stopping_rounds = 10,
    callbacks             = None,
    verbose_eval          = None,
)


booster_kwargs = dict(
    n_jobs                = 1,
    colsample_bynode      = 1,
    learning_rate         = 0.05,
    max_depth             = 10,
    num_parallel_tree     = 1,
    objective             = 'reg:squarederror',
    subsample             = 0.5,
    tree_method           = 'hist',
    min_child_weight      = 5,
    random_state          = None,
)

tr_data = xr.open_dataset(f"data/training.nc")
eval_data = xr.open_dataset(f"data/valid.nc")
test_data = xr.open_dataset(f"data/test.nc")

if SUBSET2017:
    tr_data = tr_data.where(tr_data.time.dt.year == 2017, drop=True)
    eval_data = eval_data.where(eval_data.time.dt.year == 2017, drop=True)
    test_data = test_data.where(test_data.time.dt.year == 2017, drop=True)

if SUBSET100:
    assert eval_data.site.isin(G2).all()
    assert test_data.site.isin(G1).all()
    tr_data = tr_data.where(tr_data.site.isin(G3 + G4), drop=True)

# MAKES NO DIFFERENCE in SUBSET2017 = True, makes a different when True
if OUR_SETTING:
    # only keep where mask is True
    tr_data = tr_data.where(tr_data.mask, drop=True)
    eval_data = eval_data.where(eval_data.mask, drop=True)
    test_data = test_data.where(test_data.mask, drop=True)

    # only keep sites with at least 100 training samples
    num_obs_per_site = tr_data.mask.groupby(tr_data.site).sum()
    valid_sites = num_obs_per_site.site.values[num_obs_per_site.values >= 100]
    tr_data = tr_data.where(tr_data.site.isin(valid_sites), drop=True)
    print(f"Training sites after filtering: {len(valid_sites)}")

    num_obs_per_site = eval_data.mask.groupby(eval_data.site).sum()
    valid_sites = num_obs_per_site.site.values[num_obs_per_site.values >= 100]
    eval_data = eval_data.where(eval_data.site.isin(valid_sites), drop=True)
    print(f"Evaluation sites after filtering: {len(valid_sites)}")

    num_obs_per_site = test_data.mask.groupby(test_data.site).sum()
    valid_sites = num_obs_per_site.site.values[num_obs_per_site.values >= 100]
    test_data = test_data.where(test_data.site.isin(valid_sites), drop=True)  
    print(f"Test sites after filtering: {len(valid_sites)}")  
    print(f"Valid test samples: {test_data.mask.sum().item()}")

# print y columns
print(f"#### {target} ####")
if OUR_SETTING:
    # combine train and eval for training
    combined_data = xr.concat([tr_data, eval_data], dim='samples')
    mask = np.where(combined_data.mask.values)[0]
    x = combined_data.x[mask]
    y = combined_data.y.sel(target=target)[mask]
else:
    mask = np.where(tr_data.mask.values)[0]
    x = tr_data.x[mask]
    y = tr_data.y.sel(target=target)[mask]

print(f"Training samples: {x.shape[0]}")
print(f"Training features: {x.shape[1]}")

# # MAKES NO DIFFERENCE
# if OUR_SETTING:
#     mask = x.notnull().all(dim='features') 
#     x = x[mask]
#     y = y[mask]

dtrain = xgb.DMatrix(x.values, label=y.values)

mask = np.where(eval_data.mask.values)[0]
deval = xgb.DMatrix(eval_data.x[mask].values,
                    label = eval_data.y.sel(target=target)[mask].values)

vallist = [(dtrain, 'train'), (deval, 'eval')]

xbg_model = xgb.train(booster_kwargs, dtrain, evals=vallist, **training_kwargs)
lrg_model = LinearRegression().fit(x, y)

if EVAL_ON_TRAIN:
    if OUR_SETTING:
        x = combined_data.x
        y = combined_data.y.sel(target=target)
    else:
        x = tr_data.x
        y = tr_data.y.sel(target=target)
else:
    x = test_data.x
    y = test_data.y.sel(target=target)

print(f"Testing samples: {x.shape[0]}")

# Get mask for valid data points
mask = x.notnull().all(dim='features') 
print(f"Valid testing samples: {mask.sum().item()}")

# Make predictions
pred = (y * np.nan).load()
dpred =  xgb.DMatrix(x.values[mask])
pred[mask] = xbg_model.predict(dpred).reshape(*pred[mask].shape)
xgb_pred = pred.set_index(samples=["site", "time"]).unstack('samples')

pred = (y * np.nan).load()
pred[mask] = lrg_model.predict(x[mask]).reshape(*pred[mask].shape)
lrg_pred = pred.set_index(samples=["site", "time"]).unstack('samples')#.isel(target=0)

# collect model outputs
obs  = y.set_index(samples=["site", "time"]).unstack('samples')
if EVAL_ON_TRAIN:
    if OUR_SETTING:
        mask = combined_data.mask.set_index(samples=["site", "time"]).unstack('samples')
    else:
        mask = tr_data.mask.set_index(samples=["site", "time"]).unstack('samples')
else:
    mask = test_data.mask.set_index(samples=["site", "time"]).unstack('samples')

model_outpus = xr.merge([
    xgb_pred.rename("xgb_pred").assign_attrs(**y.attrs),
    lrg_pred.rename("lrg_pred").assign_attrs(**y.attrs),
    obs.rename("obs").assign_attrs(**y.attrs),
    mask.rename("mask")
])

### Compare each site individually
if SUBSET2017:
    aggs = ['daily', 'weekly', 'monthly'] 
else:
    aggs = ['daily', 'weekly', 'monthly', 'seasonal', 'iav', 'anom'] #, 'spatial'

all_diag_xgb = compute_diagnostics(model_outpus.obs, model_outpus.xgb_pred, 
                                    model_outpus.mask, dynamic_scales=aggs)
all_diag_lrg = compute_diagnostics(model_outpus.obs, model_outpus.lrg_pred, 
                                    model_outpus.mask, dynamic_scales=aggs)

# if SUBSET2017:
#     fig, ax = pl.subplots(1,1, figsize=(4,4))
# else:
#     fig, ax = pl.subplots(1,1, figsize=(4,8))

# if metric_range is not None:
#      fill_min, fill_max = metric_range
# else:
#     fill_min, fill_max = None, None

# for p, agg in enumerate(aggs):
#     _data = all_diag_xgb[agg][metric].values
#     # print(f"{len(_data[np.isfinite(_data)])} sites with valid {agg} {metric} for XGB")
#     if agg == 'daily':
#         median_p = np.median(_data[np.isfinite(_data)])
#         mean_p = np.mean(_data[np.isfinite(_data)])
#         max_p = np.max(_data[np.isfinite(_data)])
#         print(f"{agg} XGB {metric}: median={median_p:.3f}, mean={mean_p:.3f}, max={max_p:.3f}")

#     pointx, pointy, fillx, filly = violins(_data[np.isfinite(_data)], pos=p, side=1, spread=0.8, fill_min=fill_min, fill_max=fill_max)
#     ax.fill_between(fillx, filly[0], filly[1], alpha=0.2, color="CornFlowerBlue")
#     ax.scatter(pointx, pointy, marker='o', s=20, alpha=0.2, color="CornFlowerBlue")
#     ax.scatter(np.median(_data[np.isfinite(_data)]), p+0.25, marker="d", s=400, alpha=0.75, color="CornFlowerBlue")

#     _data = all_diag_lrg[agg][metric].values
#     # print(f"{len(_data[np.isfinite(_data)])} sites with valid {agg} {metric} for LR")
#     if agg == 'daily':
#         median_p = np.median(_data[np.isfinite(_data)])
#         mean_p = np.mean(_data[np.isfinite(_data)])
#         max_p = np.max(_data[np.isfinite(_data)])
#         print(f"{agg} LR {metric}: median={median_p:.3f}, mean={mean_p:.3f}, max={max_p:.3f}")
#     pointx, pointy, fillx, filly = violins(_data[np.isfinite(_data)], pos=p, side=-1, spread=0.8, fill_min=fill_min, fill_max=fill_max)
#     ax.fill_between(fillx, filly[0], filly[1], alpha=0.2, color="IndianRed")
#     ax.scatter(pointx, pointy, marker='o', s=20, alpha=0.2, color="IndianRed")
#     ax.scatter(np.median(_data[np.isfinite(_data)]), p-0.25, marker="d", s=400, alpha=0.75, color="IndianRed")

# if metric_range is not None:
#     ax.set_xlim(*metric_range)
# ax.set_yticks(list(range(len(aggs))))
# ax.set_yticklabels(aggs)
# ax.set_xlabel(metric)
# ax.axvline(0, color='k', lw=1, dashes=(6,4))
# ax.set_title(target)


# legend_elements = [Patch(facecolor='CornFlowerBlue', edgecolor='CornFlowerBlue',
#                          label='XGBoost'),
#                    Patch(facecolor='IndianRed', edgecolor='IndianRed',
#                          label='Linear')]

# ax.legend(handles=legend_elements, loc='upper right')

# pl.show()

# -----------------------------------------------------------------------
# Save results for comparison with our refactored code
# -----------------------------------------------------------------------

print(all_diag_lrg['daily'])
results = all_diag_lrg['daily'].to_dataframe().reset_index()
# rename columns
results = results.rename(columns={
    'MSE': 'mse',
    'RMSE': 'rmse',
    'R2': 'r2_score',
    'Relative_Error': 'relative_error',
    'MAE': 'mae',
    'NSE': 'nse',
    'site': 'env'
})

results['group'] = str(np.sort(np.unique(results['env'])).tolist())
results.to_csv(results_file, index=False)
print(f"\n[SAVED] Results saved to: {results_file}")