import os
import numpy as np
import pandas as pd
import torch

from utils.utils import setup_logging, get_predictions_path, load_csv, save_csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

logger = setup_logging(__name__)

# -----------------------------------------------------------------------
# --------- Predefined spatial groups for "spatial" CV setting ----------
# -----------------------------------------------------------------------

# 25 most southern sites

SOUTHERN_SITES = [
    'AR-CCg', 'AR-TF1',
    'AU-ASM', 'AU-Boy', 'AU-Cpr', 'AU-Cum', 'AU-DaS',
    'AU-Dry', 'AU-GWW', 'AU-Lit', 'AU-Lon' ,'AU-Rgf', 
    'AU-War', 'AU-Whr', 'AU-Wom',
    'BR-Npw', # Brazil
    'CL-SDF', # Chile
    'IL-Yat', # Israel
    'PE-QFR', # Peru
    'US-xPU', 'US-ONA', 'US-SRS', 'US-KS3', 'US-SRM', 'US-SP1'
]

# Spatial split for 100 sites

G1 = ['US-Tw1', 'US-Snf', 'CH-Cha', 'DE-Rns', 'US-xDL', 'US-Tw5', 'US-KLS', 'US-Rpf', 'CL-SDF', 'US-Tw3', 'AU-Dry', 'UK-AMo', 'US-SSH', 'US-Tw4', 'DK-Skj', 'US-Sne', 'US-Seg', 'US-xGR', 'RU-Fy2', 'AR-CCg', 'US-CGG', 'RU-Fyo', 'FR-Hes', 'FR-Bil', 'DE-Hai']
G2 = ['US-CS3', 'US-A74', 'CZ-BK1', 'CA-TP3', 'SE-Nor', 'DE-Gri', 'US-CS2', 'CA-HPC', 'CA-BOU', 'US-StJ', 'DE-Hzd', 'FR-Tou', 'AU-Whr', 'ES-Abr', 'US-xWD', 'US-CS5', 'AU-GWW', 'DE-RuR', 'US-RGo', 'BE-Dor', 'US-Syv', 'US-EDN', 'US-CS1', 'IT-BCi', 'US-CS4']
G3 = ['AR-TF1', 'AT-Neu', 'AU-ASM', 'AU-Boy', 'AU-Cpr', 'AU-Cum', 'AU-DaS', 'AU-Lit', 'AU-Lon', 'AU-Rgf', 'AU-War', 'AU-Wom', 'BR-Npw', 'CA-Cbo', 'NL-Loo', 'PE-QFR', 'CA-DB2', 'CA-DBB', 'CA-EM1', 'CA-ER1', 'CA-LP1', 'CA-SCB', 'CA-SCC', 'CA-TP1', 'CA-TPD']
G4 = ['CH-Dav', 'CZ-KrP', 'CZ-Lnz', 'CZ-RAJ', 'BE-Bra', 'BE-Lon', 'CZ-Stn', 'CZ-wet', 'DE-Geb', 'DE-HoH', 'DE-Kli', 'DE-Msr', 'DE-Obe', 'DE-RuS', 'DE-Tha', 'DK-Sor', 'DK-Vng', 'ES-Agu', 'ES-LJu', 'ES-LM1', 'ES-LM2', 'FI-Hyy', 'FI-Let', 'FI-Qvd', 'FI-Sii']
G1.sort()
G2.sort()
G3.sort()
G4.sort()

# For all sites we also use the following

G5 = ['BE-Maa', 'BE-Vie', 'FI-Var', 'FR-Aur', 'FR-CLt', 'FR-EM2', 'FR-FBn', 'FR-Fon', 'FR-Gri', 'FR-Lam', 'FR-Mej', 'FR-Pue', 'IE-Cra', 'IL-Yat', 'IT-BFt', 'IT-Cp2', 'IT-Lav', 'IT-Lsn', 'IT-MBo', 'IT-MtM', 'IT-MtP', 'IT-Noe', 'IT-Ren', 'IT-SR2', 'IT-Tor']
G6 = ['IT-TrF', 'JP-BBY', 'RU-Fy3', 'SE-Htm', 'SE-Lnn', 'SE-Ros', 'SE-St1', 'SE-Svb', 'US-A32', 'US-ALQ', 'US-ARM', 'US-Akn', 'US-BRG', 'US-BZB', 'US-BZF', 'US-BZS', 'US-BZo', 'US-Bar', 'US-Bi1', 'US-Bi2', 'US-CF1', 'US-CF2', 'US-CF3', 'US-CF4', 'US-CS6']
G7 = ['US-CdM', 'US-Cst', 'US-DFC', 'US-DS3', 'US-Dmg', 'US-EML', 'US-GLE', 'US-HB1', 'US-HB2', 'US-HB3', 'US-HWB', 'US-Hn2', 'US-Hn3', 'US-Ho1', 'US-Ho2', 'US-Jo2', 'US-KFS', 'US-KS3', 'US-Kon', 'US-MMS', 'US-MOz', 'US-Me2', 'US-Me6', 'US-Mo1', 'US-Mo2']
G8 = ['US-Mo3', 'US-Mpj', 'US-NC3', 'US-NC4', 'US-NGB', 'US-NGC', 'US-NR1', 'US-Ne1', 'US-ONA', 'US-ORv', 'US-Pnp', 'US-RGA', 'US-RGB', 'US-RGW', 'US-RRC', 'US-Rls', 'US-Rms', 'US-Ro1', 'US-Ro2', 'US-Ro4', 'US-Ro5', 'US-Ro6', 'US-Rwf', 'US-Rws', 'US-SP1']
G9 = ['US-SRG', 'US-SRM', 'US-SRS', 'US-Ses', 'US-Srr', 'US-Ton', 'US-UC1', 'US-UC2', 'US-UMB', 'US-UMd', 'US-Var', 'US-Vcm', 'US-Vcp', 'US-Whs', 'US-Wkg', 'US-YK2', 'US-xAB', 'US-xAE', 'US-xBA', 'US-xBL', 'US-xBN', 'US-xBR', 'US-xCL', 'US-xCP', 'US-xDC']
G10 = ['US-xDJ', 'US-xDS', 'US-xHE', 'US-xJE', 'US-xJR', 'US-xKA', 'US-xKZ', 'US-xMB', 'US-xML', 'US-xNG', 'US-xNQ', 'US-xRM', 'US-xSB', 'US-xSC', 'US-xSE', 'US-xSJ', 'US-xSL', 'US-xSR', 'US-xST', 'US-xTA', 'US-xTR', 'US-xUK', 'US-xUN', 'US-xYE']
G5.sort()
G6.sort()
G7.sort()
G8.sort()
G9.sort()
G10.sort()

# ---------------------------------------------------
# ------------------ Loading data -------------------
# ---------------------------------------------------

DROP_COLS = ['PFT_BSV', 'PFT_SNO', 'PFT_URB']

def load_data(path):
    """Load the data from the specified path."""
    # each file in this folder is a site, and we want to load them all and concatenate into one dataframe
    path = os.path.join(path, "sites")
    dfs = []
    for filename in os.listdir(path):
        if filename.endswith(".csv"):
            site_id = filename.split(".")[0]
            df_site = pd.read_csv(os.path.join(path, filename))
            df_site["site_id"] = site_id
            dfs.append(df_site)
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop(columns=DROP_COLS)

    bool_cols = df.select_dtypes(include='bool').columns
    for col in bool_cols:
        nunique = df[col].nunique(dropna=False)
        assert nunique == 2, f"Expected boolean column {col} to have exactly 2 unique values, but found {nunique}"
    return df


# -----------------------------------------------------------------------
# ------------------ Functions for getting fold data --------------------
# -----------------------------------------------------------------------

def get_data_split(
    df,
    setting,
    path,
    target="GPP",
    remove_missing_target=False,
    keep_lonlat=False,
    keep_time=False,
    astorch=False,
    return_colnames=False
):
    """
    Get the train/test data for a specific setting.
    Args:
        df (pd.DataFrame): The input dataframe containing the data.
        setting (str): The cross-validation setting.
        target (str, optional): The target variable name.
            Defaults to "GPP".
        remove_missing_target (bool, optional): Whether to remove rows with
            missing target values. Defaults to False.
        keep_lonlat (bool, optional): Whether to keep longitude and latitude
            features. Defaults to False.
        keep_time (bool, optional): Whether to keep time feature. Defaults to False.
        astorch (bool, optional): Whether to return data as PyTorch tensors.
            Defaults to False.
    Returns:
        tuple: xtrain, ytrain, envs_train, xtest, ytest, envs_test
    """
    # Subset the correct data
    if setting == "time-split":
        sites_to_keep = pd.read_csv(os.path.join(BASE_DIR, "data/sites_with_2018.csv"))
        df_out = df.loc[df["site_id"].isin(sites_to_keep['site_id'].values)].copy()
    else: # setting in ["spatial-easy", "spatial-hard"]:
        df_out = df.copy()
    # else:
    #     raise ValueError(f"Setting `{setting}` not recognized in get_data_split")

    # Preserve time column if needed for metadata
    time_col = df_out["time"].copy()

    # drop columns
    cols_to_drop = []
    if not keep_lonlat:
        cols_to_drop += ["tower_lat", "tower_lon"]
    if not keep_time:
        cols_to_drop += ["time"]
    for col in cols_to_drop:
        if col in df_out.columns:
            df_out.drop(columns=[col], inplace=True)

    # split into train/test
    if setting == "time-split":
        df_out['site_year'] = list(zip(df_out['site_id'], df_out['year']))
        # split years chronologically
        train = df_out.loc[df_out["year"] < 2018].copy()
        val = df_out.loc[df_out["year"] == 2018].copy()
        test = df_out.loc[df_out["year"] > 2018].copy()
    else:
        # get held-out group
        if setting == "spatial-easy":
            test_group, val_group = G1, G2
        elif setting == "spatial-hard":
            test_group = SOUTHERN_SITES
            # for val group, we can use the sites that are in G1-G4 but not in the test group
            val_group = [site for group in [G1, G2, G3, G4] 
                         for site in group if site not in test_group][:25]
            assert len(test_group) == len(val_group) == 25,\
                f"Expected 25 sites in test and val groups, got {len(test_group)} and {len(val_group)}"
        else:
            raise ValueError(f"Setting `{setting}` not recognized in get_data_split")

        train = df_out.loc[~df_out["site_id"].isin(test_group + val_group)].copy()
        val = df_out.loc[df_out["site_id"].isin(val_group)].copy()
        test = df_out.loc[df_out["site_id"].isin(test_group)].copy()
        if test.shape[0] == 0:
            logger.warning(f"* SKIPPING {test_group}: no test data")
            raise ValueError(f"No test data for group {test_group}")
    del df_out

    #  for columns GPP, NEE, ET, make the values np.nan where qc_mask==0
    for col in ["GPP", "NEE", "ET"]:
        train.loc[train["qc_mask"] == 0, col] = np.nan
        val.loc[val["qc_mask"] == 0, col] = np.nan
        if remove_missing_target:
            train = train.dropna(subset=[col])
            val = val.dropna(subset=[col])

    # drop rows with any missing values (excluding target if remove_missing_target is False)
    feature_cols = [col for col in train.columns if col != target]
    incomplete_train = train[feature_cols].isna().any(axis=1).sum()
    incomplete_val = val[feature_cols].isna().any(axis=1).sum()
    incomplete_test = test[feature_cols].isna().any(axis=1).sum()
    assert incomplete_train == incomplete_val == incomplete_test == 0, \
        f"Expected no missing values in features, but found {incomplete_train} in train, {incomplete_val} in val, and {incomplete_test} in test"

    # clean up
    if setting == "time-split":
        env_col = "site_year"
    else:
        env_col = "site_id"
    envs_train = train[env_col]
    envs_val = val[env_col].copy()
    envs_test = test[env_col].copy()

    # Extract metadata before dropping columns
    sites_test = test["site_id"].copy()
    times_test = time_col.loc[test.index]

    for col in ["site_id", "year", "site_year", "qc_mask"]:
        if col in train.columns:
            train = train.drop(columns=[col])
            val = val.drop(columns=[col])
            test = test.drop(columns=[col])
    train = train.astype(np.float64)
    val = val.astype(np.float64)
    test = test.astype(np.float64) 

    xcols = ~train.columns.isin(['GPP', 'NEE', 'ET'])
    ycol = train.columns == target

    # split into x,y
    xtrain, ytrain = train.values[:, xcols], train.values[:, ycol].ravel()
    xval, yval = val.values[:, xcols], val.values[:, ycol].ravel()
    xtest, ytest = test.values[:, xcols], test.values[:, ycol].ravel()

    if astorch:
        xtrain = torch.tensor(xtrain, dtype=torch.float32)
        ytrain = torch.tensor(ytrain, dtype=torch.float32).view(-1, 1)
        xval = torch.tensor(xval, dtype=torch.float32)
        yval = torch.tensor(yval, dtype=torch.float32).view(-1, 1)
        xtest = torch.tensor(xtest, dtype=torch.float32)
        ytest = torch.tensor(ytest, dtype=torch.float32).view(-1, 1)

    out = (
        (xtrain, ytrain, envs_train), 
        (xval, yval, envs_val),
        (xtest, ytest, envs_test, sites_test, times_test)
    )
    if return_colnames:
        out = out + (train.columns[xcols].tolist(), train.columns[ycol].tolist()[0])
    return out

# -----------------------------------------------------------------------
# -------------------------- Predictions I/O ----------------------------
# -----------------------------------------------------------------------


def load_predictions(setting, target, model_name, val_strategy):
    """
    Load predictions file for a given experiment.

    Args:
        setting: Experiment setting (e.g., 'spatial-easy', 'time-split')
        target: Target variable (e.g., 'GPP', 'NEE')
        model_name: Model name (e.g., 'lr', 'xgb')
        val_strategy: Validation strategy used for model selection ('mean', 'max', 'discrepancy')

    Returns:
        pd.DataFrame with y_true, y_pred, and env columns
    """
    pred_path = get_predictions_path(setting, target, model_name, val_strategy)
    df = load_csv(pred_path)
    if df is None:
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")
    return df


def save_predictions(test, ypred, setting, target, model_name, val_strategy):
    """Save predictions DataFrame to CSV."""
    # TODO: add mask?
    xtest, ytest, envs_test, sites_test, times_test = test
    predictions_df = pd.DataFrame({
        'y_true': ytest,
        'y_pred': ypred,
        'env': envs_test,
        'site_id': sites_test,
        'time': times_test,
        # 'mask': mask,
    })

    pred_path = get_predictions_path(setting, target, model_name, val_strategy)
    save_csv(predictions_df, pred_path)
    return predictions_df