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

# -----------------------------------------------------------------------
# ------------------ Functions for CV fold generation -------------------
# -----------------------------------------------------------------------

def generate_fold_info(df, setting, fold_size=5, seed=42):
    """
    Generate fold information based on the setting.
    Args:
        df (pd.DataFrame): The input dataframe containing the data.
        setting (str): The cross-validation setting.
        fold_size (int, optional): The size of each fold for certain settings. Defaults to 5.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
    Returns:
        list: A list of groups for cross-validation.
    """
    if setting in ["time-split"]:
        sites = df["site_id"].dropna().unique()
        # only keep sites with >=7 years of data
        # TODO: it should also be where each (site, year) has enough samples
        site_years = df.groupby("site_id")["year"].nunique()
        sites = site_years[site_years >= 7].index.values
        sites = sorted(sites)
        groups = [sites]

    elif setting == "spatial-easy":
        sites = pd.Series(df["site_id"].dropna().unique())
        print(f"Total sites: {len(sites)}")
        if len(sites) <= 100:
            groups = [G1, G2, G3, G4]
        else:
            groups = [G1, G2, G3, G4, G5, G6, G7, G8, G9, G10]
        
    elif setting == "spatial-hard":
        raise NotImplementedError("spatial-hard setting not implemented yet")

    return groups


# -----------------------------------------------------------------------
# ------------------ Functions for getting fold data --------------------
# -----------------------------------------------------------------------

def get_data_split(
    df,
    setting,
    target="GPP",
    remove_missing_features=False,
    remove_missing_target=False,
    keep_lonlat=False,
    astorch=False,
):
    """
    Get the train/test data for a specific setting.
    Args:
        df (pd.DataFrame): The input dataframe containing the data.
        setting (str): The cross-validation setting.
        target (str, optional): The target variable name.
            Defaults to "GPP".
        remove_missing_features (bool, optional): Whether to remove rows with 
            missing values.
            Defaults to False.
        remove_missing_target (bool, optional): Whether to remove rows with
            missing target values. Defaults to False.
        keep_lonlat (bool, optional): Whether to keep longitude and latitude
            features. Defaults to False.
        astorch (bool, optional): Whether to return data as PyTorch tensors.
            Defaults to False.
    Returns:
        tuple: xtrain, ytrain, envs_train, xtest, ytest, envs_test
    """
    # Determine min_samples based on setting
    # TODO[LATER]: should be handled by the cleaned data
    min_samples = 100

    # Subset the correct data
    if setting in ["spatial-easy", "spatial-hard"]:
        df_out = df.copy()
    elif setting == "time-split":
        # only keep sites with >=7 years of data
        # TODO: it should also be where each (site, year) has enough samples
        site_years = df.groupby("site_id")["year"].nunique()
        sites = site_years[site_years >= 7].index.values
        df_out = df.loc[df["site_id"].isin(sites)].copy()
    else:
        raise ValueError(f"Setting `{setting}` not recognized in get_data_split")

    # create time features
    if "season" in df_out.columns:
        df_out["season"] = df_out["season"].astype(int)
        df_out = pd.get_dummies(
            df_out, columns=["season"], prefix="season", dtype=np.float64
        )
    if "month" in df_out.columns:
        df_out["month_sin"] = np.sin(2 * np.pi * df_out["month"] / 12)
        df_out["month_cos"] = np.cos(2 * np.pi * df_out["month"] / 12)
    if "hour" in df_out.columns:
        df_out["hour_sin"] = np.sin(2 * np.pi * df_out["hour"] / 24)
        df_out["hour_cos"] = np.cos(2 * np.pi * df_out["hour"] / 24)

    # Preserve time column if needed for metadata
    if "time" in df_out.columns:
        time_col = df_out["time"].copy()

    # drop columns
    cols_to_drop = ["date", "hour", "time"]
    if not keep_lonlat:
        cols_to_drop += ["longitude", "latitude"]
    for col in cols_to_drop:
        if col in df_out.columns:
            df_out.drop(columns=[col], inplace=True)

    # split into train/test
    if setting == "time-split":
        df_out['site_year'] = list(zip(df_out['site_id'], df_out['year']))
        # split years chronologically
        unique_years = np.sort(df_out["year"].unique())
        train_years, val_years, test_years = unique_years[:3], unique_years[3], unique_years[4:7]
        train = df_out.loc[df_out["year"].isin(train_years)].copy()
        val = df_out.loc[df_out["year"] == val_years].copy()
        test = df_out.loc[df_out["year"].isin(test_years)].copy()
    elif setting in ["spatial-easy", "spatial-hard"]:
        # split by group of sites
        if min_samples is not None:
            site_counts = df_out["site_id"].value_counts()
            valid_sites = site_counts[site_counts >= min_samples].index
            logger.info(f"Keeping {len(valid_sites)}/{len(site_counts)} sites with >= {min_samples} samples")
            df_out = df_out.loc[df_out["site_id"].isin(valid_sites)].copy()

        # get held-out group
        if setting == "spatial-easy":
            test_group, val_group = G1, G2
        elif setting == "spatial-hard":
            test_group = SOUTHERN_SITES
            # for val group, we can use the sites that are in G1-G4 but not in the test group
            val_group = [site for group in [G1, G2, G3, G4] for site in group if site not in test_group][:25]
            assert len(test_group) == len(val_group) == 25, f"Expected 25 sites in test and val groups, got {len(test_group)} and {len(val_group)}"

        train = df_out.loc[~df_out["site_id"].isin(test_group + val_group)].copy()
        val = df_out.loc[df_out["site_id"].isin(val_group)].copy()
        test = df_out.loc[df_out["site_id"].isin(test_group)].copy()
        if test.shape[0] == 0:
            logger.warning(f"* SKIPPING {test_group}: no test data")
            raise ValueError(f"No test data for group {test_group}")
    del df_out

    #  for columns GPP, NEE, Qle, make the values np.nan where mask==0
    for col in ["GPP", "NEE", "Qle"]:
        train.loc[train["mask"] == 0, col] = np.nan
        val.loc[val["mask"] == 0, col] = np.nan
        if remove_missing_target:
            train = train.dropna(subset=[col])
            val = val.dropna(subset=[col])

    # drop rows with any missing values (excluding target if remove_missing_target is False)
    if remove_missing_features:
        train = train.dropna(subset=[col for col in train.columns if col != target])
        val = val.dropna(subset=[col for col in val.columns if col != target])
        test = test.dropna(subset=[col for col in test.columns if col != target])

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
    times_test = time_col.loc[test.index] if time_col is not None else None

    for col in ["site_id", "year", "month", "site_year", "mask"]:
        if col in train.columns:
            train = train.drop(columns=[col])
            val = val.drop(columns=[col])
            test = test.drop(columns=[col])
    train = train.astype(np.float64)
    val = val.astype(np.float64)
    test = test.astype(np.float64) 

    xcols = ~train.columns.isin(['GPP', 'NEE', 'Qle'])
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

    return (
        (xtrain, ytrain, envs_train), 
        (xval, yval, envs_val),
        (xtest, ytest, envs_test, sites_test, times_test)
    )

# -----------------------------------------------------------------------
# -------------------------- Predictions I/O ----------------------------
# -----------------------------------------------------------------------


def load_predictions(setting, target, model_name):
    """
    Load predictions file for a given experiment.

    Args:
        setting: Experiment setting (e.g., 'spatial-easy', 'time-split')
        target: Target variable (e.g., 'GPP', 'NEE')
        model_name: Model name (e.g., 'lr', 'xgb')

    Returns:
        pd.DataFrame with y_true, y_pred, and env columns
    """
    pred_path = get_predictions_path(setting, target, model_name)
    df = load_csv(pred_path)
    if df is None:
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")
    return df


def save_predictions(test, ypred, setting, target, model_name):
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
    
    pred_path = get_predictions_path(setting, target, model_name)
    save_csv(predictions_df, pred_path)
    return predictions_df