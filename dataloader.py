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
    remove_missing=False,
    astorch=False,
    return_metadata=False,
):
    """
    Get the train/test data for a specific setting.
    Args:
        df (pd.DataFrame): The input dataframe containing the data.
        setting (str): The cross-validation setting.
        target (str, optional): The target variable name.
            Defaults to "GPP".
        remove_missing (bool, optional): Whether to remove rows with missing values.
            Defaults to False.
        astorch (bool, optional): Whether to return data as PyTorch tensors.
            Defaults to False.
        return_metadata (bool, optional): If True, also return site_id and time
            arrays for the test set. Defaults to False.
    Returns:
        tuple: xtrain, ytrain, envs_train, xtest, ytest, envs_test
            If return_metadata=True: also includes sites_test, times_test
    """
    # Generate fold info and get group
    

    # Determine min_samples based on setting
    # TODO[LATER]: should be handled by the cleaned data
    min_samples = 100
    if setting == "time-split":
        raise NotImplementedError("time-split setting not implemented yet")
    elif setting == "spatial-easy":
        group = generate_fold_info(df, setting)[0]
    elif setting == "spatial-hard":
        # group = [
        #     'AU-ASM', 'AU-Cpr', 'AU-Cum', 'AU-DaS', 'AU-GWW', 'AU-Lit', 
        #     'AU-Rgf', 'AU-War', 'AU-Wom', 'AU-Whr', 'AU-Dry', 'AU-Boy', 
        #     'AU-Lon', 'AR-TF1', 'CL-SDF', 'PE-QFR', 'BR-Npw', 'US-Bar', 
        #     'RU-Fy2', 'IL-Yat', 'ES-LJu', 'US-Tw4', 'FI-Sii', 'CZ-wet'
        # ]
        # group = [
        #     'BR-Npw', 'FR-Lam', 'PE-QFR', 'FR-Mej', 'AU-DaS', 'AU-ASM', 
        #     'IL-Yat', 'US-SRM', 'US-Whs', 'ES-LJu', 'CZ-wet', 'FI-Sii', 
        #     'US-Tw1', 'US-Tw4', 'CA-SCC', 'US-Bar', 'US-NR1', 'AR-TF1', 
        #     'RU-Fy2', 'FI-Var', 'US-ARM', 'US-Kon', 'JP-BBY', 'CL-SDF'
        # ]
        group = [ # 25 southern most sites
            'AU-ASM', 'AU-Cpr', 'AU-Cum', 'AU-DaS', 'AU-GWW', 'AU-Lit', 
            'AU-Rgf', 'AU-War', 'BR-Npw', 'AU-Wom', 'AU-Whr', 'AR-TF1', 
            'AU-Dry', 'CL-SDF', 'AU-Boy', 'AR-CCg', 'AU-Lon', 'PE-QFR', 
            'FR-Mej', 'FR-Lam', 'US-ONA', 'US-KS3', 'US-SP1', 'IL-Yat' 
        ]
    else:
        raise ValueError(f"Setting `{setting}` not recognized in get_data_split")

    # Subset the correct data
    if setting in ["spatial-easy", "spatial-hard"]:
        df_out = df.copy()
    elif setting == "time-split":
        # group is a list of sites with enough years
        df_out = df.loc[df["site_id"].isin(group)].copy()
    else:
        raise ValueError(f"Setting `{setting}` not recognized in get_data_split")

    # drop rows where target is missing
    nstart = df_out.shape[0]
    df_out = df_out.dropna(subset=[target])
    nout = df_out.shape[0]
    if nstart > nout:
        diff = nstart - nout
        logger.info(
            f"* Dropped {diff}/{nstart} ({diff/nstart*100:.2f}%) rows due to missing target `{target}`"
        )

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

    # drop any columns that only have missing values
    if any(df_out.isna().mean() == 1):
        logger.warning(
            f"Column `{df_out.columns[df_out.isna().mean() == 1][0]}` is missing for group {group}: it is being dropped"
        )
        df_out = df_out.dropna(axis=1, how="all")
    
    # drop rows with any missing values
    if remove_missing:
        nstart = df_out.shape[0]
        df_out = df_out.dropna(axis=0, how="any")
        nout = df_out.shape[0]
        if nstart > nout:
            logger.info(
                f"* Dropped {nstart-nout}/{nstart} ({(nstart-nout)/nstart * 100:.2f}%) rows due to missingness"
            )

    # Preserve time column if needed for metadata
    if return_metadata and "time" in df_out.columns:
        time_col = df_out["time"].copy()
    else:
        time_col = None

    # drop columns
    for col in ["date", "hour", "time", "longitude", "latitude"]:
        if col in df_out.columns:
            df_out.drop(columns=[col], inplace=True)

    # split into train/test
    if setting == "time-split":
        # split years chronologically
        min_years = 7 
        site_years = df_out["year"].value_counts().sort_index()
        site_years = site_years.index[site_years >= min_samples]
        n_years = len(site_years)
        if n_years < min_years:
            logger.warning(
                f"* SKIPPING {group}: only {n_years} years with >= {min_samples} samples"
            )
            if return_metadata:
                return None, None, None, None, None, None, None, None
            return None, None, None, None, None, None
        unique_years = np.sort(site_years)
        train_years, test_years = unique_years[:3], unique_years[3:7]
        train = df_out.loc[df_out["year"].isin(train_years)].copy()
        test = df_out.loc[df_out["year"].isin(test_years)].copy()
    elif setting in ["spatial-easy", "spatial-hard"]:
        # split by group of sites
        if min_samples is not None:
            site_counts = df_out["site_id"].value_counts()
            valid_sites = site_counts[site_counts >= min_samples].index
            logger.info(f"Keeping {len(valid_sites)}/{len(site_counts)} sites with >= {min_samples} samples")
            df_out = df_out.loc[df_out["site_id"].isin(valid_sites)].copy()

        train = df_out.loc[~df_out["site_id"].isin(group)].copy()
        test = df_out.loc[df_out["site_id"].isin(group)].copy()
        if test.shape[0] == 0:
            logger.warning(f"* SKIPPING {group}: no test data")
            if return_metadata:
                return None, None, None, None, None, None, None, None
            return None, None, None, None, None, None
    del df_out

    # clean up
    if setting == "time-split":
        env_col = "year"
    else:
        env_col = "site_id"
    envs_train = train[env_col]
    envs_test = test[env_col].copy()

    # Extract metadata before dropping columns
    if return_metadata:
        sites_test = test["site_id"].copy()
        times_test = time_col.loc[test.index] if time_col is not None else None

    for col in ["site_id", "year", "month"]:
        if col in train.columns:
            train = train.drop(columns=[col])
        if col in test.columns:
            test = test.drop(columns=[col])
    train = train.astype(np.float64)
    test = test.astype(np.float64) 

    # select x and y columns
    if (setting == "time-split") and (target == "GPP"):
        # remove remote sensing variables for GPP in time-split and insite
        remote_sensing_vars = [
            "EVI",
            "NDWI_SWIR1",
            "NIRv",
            "LST_Day",
            "LST_Night"
        ]
        xcols = ~train.columns.isin(remote_sensing_vars + ['GPP', 'NEE', 'Qle'])
    else:
        xcols = ~train.columns.isin(['GPP', 'NEE', 'Qle'])
    ycol = train.columns == target

    # split into x,y
    xtrain, ytrain = train.values[:, xcols], train.values[:, ycol].ravel()
    xtest, ytest = test.values[:, xcols], test.values[:, ycol].ravel()

    if astorch:
        xtrain = torch.tensor(xtrain, dtype=torch.float32)
        ytrain = torch.tensor(ytrain, dtype=torch.float32).view(-1, 1)
        xtest = torch.tensor(xtest, dtype=torch.float32)
        ytest = torch.tensor(ytest, dtype=torch.float32).view(-1, 1)

    # Filter out rows with NaN in test features
    # TODO[LATER]: handle missing values properly in data already
    feature_mask = ~np.isnan(xtest).any(axis=1)
    xtest = xtest[feature_mask]
    ytest = ytest[feature_mask]
    envs_test = envs_test.values[feature_mask]

    if return_metadata:
        sites_test = sites_test.values[feature_mask]
        times_test = times_test.values[feature_mask] if times_test is not None else None
        return sites_test, times_test, envs_test, ytest

    return xtrain, ytrain, envs_train, xtest, ytest, envs_test

# -----------------------------------------------------------------------
# -------------------------- Predictions I/O ----------------------------
# -----------------------------------------------------------------------


def get_test_metadata(df, setting, target="GPP"):
    """
    Get test set metadata (site_id, time, env, y_true) for a setting.

    Args:
        df: DataFrame with the data
        setting: Experiment setting ('spatial-easy', 'time-split', etc.)
        target: Target variable name

    Returns:
        tuple: (site_id, time, env, y_true) arrays for the test set
    """
    result = get_data_split(df, setting, target=target, return_metadata=True)
    if result[0] is None:
        return None, None, None, None
    sites_test, times_test, envs_test, ytest = result
    return sites_test, times_test, envs_test, ytest


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


def save_predictions(df, ypred, setting, target, model_name):
    """Save predictions DataFrame to CSV."""
    site_id, time, env, y_true = get_test_metadata(df, setting, target)
    predictions_df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': ypred,
        'env': env,
        'site_id': site_id,
        'time': time,
    })
    
    pred_path = get_predictions_path(setting, target, model_name)
    save_csv(predictions_df, pred_path)
    return predictions_df