import os
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import RobustScaler
from utils.utils import setup_logging, get_predictions_path, load_csv, save_csv

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
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data path not found: {path}")
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


# -------------------------------------------------------------------------
# ------------------ helpers for the overlap experiments ------------------
# -------------------------------------------------------------------------

def get_ta_overlap_site_groups(df_out, regime, val_size=30):
    """
    Deterministic TA-overlap split.

    Idea:
    - Fix a hot test set (top-60 TA sites → every 2nd site = 30 test sites)
    - Replace nearby "hot neighbours" (S1) with cold sites (S2)
    - Control overlap via k ∈ {30,20,10,0}

    Regimes:
        k = 30 → full overlap (all hot neighbours included)
        k = 0  → strong gap (only cold sites instead of neighbours)
    """

    # Parse regime (e.g. "TA-overlap-20" → k=20)
    k = int(str(regime).split("-")[-1])
    assert k in {30, 20, 10, 0}, f"Invalid TA-overlap regime: {regime}"

    # Rank sites by mean TA
    site_ta = df_out.groupby("site_id")["TA"].mean().sort_values(ascending=False)
    sites = site_ta.index.tolist()

    # Define base partitions (X = hottest 60, Y = rest)
    hottest, remaining = sites[:60], sites[60:]

    # TEST: every second site from X, S1: remaining hot sites
    test_group = hottest[::2]
    hot_set = hottest[1::2]

    # S2: 30 coldest sites (used to replace S1 progressively)
    cold_set = remaining[-30:]

    # VAL: randomly sample from remaining sites (excluding the coldest 30 reserved for S2)
    np.random.seed(42)
    val_candidates = remaining[:-30]
    val_group = np.random.choice(val_candidates, size=val_size, replace=False).tolist()

    # T: all remaining sites (always included in training)
    remaining = set(sites) - set(test_group) - set(val_group)

    # assert that hot_set and cold_set are ordered by TA
    assert site_ta[hot_set].is_monotonic_decreasing, "Hot neighbours not ordered by TA"
    assert site_ta[cold_set].is_monotonic_decreasing, "Cold sites not ordered by TA"

    # Train: Keep k closest hot neighbours, replace the rest with cold sites
    train_group = hot_set[:k] + cold_set[:30-k] + list(remaining)

    # Sanity checks: disjointness
    assert not (set(train_group) & set(val_group))
    assert not (set(train_group) & set(test_group))
    assert not (set(val_group) & set(test_group))

    return train_group, val_group, test_group


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
    astorch=False,
    return_colnames=False,
    standardize=False,
):
    """
    Get the train/test data for a specific setting.
    Args:
        df (pd.DataFrame): The input dataframe containing the data.
        setting (str): The cross-validation setting.
        path (str): The path to the data directory (used for loading site lists).
        target (str, optional): The target variable name.
            Defaults to "GPP".
        remove_missing_target (bool, optional): Whether to remove rows with
            missing target values. Defaults to False.
        keep_lonlat (bool, optional): Whether to keep longitude and latitude
            features. Defaults to False.
        astorch (bool, optional): Whether to return data as PyTorch tensors.
            Defaults to False.
        return_colnames (bool, optional): Whether to return column names of
            features. Defaults to False.
        standardize (bool, optional): Whether to standardize features using
            training set statistics. Defaults to False.
    Returns:
        tuple: xtrain, ytrain, envs_train, xtest, ytest, envs_test
    """
    # Subset the correct data
    if setting in ["time-split", "time-space"]:
        sites_to_keep = pd.read_csv(os.path.join(path, "sites_with_2018.csv"))
        df_out = df.loc[df["site_id"].isin(sites_to_keep['site_id'].values)].copy()
    else: # setting in ["spatial-easy", "spatial-hard"]:
        df_out = df.copy()
    # else:
    #     raise ValueError(f"Setting `{setting}` not recognized in get_data_split")

    # Preserve time column if needed for metadata
    time_col = df_out["time"].copy()

    # drop columns
    cols_to_drop = ["time"]
    if not keep_lonlat:
        cols_to_drop += ["tower_lat", "tower_lon"]
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

    elif setting == "time-space":
        # 1. Define site groups based on "hard-1"
        test_group = ['AU-Rgf', 'AU-War', 'BR-Npw', 'CA-Cbo', 'DK-Vng', 'FR-Gri', 'FR-Lam', 'IT-BCi', 'IT-Cp2', 'IT-Lav', 'IT-TrF', 'US-A32', 'US-ARM', 'US-Bi1', 'US-Bi2', 'US-DFC', 'US-Kon', 'US-Ne1', 'US-Pnp', 'US-RGA', 'US-RGo', 'US-SP1', 'US-Sne', 'US-Snf', 'US-Tw4']
        
        all_sites = df_out["site_id"].unique().tolist()
        remaining_sites = [site for site in all_sites if site not in test_group]
        
        np.random.seed(42)
        np.random.shuffle(remaining_sites)
        val_group = remaining_sites[:25]
        
        # 2. Split by both time AND space
        df_out['site_year'] = list(zip(df_out['site_id'], df_out['year']))
        train = df_out.loc[(df_out["year"] < 2018) & (~df_out["site_id"].isin(test_group + val_group))].copy()
        val = df_out.loc[(df_out["year"] == 2018) & (df_out["site_id"].isin(val_group))].copy()
        test = df_out.loc[(df_out["year"] > 2018) & (df_out["site_id"].isin(test_group))].copy()
        
        if test.shape[0] == 0:
            raise ValueError("No test data for time-space setting")
        
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
        elif setting[:6] == "random":
            seed = int(setting.split("-")[1]) if "-" in setting else 42
            np.random.seed(seed)
            site_ids = df_out["site_id"].unique().tolist()
            np.random.shuffle(site_ids)
            test_group = site_ids[:25]
            val_group = site_ids[25:50]
        elif setting.startswith("TA-overlap"):
            train_group, val_group, test_group = get_ta_overlap_site_groups(
                df_out, setting, val_size=30)
        else:
            if setting == 'spatial-easy40':
                # G1 and G2 and 3 extra (to make 40)
                test_group = ['US-Tw1', 'DE-Hai', 'US-Seg', 'US-Sne', 'US-Tw4', 'US-xDL', 'UK-AMo', 'AU-Dry', 'US-CGG', 'FR-Bil', 'US-Rpf', 'DK-Skj', 'RU-Fy2', 'DE-Rns', 'US-Tw3', 'RU-Fyo', 'US-Snf', 'CH-Cha', 'AR-CCg', 'CL-SDF', 'DE-Gri', 'FR-Tou', 'AU-Whr', 'AU-GWW', 'US-RGo', 'IT-BCi', 'ES-Abr', 'SE-Nor', 'DE-Hzd', 'US-CS2', 'US-StJ', 'CA-TP3', 'BE-Dor', 'US-xWD', 'US-Syv', 'DE-RuR', 'CZ-BK1', 'BE-Maa', 'BE-Vie', 'FI-Var']
            elif setting in ["PFT_CRO", "PFT_ENF", "PFT_GRA", "PFT_WET"]:
                test_group = df_out.loc[df_out[setting] == 1, "site_id"].unique().tolist()
            elif setting == "forest":
                forest_columns = ["PFT_DBF", "PFT_DNF", "PFT_EBF"]
                test_group = df_out.loc[df_out[forest_columns].sum(axis=1) > 0, "site_id"].unique().tolist()
            elif setting == "schrub-savanna":
                shrub_savanna_columns = ["PFT_OSH", "PFT_SAV", "PFT_WSA"]
                test_group = df_out.loc[df_out[shrub_savanna_columns].sum(axis=1) > 0, "site_id"].unique().tolist()
            elif setting == 'grass-savanna':
                grass_savanna_columns = ["PFT_GRA", "PFT_SAV", "PFT_WSA"]
                test_group = df_out.loc[df_out[grass_savanna_columns].sum(axis=1) > 0, "site_id"].unique().tolist()
            elif setting == 'TA':
                test_group = ['AU-Dry', 'AU-DaS', 'AU-Lit', 'BR-Npw', 'AU-Lon', 'AU-ASM', 'US-xDS', 'US-ONA', 'US-SP1', 'US-xJE', 'US-SRM', 'US-HB2', 'AU-GWW', 'US-SRS', 'US-SRG', 'IL-Yat', 'US-HB3', 'US-HB1', 'US-xDL', 'US-RGA', 'AU-Cum', 'US-xTA', 'AU-Cpr', 'US-Whs', 'US-Cst']
            elif setting == 'TA40':
                test_group = ['AU-Dry', 'AU-DaS', 'AU-Lit', 'BR-Npw', 'AU-Lon', 'AU-ASM', 'US-xDS', 'US-ONA', 'US-SP1', 'US-xJE', 'US-SRM', 'US-HB2', 'AU-GWW', 'US-SRS', 'US-SRG', 'IL-Yat', 'US-HB3', 'US-HB1', 'US-xDL', 'US-RGA', 'AU-Cum', 'US-xTA', 'AU-Cpr', 'US-Whs', 'US-Cst', 'US-Wkg', 'IT-BCi', 'US-Jo2', 'IT-Cp2', 'US-RGo', 'ES-Abr', 'US-NC4', 'ES-Agu', 'US-Akn', 'US-xJR', 'ES-Pdu', 'US-Ton', 'ES-LM2', 'IT-Noe', 'ES-LM1']
            elif setting == "VPD":
                test_group = ['AU-ASM', 'AU-Lon', 'AU-Dry', 'US-SRM', 'US-SRG', 'US-Jo2', 'AU-DaS', 'US-xJR', 'US-SRS', 'US-Whs', 'US-Wkg', 'AU-GWW', 'AU-Cpr', 'US-Ses', 'US-CdM', 'US-Seg', 'US-Ton', 'US-RGo', 'AU-Lit', 'IL-Yat', 'ES-Abr', 'ES-LM2', 'ES-LM1', 'US-CGG', 'US-Hn2']
            elif setting == "LST":
                test_group = ['AU-Lon', 'AU-Dry', 'AU-ASM', 'AU-DaS', 'US-xJR', 'AU-GWW', 'AU-Lit', 'US-SRM', 'US-Whs', 'AU-Cpr', 'US-Jo2', 'US-Ses', 'US-Seg', 'US-SRS', 'US-Wkg', 'US-SRG', 'AU-Rgf', 'BR-Npw', 'IL-Yat', 'ES-Abr', 'ES-Agu', 'US-CGG', 'AU-Boy', 'US-CdM', 'US-ONA']
            elif setting == "LST40":
                test_group = ['AU-Lon', 'AU-Dry', 'AU-ASM', 'AU-DaS', 'US-xJR', 'AU-GWW', 'AU-Lit', 'US-SRM', 'US-Whs', 'AU-Cpr', 'US-Jo2', 'US-Ses', 'US-Seg', 'US-SRS', 'US-Wkg', 'US-SRG', 'AU-Rgf', 'BR-Npw', 'IL-Yat', 'ES-Abr', 'ES-Agu', 'US-CGG', 'AU-Boy', 'US-CdM', 'US-ONA', 'US-xDS', 'US-Ton', 'AU-Whr', 'ES-LM2', 'US-Bi2', 'US-Dmg', 'ES-LM1', 'US-SP1', 'US-xCP', 'AU-Cum', 'US-ARM', 'US-RGo', 'US-xJE', 'ES-LJu', 'US-Srr']
            elif setting == "europe":
                europe = [
                    'IT', 'DE', 'FR', 'ES', 'SE', 'CZ', 
                    'FI', 'BE', 'DK', 'RU', 'CH', 'IE', 
                    'NL', 'UK'
                ]
                test_group = [site for site in df_out["site_id"].unique().tolist() if site[:2] in europe]
            elif setting == "rest-of-world":
                rest = ['AU', 'AR', 'CL', 'IL', 'JP', 'BR'] + ['CA']
                test_group = [site for site in df_out["site_id"].unique().tolist() if site[:2] in rest]
            elif setting[:4] == "hard":
                if setting == "hard-1":
                    # highest RMSE (avg ranking across targets) 
                    test_group = ['AU-Rgf', 'AU-War', 'BR-Npw', 'CA-Cbo', 'DK-Vng', 'FR-Gri', 'FR-Lam', 'IT-BCi', 'IT-Cp2', 'IT-Lav', 'IT-TrF', 'US-A32', 'US-ARM', 'US-Bi1', 'US-Bi2', 'US-DFC', 'US-Kon', 'US-Ne1', 'US-Pnp', 'US-RGA', 'US-RGo', 'US-SP1', 'US-Sne', 'US-Snf', 'US-Tw4']
                elif setting == "hard-2":
                    # highest increase (avg ranking across targets) 
                    test_group = ['AU-Rgf', 'BE-Lon', 'BE-Maa', 'CA-ER1', 'FR-Aur', 'FR-Lam', 'IT-BCi', 'IT-Cp2', 'IT-MtP', 'US-A32', 'US-ARM', 'US-Bi2', 'US-DS3', 'US-Dmg', 'US-Kon', 'US-Ne1', 'US-Pnp', 'US-RGA', 'US-RGo', 'US-Ro1', 'US-Ro6', 'US-Sne', 'US-Snf', 'US-Tw1', 'US-Tw4']
                elif setting == "hard-3":
                    # highest % increase (avg ranking across targets) 
                    test_group = ['AR-TF1', 'AU-ASM', 'AU-Lon', 'AU-Rgf', 'BE-Maa', 'CA-DB2', 'CA-DBB', 'CA-ER1', 'FI-Sii', 'FR-Aur', 'IL-Yat', 'IT-MtP', 'UK-AMo', 'US-A32', 'US-Bi2', 'US-DS3', 'US-Pnp', 'US-RGA', 'US-RGo', 'US-Sne', 'US-Snf', 'US-Srr', 'US-Tw1', 'US-Tw4', 'US-xJR']
                elif setting == "hard-4":
                    # highest % increase (min rank)
                    test_group = ['AR-TF1', 'AU-ASM', 'AU-Rgf', 'BE-Maa', 'CA-DBB', 'CA-SCB', 'FI-Sii', 'FR-FBn', 'IT-Cp2', 'SE-Lnn', 'UK-AMo', 'US-Bi2', 'US-DS3', 'US-ICh', 'US-Pnp', 'US-Ro1', 'US-Sne', 'US-Snf', 'US-Srr', 'US-StJ', 'US-Tw1', 'US-Tw4', 'US-Vcm', 'US-xHE', 'US-xSL']
                elif setting == "hard-5":
                    # hardest for ET
                    test_group = ['AU-Lit', 'BR-Npw', 'FR-FBn', 'IT-BCi', 'IT-Cp2', 'IT-Ren', 'IT-TrF', 'US-DS3', 'US-Dmg', 'US-HB1', 'US-HB2', 'US-HB3', 'US-KFS', 'US-NC3', 'US-NC4', 'US-ORv', 'US-Pnp', 'US-RGA', 'US-RGB', 'US-RGo', 'US-SP1', 'US-Sne', 'US-Snf', 'US-StJ', 'US-Tw4']
            else:
                raise ValueError(f"Setting `{setting}` not recognized in get_data_split")
            
            all_sites = df_out["site_id"].unique().tolist()
            remaining_sites = [site for site in all_sites if site not in test_group]
            np.random.seed(42)
            np.random.shuffle(remaining_sites)
            if setting[-2:] == "40":
                val_group = remaining_sites[:20]
            else:
                val_group = remaining_sites[:25]
        
        val = df_out.loc[df_out["site_id"].isin(val_group)].copy()
        test = df_out.loc[df_out["site_id"].isin(test_group)].copy()
        if setting.startswith("TA-overlap"):
            # For TA-overlap, we have a predefined train group 
            # it is *not* just the complement of test+val
            train = df_out.loc[~df_out["site_id"].isin(train_group)].copy()
        else:
            train = df_out.loc[~df_out["site_id"].isin(test_group + val_group)].copy()
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
    feature_cols = [col for col in train.columns if col not in ['GPP', 'NEE', 'ET']]
    incomplete_train = train[feature_cols].isna().any(axis=1).sum()
    incomplete_val = val[feature_cols].isna().any(axis=1).sum()
    incomplete_test = test[feature_cols].isna().any(axis=1).sum()
    assert incomplete_train == incomplete_val == incomplete_test == 0, \
        f"Expected no missing values in features, but found {incomplete_train} in train and {incomplete_val} in val, and {incomplete_test} in test"

    # clean up
    if setting in ["time-split", "time-space"]:
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

    if standardize:
        scaler = RobustScaler()
        xtrain = scaler.fit_transform(xtrain)
        xval = scaler.transform(xval)
        xtest = scaler.transform(xtest)

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
        'y_true': ytest.ravel(),
        'y_pred': ypred,
        'env': envs_test,
        'site_id': sites_test,
        'time': times_test,
        # 'mask': mask,
    })

    pred_path = get_predictions_path(setting, target, model_name, val_strategy)
    save_csv(predictions_df, pred_path)
    return predictions_df