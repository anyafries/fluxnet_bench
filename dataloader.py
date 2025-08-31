import logging
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)

def generate_fold_info(df, setting, start, stop=None):
    if setting in ['insite', 'insite-random', 'loso']:
        sites = df['site_id'].unique()
        if stop is None: stop = len(sites)
        groups = sorted(sites)[start:stop]
    elif setting == 'logo':
        if stop is None or stop > 10: stop = 10
        groups = range(start, stop)
    return groups


def get_fold_df(df, setting, group, target='GPP', remove_missing=False,
                astorch=False):
    ## Get the correct data
    if setting == 'insite' or setting == 'insite-random':
        df_out = df.loc[df['site_id'] == group].copy()
    elif setting == 'logo' or setting == 'loso':
        df_out = df.copy()

    ## drop columns
    df_out.drop(columns=['time', 'longitude', 'latitude'], inplace=True)
    if 'date' in df.columns: 
        df_out.drop(columns='date', inplace=True)

    ## drop missing
    if any(df_out.isna().mean() == 1):
        logger.warning(
            f"Column `{df_out.columns[df_out.isna().mean() == 1][0]}` is missing for group {group}: it is being dropped")
        df_out = df_out.dropna(axis=1, how='all')
    if remove_missing:
        nstart = df_out.shape[0]
        df_out = df_out.dropna(axis = 0, how = 'any')
        nout = df_out.shape[0]
        if nstart > nout:
            logger.info(f"* Dropped {nstart-nout}/{nstart} ({(nstart-nout)/nstart * 100:.2f}%) rows due to missingness")

    ## split into train/test
    if setting == 'insite':       
        # split it chronologically
        n_train = int(df_out.shape[0] * 0.8)
        train = df_out.iloc[:n_train]
        test  = df_out.iloc[n_train:]
    elif setting == 'insite-random':
        # split it randomly
        n_train = int(df_out.shape[0] * 0.8)
        train = df_out.sample(n=n_train, random_state=1)
        test  = df_out.drop(train.index)
    elif setting == 'logo':
        sites = pd.read_csv('data_cleaned/grouping_equal_size.csv')
        sites = sites.loc[sites['balanced_cluster'] == group,'site']
        train = df_out.loc[~df_out['site_id'].isin(sites)].copy()
        test = df_out.loc[df_out['site_id'].isin(sites)].copy()
        fold = None
    elif setting == 'loso':
        # split it by site
        train = df_out.loc[df_out["site_id"] != group].copy()
        test = df_out.loc[df_out["site_id"] == group].copy()
        if test.shape[0] == 0:
            logger.warning(f"* SKIPPING {group}: no test data")
            return None, None, None, None, None
        fold = None
    del df_out

    # clean up
    train_ids = train['site_id']
    train.drop(columns='site_id', inplace=True)
    test.drop(columns='site_id', inplace=True)
    xcols = train.columns != target
    ycol = train.columns == target

    ## drop outliers
    q1, q99 = train['GPP'].quantile([0.01, 0.99])
    ndrop = np.mean((train['GPP'] < q1) | (train['GPP'] > q99))
    logger.info(f"* Dropping {ndrop*100:.2f}% training outliers")
    train = train.loc[(train['GPP'] > q1) & (train['GPP'] < q99)]
    ndrop_test = np.mean((test['GPP'] < q1) | (test['GPP'] > q99))
    logger.info(f"* Dropping {ndrop_test*100:.2f}% test outliers")
    test = test.loc[(test['GPP'] > q1) & (test['GPP'] < q99)]

    # scale
    scaler_X = MinMaxScaler()
    # scaler_X = StandardScaler()
    train = scaler_X.fit_transform(train)
    test  = scaler_X.transform(test)

    # TODO: float64 -> float32
    # X_train_scaled = np.asarray(X_train_scaled, dtype=np.float32)

    # split into x,y
    xtrain, ytrain = train[:, xcols], train[:, ycol].ravel()
    xtest, ytest = test[:, xcols], test[:, ycol].ravel()
    if astorch:
        xtrain = torch.tensor(xtrain, dtype=torch.float32)
        ytrain = torch.tensor(ytrain, dtype=torch.float32).view(-1, 1)
        xtest = torch.tensor(xtest, dtype=torch.float32)
        ytest = torch.tensor(ytest, dtype=torch.float32).view(-1, 1)
    return xtrain, ytrain, xtest, ytest, train_ids