# Code slightly modified from https://github.com/anyafries/fluxnet_bench.git
import argparse
import pandas as pd
import numpy as np
import os
from xarray import open_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Unique vegetation types
veg_types = [
    "MF",
    "GRA",
    "ENF",
    "SAV",
    "EBF",
    "WSA",
    "DBF",
    "OSH",
    "CRO",
    "CSH",
    "WET",
    "CVM",
]

# Column groupings
flux_cols = ["GPP", "NEE", "Qle", "longitude", "latitude"]
flux_cols_qc = ["GPP_qc", "NEE_qc", "Qle_qc"]
meteo_cols = [
    "Tair",
    "vpd",
    "SWdown",
    "LWdown",
    "SWdown_clearsky",
    "IGBP_veg_short",
]
meteo_cols_qc = ["Tair_qc", "vpd_qc", "SWdown_qc", "LWdown_qc"]
rs_cols = [
    "LST_TERRA_Day",
    "LST_TERRA_Night",
    "EVI",
    "NIRv",
    "NDWI_band7",
    "LAI",
    "fPAR",
]
meteo_cols_processed = meteo_cols[:-1] + ["IGBP_veg_" + v for v in veg_types]


def print_progress(completed, total, bar_length=20):
    percentage = completed / total
    num_hashes = int(percentage * bar_length)
    bar = "#" * num_hashes + "-" * (bar_length - num_hashes)
    print(f"\r[{bar}] {percentage * 100:.1f}%", end="", flush=True)


def read_file(filename, columns, columns_qc=None):
    keep_cols = columns + columns_qc if columns_qc else columns
    df = open_dataset(filename, engine="netcdf4", decode_times=False)
    # time_units = df["time"].attrs.get("units", "No units found")
    df = df[keep_cols].to_dataframe().reset_index().drop(["x", "y"], axis=1)
    df.dropna(subset=columns, how="all", inplace=True)
    return df


def initialize_dataframe(site, path):
    # Read data
    df_f = read_file(
        os.path.join(path, site + "_flux.nc"), flux_cols + flux_cols_qc
    ).dropna(subset=flux_cols, how="any")
    df_m = read_file(
        os.path.join(path, site + "_meteo.nc"), meteo_cols + meteo_cols_qc
    )
    df_m["IGBP_veg_short"] = df_m["IGBP_veg_short"].str.decode("utf-8")
    df_m["IGBP_veg_short"] = pd.Categorical(
        df_m["IGBP_veg_short"], categories=veg_types
    )
    df_m = pd.get_dummies(df_m, columns=["IGBP_veg_short"], prefix="IGBP_veg")
    df_r = read_file(os.path.join(path, site + "_rs.nc"), rs_cols)

    df = df_f.merge(df_m, on=["time"], how="left").merge(
        df_r, on=["time"], how="left"
    )
    del df_f, df_m, df_r

    # Remove bad quality data
    qc_cols = flux_cols_qc + meteo_cols_qc
    for qc_col in qc_cols:
        bad_quality_mask = df[qc_col].isin([2, 3])
        column = qc_col[:-3]
        df.loc[bad_quality_mask, column] = np.nan

    df = (
        df.drop(columns=qc_cols)
        .dropna(subset=meteo_cols_processed + rs_cols, how="all")
        .dropna(subset=flux_cols, how="any")
    )

    # Extract and process time information
    if "time" in df.columns:
        df["time"] = pd.to_datetime(
            df["time"], origin="1990-01-01", unit="m"
        )  # 00:15:00
        df["date"] = df["time"].dt.date
        df["year"] = df["time"].dt.year
        df["season"] = df["time"].dt.quarter
        df["month"] = df["time"].dt.month
        df["hour"] = df["time"].dt.hour
    else:
        raise ValueError("Time column not found in dataframe")
    
    # NDWI should be between -1 and 1 (allow some margin)
    df.loc[df.NDWI_band7 < -1.1, 'NDWI_band7'] = np.nan
    df.loc[df.NDWI_band7 > 1.1, 'NDWI_band7'] = np.nan

    # NIRv should be between 0 and 1 (allow some margin)
    df.loc[df.NIRv < -0.2, 'NIRv'] = np.nan
    df.loc[df.NIRv > 1.2, 'NIRv'] = np.nan

    # EVI should be between -1 and 1, but theoretically can be between +/- 2.5
    df.loc[df.EVI < -2.5, 'EVI'] = np.nan
    df.loc[df.EVI > 2.5, 'EVI'] = np.nan

    # Drop rows with NaN in the target variable
    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default=os.path.join(BASE_DIR, "data2024")
    )
    parser.add_argument("--override", action="store_true", default=False)
    parser.add_argument("--agg", type=str, default="all")
    path = parser.parse_args().path
    override = parser.parse_args().override
    agg = parser.parse_args().agg
    files = [
        f
        for f in os.listdir(path)
        if f.endswith(".nc") and not f.startswith(".")
    ]
    sites = np.unique([f[:6] for f in files])

    cleaned_path = os.path.join(
        os.path.dirname(path.rstrip("/\\")), "data_cleaned"
    )
    os.makedirs(cleaned_path, exist_ok=True)
    outfile_raw = os.path.join(cleaned_path, "raw.csv")
    outfile_daily = os.path.join(cleaned_path, "daily.csv")
    outfile_seasonal = os.path.join(cleaned_path, "seasonal.csv")

    if os.path.exists(outfile_raw) and not override:
        print(
            f"Raw data already exists, reading from {outfile_raw}...",
            end=" ",
            flush=True,
        )
        full_dataframe = pd.read_csv(outfile_raw)
        print("Done!", flush=True)
    else:
        print("Cleaning raw data...")
        dataframes = []
        i = 0
        for site in sites:
            i += 1
            if i % 10 == 0:
                print_progress(i, len(sites))
            df = initialize_dataframe(site, path=path)
            df["site_id"] = site
            # df['cluster'] = dg['balanced_cluster'][site]
            dataframes.append(df)
        full_dataframe = pd.concat(dataframes, ignore_index=True)

        if agg == "all" or agg == "raw":
            print(
                f"\nSaving cleaned raw data to {outfile_raw}...",
                end=" ",
                flush=True,
            )
            full_dataframe.to_csv(outfile_raw)
            print("Done!", flush=True)

    if agg == "all" or agg == "daily":
        if not os.path.exists(outfile_daily) or override:
            print("Aggregating daily data...", end=" ", flush=True)
            daily = (
                full_dataframe.groupby(["site_id", "date"])
                .mean()
                .reset_index()
            )
            print("Done!", flush=True)
            print(
                f"Saving daily data to {outfile_daily}...", end=" ", flush=True
            )
            daily.to_csv(outfile_daily)
            print("Done!", flush=True)
        else:
            print("Daily data already exists, nothing to do...")

    if agg == "all" or agg == "seasonal":
        if not os.path.exists(outfile_seasonal) or override:
            print("Aggregating seasonal data...", end=" ", flush=True)
            seasonal = (
                full_dataframe.drop(columns=["date", "hour"])
                .groupby(["site_id", "year", "season"])
                .mean()
                .reset_index()
            )
            print("Done!", flush=True)
            print(
                f"Saving seasonal data to {outfile_seasonal}...",
                end=" ",
                flush=True,
            )
            seasonal.to_csv(outfile_seasonal)
            print("Done!", flush=True)
        else:
            print("Seasonal data already exists, nothing to do...")
