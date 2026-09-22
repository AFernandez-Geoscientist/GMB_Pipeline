import os
import re
import numpy as np
import pandas as pd


def MB_stats(periods_csv_path):
    # Path configuration
    output_dir = "Outputcsv"
    integrated_csv_path = os.path.join(
        output_dir, "03_DH_DMB_Unc_Integrated_All_Periods.csv"
    )
    final_output_path = os.path.join(output_dir, "03_DH_MB_final_stats.csv")

    # 1. Read input CSVs
    periods_df = pd.read_csv(periods_csv_path)
    integrated_df = pd.read_csv(integrated_csv_path)

    # 2. Filter integrated dataset for periods present in periods_df
    target_periods = periods_df["Period"].dropna().astype(str).unique()
    filtered_df = integrated_df[
        integrated_df["Period"].astype(str).isin(target_periods)
    ].copy()

    # 3. Sort by the start year of the "Period"
    def extract_start_year(period_str):
        match = re.search(r"^(\d+)", str(period_str).strip())
        return int(match.group(1)) if match else 0

    filtered_df["Start_Year"] = filtered_df["Period"].apply(extract_start_year)
    sorted_df = filtered_df.sort_values(by="Start_Year").drop(
        columns=["Start_Year"]
    )

    # Calculate cumulative dt (a)
    cum_dt = sorted_df["dt (a)"].cumsum()

    # (3.a) Cumulative sum of dh (m)
    dh = sorted_df["dh (m)"]
    sorted_df["Cum dh (m)"] = dh.cumsum()

    # (3.b) Cumulative quadrature of σ_dh (m)
    sig_dh = sorted_df["σ_dh (m)"]
    sorted_df["Cum σ_dh (m)"] = np.sqrt((sig_dh**2).cumsum())

    # (3.c) Rate of Cum dh (m) -> Cum dh/dt (m/a)
    sorted_df["Cum dh/dt (m/a)"] = sorted_df["Cum dh (m)"] / cum_dt

    # (3.d) Rate of Cum σ_dh (m) -> Cum σ_dh (m)/dt (m/a)
    sorted_df["Cum σ_dh (m)/dt (m/a)"] = sorted_df["Cum σ_dh (m)"] / cum_dt

    # (3.e) Cumulative sum of MB (m w.e.)
    mb = sorted_df["MB (m w.e.)"]
    sorted_df["Cum MB (m w.e.)"] = mb.cumsum()

    # (3.f) Cumulative quadrature of σ_MB (m w.e.)
    sig_mb = sorted_df["σ_MB (m w.e.)"]
    sorted_df["Cum σ_MB (m w.e.)"] = np.sqrt((sig_mb**2).cumsum())

    # (3.g) Rate of Cum MB (m w.e.) -> Cum MB/dt (m w.e./a)
    sorted_df["Cum MB/dt (m w.e./a)"] = sorted_df["Cum MB (m w.e.)"] / cum_dt

    # (3.h) Rate of Cum σ_MB (m w.e.) -> Cum σ_MB/dt (m w.e./a)
    sorted_df["Cum σ_MB/dt (m w.e./a)"] = sorted_df["Cum σ_MB (m w.e.)"] / cum_dt

    # 4. Organize existing and newly created fields
    base_columns = [
        "Period",
        "dt (a)",
        "Area (km²)",
        "Area (%)",
        "dh (m)",
        "σ_dh (m)",
        "dh/dt (m/a)",
        "σ_dh_rate (m/a)",
        "MB (m w.e.)",
        "σ_MB (m w.e.)",
        "MB rate (m w.e./a)",
        "σ_MB rate (m w.e./a)",
    ]

    new_columns = [
        "Cum dh (m)",
        "Cum σ_dh (m)",
        "Cum dh/dt (m/a)",
        "Cum σ_dh (m)/dt (m/a)",
        "Cum MB (m w.e.)",
        "Cum σ_MB (m w.e.)",
        "Cum MB/dt (m w.e./a)",
        "Cum σ_MB/dt (m w.e./a)",
    ]

    # Keep base columns present in input file, followed by new columns
    existing_base_cols = [
        col for col in base_columns if col in sorted_df.columns
    ]
    final_df = sorted_df[existing_base_cols + new_columns]

    # Round all numerical values to 2 decimal places
    final_df = final_df.round(2)

    # 5. Save output
    os.makedirs(output_dir, exist_ok=True)
    final_df.to_csv(final_output_path, index=False)
    print(f"File successfully created: {final_output_path}")