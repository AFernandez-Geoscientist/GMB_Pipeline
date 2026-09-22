import math
from pathlib import Path
import numpy as np
import pandas as pd
import helper
from MB_stats import MB_stats 

# ==============================================================================
# HELPER FUNCTION FOR FILE LOCATION
# ==============================================================================

def locate_file(filename, fallback_dirs=["Outputcsv", "Output", "Outputcsv/01_Co-registration"]):
    """Finds file in current directory or checks common fallback subdirectories."""
    p = Path(filename)
    if p.exists():
        return p
    for d in fallback_dirs:
        p_sub = Path(d) / filename
        if p_sub.exists():
            return p_sub
    return p

# ==============================================================================
# SECTION 1: PRIMARY COMPUTATION (VARIOGRAM / PIXEL INTEGRATION METHOD)
# ==============================================================================

# 1. Load CSV files dynamically
dh_stats_path = locate_file("02_DH_stats.csv")
coreg_unc_path = locate_file("01_Co-registration_uncertainty.csv")

df_dh = pd.read_csv(dh_stats_path)
df_var = pd.read_csv(coreg_unc_path)

# 2. Ensure Period column exists in DH_stats.csv
if "Period" not in df_dh.columns:
    df_dh["Period"] = df_dh["Ini_Period"].astype(str) + "-" + df_dh["End_Period"].astype(str)

# 3. Calculate Area % relative to Total_Pixels_In_Polygon
df_dh["Area_pct"] = (df_dh["Count"] / df_dh["Total_Pixels_In_Polygon"]) * 100

# Maintain order defined in DH_stats.csv
dh_period_order = df_dh["Period"].tolist()

# 4. Merge datasets on Period (preserving original order of df_dh)
df = pd.merge(df_dh, df_var, on="Period", how="left")

# Constants
PIXEL_SIZE = 30  # meters
PIXEL_AREA = PIXEL_SIZE * PIXEL_SIZE  # 900 m^2 per pixel
RHO_ICE = 850.0  # kg/m^3
RHO_WATER = 1000.0  # kg/m^3
SIGMA_RHO = 60.0  # kg/m^3

# Temporal setup
df["iniyear"] = df["Ini_Period"].astype(int)
df["endyear"] = df["End_Period"].astype(int)
df["dt"] = df["endyear"] - df["iniyear"]

# Area calculation
df["A_m2"] = df["Count"] * PIXEL_AREA
df["A_km2"] = df["A_m2"] / 1e6

# Spatial correlation & Effective sample size
df["Acorr_m2"] = df["Acorr"] * 1e6
df["N_eff"] = np.where(
    df["A_m2"] <= df["Acorr_m2"], 1.0, df["A_m2"] / (np.pi * (df["L"] ** 2))
)

# Height change (dh) & Mass Balance (MB) calculations
df["sigma_dh"] = df["NMAD"] / np.sqrt(df["N_eff"])
df["dh_rate"] = df["Mean"] / df["dt"]
df["sigma_dh_rate"] = df["sigma_dh"] / df["dt"]

df["MB_mwe"] = df["Mean"] * (RHO_ICE / RHO_WATER)
df["MB_rate_mwe"] = df["MB_mwe"] / df["dt"]

df["sigma_dMB"] = np.sqrt(
    (df["sigma_dh"] * (RHO_ICE / RHO_WATER)) ** 2
    + (df["Mean"] * (SIGMA_RHO / RHO_WATER)) ** 2
)
df["sigma_dMB_rate"] = df["sigma_dMB"] / df["dt"]

# Summary table for primary method
summary_primary = df[[
    "Period",
    "dt",
    "A_km2",
    "Area_pct",
    "Mean",
    "sigma_dh",
    "dh_rate",
    "sigma_dh_rate",
    "MB_mwe",
    "sigma_dMB",
    "MB_rate_mwe",
    "sigma_dMB_rate",
]].copy()

summary_primary.columns = [
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

# Create lookup mappings to retrieve primary data
dh_map = dict(zip(df["Period"], df["Mean"]))
sigma_dh_map = dict(zip(df["Period"], df["sigma_dh"]))
area_map = dict(zip(df["Period"], df["A_km2"]))
area_pct_map = dict(zip(df["Period"], df["Area_pct"]))

# ==============================================================================
# SECTION 2: SECONDARY COMPUTATION (DEM STACKING & TEMPORAL ERROR PROPAGATION)
# ==============================================================================
print("Computing uncertainty for periods centered in 2010")
print("Using hard-coded values from metadata except σ_dh (m)")
# Configurations retrieving dh/MB source data from Section 1
sec_config = [
    {
        "Period": "2000-2008",
        "dem1": 2000.0,
        "dem2": 2008.5,
        "sigma_aw3d30": 2.01273,
        "sigma_dt": 4.5,
        "src": "2000-2010",
    },
    {
        "Period": "2000-2010",
        "dem1": 2000.0,
        "dem2": 2010.0,
        "sigma_aw3d30": 2.01273,
        "sigma_dt": 4.5,
        "src": "2000-2010",
    },
    {
        "Period": "2008-2025",
        "dem1": 2008.5,
        "dem2": 2025.0,
        "sigma_aw3d30": 2.01273,
        "sigma_dt": 4.5,
        "src": "2010-2025",
    },
    {
        "Period": "2010-2025",
        "dem1": 2010.0,
        "dem2": 2025.0,
        "sigma_aw3d30": 2.01273,
        "sigma_dt": 4.5,
        "src": "2010-2025",
    },
    {
        "Period": "2000-2025",
        "dem1": 2000.0,
        "dem2": 2025.0,
        "sigma_aw3d30": 2.01273,
        "sigma_dt": 4.5,
        "src": "2000-2025",
    },
]

sec_rows = []
for p in sec_config:
    dh_val = dh_map[p["src"]]
    sigma_coreg = sigma_dh_map[p["src"]]
    area_val = area_map[p["src"]]
    area_pct_val = area_pct_map[p["src"]]
    dt = p["dem2"] - p["dem1"]

    # Spatial height uncertainty (quadrature)
    sigma_dh = math.sqrt(sigma_coreg ** 2 + p["sigma_aw3d30"] ** 2)
    dh_rate = dh_val / dt

    # Quadrature rate uncertainty with temporal uncertainty (sigma_dt)
    rel_dh_err = sigma_dh / abs(dh_val)
    rel_dt_err = p["sigma_dt"] / dt
    sigma_dh_rate = abs(dh_rate) * math.sqrt(rel_dh_err**2 + rel_dt_err**2)

    # Mass balance and uncertainty
    MB_mwe = dh_val * (RHO_ICE / RHO_WATER)
    MB_rate_mwe = MB_mwe / dt

    sigma_dMB = np.sqrt(
        (sigma_dh * (RHO_ICE / RHO_WATER)) ** 2
        + (dh_val * (SIGMA_RHO / RHO_WATER)) ** 2
    )
    rel_dMB_err = sigma_dMB / abs(MB_mwe)
    sigma_dMB_rate = abs(MB_rate_mwe) * math.sqrt(
        rel_dMB_err**2 + rel_dt_err**2
    )

    sec_rows.append({
        "Period": p["Period"],
        "dt (a)": dt,
        "Area (km²)": area_val,
        "Area (%)": area_pct_val,
        "dh (m)": dh_val,
        "σ_dh (m)": sigma_dh,
        "dh/dt (m/a)": dh_rate,
        "σ_dh_rate (m/a)": sigma_dh_rate,
        "MB (m w.e.)": MB_mwe,
        "σ_MB (m w.e.)": sigma_dMB,
        "MB rate (m w.e./a)": MB_rate_mwe,
        "σ_MB rate (m w.e./a)": sigma_dMB_rate,
    })

summary_secondary = pd.DataFrame(sec_rows)

# ==============================================================================
# SECTION 3: INTEGRATED TABLE GENERATION
# ==============================================================================

# Extract historical periods dynamically where end year <= 2000
historical_periods = df.loc[df["endyear"] <= 2000, "Period"].tolist()
summary_unchanged = summary_primary[
    summary_primary["Period"].isin(historical_periods)
].copy()

# Combine unchanged historical periods with updated DEM-stacking periods
summary_integrated = pd.concat(
    [summary_unchanged, summary_secondary], ignore_index=True
)

# Ordering integrated table consistently with DH_stats.csv order sequence
summary_integrated["start_year"] = (
    summary_integrated["Period"].str.split("-").str[0].astype(float)
)
summary_integrated = summary_integrated.sort_values(
    by=["start_year", "dt (a)"]
).drop(columns=["start_year"])

# Round all numerical values in tables to 2 decimal places
num_cols_primary = summary_primary.columns.drop("Period")
summary_primary[num_cols_primary] = summary_primary[num_cols_primary].round(2)

num_cols_sec = summary_secondary.columns.drop("Period")
summary_secondary[num_cols_sec] = summary_secondary[num_cols_sec].round(2)

num_cols_int = summary_integrated.columns.drop("Period")
summary_integrated[num_cols_int] = summary_integrated[num_cols_int].round(2)

# Ensure Output directory exists for CSV export
output_dir = Path("Outputcsv/")
output_dir.mkdir(parents=True, exist_ok=True)

# Print Output Tables
print("=" * 110)
print("TABLE 1: PRIMARY VARIOGRAM METHOD")
print("=" * 110)
print(summary_primary.to_string(index=False))

print("\n" + "=" * 110)
print("TABLE 2: DEM STACKING METHOD (WITH TEMPORAL UNCERTAINTIES)")
print("=" * 110)
print(summary_secondary.to_string(index=False))

print("\n" + "=" * 110)
print("TABLE 3: INTEGRATED ALL-PERIODS SUMMARY TABLE")
print("=" * 110)
print(summary_integrated.to_string(index=False))

# Save Table 1 and Table 3 to CSV files
summary_primary.to_csv(output_dir / "03_DH_DMB_Unc_Primary_Variogram_Method.csv", index=False)
summary_integrated.to_csv(output_dir / "03_DH_DMB_Unc_Integrated_All_Periods.csv", index=False)

print("\nTable 1 and Table 3 successfully exported to Outputcsv/ folder!")
print("Getting full MB stats")
MB_stats("03_MB_stats_in.csv")