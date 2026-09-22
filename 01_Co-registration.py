"""
Scripts used to estimate elevation changes and error for Universidad Glacier using XDEM (Hugonnet et al., 2020)

By Jorge Andrés Berkhoff
Department of Geography and Geosciences, FAU Erlangen–Nürnberg
"""

import os
import re
import warnings
from datetime import datetime
from pathlib import Path

import geoutils as gu
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist
from scipy.stats import binned_statistic
import xdem


# =============================================================================
# Helper Functions
# =============================================================================

def nmad(x):
    """Calculate Normalized Median Absolute Deviation (NMAD)."""
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    m = np.median(x)
    return float(1.4826 * np.median(np.abs(x - m)))


def residuo_aspecto(dh, mask, slope, aspect):
    """Calculate aspect residuals for DEM co-registration diagnostics."""
    m = mask & np.isfinite(dh) & (slope > 5)
    if np.sum(m) == 0:
        return np.nan
    y = (dh[m] - np.median(dh[mask & np.isfinite(dh)])) / np.tan(np.deg2rad(slope[m]))
    x = aspect[m]
    k = np.abs(y) < 150
    if np.sum(k) == 0:
        return np.nan
    p, _ = curve_fit(
        lambda a, A, B, C: A * np.cos(np.deg2rad(B - a)) + C,
        x[k],
        y[k],
        p0=[5, 180, 0],
        maxfev=9000,
    )
    return abs(p[0])


def spherical(h, nugget, psill, L):
    """Spherical variogram model."""
    h = np.asarray(h, float)
    out = nugget + psill * (1.5 * h / L - 0.5 * (h / L) ** 3)
    out[h >= L] = nugget + psill
    return out


def variogram_sigma(dh2d, mask, transform, res, maxlag=2000.0, n_sample=7000, nbins=25, seed=42):
    """
    Calculate NMAD, spatial correlation range (L), correlation area, and 
    stable terrain area directly from elevation differences in memory.
    """
    m = np.asarray(mask, bool) & np.isfinite(dh2d)
    med = np.median(dh2d[m])
    sdh = nmad(dh2d[m])
    keep = m & (np.abs(dh2d - med) < 3 * sdh)
    rows, cols = np.where(keep)

    xs = transform.c + (cols + 0.5) * transform.a
    ys = transform.f + (rows + 0.5) * transform.e
    vals = dh2d[rows, cols]

    rng = np.random.default_rng(seed)
    n = min(n_sample, len(vals))
    idx = rng.choice(len(vals), n, replace=False)

    xy = np.column_stack([xs[idx], ys[idx]])
    v = vals[idx]

    d = pdist(xy)
    g = pdist(v[:, None]) ** 2

    bins = np.linspace(0, maxlag, nbins + 1)
    bi = np.digitize(d, bins)
    lags, gamma = [], []

    for b in range(1, nbins + 1):
        s = bi == b
        if s.sum() > 50:
            lags.append((bins[b - 1] + bins[b]) / 2)
            gamma.append(0.5 * np.mean(g[s]))

    lags, gamma = np.array(lags), np.array(gamma)
    popt, _ = curve_fit(
        spherical,
        lags,
        gamma,
        p0=[sdh**2, np.var(v), maxlag / 4],
        bounds=([0, 0, res], [np.inf, np.inf, maxlag]),
        maxfev=20000,
    )

    return dict(
        nmad=sdh,
        L=popt[2],
        A_corr=np.pi * popt[2] ** 2,
        A_stable=int(m.sum()) * res**2,
    )


def variograma_plot(dh2d, mask, transform, res, maxlag=2000.0, n_sample=7000, nbins=25, seed=42):
    """Generate empirical variogram parameters and fit optimized curve for plotting."""
    m = np.asarray(mask, bool) & np.isfinite(dh2d)
    med = np.median(dh2d[m])
    sdh = nmad(dh2d[m])
    keep = m & (np.abs(dh2d - med) < 3 * sdh)
    rows, cols = np.where(keep)

    xs = transform.c + (cols + 0.5) * transform.a
    ys = transform.f + (rows + 0.5) * transform.e
    vals = dh2d[rows, cols]

    rng = np.random.default_rng(seed)
    n = min(n_sample, len(vals))
    idx = rng.choice(len(vals), n, replace=False)

    xy = np.column_stack([xs[idx], ys[idx]])
    v = vals[idx]

    d = pdist(xy)
    g = pdist(v[:, None]) ** 2

    bins = np.linspace(0, maxlag, nbins + 1)
    bi = np.digitize(d, bins)
    lags, gamma = [], []

    for b in range(1, nbins + 1):
        s = bi == b
        if s.sum() > 50:
            lags.append((bins[b - 1] + bins[b]) / 2)
            gamma.append(0.5 * np.mean(g[s]))

    lags, gamma = np.array(lags), np.array(gamma)
    popt, _ = curve_fit(
        spherical,
        lags,
        gamma,
        p0=[sdh**2, np.var(v), maxlag / 4],
        bounds=([0, 0, res], [np.inf, np.inf, maxlag]),
        maxfev=20000,
    )

    return sdh, lags, gamma, popt


def clean_dem(dem, lo=-5, hi=5000):
    """Filter out NoData or unphysical values in DEM."""
    dem.data[dem.data <= lo] = np.nan
    dem.data[dem.data > hi] = np.nan
    return dem


def parse_year(filename):
    """Extract a 4-digit year integer from a filename."""
    match = re.search(r"\b(19\d{2}|20\d{2})\b", Path(filename).name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract 4-digit year from filename: {filename}")


# =============================================================================
# Main Processing Script
# =============================================================================

def main():
    # Set Matplotlib parameters for plotting
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    script_name = Path(__file__).stem
    csv_table_path = "01_dem_pairs_co-registration_in.csv"  # File path to CSV input table

    # Static vector input files
    LAKES = "InputLakes/Lakes.shp"
    GLACIERS_EXTRA = "inputOtherGlaciers/Peripheral_Glaciers.shp"

    # Base output directories
    output_dems_base = Path("outputDEMs")
    output_masks_base = Path("outputMasks")
    cor_figures_base = Path("CoR_Figures")
    output_csv_dir = Path("Outputcsv") / script_name

    output_dems_base.mkdir(parents=True, exist_ok=True)
    output_masks_base.mkdir(parents=True, exist_ok=True)
    cor_figures_base.mkdir(parents=True, exist_ok=True)
    output_csv_dir.mkdir(parents=True, exist_ok=True)

    # Read pairs CSV
    pairs_df = pd.read_csv(csv_table_path, sep=r"\s+|,", engine="python")

    # List to store summary statistics for all pairs
    results_list = []

    for idx, row in pairs_df.iterrows():
        slave_file = str(row["Slave"]).strip()
        master_file = str(row["Master"]).strip()
        outline_file = str(row["outline"]).strip()

        # NEW (FIXED)
        slave_year = parse_year(slave_file)
        master_year = parse_year(master_file)

        # Formula for period folder naming: older_year-younger_year
        older_year = min(slave_year, master_year)
        younger_year = max(slave_year, master_year)
        period = f"{older_year}-{younger_year}"

        print("\n" + "=" * 60)
        print(f"PROCESSING PAIR {idx + 1}/{len(pairs_df)}: Period {period}")
        print(f"Slave: {slave_file} | Master: {master_file} | Outline: {outline_file}")
        print("=" * 60)

        # Create period subfolders within base output folders
        dem_subfolder = output_dems_base / period
        mask_subfolder = output_masks_base / period
        fig_subfolder = cor_figures_base / period

        dem_subfolder.mkdir(parents=True, exist_ok=True)
        mask_subfolder.mkdir(parents=True, exist_ok=True)
        fig_subfolder.mkdir(parents=True, exist_ok=True)

        DEM_slave_path = Path("inputDEMs") / slave_file
        DEM_master_path = Path("inputDEMs") / master_file
        GLACIER_path = Path("Outlines") / outline_file

        # Load & clean DEMs
        dem_slave = clean_dem(xdem.DEM(DEM_slave_path))
        dem_master = clean_dem(xdem.DEM(DEM_master_path))

        # Save original master DEM copy in the subfolder
        dem_master.to_file(dem_subfolder / Path(master_file).name)

        # Reproject slave DEM to match master grid
        dem_slave_r = dem_slave.reproject(dem_master, resampling="average")

        # Stable terrain masking
        glacier_mask = gu.Vector(GLACIER_path).create_mask(dem_master)
        lakes_mask = gu.Vector(LAKES).create_mask(dem_master) if Path(LAKES).exists() else glacier_mask & False
        slope = xdem.terrain.slope(dem_master)
        aspect = xdem.terrain.aspect(dem_master)

        extra_mask = (
            gu.Vector(GLACIERS_EXTRA).create_mask(dem_master)
            if Path(GLACIERS_EXTRA).exists()
            else glacier_mask & False
        )

        inlier_mask = (~glacier_mask & ~lakes_mask & ~extra_mask & (slope < 10)).data.squeeze()
        inlier_mask = np.asarray(inlier_mask, dtype=bool)

        # Save stable terrain mask GeoTIFF in period subfolder
        slave_stem = Path(slave_file).stem
        # Convert zeros (outside mask) to masked values prior to produce the Raster
        mask_array = np.ma.masked_where(~inlier_mask, inlier_mask.astype(np.uint8))
        stable_terrain_mask = gu.Raster.from_array(data=mask_array, transform=dem_master.transform, crs=dem_master.crs, nodata=0,)
        #stable_terrain_mask = gu.Raster.from_array(
        #    data=inlier_mask.astype(np.uint8),
        #    transform=dem_master.transform,
        #    crs=dem_master.crs,
        #    nodata=0,
        #)
        mask_output_path = mask_subfolder / f"stable_terrain_{slave_stem}.tif"
        stable_terrain_mask.to_file(mask_output_path)

        # Co-registration pipeline
        deramp = xdem.coreg.Deramp(poly_order=1)
        deramp.fit(reference_elev=dem_master, to_be_aligned_elev=dem_slave_r, inlier_mask=inlier_mask, subsample=100000)
        dem_slave_d = deramp.apply(dem_slave_r)

        registros, best = [], None
        slope_2d = slope.data.squeeze().filled(np.nan)
        aspect_2d = aspect.data.squeeze().filled(np.nan)

        for k in range(1, 15):
            nk = xdem.coreg.NuthKaab(max_iterations=k)
            nk.fit(reference_elev=dem_master, to_be_aligned_elev=dem_slave_d, inlier_mask=inlier_mask, subsample=100000)
            dh = (dem_slave_d - nk.apply(dem_master)).data.squeeze().filled(np.nan)
            dh = dh - np.median(dh[inlier_mask & np.isfinite(dh)])
            nm = nmad(dh[inlier_mask & np.isfinite(dh)])
            am = residuo_aspecto(dh, inlier_mask, slope_2d, aspect_2d)
            registros.append((k, nm, am))
            if best is None or am < best[2]:
                best = (k, nm, am)

        best_k = best[0]
        nk_best = xdem.coreg.NuthKaab(max_iterations=best_k)
        nk_best.fit(reference_elev=dem_master, to_be_aligned_elev=dem_slave_d, inlier_mask=inlier_mask, subsample=100000)
        dem_slave_coreg = nk_best.apply(dem_slave_d)

        # Median debias over stable terrain
        dh_st = (dem_slave_coreg - dem_master).data.squeeze().filled(np.nan)[inlier_mask]
        median_offset = float(np.nanmedian(dh_st))
        dem_slave_final = dem_slave_coreg + median_offset

        # Histogram Plot
        dh_before = (dem_slave_r - dem_master).data.squeeze().filled(np.nan)[inlier_mask]
        dh_after = (dem_slave_final - dem_master).data.squeeze().filled(np.nan)[inlier_mask]
        dh_before = dh_before[np.isfinite(dh_before)]
        dh_after = dh_after[np.isfinite(dh_after)]

        XR = 45
        bins = np.linspace(-XR, XR, 91)

        fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
        for a, v, letter, col, label in [(ax[0], dh_before, "a", "black", "no coregistration"), (ax[1], dh_after, "b", "red", "coregistration")]:
            a.hist(v, bins=bins, histtype="step", color=col, linewidth=1.2, zorder=3)
            a.axvline(np.mean(v), color="k", ls="--", lw=1.0, zorder=4)
            a.axvline(np.median(v), color="navy", ls=":", lw=1.2, zorder=4)
            stats = (f"{label}\nMean = {np.mean(v):+.2f} m\nMedian = {np.median(v):+.2f} m\nNMAD = {nmad(v):.2f} m")
            a.text(0.97, 0.97, stats, transform=a.transAxes, ha="right", va="top", fontsize=7.5, bbox=dict(boxstyle="square,pad=0.35", fc="white", ec="0.5", lw=0.7))
            a.text(0.03, 0.97, letter, transform=a.transAxes, fontsize=12, fontweight="bold", va="top", ha="left")
            a.set_xlabel("dh over stable terrain (m)")
            a.spines[["top", "right"]].set_visible(False)
            a.set_xlim(-XR, XR)

        ax[0].set_ylabel("Frequency")
        fig.tight_layout()
        fig.savefig(fig_subfolder / f"hist_control_step_{slave_stem}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

        # Fit & Aspect Scatter Plot
        dh_before_2d = (dem_slave_r - dem_master).data.squeeze().filled(np.nan)
        dh_after_2d = (dem_slave_final - dem_master).data.squeeze().filled(np.nan)

        def nk(a, A, B, C):
            return A * np.cos(np.deg2rad(B - a)) + C

        def panel(ax, dh2d, letter, color):
            m = inlier_mask & np.isfinite(dh2d) & (slope_2d > 5)
            base = np.median(dh2d[inlier_mask & np.isfinite(dh2d)])
            y = dh2d[m] / np.tan(np.deg2rad(slope_2d[m]))
            x = aspect_2d[m]
            k = np.abs(y) < 250
            x, y = x[k], y[k]

            p, _ = curve_fit(nk, x, y, p0=[5, 180, np.median(y)], maxfev=12000)
            A, B, C = p
            B %= 360
            dx, dy = A * np.sin(np.deg2rad(B)), A * np.cos(np.deg2rad(B))

            rng = np.random.default_rng(1)
            idx = rng.choice(len(x), min(12000, len(x)), replace=False)
            ax.scatter(x[idx], y[idx], s=1.2, color="0.35", alpha=0.35, edgecolors="none")

            xx = np.linspace(0, 360, 361)
            ax.plot(xx, nk(xx, *p), color=color, lw=2.4)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xlim(0, 360)
            ax.set_ylim(-200, 320)
            ax.set_xlabel("Aspect (°)")
            ax.spines[["top", "right"]].set_visible(False)
            ax.text(0.03, 0.97, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="left")
            ax.text(0.03, 0.05, f"Δx: {dx:+.1f} m\nΔy: {dy:+.1f} m\nΔz: {base:+.1f} m", transform=ax.transAxes, color="red", fontsize=9, va="bottom", fontweight="bold")

        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
        panel(ax[0], dh_before_2d, "a", "red")
        panel(ax[1], dh_after_2d, "b", "red")
        ax[0].set_ylabel(r"$dh/\tan(\alpha)$ [m]")

        fig.tight_layout()
        fig.savefig(fig_subfolder / f"initial_final_fit_{slave_stem}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

        # Variogram & Uncertainty Statistics
        transform = dem_master.transform
        resolution = dem_master.res[0]

        sdh, lags, gamma, popt = variograma_plot(
            dh_after_2d, inlier_mask, transform, resolution, maxlag=2000.0, n_sample=7000
        )

        stats_var = variogram_sigma(
            dh_after_2d, inlier_mask, transform, resolution, maxlag=2000.0, n_sample=7000
        )
        nmad_v = stats_var['nmad']
        L_v = stats_var['L']
        A_corr_km2 = stats_var['A_corr'] / 1e6

        # Plot Variogram
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.plot(lags, gamma, 'o', color='#4a4a4a', markersize=5, label='empirical')
        fine_lags = np.linspace(0, 2000, 200)
        ax.plot(fine_lags, spherical(fine_lags, *popt), '-', color='#c0392b', lw=2, label='spherical fit')

        sill_val = popt[0] + popt[1]
        ax.axhline(sill_val, color='#666666', linestyle=':', lw=1.2)
        ax.axvline(L_v, color='darkblue', linestyle='--', lw=1.2)

        ax.text(L_v + 40, 0.22 * sill_val, f"range L = {L_v:.0f} m", color='darkblue', fontsize=9, va='center')
        stats_text = f"NMAD = {nmad_v:.2f} m\n$A_{{corr}} = \\pi L^2 = {A_corr_km2:.2f}$ km²"
        ax.text(0.95, 0.05, stats_text, transform=ax.transAxes, fontsize=8.5, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9))

        ax.set_xlabel("lag distance (m)", fontsize=9)
        ax.set_ylabel(r"semivariance $\gamma$ (m²)", fontsize=9)
        ax.set_xlim(-50, 2150)
        ax.legend(frameon=False, loc='center right', bbox_to_anchor=(0.9, 0.55))
        ax.spines[['top', 'right']].set_visible(False)

        fig.tight_layout()
        fig.savefig(fig_subfolder / f"semivariogram_{slave_stem}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

        # Save output coregistered slave DEM in period subfolder
        output_dem_path = dem_subfolder / f"{slave_stem}_coreg.tif"
        dem_slave_final.to_file(output_dem_path)

        # Append statistics to summary table
        results_list.append({
            "Period": period,
            "L": round(L_v, 2),
            "NMAD": round(nmad_v, 2),
            "Acorr": round(A_corr_km2, 2)
        })

    # Save summary CSV table inside Outputcsv/<script_name>/
    results_df = pd.DataFrame(results_list)
    output_csv_file = output_csv_dir / f"{script_name}_uncertainty.csv"
    results_df.to_csv(output_csv_file, index=False)

    print("\n" + "=" * 60)
    print("ALL PAIRS PROCESSED SUCCESSFULLY")
    print(f"Uncertainty summary table saved to: {output_csv_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()