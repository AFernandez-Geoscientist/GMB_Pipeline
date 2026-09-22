"""
Script to compute elevation differences (dh), statistics, export rasters, 
and generate multi-panel spatial plots grouped by plot ID.

 FAU Erlangen-Nürnberg / Universidad Glacier Research
"""

import os
import math
import warnings
from pathlib import Path

import geoutils as gu
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.features import geometry_mask
from pyproj import Transformer
import xdem
import cmaps


# =============================================================================
# Helper Functions
# =============================================================================

def find_dem_file(folder_path: Path, year: int) -> Path:
    """Find DEM file for a given year inside folder_path matching standard or coreg pattern."""
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Search candidates: exact filename or _coreg suffix
    candidates = list(folder_path.glob(f"*{year}*.tif"))
    if not candidates:
        raise FileNotFoundError(f"No DEM found for year {year} in {folder_path}")

    # Prefer _coreg file if multiple match
    coreg_candidates = [f for f in candidates if "_coreg" in f.stem]
    if coreg_candidates:
        return coreg_candidates[0]
    return candidates[0]


def compute_stats_dict(period_text, data_array, total_polygon_pixels=np.nan):
    """Computes statistical summary dictionary for pixels."""
    if data_array.size == 0 or not np.any(np.isfinite(data_array)):
        return {
            'Ini_Period': period_text.split('-')[0],
            'End_Period': period_text.split('-')[1],
            'Total_Pixels_In_Polygon': total_polygon_pixels,
            'Count': 0, 'Sum': np.nan, 'Mean': np.nan, 'Median': np.nan,
            'Std': np.nan, 'Min': np.nan, 'Max': np.nan,
            'Cutoff_Lower_3Std': np.nan, 'Cutoff_Upper_3Std': np.nan,
            'Pixels_Within_3Std': 0, 'Mean_Within_3Std': np.nan
        }
    
    valid_data = data_array[np.isfinite(data_array)]
    if valid_data.size == 0:
        return {
            'Ini_Period': period_text.split('-')[0],
            'End_Period': period_text.split('-')[1],
            'Total_Pixels_In_Polygon': total_polygon_pixels,
            'Count': 0, 'Sum': np.nan, 'Mean': np.nan, 'Median': np.nan,
            'Std': np.nan, 'Min': np.nan, 'Max': np.nan,
            'Cutoff_Lower_3Std': np.nan, 'Cutoff_Upper_3Std': np.nan,
            'Pixels_Within_3Std': 0, 'Mean_Within_3Std': np.nan
        }

    mean_val = float(np.mean(valid_data))
    std_val = float(np.std(valid_data))
    cutoff_lower = mean_val - (3 * std_val)
    cutoff_upper = mean_val + (3 * std_val)
    data_3std = valid_data[(valid_data >= cutoff_lower) & (valid_data <= cutoff_upper)]

    parts = period_text.split('-')
    return {
        'Ini_Period': parts[0],
        'End_Period': parts[1],
        'Total_Pixels_In_Polygon': int(total_polygon_pixels) if not np.isnan(total_polygon_pixels) else np.nan,
        'Count': int(valid_data.size),
        'Sum': float(np.sum(valid_data)),
        'Mean': mean_val,
        'Median': float(np.median(valid_data)),
        'Std': std_val,
        'Min': float(np.min(valid_data)),
        'Max': float(np.max(valid_data)),
        'Cutoff_Lower_3Std': cutoff_lower,
        'Cutoff_Upper_3Std': cutoff_upper,
        'Pixels_Within_3Std': int(data_3std.size),
        'Mean_Within_3Std': float(np.mean(data_3std)) if data_3std.size > 0 else np.nan
    }


def setup_axis_formatting(ax, period_text):
    """Applies spatial bounds, tick labels, and period tags."""
    lat_ticks = [-(34 + 42/60), -(34 + 40/60), -(34 + 38/60)]
    lat_labels = ["34º42'S", "34º40'S", "34º38'S"]
    lon_ticks = [-(70 + 22/60), -(70 + 19/60)]
    lon_labels = ["70º22'W", "70º19'W"]

    ax.text(0.03, 0.97, period_text, transform=ax.transAxes, 
            ha='left', va='top', fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.7, pad=0.2, edgecolor="white"), zorder=4)
    ax.set_xlim(-70.38, -70.30)
    ax.set_ylim(-34.722, -34.625)
    ax.set_yticks(lat_ticks)
    ax.set_yticklabels(lat_labels, rotation='vertical', va='center', fontsize=9)
    ax.set_xticks(lon_ticks)
    ax.set_xticklabels(lon_labels, fontsize=9)


def determine_grid_dimensions(n_items):
    """Determines rows and columns based on item count."""
    if n_items == 1:
        return 1, 1
    elif n_items == 2:
        return 1, 2
    elif n_items <= 4:
        return 2, 2
    elif n_items <= 6:
        return 2, 3
    elif n_items <= 9:
        return 3, 3
    else:
        cols = 3
        rows = math.ceil(n_items / cols)
        return rows, cols


# =============================================================================
# Main Script Execution
# =============================================================================

def main():
    # Input CSV table path
    csv_table_path = "02_DH_dem_pairs_in.csv"
    
    # Path Directories
    output_dems_dir = Path("outputDEMs")
    outlines_dir = Path("Outlines")
    output_csv_dir = Path("Outputcsv")
    output_dh_dir = Path("outputDH")
    figures_dir = Path("Figures")

    # Create missing directories
    output_csv_dir.mkdir(parents=True, exist_ok=True)
    output_dh_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Visualization settings
    cmap = cmaps.MPL_PuOr
    norm = Normalize(vmin=-50, vmax=50)
    no_data_patch = Patch(facecolor='none', edgecolor='grey', hatch='xxx', label='No data')

    # Read CSV table
    pairs_df = pd.read_csv(csv_table_path, sep=r"\s+|,", engine="python")

    dh_stats_list = []
    plot_data_registry = {}

    for idx, row in pairs_df.iterrows():
        end_year = int(row["End_year"])
        ini_year = int(row["Ini_year"])
        outline_file = str(row["outline"]).strip()
        plot_id = int(row["plot"])

        period_key = f"{ini_year}-{end_year}"

        # Resolve folder name: older_year-younger_year
        older_year = min(ini_year, end_year)
        younger_year = max(ini_year, end_year)
        folder_name = f"{older_year}-{younger_year}"
        subfolder_path = output_dems_dir / folder_name

        # Find DEM files for initial and final years inside folder
        ini_dem_path = find_dem_file(subfolder_path, ini_year)
        end_dem_path = find_dem_file(subfolder_path, end_year)

        print(f"\n[{idx+1}/{len(pairs_df)}] Processing dh for period {period_key}...")
        print(f"  Ini DEM: {ini_dem_path.name}")
        print(f"  End DEM: {end_dem_path.name}")

        # Load DEM rasters using xdem
        dem_ini = xdem.DEM(str(ini_dem_path))
        dem_end = xdem.DEM(str(end_dem_path))

        # Reproject end DEM to match initial DEM grid if geometries differ
        if dem_end.shape != dem_ini.shape or dem_end.transform != dem_ini.transform:
            dem_end = dem_end.reproject(dem_ini, resampling="average")

        # Compute full elevation difference: dh = DEM_end - DEM_ini
        dh_raster = dem_end - dem_ini
        dh_data = dh_raster.data.squeeze().filled(np.nan)

        # Save Full dh GeoTIFF
        full_dh_filename = output_dh_dir / f"dh_{period_key}.tif"
        dh_raster.to_file(str(full_dh_filename))

        # Vector outline masking
        shp_path = outlines_dir / outline_file
        shape_gpd = gpd.read_file(shp_path).to_crs(epsg=4326)

        glacier_vector = gu.Vector(str(shp_path))
        outline_mask_raster = glacier_vector.create_mask(dh_raster)
        outline_mask = np.squeeze(np.asarray(outline_mask_raster.data))

        # Apply mask: keep values INSIDE polygon boundary.
        # IMPORTANT: use dh_data (already .filled(np.nan)), not a raw np.asarray()
        # of the masked array — the latter exposes the underlying nodata sentinel
        # values instead of NaN, which inflates pixel counts and can blow up
        # Mean/Std into inf.
        dh_masked_data = np.where(outline_mask, dh_data, np.nan)

        # Save Masked dh GeoTIFF
        masked_dh_raster = gu.Raster.from_array(
            data=dh_masked_data,
            transform=dh_raster.transform,
            crs=dh_raster.crs,
            nodata=np.nan
        )
        masked_dh_filename = output_dh_dir / f"dh_{period_key}_masked.tif"
        masked_dh_raster.to_file(str(masked_dh_filename))

        # Compute Statistics for Masked dh
        total_pixels_in_polygon = int(np.sum(outline_mask))
        valid_masked_pixels = dh_masked_data[np.isfinite(dh_masked_data)]

        stats = compute_stats_dict(period_key, valid_masked_pixels, total_polygon_pixels=total_pixels_in_polygon)
        dh_stats_list.append(stats)

        # Process spatial visualization bounds for plotting
        if plot_id > 0:
            with rasterio.open(str(full_dh_filename)) as src:
                out_image, out_transform = mask(src, shape_gpd.to_crs(src.crs).geometry, crop=True, nodata=np.nan, filled=True)
                raster_arr = out_image[0].astype(float)
                
                raw_masked_img = np.ma.masked_where(np.isnan(raster_arr), raster_arr)

                left_native = out_transform.c
                top_native = out_transform.f
                right_native = left_native + out_transform.a * raster_arr.shape[1]
                bottom_native = top_native + out_transform.e * raster_arr.shape[0]

                transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                left_lon, top_lat = transformer.transform(left_native, top_native)
                right_lon, bottom_lat = transformer.transform(right_native, bottom_native)
                extent = [left_lon, right_lon, bottom_lat, top_lat]

                if plot_id not in plot_data_registry:
                    plot_data_registry[plot_id] = []

                plot_data_registry[plot_id].append({
                    'period_text': period_key,
                    'shape': shape_gpd,
                    'img': np.clip(raw_masked_img, -50, 50),
                    'extent': extent
                })

    # Save Stats CSV rounded to 2 decimal places
    stats_df = pd.DataFrame(dh_stats_list)
    output_stats_csv = output_csv_dir / "02_DH_stats.csv"
    stats_df.round(2).to_csv(output_stats_csv, index=False, float_format="%.2f")
    print(f"\nSaved statistics table to: {output_stats_csv}")

    # Generate Figures dynamically by plot_id group
    for plot_id, items in plot_data_registry.items():
        n_items = len(items)
        rows, cols = determine_grid_dimensions(n_items)

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.5 * rows), sharex=True, sharey=True, layout='constrained')
        
        # Format axes as 2D array
        if rows == 1 and cols == 1:
            axes_grid = np.array([[axes]])
        elif rows == 1 or cols == 1:
            axes_grid = np.atleast_2d(axes)
            if rows == 1:
                axes_grid = axes_grid.reshape(1, -1)
            else:
                axes_grid = axes_grid.reshape(-1, 1)
        else:
            axes_grid = axes

        im_ref = None
        for i, item in enumerate(items):
            r, c = divmod(i, cols)
            ax = axes_grid[r, c]

            shape = item['shape']
            img = item['img']
            period_text = item['period_text']
            extent = item['extent']

            shape.plot(ax=ax, facecolor='none', edgecolor='grey', hatch='xxx', linewidth=0.5, zorder=1)
            if i == 0:
                ax.legend(handles=[no_data_patch], loc='lower right', edgecolor='white')

            im_ref = ax.imshow(img, cmap=cmap, norm=norm, extent=extent, zorder=2, origin='upper')
            shape.boundary.plot(ax=ax, edgecolor='gray', linewidth=2.5, zorder=3)
            setup_axis_formatting(ax, period_text)

        # Hide any unused axis panels
        for i in range(n_items, rows * cols):
            r, c = divmod(i, cols)
            axes_grid[r, c].axis('off')

        # Add shared colorbar centered below plots
        if im_ref:
            cbar = fig.colorbar(im_ref, ax=axes_grid, orientation='horizontal', shrink=0.6, aspect=30, pad=0.03)
            cbar.set_label('Thickness Change Δh (m)', fontsize=11)

        fig_filename = figures_dir / f"DH_plot_{plot_id}.png"
        plt.savefig(fig_filename, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Saved figure for plot group {plot_id} to: {fig_filename}")


if __name__ == "__main__":
    main()