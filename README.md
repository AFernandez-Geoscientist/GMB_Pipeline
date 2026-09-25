# GMB_Pipeline
Mahmud et al paper - Universidad Glacier
**GMB_Pipeline** is a Python-based processing pipeline designed to calculate geodetic glacier mass balance, elevation change ($\Delta h$), spatial co-registration using `xdem`, variogram uncertainty modeling, and cumulative mass balance statistics.

---

## 📋 Table of Contents
1. [Repository Structure](#-repository-structure)
2. [Pipeline Overview](#-pipeline-overview)
3. [Prerequisites & Environment Setup](#-prerequisites--environment-setup)
4. [Input Requirements](#-input-requirements)
5. [Execution Sequence](#-execution-sequence)
6. [Cross-Platform Compatibility (macOS & Windows)](#-cross-platform-compatibility-macos--windows)

---

## 📂 Repository Structure

To ensure relative paths resolve seamlessly across different operating systems (macOS, Windows, Linux), set up your workspace directory as follows:

```text
GMB_Pipeline/
├── inputDEMs/                  # Input DEM raster files (.tif)
├── Outlines/                   # Glacier outline shapefiles (.shp)
├── InputLakes/                 # Off-glacier lake shapefiles (.shp)
├── inputOtherGlaciers/         # Peripheral glacier shapefiles (.shp)
├── outputDEMs/                 # Coregistered DEM outputs (auto-created)
├── outputMasks/                # Stable terrain mask outputs (auto-created)
├── CoR_Figures/                # Co-registration diagnostics plots (auto-created)
├── outputDH/                   # Elevation difference rasters (auto-created)
├── Outputcsv/                  # Generated CSV tables (auto-created)
├── Figures/                    # Multi-panel spatial plots (auto-created)
├── 01_Co-registration.py       # Step 1: Sub-pixel co-registration & spatial error
├── 02_DH.py                    # Step 2: Elevation difference & spatial figures
├── 03_DH_DMB_Uncertainty.py    # Step 3: Mass balance & temporal uncertainty
├── MB_stats.py                 # Module called within script 03
├── 01_dem_pairs_co-registration_in.csv
├── 02_DH_dem_pairs_in.csv
├── 03_MB_stats_in.csv
├── environment.yml             # Conda environment file
└── README.md
```

---

## 🔄 Pipeline Overview

The workflow runs in three main steps:

1. **`01_Co-registration.py`**:
   - Performs sub-pixel DEM co-registration (deramping + Nuth & Kääb iteration) using stable terrain.
   - Filters out non-stable terrain using glacier, lake, and peripheral glacier vectors.
   - Calculates spatial autocorrelation range ($L$) and empirical variogram models.
   - Outputs coregistered DEM rasters, stable terrain masks, summary tables, and diagnostic figures.

2. **`02_DH.py`**:
   - Computes spatial elevation differences ($\Delta h$ = $\text{DEM}_{\text{end}}$ - $\text{DEM}_{\text{ini}}$).
   - Exports masked/unmasked elevation change GeoTIFFs.
   - Calculates statistical metrics (Mean, Median, Std, $3\sigma$ cutoffs) over glacier outlines.
   - Generates multi-panel spatial plots grouped by plot ID.

3. **`03_DH_DMB_Uncertainty.py`**:
   - Integrates spatial variogram models from Step 1 with temporal error propagation (DEM stacking).
   - Computes geodetic mass balance ($\text{m w.e.}$ and $\text{m w.e./a}$) and uncertainties.
   - Automatically executes `MB_stats.py` as an embedded routine to compute cumulative time-series stats (`Cum dh`, `Cum MB`).

---

## 📦 Prerequisites & Environment Setup

It is strongly recommended to use **Conda** or **Mamba** to manage geospatial dependencies (`gdal`, `rasterio`, `geopandas`, `xdem`).

### 1. Clone the repository
```bash
git clone https://github.com/AFernandez-Geoscientist/GMB_Pipeline.git
cd GMB_Pipeline
```

### 2. Create and activate the Conda environment

You can create an environment using the provided `environment.yml` file:

```bash
conda env create -f gmb_env.yml
conda activate gmb_env
```

*If creating manually, install the following key dependencies:*
```bash
conda create -n gmb_env -c conda-forge python=3.10 xdem geoutils geopandas rasterio scipy matplotlib pandas numpy cmaps pyproj
conda activate gmb_env
```

---

## 📄 Input Requirements

Provide the required input datasets offline in their designated folders:

### 1. Raster & Vector Input Directories
* **`inputDEMs/`**: Place raw DEM `.tif` files.
* **`Outlines/`**: Place primary glacier boundary `.shp` files.
* **`InputLakes/`**: Place lake vector `.shp` files (optional, for stable terrain masking).
* **`inputOtherGlaciers/`**: Place surrounding glacier vector `.shp` files (optional).

### 2. CSV Configuration Tables
* **`01_dem_pairs_co-registration_in.csv`**: Specifies slave/master DEM filename pairs and corresponding outline shapefile names.
* **`02_DH_dem_pairs_in.csv`**: Specifies initial/final years, outline shapefile names, and plot ID groupings for figure generation.
* **`03_MB_stats_in.csv`**: Specifies periods to extract cumulative mass balance statistics.

---

## 🚀 Execution Sequence

Execute the pipeline in sequential order from the root directory of the repository:

### Step 1: DEM Co-registration & Spatial Autocorrelation
```bash
python 01_Co-registration.py
```

### Step 2: Elevation Difference ($\Delta h$) Computation & Plotting
```bash
python 02_DH.py
```

### Step 3: Mass Balance & Cumulative Uncertainty Statistics
```bash
python 03_DH_DMB_Uncertainty.py
```
*(Note: `MB_stats.py` is called internally by `03_DH_DMB_Uncertainty.py` and does not need to be run separately).*

---

## 💻 Cross-Platform Compatibility (macOS & Windows)

All scripts utilize Python's `pathlib.Path` library to ensure cross-platform compatibility across Windows, macOS, and Linux.

When running on **Windows**:
1. **Path Separators**: Use standard forward slashes (`/`) in CSV configuration tables (e.g., `inputDEMs/DEM_2000.tif`). `pathlib` automatically converts paths to Windows backslashes (`\`) internally.
2. **Execution Environment**: Always execute scripts within **Anaconda Prompt** or a terminal where your Conda environment is activated. Running directly from standard Windows PowerShell or Command Prompt without activation may trigger GDAL C-library DLL binding errors.
3. **File Encoding**: Save all input `.csv` text files with `UTF-8` encoding to prevent line-ending or encoding mismatch issues.
