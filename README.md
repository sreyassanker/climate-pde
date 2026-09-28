# Climate-PDE: Multi-Variable SINDy for Regional Climate Dynamics

Discovering interpretable, physically consistent governing partial differential equations (PDEs) for regional multivariate climate dynamics from gridded satellite and reanalysis observations using sparse regression (SINDy and WSINDy).

**Manuscript:** Sreyas S. S., Saied Pirasteh (2026). *Multi-Variable SINDy: Discovering Coupled Governing Equations for Climate Dynamics from Gridded Observations*. Submitted to **Computers & Geosciences** (Elsevier).

---

## Overview & Implementation

This repository provides an end-to-end framework to:
1. **Acquire & Harmonize Multi-Source Climate Data:** Download and align monthly ERA5 reanalysis ($t_{2m}$, $q$, $tp$, $d_{2m}$), CHIRPS v2.0 precipitation, and MODIS Terra/Aqua Land Surface Temperature onto a common 0.05° spatial grid over the southeastern United States (2000–2023, 288 monthly time steps).
2. **Discover Single-Variable PDEs:** Identify dominant thermodynamic and advection terms ($\sin(2\pi t/12)$, $\partial T/\partial y$, $T \cdot \partial T/\partial y$) using Sequential Thresholded Least Squares (STLSQ) on raw fields with explicit seasonal forcing.
3. **Discover Multivariate Coupled Equations:** Model coupled interactions between temperature ($t_{2m}$) and specific humidity ($q$), capturing advective and moisture-coupling feedback terms ($T \cdot q$, $T \cdot \partial q/\partial y$, $q \cdot \partial T/\partial y$).
4. **Evaluate Spatial Generalization:** Perform spatial cross-validation across three distinct climate sub-regions (Florida peninsula, Georgia/Alabama coastal plain, and Carolinas piedmont).
5. **Assess Forward Predictive Skill:** Numerically integrate discovered equations up to 12 months ahead across 22 evaluation start dates, benchmarking against persistence, monthly climatology, and fit-once seasonal ARIMA baselines.
6. **Quantify Robustness & Sensitivity:** Evaluate weak-form (WSINDy) recovery on deseasonalized anomalies, Gaussian noise perturbation sensitivity, spatial resolution effects (0.05° vs. 0.25° grid), and block-bootstrap confidence intervals (100 replicates, 12-month blocks).
7. **Generate Publication Figures:** Produce camera-ready 600 dpi PNG, vector PDF, and TIFF figures matching all manuscript visualizations (Figures 1–6).

---

## Repository Structure

```
├── run_all.py                     # Full pipeline orchestrator (root level)
├── requirements.txt               # Python package dependencies
├── .env(example)                  # Template for NASA Earthdata & CDS credentials
├── README.md                      # Documentation and usage guide
├── LICENSE                        # MIT License
├── src/
│   ├── preprocessing/             # Data acquisition & harmonization
│   │   ├── download_era5.py       # Download ERA5 monthly reanalysis via Copernicus CDS API
│   │   ├── download_chirps.py     # Download & subset CHIRPS v2.0 precipitation
│   │   ├── download_modis.py      # Download MODIS monthly LST (MOD11C3 / MYD11C3)
│   │   └── merge_datasets.py      # Merge into unified data/climate_multivariate.nc
│   ├── discovery/                 # PDE discovery, sensitivity, and validation
│   │   ├── pde_discovery.py       # PySINDy weak-form discovery & threshold sensitivity (Tables 1-2)
│   │   ├── validate_pde.py        # Single-variable SINDy & spatial cross-validation (Table 3)
│   │   ├── pde_multivar.py        # Multivariate coupled PDE discovery (Section 3.2)
│   │   ├── robustness_checks.py   # Weak-form confirmation & noise sensitivity (Sections 2.2/3.1)
│   │   ├── bootstrap_ci.py        # 100-replicate block bootstrap confidence intervals (Section 3.1)
│   │   ├── resolution_compare.py  # 0.05° vs 0.25° resolution sensitivity (Table 4)
│   │   └── tune_regularization.py # Regularization threshold scan for forward integration stability
│   └── prediction/                # Forecasting & baselines
│       ├── forward_integrate.py   # 12-month forward integration skill vs persistence/climatology
│       └── arima_baseline.py      # Fit-once seasonal ARIMA baseline comparison
├── figures/                       # Publication figures (fig1–fig6)
│   ├── scripts/                   # Figure generation scripts (run from project root)
│   │   ├── fig1_study_area.py     # Study domain map and continental US overview (Figure 1)
│   │   ├── fig2_methodology.drawio# Methodological flowchart diagram (Figure 2 source)
│   │   ├── fig3_pde_coefficient_maps.py # 120,000-cell spatial coefficient maps (Figure 3)
│   │   ├── fig4_cross_validation.py     # 3×3 hexbin spatial cross-validation matrix (Figure 4)
│   │   ├── fig5_multi_var_coupling.py   # Domain-wide multivariate coupling maps (Figure 5)
│   │   └── fig6_forward_skill.py        # 9-panel forecast skill & spatial error maps (Figure 6)
│   └── *.png/.pdf/.tiff           # Generated publication figures (600 dpi, vector, TIFF)
├── data/                          # Gridded climate NetCDF datasets (see Zenodo)
│   ├── climate_multivariate.nc    # Unified multi-variable dataset (296 MB)
│   ├── era5/                      # ERA5 monthly reanalysis NetCDF (487 MB)
│   ├── chirps/                    # CHIRPS monthly precipitation NetCDF (35 MB)
│   └── modis/                     # MODIS monthly LST NetCDF (69 MB)
├── output/                        # Discovery outputs (.npz files containing fitted coefficients)
└── templates/                     # Official Computers & Geosciences templates (LaTeX & Word)
```

---

## Quick Start & Reproducibility

### 1. Environment Setup

Ensure Python 3.10+ (tested on Python 3.14) is installed:

```bash
git clone https://github.com/sreyassanker/climate-pde.git
cd climate-pde
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Data

Download the unified dataset `climate_multivariate.nc` (296 MB) from Zenodo:
* **DOI:** [10.5281/zenodo.20805499](https://doi.org/10.5281/zenodo.20805499)
* Place the file into the `data/` directory:
  ```bash
  mkdir -p data
  # Move or symlink downloaded climate_multivariate.nc to data/climate_multivariate.nc
  ```

Alternatively, raw datasets can be re-downloaded using scripts in `src/preprocessing/` (requires CDS API credentials in `~/.cdsapirc` and NASA Earthdata credentials in `.env`).

### 3. Running the Pipeline

All scripts should be executed from the **project root directory**:

* **Quick Sanity Check (~30 seconds):**
  Runs single-variable discovery, cross-validation, and multivariate discovery:
  ```bash
  python3 run_all.py --fast
  ```

* **Full Pipeline Execution:**
  Executes all discovery routines, sensitivity scans, forward integration, and figure generators:
  ```bash
  python3 run_all.py
  ```

* **Individual Module Execution:**
  Each component can be run independently:
  ```bash
  python3 src/discovery/validate_pde.py          # Table 3 (Spatial cross-validation)
  python3 src/discovery/pde_discovery.py --sensitivity # Table 2 (Threshold robustness)
  python3 src/discovery/pde_multivar.py          # Section 3.2 (Coupled multivariate equations)
  python3 src/discovery/resolution_compare.py    # Table 4 (0.05° vs 0.25° sensitivity)
  python3 src/discovery/bootstrap_ci.py          # Section 3.1 (Bootstrap confidence intervals)
  python3 src/prediction/forward_integrate.py    # Section 3.3 (Forward integration skill)
  python3 src/prediction/arima_baseline.py       # Section 3.3 (ARIMA comparison)
  python3 figures/scripts/fig4_cross_validation.py # Render Figure 4
  python3 figures/scripts/fig6_forward_skill.py    # Render Figure 6
  ```

---

## Data Sources & Provenance

| Dataset | Provider | Variables | Resolution | Temporal Range | Citation / License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ERA5 Monthly** | ECMWF / Copernicus Climate Change Service | $t_{2m}$, $q$, $tp$, $d_{2m}$ | 0.25° (regridded to 0.05°) | 2000–2023 | Hersbach et al. (2020) |
| **CHIRPS v2.0** | UCSB Climate Hazards Center | Precipitation | 0.05° | 2000–2023 | Funk et al. (2015) |
| **MODIS Terra/Aqua** | NASA EOSDIS LP DAAC | Land Surface Temp (Day/Night) | 0.05° CMG | 2000–2023 | Wan et al. (2021) |

The complete harmonized dataset is archived on Zenodo:  
**Zenodo Record:** [https://doi.org/10.5281/zenodo.20805499](https://doi.org/10.5281/zenodo.20805499)

---

## Scope & Methodological Limitations

In the interest of scientific rigor and complete transparency, users and reviewers should note the following physical and methodological boundaries:

1. **Monthly Temporal Aggregation:**  
   The dataset operates at monthly resolution over a 24-year period (2000–2023). High-frequency atmospheric phenomena (e.g., diurnal heating, turbulent boundary layer eddies, individual convective frontal systems) are averaged out. The discovered equations govern large-scale monthly thermal tendencies and regional advective balancing rather than instantaneous fluid dynamics.
2. **Derivative Noise Sensitivity:**  
   Numerical finite differences applied to gridded observations are sensitive to high-frequency noise. While weak-form SINDy (WSINDy) mitigates this by integrating against smooth test functions, strong-form STLSQ can degrade or identify spurious higher-order terms under severe noise perturbations (as explicitly quantified in `src/discovery/robustness_checks.py`).
3. **Geographic Specificity:**  
   The empirical governing equations and cross-validation matrices were derived for the subtropical southeastern United States (25°N–35°N, 95°W–65°W). Transferability to regions with distinct topographic forcing (e.g., complex mountainous terrain) or different atmospheric circulation regimes will require retraining regional coefficients.
4. **Boundary Effects in Rollout:**  
   Spatial derivative stencils require interior grid points. During decoupled multi-month forward rollouts, boundary values are held consistent with interior advection, and long-term numerical stability requires regularization thresholding to prevent non-physical drift.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
