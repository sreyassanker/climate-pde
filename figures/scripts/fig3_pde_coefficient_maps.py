#!/usr/bin/env python3
"""
Figure 3: Spatially resolved coefficient stability maps across 120,000 grid cells
of the study domain: (a) seasonal forcing amplitude sin(2πt/12); (b) meridional
advection coefficient ∂T/∂y; (c) seasonal phase adjustment cos(2πt/12); and (d) R².
"""
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shapereader
from shapely.geometry import box
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as mticker

FIGS_PATH = PROJECT_ROOT / "figures"
FIGS_PATH.mkdir(exist_ok=True)
DATA_PATH = PROJECT_ROOT / "data/climate_multivariate.nc"

PROJ = ccrs.PlateCarree()
STUDY_LAT = (25, 35)
STUDY_LON = (-95, -65)


def main():
    print("Loading data...")
    ds = xr.open_dataset(DATA_PATH)
    T_raw = ds["t2m"].values.astype(np.float64)  # (time, lat, lon)
    T_raw = np.where(np.isnan(T_raw), np.nan, T_raw)
    nt, ny, nx = T_raw.shape
    lats = ds.lat.values
    lons = ds.lon.values
    dx = float(np.abs(lons[1] - lons[0]))
    dy = float(np.abs(lats[1] - lats[0]))
    ds.close()

    # Land mask
    land_mask = ~np.isnan(T_raw[0])

    # Normalize T
    T_mu = np.nanmean(T_raw, axis=0, keepdims=True)
    T_std = np.nanstd(T_raw, axis=0, keepdims=True)
    T_std = np.where(T_std < 1e-10, 1.0, T_std)
    T = (T_raw - T_mu) / T_std
    T = np.where(np.isnan(T), 0.0, T)

    # Target: dT/dt via central differences
    dT_dt = np.zeros_like(T)
    dT_dt[1:-1] = (T[2:] - T[:-2]) / 2.0

    # Candidate terms (all same shape as T)
    t_idx = np.arange(nt, dtype=float)
    sin_t = np.sin(2 * np.pi * t_idx / 12)[:, None, None] * np.ones((1, ny, nx))
    cos_t = np.cos(2 * np.pi * t_idx / 12)[:, None, None] * np.ones((1, ny, nx))

    # T_y: meridional gradient
    T_y = np.zeros_like(T)
    for t_i in range(nt):
        f = T[t_i]
        T_y[t_i, 1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * dy)

    # Build design matrix per pixel: [sin, cos, T, T_y]
    n_features = 4
    feature_names = ["sin(2πt/12)", "cos(2πt/12)", "T", "T_y"]

    print(f"Computing per-pixel OLS ({ny}×{nx} = {ny*nx} pixels)...")
    # Stack features: (nt, n_features) — we'll matmul against (nt, ny*nx) targets
    X = np.column_stack([sin_t.reshape(nt, -1)[:, 0],
                         cos_t.reshape(nt, -1)[:, 0]])  # sin, cos don't vary spatially
    # Actually sin_t and cos_t don't vary spatially, so X for spatial terms is different per pixel
    # We need to handle per-pixel features that vary in space AND time
    
    # Vectorized approach: for each pixel, we have a local design matrix
    # X_pixel = [sin_t, cos_t, T[:, y, x], T_y[:, y, x]]  shape (nt, 4)
    # y_pixel = dT_dt[:, y, x]  shape (nt,)
    # coef[y, x] = (X_pixel^T X_pixel)^{-1} X_pixel^T y_pixel

    # Precompute sin, cos (same for all pixels)
    s = sin_t[:, 0, 0]  # (nt,)
    c = cos_t[:, 0, 0]  # (nt,)

    # Initialize coefficient maps
    coef_maps = np.full((n_features, ny, nx), np.nan)
    r2_map = np.full((ny, nx), np.nan)
    nz_map = np.full((ny, nx), -1, dtype=int)

    # Process pixel by pixel (can be vectorized with batched operations)
    valid_pixels = np.argwhere(land_mask)
    n_valid = len(valid_pixels)
    print(f"  {n_valid} valid land pixels")

    for idx, (y_i, x_i) in enumerate(valid_pixels):
        if idx % 5000 == 0:
            print(f"  Processing pixel {idx}/{n_valid}...")

        # Local design matrix
        T_loc = T[:, y_i, x_i]
        Ty_loc = T_y[:, y_i, x_i]
        y_loc = dT_dt[:, y_i, x_i]

        X_loc = np.column_stack([s, c, T_loc, Ty_loc])

        # OLS
        coef = np.linalg.lstsq(X_loc, y_loc, rcond=None)[0]

        # Iterative thresholding (STLSQ-style)
        for thresh in [0.1, 0.05, 0.02, 0.01]:
            support = np.abs(coef) > thresh
            if support.sum() < 2:
                continue
            coef_new = np.zeros(n_features)
            coef_new[support] = np.linalg.lstsq(X_loc[:, support], y_loc, rcond=None)[0]
            coef = coef_new

        # Final refit
        support = np.abs(coef) > 1e-10
        if support.sum() > 0:
            coef[support] = np.linalg.lstsq(X_loc[:, support], y_loc, rcond=None)[0]

        y_pred = X_loc @ coef
        ss_res = np.sum((y_loc - y_pred)**2)
        ss_tot = np.sum((y_loc - np.mean(y_loc))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        coef_maps[:, y_i, x_i] = coef
        r2_map[y_i, x_i] = r2
        nz_map[y_i, x_i] = np.sum(np.abs(coef) > 1e-10)

    # Count term frequency
    print("\nTerm frequency across valid pixels:")
    for i, name in enumerate(feature_names):
        n_active = np.sum(np.abs(coef_maps[i]) > 1e-10)
        mu = np.nanmean(np.abs(coef_maps[i][np.abs(coef_maps[i]) > 1e-10]))
        print(f"  {name:20s}: active in {n_active}/{n_valid} pixels, mean |coef| = {mu:.4f}")
    print(f"  R²: mean = {np.nanmean(r2_map):.3f}, median = {np.nanmedian(r2_map):.3f}")

    # ────────────────────────────────────────────
    # Plotting
    # ────────────────────────────────────────────
    print("\nRendering figure...")
    fig_h = 5.2
    fig = plt.figure(figsize=(7.48, fig_h), facecolor='white')

    p_w = 0.35
    p_h = 0.34
    cb_w = 0.015
    cb_gap = 0.008

    x_left = 0.07
    x_right = x_left + p_w + cb_gap + cb_w + 0.12
    y_top = 0.563
    y_bot = 0.08

    x_positions = [x_left, x_right]
    y_positions = [y_top, y_bot]

    panel_names = [
        ("(a) Seasonal forcing amplitude", "(b) Meridional advection coefficient"),
        ("(c) Seasonal phase adjustment",  "(d) Coefficient of determination (R²)"),
    ]

    panels = [
        (0, "sin(2πt/12)", coef_maps[0], 'RdBu_r'),
        (1, "T_y",          coef_maps[3], 'RdBu_r'),
        (2, "cos(2πt/12)",  coef_maps[1], 'RdBu_r'),
        (3, "R²",           r2_map,       'viridis'),
    ]

    # Load US states shapefile once
    states_reader = shapereader.Reader(shapereader.natural_earth(
        resolution='50m', category='cultural', name='admin_1_states_provinces'))
    us_states_data = [
        (r.attributes.get('postal', ''), r.attributes.get('name', ''), r.geometry)
        for r in states_reader.records()
        if r.attributes.get('admin') == 'United States of America'
    ]

    for idx, (panel_idx, data_label, data, cmap) in enumerate(panels):
        row = idx // 2
        col = idx % 2
        x0 = x_positions[col]
        y0 = y_positions[row]

        ax = fig.add_axes([x0, y0, p_w, p_h], projection=PROJ)
        ax.set_extent([STUDY_LON[0], STUDY_LON[1],
                       STUDY_LAT[0], STUDY_LAT[1]], crs=PROJ)
        ax.set_aspect('auto')

        ax.add_feature(cfeature.OCEAN, color='#DEEBF7', zorder=0)
        ax.add_feature(cfeature.LAND, color='#F0F0F0', zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor='#5A6B7C', zorder=2)

        usa_geoms = [g for _, _, g in us_states_data]
        ax.add_geometries(usa_geoms, crs=PROJ, facecolor='none',
                          edgecolor='#4A5B6C', linewidth=0.5, zorder=2)

        # State abbreviation labels
        bbox = box(STUDY_LON[0], STUDY_LAT[0], STUDY_LON[1], STUDY_LAT[1])
        margin = 1.0
        for abbr, name, geom in us_states_data:
            if geom is None or not abbr:
                continue
            if not geom.intersects(bbox):
                continue
            inter = geom.intersection(bbox)
            if inter.is_empty:
                continue
            pt = inter.centroid
            if not (STUDY_LON[0] + margin < pt.x < STUDY_LON[1] - margin and
                    STUDY_LAT[0] + margin < pt.y < STUDY_LAT[1] - margin):
                continue
            txt = ax.text(pt.x, pt.y, abbr, transform=PROJ, fontsize=5,
                    fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.08', facecolor='white',
                              edgecolor='none', alpha=0.7))
            txt.set_clip_path(ax.patch)

        lat_idx = np.where((lats >= STUDY_LAT[0]) & (lats <= STUDY_LAT[1]))[0]
        lon_idx = np.where((lons >= STUDY_LON[0]) & (lons <= STUDY_LON[1]))[0]
        lats_trim = lats[lat_idx]
        lons_trim = lons[lon_idx]
        data_trim = data[np.ix_(lat_idx, lon_idx)]
        land_trim = land_mask[np.ix_(lat_idx, lon_idx)]
        data_plot = np.ma.array(data_trim, mask=~land_trim)

        if panel_idx < 3:
            vmax = np.nanpercentile(np.abs(data_trim[land_trim]), 95)
            vmin, vmax = -vmax, vmax
        else:
            vmin, vmax = 0, 1

        im = ax.pcolormesh(lons_trim, lats_trim, data_plot,
                          transform=PROJ, cmap=cmap,
                          vmin=vmin, vmax=vmax, shading='auto')

        gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='gray', alpha=0.4,
                         linestyle=':')
        gl.top_labels = True
        gl.right_labels = True
        gl.ylocator = mticker.MultipleLocator(2)
        gl.xlabel_style = {'size': 6, 'color': '#444444'}
        gl.ylabel_style = {'size': 6, 'color': '#444444', 'rotation': 90}

        unit_labels = ["", "", "", ""]
        cbar_ax = fig.add_axes([x0 + p_w + cb_gap, y0, cb_w, p_h])
        cb = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
        cb.ax.tick_params(labelsize=6)
        if unit_labels[panel_idx]:
            cb.set_label(unit_labels[panel_idx], fontsize=7)

        label_off = 0.055
        label_y = y0 + p_h + label_off
        fig.text(x0 + (p_w + cb_gap + cb_w) / 2, label_y, panel_names[row][col],
                 fontsize=8, ha='center', va='bottom', fontweight='bold')

    fig.savefig(FIGS_PATH / "fig3_pde_coefficient_maps.png", dpi=600,
                facecolor='white', transparent=False)
    fig.savefig(FIGS_PATH / "fig3_pde_coefficient_maps.pdf",
                facecolor='white', transparent=False)
    print(f"Saved {FIGS_PATH}/fig3_pde_coefficient_maps.png (600 dpi)")
    print(f"Saved {FIGS_PATH}/fig3_pde_coefficient_maps.pdf")
    plt.close(fig)

    # Caption
    print("\n── Caption for manuscript ──")
    print("Figure 3. Spatial maps of discovered PDE coefficients")
    print("for the southeastern United States (2000–2023).")
    print("(a) Seasonal forcing amplitude [sin(2πt/12) coefficient],")
    print("(b) meridional advection coefficient [∂T/∂y],")
    print("(c) seasonal phase adjustment [cos(2πt/12) coefficient], and")
    print("(d) coefficient of determination (R²). Coefficients are")
    print("dimensionless due to z-score normalization. White areas")
    print("indicate ocean. Map lines and geographic boundaries are")
    print("for illustration only and do not imply any political")
    print("position. (Natural Earth Data)")

    # Also save as TIFF (LZW-compressed, submission-ready)
    from PIL import Image as PILImage
    png_path = FIGS_PATH / "fig3_pde_coefficient_maps.png"
    tiff_path = FIGS_PATH / "fig3_pde_coefficient_maps.tiff"
    img_pil = PILImage.open(png_path).convert('RGB')
    img_pil.save(tiff_path, dpi=(600, 600), compression='tiff_lzw')
    print(f"Saved {tiff_path}")


if __name__ == "__main__":
    main()
