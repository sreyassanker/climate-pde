#!/usr/bin/env python3
"""
Figure 5: Multi-variable PDE coefficient maps across the southeastern United States
(2000–2023): (a) temperature linear coefficient; (b) specific humidity linear
coefficient; (c) temperature-humidity coupling coefficient (T·q); and (d) R²
improvement relative to the single-variable PDE.
"""
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shapereader
from shapely.geometry import box
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

FIGS_PATH = PROJECT_ROOT / "figures"
FIGS_PATH.mkdir(exist_ok=True)
DATA_PATH = PROJECT_ROOT / "data/climate_multivariate.nc"

PROJ = ccrs.PlateCarree()
STUDY_LAT = (25, 35)
STUDY_LON = (-95, -65)


def main():
    print("Loading data...")
    ds = xr.open_dataset(DATA_PATH)
    T_raw = ds["t2m"].values.astype(np.float64)
    q_raw = ds["q"].values.astype(np.float64)
    nt, ny, nx = T_raw.shape
    lats = ds.lat.values
    lons = ds.lon.values
    dy = float(np.abs(lats[1] - lats[0]))
    ds.close()

    land_mask = ~np.isnan(T_raw[0])
    nan_mask = np.isnan(T_raw[0])

    # Compute derivatives on raw data
    dT_dt = np.zeros_like(T_raw)
    dT_dt[1:-1] = (T_raw[2:] - T_raw[:-2]) / 2.0
    # dT_dt[0] and dT_dt[-1] stay 0, will be excluded

    t_idx = np.arange(nt, dtype=float)
    sin_t = np.sin(2 * np.pi * t_idx / 12)[:, None, None] * np.ones((1, ny, nx))
    cos_t = np.cos(2 * np.pi * t_idx / 12)[:, None, None] * np.ones((1, ny, nx))

    T_y = np.zeros_like(T_raw)
    for t in range(nt):
        T_y[t, 1:-1, :] = (T_raw[t, 2:, :] - T_raw[t, :-2, :]) / (2 * dy)

    s = sin_t[:, 0, 0]
    c = cos_t[:, 0, 0]

    # Library: single = [sin, cos, T_y], multi = [sin, cos, T_y, T, q, T×q]
    single_n = 3
    multi_n = 6

    single_coef = np.full((single_n, ny, nx), np.nan)
    multi_coef = np.full((multi_n, ny, nx), np.nan)
    r2_single = np.full((ny, nx), np.nan)
    r2_multi = np.full((ny, nx), np.nan)

    valid = np.argwhere(~nan_mask)
    nv = len(valid)
    print(f"  {nv} valid land pixels")

    for idx, (y_i, x_i) in enumerate(valid):
        if idx % 5000 == 0:
            print(f"  pixel {idx}/{nv}...")

        T_pix = T_raw[:, y_i, x_i]
        Ty_pix = T_y[:, y_i, x_i]
        q_pix = q_raw[:, y_i, x_i]
        y_vec = dT_dt[:, y_i, x_i]

        # exclude t=0, t=nt-1 where dT_dt=0
        good = (y_vec != 0)
        y_g = y_vec[good]
        if y_g.size < 6:
            continue

        # Single: sin, cos, T_y
        X_s = np.column_stack([s[good], c[good], Ty_pix[good]])
        cs = np.linalg.lstsq(X_s, y_g, rcond=None)[0]
        single_coef[:3, y_i, x_i] = cs
        yp = X_s @ cs
        r2_s = 1 - np.sum((y_g - yp)**2) / np.sum((y_g - y_g.mean())**2)
        r2_single[y_i, x_i] = r2_s

        # Multi: sin, cos, T_y, T, q, T×q
        X_m = np.column_stack([s[good], c[good], Ty_pix[good],
                               T_pix[good], q_pix[good],
                               T_pix[good] * q_pix[good]])
        cm = np.linalg.lstsq(X_m, y_g, rcond=None)[0]
        multi_coef[:6, y_i, x_i] = cm
        yp = X_m @ cm
        r2_m = 1 - np.sum((y_g - yp)**2) / np.sum((y_g - y_g.mean())**2)
        r2_multi[y_i, x_i] = r2_m

    # Stats
    print("\n── Single-variable (raw OLS, no threshold) ──")
    for i, nm in enumerate(["sin", "cos", "T_y"]):
        nz = np.sum(~np.isnan(single_coef[i]))
        mu = np.nanmean(single_coef[i])
        sd = np.nanstd(single_coef[i])
        print(f"  {nm}: mean={mu:.6f}, std={sd:.6f}, pixels={nz}")
    print(f"  R²: mean={np.nanmean(r2_single):.4f}")

    print("\n── Multi-variable (raw OLS, no threshold) ──")
    for i, nm in enumerate(["sin", "cos", "T_y", "T", "q", "Txq"]):
        nz = np.sum(~np.isnan(multi_coef[i]))
        mu = np.nanmean(multi_coef[i])
        sd = np.nanstd(multi_coef[i])
        print(f"  {nm}: mean={mu:.6f}, std={sd:.6f}, pixels={nz}")
    print(f"  R²: mean={np.nanmean(r2_multi):.4f}")
    r2_imp = r2_multi - r2_single
    print(f"  R² improvement: mean={np.nanmean(r2_imp):.4f}, max={np.nanmax(r2_imp):.4f}")

    # ────────────────────────────────────────
    # Plotting — matching Fig 3 layout exactly
    # ────────────────────────────────────────
    print("\nRendering figure...")

    # Robust ranges for each coefficient
    t_vals = multi_coef[3]  # T
    q_vals = multi_coef[4]  # q
    tq_vals = multi_coef[5]  # T×q
    r2_d = r2_multi - r2_single

    t_finite = t_vals[np.isfinite(t_vals)]
    t_lim = np.percentile(np.abs(t_finite), 95)
    q_finite = q_vals[np.isfinite(q_vals)]
    q_lim = np.percentile(np.abs(q_finite), 95)
    tq_finite = tq_vals[np.isfinite(tq_vals)]
    tq_lim = np.percentile(np.abs(tq_finite), 95)
    r2d_finite = r2_d[np.isfinite(r2_d)]
    r2d_lim = np.percentile(np.abs(r2d_finite), 95)

    print(f"\nRanges: T=±{t_lim:.5f}, q=±{q_lim:.0f}, T·q=±{tq_lim:.2f}, R² diff=±{r2d_lim:.4f}")

    fig_h = 5.2
    fig = plt.figure(figsize=(7.48, fig_h), facecolor='white')

    p_w = 0.33
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
        ("(a) Temperature coefficient",           "(b) Humidity coefficient"),
        ("(c) Temperature \u00d7 Humidity coefficient", "(d) R\u00b2 improvement"),
    ]

    panels = [
        (0, "T coeff",  t_vals, 'RdBu_r'),
        (1, "q coeff",  q_vals, 'RdBu_r'),
        (2, "Txq coeff", tq_vals, 'RdBu_r'),
        (3, "R² diff",  r2_d,  'RdBu_r'),
    ]

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

        margin = 1.0
        bbox = box(STUDY_LON[0], STUDY_LAT[0], STUDY_LON[1], STUDY_LAT[1])
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

        if panel_idx == 0:
            vmin, vmax = -t_lim, t_lim
        elif panel_idx == 1:
            vmin, vmax = -q_lim, q_lim
        elif panel_idx == 2:
            vmin, vmax = -tq_lim, tq_lim
        else:
            vmin, vmax = -r2d_lim, r2d_lim

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

        unit_labels = ["K$^{-1}$", "(kg/kg)$^{-1}$", "(K\u00b7kg/kg)$^{-1}$", ""]

        cbar_ax = fig.add_axes([x0 + p_w + cb_gap, y0, cb_w, p_h])
        cb = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
        cb.ax.tick_params(labelsize=6)
        if unit_labels[panel_idx]:
            cb.set_label(unit_labels[panel_idx], fontsize=7)

        label_off = 0.055
        label_y = y0 + p_h + label_off
        fig.text(x0 + (p_w + cb_gap + cb_w) / 2, label_y, panel_names[row][col],
                 fontsize=8, ha='center', va='bottom', fontweight='bold')

    fig.savefig(FIGS_PATH / "fig5_multi_var_coupling.png", dpi=600,
                facecolor='white', transparent=False)
    fig.savefig(FIGS_PATH / "fig5_multi_var_coupling.pdf",
                facecolor='white', transparent=False)
    print(f"Saved png & pdf")

    from PIL import Image as PILImage
    PILImage.open(FIGS_PATH / "fig5_multi_var_coupling.png").convert('RGB').save(
        FIGS_PATH / "fig5_multi_var_coupling.tiff", dpi=(600, 600), compression='tiff_lzw')
    print(f"Saved tiff")

    print("\n── Caption for manuscript ──")
    print("Figure 5. Multi-variable PDE coefficient maps for the")
    print("southeastern United States (2000–2023).")
    print("(a) Temperature linear coefficient,")
    print("(b) specific humidity linear coefficient,")
    print("(c) temperature-humidity interaction coefficient (T×q),")
    print("and (d) improvement in R² relative to single-variable")
    print("PDE. Coefficients in month⁻¹ (a), K·month⁻¹·(kg/kg)⁻¹ (b), and")
    print("month⁻¹·(kg/kg)⁻¹ (c). Map lines and geographic boundaries")
    print("are for illustration only and do not imply any")
    print("political position. (Natural Earth Data)")


if __name__ == "__main__":
    main()
