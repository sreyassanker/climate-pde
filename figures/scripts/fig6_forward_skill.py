#!/usr/bin/env python3
"""
Figure 6: Forward integration predictive skill across lead times of 1–12 months
for single-variable PDE, multivariate PDE, fit-once ARIMA, persistence, and
monthly climatology baselines across the three study regions (3×3 panel layout).
"""
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shapereader
from shapely.geometry import box
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
from statsmodels.tsa.arima.model import ARIMA as sm_ARIMA
import warnings
warnings.filterwarnings('ignore')

FIGS_PATH = PROJECT_ROOT / "figures"
FIGS_PATH.mkdir(exist_ok=True)
DATA_PATH = PROJECT_ROOT / "data/climate_multivariate.nc"

DX = 0.05
REGIONS = {
    "florida":   {"lat": (25.0, 30.0), "lon": (-83.0, -80.0)},
    "ga_al":     {"lat": (30.0, 34.0), "lon": (-88.0, -82.0)},
    "carolinas": {"lat": (33.0, 35.0), "lon": (-82.0, -75.0)},
}
PATCH_SIZE = 32
REGION_KEYS = ["florida", "ga_al", "carolinas"]
REGION_LABELS = ["Florida", "Georgia/Alabama", "Carolinas"]
PROJ = ccrs.PlateCarree()

from pmdarima import auto_arima
# ARIMA orders discovered by auto_arima on months 1-240 of T_avg
ARIMA_ORDERS = {}


def extract(ds, key):
    r = REGIONS[key]
    lat_s = ds.lat.sel(lat=slice(r["lat"][0], r["lat"][1]))
    lon_s = ds.lon.sel(lon=slice(r["lon"][0], r["lon"][1]))
    ci, cj = len(lat_s)//2, len(lon_s)//2
    h = PATCH_SIZE//2
    return ds.sel(lat=slice(lat_s.values[ci-h], lat_s.values[ci+h-1]),
                  lon=slice(lon_s.values[cj-h], lon_s.values[cj+h-1]))


def compute_T_y(T):
    d = np.zeros_like(T)
    d[1:-1, :] = (T[2:, :] - T[:-2, :]) / (2 * DX)
    return d


def rhs_multivar(T, q, t, coeffs):
    rhs = np.zeros_like(T)
    T_y = compute_T_y(T)
    for name, c in coeffs.items():
        if name == "sin":
            rhs += c * np.sin(2 * np.pi * t / 12)
        elif name == "cos":
            rhs += c * np.cos(2 * np.pi * t / 12)
        elif name == "T":
            rhs += c * T
        elif name == "q":
            rhs += c * q
        elif name == "T_y":
            rhs += c * T_y
        elif name == "T*q":
            rhs += c * T * q
    return rhs


def main():
    print("Computing results for all regions...")
    all_data = {}

    for rk in REGION_KEYS:
        print(f"\n  {rk}...")
        ds = xr.open_dataset(DATA_PATH)
        patch = extract(ds, rk)
        T_raw = patch["t2m"].values.copy()
        q_raw = patch["q"].values.copy()
        T_raw = np.where(np.isnan(T_raw), np.nanmean(T_raw), T_raw)
        q_raw = np.where(np.isnan(q_raw), 0.0, q_raw)
        lats_p = patch.lat.values
        lons_p = patch.lon.values
        ds.close()

        T = (T_raw - T_raw.mean()) / (T_raw.std() + 1e-10)
        q = (q_raw - q_raw.mean()) / (q_raw.std() + 1e-10)
        nt, ny, nx = T.shape
        T_avg = T.mean(axis=(1, 2))
        land_mask = ~np.isnan(patch["t2m"].isel(time=0).values)

        # Discover ARIMA order if not known
        if rk not in ARIMA_ORDERS:
            result = auto_arima(T_avg[:240], seasonal=True, m=12,
                                start_p=0, max_p=4, start_q=0, max_q=4,
                                start_P=0, max_P=2, start_Q=0, max_Q=2,
                                trace=False, error_action='ignore',
                                suppress_warnings=True, stepwise=True,
                                max_order=6, n_fits=10)
            ARIMA_ORDERS[rk] = (result.order, result.seasonal_order)
            print(f"    ARIMA order: {result.order} x {result.seasonal_order}")

        # Common terms
        sin_t = np.sin(2 * np.pi * np.arange(nt) / 12)
        cos_t = np.cos(2 * np.pi * np.arange(nt) / 12)
        T_y_all = np.zeros_like(T)
        for t in range(nt):
            T_y_all[t] = compute_T_y(T[t])
        dT = np.zeros_like(T)
        dT[1:-1] = (T[2:] - T[:-2]) / 2.0

        # ── Fit sin+T_y PDE ──
        X_s = np.column_stack([sin_t[:, None, None].repeat(ny, axis=1).repeat(nx, axis=2).ravel(),
                               T_y_all.ravel()])
        y_v = dT.ravel()
        valid = ~np.isnan(y_v)
        c_s = np.linalg.lstsq(X_s[valid], y_v[valid], rcond=None)[0]
        a, b = float(c_s[0]), float(c_s[1])

        # ── Fit multi-var PDE ──
        sin_3d = sin_t[:, None, None] * np.ones((1, ny, nx))
        cos_3d = cos_t[:, None, None] * np.ones((1, ny, nx))
        terms = {"sin": sin_3d, "cos": cos_3d, "T_y": T_y_all,
                 "T": T, "q": q, "T*q": T * q}
        names = ["sin", "cos", "T_y", "T", "q", "T*q"]
        X_m = np.column_stack([terms[n].ravel() for n in names])
        c_m = np.linalg.lstsq(X_m[valid], y_v[valid], rcond=None)[0]
        # STLSQ thresholding
        for thresh in [0.05]:
            supp = np.abs(c_m) > thresh
            if supp.sum() >= 2:
                c2 = np.zeros(len(c_m))
                c2[supp] = np.linalg.lstsq(X_m[valid][:, supp], y_v[valid], rcond=None)[0]
                c_m = c2
        mv_coeffs = {}
        for i, n in enumerate(names):
            if abs(c_m[i]) > 1e-10:
                mv_coeffs[n] = float(c_m[i])

        # ── Multi-var PDE T·q coefficient map (per pixel) ──
        tq_map = np.full((ny, nx), np.nan)
        valid_ij = np.argwhere(land_mask)
        for yi, xi in valid_ij:
            T_pix = T[:, yi, xi]
            Ty_pix = T_y_all[:, yi, xi]
            q_pix = q[:, yi, xi]
            y_pix = dT[:, yi, xi]
            good = y_pix != 0
            if good.sum() < 6:
                continue
            Xm = np.column_stack([sin_t[good], cos_t[good], Ty_pix[good],
                                  T_pix[good], q_pix[good], T_pix[good]*q_pix[good]])
            cm = np.linalg.lstsq(Xm, y_pix[good], rcond=None)[0]
            tq_map[yi, xi] = cm[5]

        # Monthly climatology
        month_of_year = np.arange(nt) % 12
        clim_2d = np.zeros((12, ny, nx))
        for m in range(12):
            clim_2d[m] = T[month_of_year == m].mean(axis=0)

        # ── Forward integration (sin+T_y PDE) — per-pixel RMSE ──
        n_steps = 12
        n_starts = 4
        starts = np.linspace(12, nt - n_steps - 12, n_starts, dtype=int)
        all_res = []

        for start_idx in starts:
            # sin+T_y PDE (evolve full 2D field)
            T_pred = T[start_idx].copy()
            sin_traj_2d = [T_pred.copy()]
            for step in range(n_steps):
                t_idx = start_idx + step
                T_pred = T_pred + a * np.sin(2*np.pi*t_idx/12) + b * compute_T_y(T_pred)
                sin_traj_2d.append(T_pred.copy())
            sin_traj_2d = np.array(sin_traj_2d)

            # Multi-var PDE (evolve full 2D field)
            T_pred_mv = T[start_idx].copy()
            mv_traj_2d = [T_pred_mv.copy()]
            for step in range(n_steps):
                t_idx = start_idx + step
                q_at_t = q[min(t_idx, nt-1)]
                dT_mv = rhs_multivar(T_pred_mv, q_at_t, t_idx, mv_coeffs)
                T_pred_mv = T_pred_mv + dT_mv
                mv_traj_2d.append(T_pred_mv.copy())
            mv_traj_2d = np.array(mv_traj_2d)

            # Persistence (no change)
            actual_2d = T[start_idx:start_idx + n_steps + 1]
            persist_2d = np.tile(T[start_idx][None, :, :], (n_steps + 1, 1, 1))

            # Per-pixel RMSE
            rmse_sin = [float(np.sqrt(np.mean((sin_traj_2d[i] - actual_2d[i])**2)))
                        for i in range(n_steps + 1)]
            rmse_mv = [float(np.sqrt(np.mean((mv_traj_2d[i] - actual_2d[i])**2)))
                       for i in range(n_steps + 1)]
            rmse_persist = [float(np.sqrt(np.mean((persist_2d[i] - actual_2d[i])**2)))
                            for i in range(n_steps + 1)]

            # Climatology RMSE
            clim_pred = np.array([clim_2d[(start_idx + i) % 12] for i in range(n_steps + 1)])
            rmse_clim = [float(np.sqrt(np.mean((clim_pred[i] - actual_2d[i])**2)))
                         for i in range(n_steps + 1)]

            all_res.append({
                "start": start_idx,
                "rmse_sin": rmse_sin,
                "rmse_mv": rmse_mv,
                "rmse_persist": rmse_persist,
                "rmse_clim": rmse_clim,
                "pred_sin": sin_traj_2d,
                "pred_mv": mv_traj_2d,
                "actual": actual_2d,
                "pde_err_map": None,  # filled below on first start
            })

        # Fill PDE spatial error map (first start)
        start_idx = starts[0]
        T_pred = T[start_idx].copy()
        for step in range(6):
            t_idx = start_idx + step
            T_pred = T_pred + a * np.sin(2*np.pi*t_idx/12) + b * compute_T_y(T_pred)
        err_map_6 = T_pred - T[start_idx + 6]
        all_res[0]["pde_err_map"] = err_map_6

        # ── ARIMA RMSE curves (fit-once on months 1–240, then forecast) ──
        order, s_order = ARIMA_ORDERS[rk]
        arima_err2 = [[] for _ in range(n_steps + 1)]
        arima_at_6_per_start = [np.nan] * len(starts)
        if nt > 240:
            try:
                model = sm_ARIMA(T_avg[:240], order=order, seasonal_order=s_order)
                fitted = model.fit(low_memory=True)
                forecast = np.asarray(fitted.forecast(steps=nt - 240)).ravel()
                # Evaluate from starts >= 240
                for si, start_idx in enumerate(starts):
                    if start_idx >= 240 and start_idx + n_steps < nt:
                        offset = start_idx - 240
                        for i in range(1, n_steps + 1):
                            fc_idx = offset + i - 1
                            if fc_idx < len(forecast):
                                err = (forecast[fc_idx] - T_avg[start_idx + i])**2
                                arima_err2[i].append(err)
                        if n_steps >= 6:
                            fc_idx = offset + 5
                            if fc_idx < len(forecast):
                                arima_at_6_per_start[si] = np.sqrt((forecast[fc_idx] - T_avg[start_idx + 6])**2)
            except Exception:
                pass
        arima_rmse = [np.nan]
        for li in range(1, n_steps + 1):
            arima_rmse.append(np.sqrt(np.mean(arima_err2[li])) if arima_err2[li] else np.nan)

        all_data[rk] = {
            "all_res": all_res,
            "arima_rmse": arima_rmse,
            "arima_at_6": arima_at_6_per_start,
            "mv_coeffs": mv_coeffs,
            "a": a, "b": b,
            "lats": lats_p, "lons": lons_p,
            "land_mask": land_mask,
            "tq_map": tq_map,
        }

        impr = (1 - np.mean([r["rmse_sin"][6] for r in all_res]) /
                np.mean([r["rmse_persist"][6] for r in all_res])) * 100
        print(f"    sin+T_y PDE: a={a:.4f}, b={b:.4f} | multi-var: {mv_coeffs}")
        print(f"    Improvement at 6mo: {impr:.1f}%")

    # ────────────────────────────────────────
    # Plotting: 4 rows × 3 columns
    # ────────────────────────────────────────
    print("\nRendering figure...")
    fig = plt.figure(figsize=(7.48, 6.8), facecolor='white')

    gs = fig.add_gridspec(3, 3, left=0.06, right=0.88, bottom=0.087, top=0.93,
                          wspace=0.30, hspace=0.28,
                          height_ratios=[1, 1.1, 1.1])

    leads = np.arange(13)
    all_axes = [[None, None, None] for _ in range(3)]
    row1_mappable = [None, None, None]  # colorbar mappables for row 1 (error maps)
    row2_mappable = [None, None, None]  # colorbar mappables for row 2 (T×q maps)

    # Load US states shapefile once
    states_reader = shapereader.Reader(shapereader.natural_earth(
        resolution='50m', category='cultural', name='admin_1_states_provinces'))
    us_states_data = [
        (r.attributes.get('postal', ''), r.attributes.get('name', ''), r.geometry)
        for r in states_reader.records()
        if r.attributes.get('admin') == 'United States of America'
    ]

    def add_map_layers(ax, ad):
        """Add ocean, land, coastline, and state boundaries to a map axis."""
        ax.add_feature(cfeature.OCEAN, color='#DEEBF7', zorder=0)
        ax.add_feature(cfeature.LAND, color='#F0F0F0', zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor='#4A5B6C', zorder=2)
        usa_geoms = [g for _, _, g in us_states_data]
        ax.add_geometries(usa_geoms, crs=PROJ, facecolor='none',
                          edgecolor='#4A5B6C', linewidth=0.5, zorder=2)
        ax.set_extent([ad["lons"].min(), ad["lons"].max(),
                       ad["lats"].min(), ad["lats"].max()], crs=PROJ)

    def add_state_labels(ax, ad):
        """Add state abbreviation labels for states intersecting the subregion extent."""
        lon_min, lon_max = ad["lons"].min(), ad["lons"].max()
        lat_min, lat_max = ad["lats"].min(), ad["lats"].max()
        bbox = box(lon_min, lat_min, lon_max, lat_max)
        for abbr, name, geom in us_states_data:
            if geom is None or not abbr:
                continue
            if not geom.intersects(bbox):
                continue
            # Place label at centroid of the intersection (guaranteed inside subregion)
            inter = geom.intersection(bbox)
            if inter.is_empty:
                continue
            pt = inter.centroid
            ax.text(pt.x, pt.y, abbr, transform=PROJ, fontsize=5,
                    fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.08', facecolor='white',
                              edgecolor='none', alpha=0.7))

    for idx, rk in enumerate(REGION_KEYS):
        ad = all_data[rk]
        r_all = ad["all_res"]

        # ── Row 0: RMSE curves (4 methods) ──
        ax = fig.add_subplot(gs[0, idx])
        sin_arr = np.array([r["rmse_sin"] for r in r_all])
        mv_arr = np.array([r["rmse_mv"] for r in r_all])
        persist_arr = np.array([r["rmse_persist"] for r in r_all])
        clim_arr = np.array([r["rmse_clim"] for r in r_all])
        arima_r = np.array(ad["arima_rmse"])

        for r in r_all:
            ax.plot(leads, r["rmse_sin"], '-', color='#0072B2', alpha=0.10, lw=0.5)
        ax.plot(leads, sin_arr.mean(axis=0), '-o', color='#0072B2', lw=1.2,
                label='Single-variable PDE', markersize=2)

        for r in r_all:
            ax.plot(leads, r["rmse_mv"], '-', color='#009E73', alpha=0.10, lw=0.5)
        ax.plot(leads, mv_arr.mean(axis=0), '-s', color='#009E73', lw=1.2,
                label='Multi-variable PDE', markersize=2)

        ax.plot(leads, arima_r, '-d', color='#CC79A7', lw=1.2,
                label='ARIMA (fit-once)', markersize=2)

        for r in r_all:
            ax.plot(leads, r["rmse_persist"], '-', color='#D55E00', alpha=0.10, lw=0.5)
        ax.plot(leads, persist_arr.mean(axis=0), '--^', color='#D55E00', lw=1.2,
                label='Persistence', markersize=2)

        ax.plot(leads, clim_arr.mean(axis=0), ':', color='#000000', lw=1.2,
                label='Climatology', markersize=2)

        ax.set_xlim(0, 12)
        ax.tick_params(labelsize=6)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
        ax.grid(True, alpha=0.25, linewidth=0.3)
        if idx == 0:
            ax.set_ylabel('RMSE', fontsize=6)
            ax.legend(fontsize=5, loc='upper left',
                     framealpha=0.85, edgecolor='#CCCCCC')
        ax.set_xlabel('Lead (months)', fontsize=6)
        all_axes[0][idx] = ax

        # ── Row 1: PDE spatial error at 6mo ──
        ax = fig.add_subplot(gs[1, idx], projection=PROJ)
        all_axes[1][idx] = ax
        err = r_all[0]["pde_err_map"]
        add_map_layers(ax, ad)
        add_state_labels(ax, ad)
        vmax = max(abs(err.min()), abs(err.max()))
        im = ax.pcolormesh(ad["lons"], ad["lats"], err, transform=PROJ,
                          cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
        row1_mappable[idx] = im
        gl = ax.gridlines(draw_labels=True, linewidth=0.15, color='gray',
                         alpha=0.4, linestyle=':')
        gl.top_labels = True
        gl.right_labels = True
        gl.xlocator = mticker.MultipleLocator(0.5)
        gl.ylocator = mticker.MultipleLocator(0.5)
        gl.xlabel_style = {'size': 5, 'color': '#444444'}
        gl.ylabel_style = {'size': 5, 'color': '#444444', 'rotation': 90}

        # ── Row 2: Multi-var PDE T·q coefficient ──
        ax = fig.add_subplot(gs[2, idx], projection=PROJ)
        tq = ad["tq_map"]
        lm = ad["land_mask"]
        tq_finite = tq[np.isfinite(tq)]
        vmax = np.percentile(np.abs(tq_finite), 95) if len(tq_finite) > 0 else 1
        dp = np.ma.array(tq, mask=~lm)

        add_map_layers(ax, ad)
        add_state_labels(ax, ad)
        im2 = ax.pcolormesh(ad["lons"], ad["lats"], dp, transform=PROJ,
                           cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
        row2_mappable[idx] = im2
        gl2 = ax.gridlines(draw_labels=True, linewidth=0.15, color='gray',
                          alpha=0.4, linestyle=':')
        gl2.top_labels = True
        gl2.right_labels = True
        gl2.xlocator = mticker.MultipleLocator(0.5)
        gl2.ylocator = mticker.MultipleLocator(0.5)
        gl2.xlabel_style = {'size': 5, 'color': '#444444'}
        gl2.ylabel_style = {'size': 5, 'color': '#444444', 'rotation': 90}
        all_axes[2][idx] = ax

    # ── Per-panel letter labels ──
    panel_letters = 'abcdefghi'
    for pi in range(9):
        row, col = pi // 3, pi % 3
        ax = all_axes[row][col]
        if row in (1, 2):  # rows 1-2 are maps
            ad = all_data[REGION_KEYS[col]]
            lon0 = ad["lons"].min() + 0.05
            lat0 = ad["lats"].max() - 0.05
            ax.text(lon0, lat0, f"({panel_letters[pi]})", transform=PROJ, fontsize=7,
                   fontweight='bold', ha='left', va='top',
                   bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                            edgecolor='none', alpha=0.7))
        else:  # non-map panels: use figure coordinates
            pos = ax.get_position()
            fig.text(pos.x1 - 0.005, pos.y1 - 0.005, f"({panel_letters[pi]})",
                    fontsize=7, fontweight='bold', ha='right', va='top',
                    transform=fig.transFigure)

    # ── Capture original positions before any shift ──
    r0_bot = all_axes[0][1].get_position().y0
    r1_o = all_axes[1][1].get_position()
    r2_o = all_axes[2][1].get_position()
    shift1 = 0.5 / 2.54 / fig.get_figheight()
    shift2 = 0.9 / 2.54 / fig.get_figheight()

    # ── Shift row 1 down ──
    for col in range(3):
        ax = all_axes[1][col]
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 - shift1, pos.width, pos.height])
    # ── Shift row 2 down ──
    for col in range(3):
        ax = all_axes[2][col]
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 - shift2, pos.width, pos.height])

    # ── "6-Month Lead Error" below row 0 ──
    xlabel_off = 10 / 72 / fig.get_figheight()
    title1_y = r0_bot - xlabel_off - 0.55 / 2.54 / fig.get_figheight()
    fig.text(0.463, title1_y, "6-Month Lead Error",
            fontsize=7, fontweight='bold', ha='center', va='top',
            transform=fig.transFigure)

    # ── "T×q Coefficient" centered between row 1 bottom & row 2 top (after shifts) ──
    gap2_top = r1_o.y0 - shift1
    gap2_bot = r2_o.y1 - shift2
    fig.text(0.463, (gap2_top + gap2_bot) / 2, "T\u00d7q Coefficient",
            fontsize=7, fontweight='bold', ha='center', va='center',
            transform=fig.transFigure)

    # ── Shared row title for row 0 ──
    ax = all_axes[0][1]
    pos = ax.get_position()
    fig.text(0.463, pos.y1 + 0.014, "Forecast RMSE",
            fontsize=7, fontweight='bold', ha='center', va='bottom',
            transform=fig.transFigure)

    # ── Vertical colorbar for row 1 (error maps) ──
    r1_y0 = r1_o.y0 - shift1
    r1_y1 = r1_o.y1 - shift1
    cb1_ax = fig.add_axes([0.90, r1_y0, 0.015, r1_y1 - r1_y0])
    cb1 = fig.colorbar(row1_mappable[1], cax=cb1_ax, orientation='vertical')
    cb1.ax.tick_params(labelsize=5)
    cb1.set_label("Error (\u03c3)", fontsize=5, labelpad=1)

    # ── Vertical colorbar for row 2 (T×q maps) ──
    r2_y0 = r2_o.y0 - shift2
    r2_y1 = r2_o.y1 - shift2
    cb2_ax = fig.add_axes([0.90, r2_y0, 0.015, r2_y1 - r2_y0])
    cb2 = fig.colorbar(row2_mappable[1], cax=cb2_ax, orientation='vertical')
    cb2.ax.tick_params(labelsize=5)
    cb2.set_label("T\u00d7q coeff", fontsize=5, labelpad=1)

    fname = "fig6_forward_skill"
    fig.savefig(FIGS_PATH / f"{fname}.png", dpi=600,
                facecolor='white', transparent=False)
    fig.savefig(FIGS_PATH / f"{fname}.pdf",
                facecolor='white', transparent=False)
    print(f"\nSaved {fname}.png / .pdf")

    from PIL import Image as PILImage
    PILImage.open(FIGS_PATH / f"{fname}.png").convert('RGB').save(
        FIGS_PATH / f"{fname}.tiff", dpi=(600, 600), compression='tiff_lzw')
    print(f"Saved {fname}.tiff")

    print("\n── Caption for manuscript ──")
    print("Figure 6. Forward integration prediction skill for three")
    print("sub-regions. (a–c) RMSE vs lead time for five methods:")
    print("single-variable PDE (sin+∂T/∂y), multi-variable PDE")
    print("(sin, cos, T, q, T_y, T×q), ARIMA(1,0,0)×(1,0,1,12)")
    print("(fitted once, not walk-forward), persistence, and")
    print("monthly climatology. Thin lines: individual start-month")
    print("trajectories. (d–f) PDE prediction error at 6-month lead.")
    print("(g–i) Multi-variable PDE T×q coefficient.")
    print("Map lines and geographic boundaries are for illustration")
    print("only. (Natural Earth Data)")


if __name__ == "__main__":
    main()
