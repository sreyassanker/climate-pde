#!/usr/bin/env python3
"""
Figure 4: Spatial cross-validation hexbin density distributions comparing
predicted versus observed temperature tendencies across Florida, GA/AL, and
Carolinas subregions (3×3 transfer matrix).
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
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

FIGS_PATH = PROJECT_ROOT / "figures"
FIGS_PATH.mkdir(exist_ok=True)
DATA_PATH = PROJECT_ROOT / "data/climate_multivariate.nc"

REGIONS = {
    "florida":   {"lat": (25.0, 30.0), "lon": (-83.0, -80.0), "label": "Florida"},
    "ga_al":     {"lat": (30.0, 34.0), "lon": (-88.0, -82.0), "label": "GA/AL"},
    "carolinas": {"lat": (33.0, 35.0), "lon": (-82.0, -75.0), "label": "Carolinas"},
}
PATCH_SIZE = 32
DX = 0.05


def extract_patch(ds, region_key, patch_size=PATCH_SIZE):
    r = REGIONS[region_key]
    lat_s = ds.lat.sel(lat=slice(r["lat"][0], r["lat"][1]))
    lon_s = ds.lon.sel(lon=slice(r["lon"][0], r["lon"][1]))
    ci, cj = len(lat_s) // 2, len(lon_s) // 2
    half = patch_size // 2
    lat_p = lat_s.values[ci-half:ci+half]
    lon_p = lon_s.values[cj-half:cj+half]
    return ds.sel(lat=slice(lat_p[0], lat_p[-1]),
                  lon=slice(lon_p[0], lon_p[-1]))


def compute_terms(arr):
    nt, ny, nx = arr.shape
    T = arr
    T_y = np.zeros_like(T)
    for t in range(nt):
        f = T[t]
        T_y[t, 1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * DX)
    sin_t = np.sin(2 * np.pi * np.arange(nt) / 12)
    cos_t = np.cos(2 * np.pi * np.arange(nt) / 12)
    terms = {
        "sin": sin_t[:, None, None] * np.ones((1, ny, nx)),
        "cos": cos_t[:, None, None] * np.ones((1, ny, nx)),
        "T": T,
        "T_y": T_y,
    }
    return terms


def learn_pde(region_key):
    ds = xr.open_dataset(DATA_PATH)
    patch = extract_patch(ds, region_key)
    nt, ny, nx = len(patch.time), len(patch.lat), len(patch.lon)

    arr = patch["t2m"].values
    arr = np.where(np.isnan(arr), np.nanmean(arr), arr)
    arr = (arr - arr.mean()) / arr.std()

    terms = compute_terms(arr)
    term_keys = ["sin", "cos", "T", "T_y"]
    X = np.column_stack([terms[k].ravel() for k in term_keys])

    dT_dt = np.zeros_like(arr)
    dT_dt[1:-1] = (arr[2:] - arr[:-2]) / 2.0
    y = dT_dt.ravel()

    n_vars = len(term_keys)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    for threshold in [0.1, 0.05, 0.02, 0.01]:
        support = np.abs(coef) > threshold
        if support.sum() < 2:
            continue
        coef_new = np.zeros(n_vars)
        coef_new[support] = np.linalg.lstsq(X[:, support], y, rcond=None)[0]
        coef = coef_new

    support = np.abs(coef) > 1e-10
    if support.sum() > 0:
        coef[support] = np.linalg.lstsq(X[:, support], y, rcond=None)[0]

    y_pred = X @ coef
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    ds.close()
    return coef, term_keys, r2


def evaluate_on_region(region_key, coef, term_keys):
    ds = xr.open_dataset(DATA_PATH)
    patch = extract_patch(ds, region_key)
    nt = len(patch.time)

    arr = patch["t2m"].values
    arr = np.where(np.isnan(arr), np.nanmean(arr), arr)
    arr = (arr - arr.mean()) / arr.std()

    terms = compute_terms(arr)
    X = np.column_stack([terms[k].ravel() for k in term_keys])

    dT_dt = np.zeros_like(arr)
    dT_dt[1:-1] = (arr[2:] - arr[:-2]) / 2.0
    y = dT_dt.ravel()

    y_pred = X @ coef
    valid = ~(np.isnan(y) | np.isnan(y_pred))
    valid = valid & (np.abs(y) < 20) & (np.abs(y_pred) < 20)
    ya, yp = y[valid], y_pred[valid]

    rmse = float(np.sqrt(np.mean((yp - ya)**2)))
    ss_res = np.sum((yp - ya)**2)
    ss_tot = np.sum((ya - np.mean(ya))**2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    ds.close()
    return rmse, r2, ya, yp


def main():
    print("=" * 60)
    print("Fig 4: Cross-Validation (3×3)")
    print("=" * 60)

    region_keys = ["florida", "ga_al", "carolinas"]
    rlabels = [REGIONS[k]["label"] for k in region_keys]

    # Learn PDE on each region
    print("\nLearning PDEs...")
    models = {}
    for reg in region_keys:
        coef, keys, r2 = learn_pde(reg)
        models[reg] = {"coef": coef, "keys": keys, "r2": r2}
        active = [f"{keys[i]}:{coef[i]:+.4f}" for i in range(len(keys)) if abs(coef[i]) > 1e-10]
        print(f"  {reg}: R²={r2:.3f}, terms: {', '.join(active)}")

    # Evaluate on all regions
    print("\nEvaluating...")
    all_results = {}
    for train in region_keys:
        m = models[train]
        for test in region_keys:
            key = f"{train}_{test}"
            rmse, r2, ya, yp = evaluate_on_region(test, m["coef"], m["keys"])
            all_results[key] = {"rmse": rmse, "r2": r2, "actual": ya, "pred": yp}
            print(f"  {train}→{test}: RMSE={rmse:.3f}, R²={r2:.2f}")

    # Print matrix
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION MATRIX (RMSE / R²)")
    print("=" * 60)
    hdr = f"{'Train \\ Test':16s}"
    for r in region_keys:
        hdr += f"{r:>14s}"
    print(hdr)
    print("-" * 60)
    for train in region_keys:
        line = f"{train:16s}"
        for test in region_keys:
            key = f"{train}_{test}"
            r = all_results[key]
            line += f"{r['rmse']:6.3f}/{r['r2']:5.2f}  "
        print(line)

    # ────────────────────────────────────────
    # Plotting — clean style + hexbin + colorbar
    # ────────────────────────────────────────
    print("\nRendering figure...")
    fig = plt.figure(figsize=(7.48, 7.0), facecolor='white')
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.04],
                          hspace=0.35, wspace=0.30,
                          left=0.09, right=0.90, bottom=0.10, top=0.95)

    cmap = plt.cm.Blues
    cmap.set_bad('white')

    for i, train in enumerate(region_keys):
        for j, test in enumerate(region_keys):
            ax = fig.add_subplot(gs[i, j])
            key = f"{train}_{test}"
            r = all_results[key]

            if len(r["actual"]) > 0:
                ya, yp = r["actual"], r["pred"]
                lim = max(np.abs(ya).max(), np.abs(yp).max()) * 1.1

                hb = ax.hexbin(ya, yp, gridsize=35, cmap=cmap,
                              mincnt=1, alpha=0.85, rasterized=True,
                              extent=[-lim, lim, -lim, lim])

                ax.plot([-lim, lim], [-lim, lim], '--', color='#C0392B', lw=0.8)
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                ax.set_aspect('equal')

                txt = f"RMSE={r['rmse']:.3f}\nR²={r['r2']:.2f}"
                ax.text(0.95, 0.05, txt, transform=ax.transAxes,
                       ha='right', va='bottom', fontsize=6.5,
                       bbox=dict(facecolor='white', alpha=0.85, pad=2,
                                edgecolor='#DDDDDD', linewidth=0.3))

            ax.set_title(f"{rlabels[i]} → {rlabels[j]}", fontsize=7,
                        fontweight='bold', pad=10)
            ax.grid(True, alpha=0.2, linewidth=0.3)
            ax.tick_params(labelsize=6)

            if j == 0:
                ax.set_ylabel("Predicted", fontsize=6.5)
            if i == 2:
                ax.set_xlabel("Actual", fontsize=6.5)

    # Shared labels — centered on plot area, not page
    fig.text(0.48, 0.02, "Actual ∂T/∂t (normalized)", fontsize=8,
             ha='center', fontweight='bold')
    fig.text(0.02, 0.525, "Predicted ∂T/∂t (normalized)", fontsize=8,
             va='center', rotation=90, fontweight='bold')

    # Colorbar
    cbar_ax = fig.add_subplot(gs[:, 3])
    cb = fig.colorbar(hb, cax=cbar_ax)
    cb.set_label('Point count', fontsize=6.5, fontweight='bold')
    cb.ax.tick_params(labelsize=5.5)

    fig.savefig(FIGS_PATH / "fig4_cross_validation.png", dpi=600,
                facecolor='white', transparent=False)
    fig.savefig(FIGS_PATH / "fig4_cross_validation.pdf",
                facecolor='white', transparent=False)
    print(f"Saved {FIGS_PATH}/fig4_cross_validation.png (600 dpi)")
    print(f"Saved {FIGS_PATH}/fig4_cross_validation.pdf")
    plt.close(fig)

    # TIFF
    from PIL import Image as PILImage
    png_path = FIGS_PATH / "fig4_cross_validation.png"
    tiff_path = FIGS_PATH / "fig4_cross_validation.tiff"
    img_pil = PILImage.open(png_path).convert('RGB')
    img_pil.save(tiff_path, dpi=(600, 600), compression='tiff_lzw')
    print(f"Saved {tiff_path}")

    # Caption
    print("\n── Caption for manuscript ──")
    print("Figure 4. Spatial cross-validation of the discovered")
    print("PDE across southeastern US subregions. Rows indicate")
    print("the training region, columns the testing region.")
    print("Diagonal panels (red border) show same-region")
    print("performance, off-diagonal panels show cross-region")
    print("generalization. RMSE and coefficient of determination")
    print("(R²) are reported for each train/test combination.")


if __name__ == "__main__":
    main()
