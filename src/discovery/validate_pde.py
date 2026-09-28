"""
Single-variable PDE discovery with raw temperature data and explicit seasonal forcing.
Computes spatial cross-validation and transfer performance across Florida, GA/AL,
and Carolinas subregions (reproducing Table 3 of the manuscript).
"""
import numpy as np
import xarray as xr
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

from pysindy import SINDy
from pysindy.feature_library import WeakPDELibrary, PolynomialLibrary, CustomLibrary, ConcatLibrary
from pysindy.optimizers import STLSQ
from pysindy.differentiation import FiniteDifference
from sklearn.linear_model import LinearRegression

DATA_PATH = PROJECT_ROOT / "data/climate_multivariate.nc"

REGIONS = {
    "florida":   {"lat": (25.0, 30.0), "lon": (-83.0, -80.0)},
    "ga_al":     {"lat": (30.0, 34.0), "lon": (-88.0, -82.0)},
    "carolinas": {"lat": (33.0, 35.0), "lon": (-82.0, -75.0)},
}
PATCH_SIZE = 32
PIXEL_SCALE = 0.05


def extract_patch(ds, region_key, patch_size=PATCH_SIZE):
    r = REGIONS[region_key]
    lat_slice = ds.lat.sel(lat=slice(r["lat"][0], r["lat"][1]))
    lon_slice = ds.lon.sel(lon=slice(r["lon"][0], r["lon"][1]))
    ci, cj = len(lat_slice) // 2, len(lon_slice) // 2
    half = patch_size // 2
    lat_p = lat_slice.values[ci-half:ci+half]
    lon_p = lon_slice.values[cj-half:cj+half]
    return ds.sel(lat=slice(lat_p[0], lat_p[-1]),
                  lon=slice(lon_p[0], lon_p[-1]))


def compute_terms(arr_norm, dx=PIXEL_SCALE, dy=PIXEL_SCALE):
    """Compute all candidate terms from a (nt, ny, nx) array."""
    nt, ny, nx = arr_norm.shape
    T = arr_norm
    T_x = np.zeros_like(T); T_y = np.zeros_like(T)
    T_xx = np.zeros_like(T); T_yy = np.zeros_like(T); T_xy = np.zeros_like(T)
    for t_i in range(nt):
        f = T[t_i]
        T_x[t_i, :, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2*dx)
        T_y[t_i, 1:-1, :] = (f[2:, :] - f[:-2, :]) / (2*dy)
        T_xx[t_i, :, 1:-1] = (f[:, 2:] - 2*f[:, 1:-1] + f[:, :-2]) / dx**2
        T_yy[t_i, 1:-1, :] = (f[2:, :] - 2*f[1:-1, :] + f[:-2, :]) / dy**2
        T_xy[t_i, 1:-1, 1:-1] = (f[2:, 2:] - f[2:, :-2] - f[:-2, 2:] + f[:-2, :-2]) / (4*dx*dy)
    terms = {
        "1": np.ones_like(T),
        "T": T,
        "T^2": T**2,
        "T_x": T_x, "T_y": T_y,
        "T_xx": T_xx, "T_yy": T_yy, "T_xy": T_xy,
        "T*T_x": T*T_x, "T*T_y": T*T_y,
        "T*T_xx": T*T_xx, "T*T_yy": T*T_yy, "T*T_xy": T*T_xy,
        "T^2*T_x": T**2 * T_x, "T^2*T_y": T**2 * T_y,
        "T^2*T_xx": T**2 * T_xx, "T^2*T_yy": T**2 * T_yy, "T^2*T_xy": T**2 * T_xy,
    }
    # Tile seasonal terms to full spatial grid
    sin_t = np.sin(2*np.pi*np.arange(nt) / 12)
    cos_t = np.cos(2*np.pi*np.arange(nt) / 12)
    terms["sin"] = sin_t[:, None, None] * np.ones((1, ny, nx))
    terms["cos"] = cos_t[:, None, None] * np.ones((1, ny, nx))
    return terms


def learn_pde(region_key):
    """Learn a PDE on raw data with seasonal terms using OLS + threshold."""
    ds = xr.open_dataset(DATA_PATH)
    patch = extract_patch(ds, region_key)
    nt, ny, nx = len(patch.time), len(patch.lat), len(patch.lon)

    arr = patch["t2m"].values
    arr = np.where(np.isnan(arr), np.nanmean(arr), arr)
    arr_mu, arr_std = arr.mean(), arr.std()
    arr = (arr - arr_mu) / arr_std

    # Compute all candidate terms
    terms = compute_terms(arr)

    # Build design matrix
    term_keys = ["sin", "cos", "T", "T^2", "T_x", "T_y", "T_xx", "T_yy", "T_xy",
                 "T*T_x", "T*T_y", "T*T_xx", "T*T_yy", "T*T_xy",
                 "T^2*T_x", "T^2*T_y", "T^2*T_xx", "T^2*T_yy", "T^2*T_xy"]
    X = np.column_stack([terms[k].ravel() for k in term_keys])

    # Target: dT/dt via central differences
    dT_dt = np.zeros_like(arr)
    dT_dt[1:-1] = (arr[2:] - arr[:-2]) / 2.0
    y = dT_dt.ravel()

    # OLS with iterative thresholding (STLSQ-style)
    n_vars = len(term_keys)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    for threshold in [0.1, 0.05, 0.02, 0.01, 0.005]:
        support = np.abs(coef) > threshold
        if support.sum() < 2:
            continue
        coef_new = np.zeros(n_vars)
        coef_new[support] = np.linalg.lstsq(X[:, support], y, rcond=None)[0]
        coef = coef_new

    # Final fit on support
    support = np.abs(coef) > 1e-10
    if support.sum() > 0:
        coef[support] = np.linalg.lstsq(X[:, support], y, rcond=None)[0]

    # R²
    y_pred = X @ coef
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    print(f"\n{region_key.upper()} discovered PDE (R² = {r2:.3f}):")
    print(f"  ∂T/∂t = ", end="")
    parts = []
    for i, (k, c) in enumerate(sorted(zip(term_keys, coef), key=lambda x: -abs(x[1]))):
        if abs(c) > 1e-10:
            parts.append(f"{c:+.4f}·{k}")
    print(" ".join(parts) if parts else "0 (no terms)")

    ds.close()
    return coef, term_keys, r2, (arr_mu, arr_std)


def evaluate_on_region(region_key, coef, term_keys, train_mu, train_std):
    """Evaluate a discovered PDE on a region."""
    ds = xr.open_dataset(DATA_PATH)
    patch = extract_patch(ds, region_key)
    nt = len(patch.time)

    arr = patch["t2m"].values
    arr = np.where(np.isnan(arr), np.nanmean(arr), arr)
    arr = (arr - train_mu) / train_std

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


if __name__ == "__main__":
    print("=" * 60)
    print("Climate-PDE: Raw Data + Seasonal Forcing")
    print("=" * 60)

    regions = ["florida", "ga_al", "carolinas"]

    # Learn PDE on each region
    models = {}
    for reg in regions:
        coef, keys, r2, norm = learn_pde(reg)
        models[reg] = {"coef": coef, "keys": keys, "r2": r2, "mu": norm[0], "std": norm[1]}

    # Evaluate
    all_results = {}
    for train in regions:
        m = models[train]
        for test in regions:
            key = f"{train}_{test}"
            print(f"\n  {train}→{test}: ", end="")
            rmse, r2, ya, yp = evaluate_on_region(test, m["coef"], m["keys"], m["mu"], m["std"])
            all_results[key] = {"rmse": rmse, "r2": r2, "actual": ya, "pred": yp}

    # Summary
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION MATRIX  (RMSE / R²)")
    print("=" * 60)
    hdr = f"{'Train \\ Test':16s}"
    for r in regions:
        hdr += f"{r:>14s}"
    print(hdr)
    print("-" * 58)
    for train in regions:
        line = f"{train:16s}"
        for test in regions:
            key = f"{train}_{test}"
            r = all_results.get(key, {"rmse": np.nan, "r2": np.nan})
            line += f"{r['rmse']:6.3f}/{r['r2']:5.2f}  "
        print(line)
    line = f"{'Train R²':16s}"
    for train in regions:
        line += f"{models[train]['r2']:12.3f}  "
    print(line)

    print("Done!")
