"""
Bootstrap confidence intervals for discovered PDE coefficients.

Block bootstrap with block size = 12 months (1 year) on the Carolinas patch.
Reports coefficient CIs and selection frequencies across 100 replicates.
"""

import numpy as np
import xarray as xr
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]


DATA_PATH = PROJECT_ROOT / "data/climate_multivariate.nc"
FIGS_PATH = PROJECT_ROOT / "figures"
FIGS_PATH.mkdir(exist_ok=True)

REGION = "carolinas"
REGIONS = {
    "florida":   {"lat": (25.0, 30.0), "lon": (-83.0, -80.0)},
    "ga_al":     {"lat": (30.0, 34.0), "lon": (-88.0, -82.0)},
    "carolinas": {"lat": (33.0, 35.0), "lon": (-82.0, -75.0)},
}
PATCH_SIZE = 32
DX = 0.05
N_BOOTSTRAP = 100
BLOCK_SIZE = 12


def extract(ds, key):
    r = REGIONS[key]
    lat_s = ds.lat.sel(lat=slice(r["lat"][0], r["lat"][1]))
    lon_s = ds.lon.sel(lon=slice(r["lon"][0], r["lon"][1]))
    ci, cj = len(lat_s)//2, len(lon_s)//2
    h = PATCH_SIZE//2
    return ds.sel(lat=slice(lat_s.values[ci-h], lat_s.values[ci+h-1]),
                  lon=slice(lon_s.values[cj-h], lon_s.values[cj+h-1]))


def compute_terms(arr_norm):
    """Compute all candidate terms from a (nt, ny, nx) array."""
    nt, ny, nx = arr_norm.shape
    T = arr_norm
    T_x = np.zeros_like(T); T_y = np.zeros_like(T)
    T_xx = np.zeros_like(T); T_yy = np.zeros_like(T); T_xy = np.zeros_like(T)
    for t_i in range(nt):
        f = T[t_i]
        T_x[t_i, :, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2*DX)
        T_y[t_i, 1:-1, :] = (f[2:, :] - f[:-2, :]) / (2*DX)
        T_xx[t_i, :, 1:-1] = (f[:, 2:] - 2*f[:, 1:-1] + f[:, :-2]) / DX**2
        T_yy[t_i, 1:-1, :] = (f[2:, :] - 2*f[1:-1, :] + f[:-2, :]) / DX**2
        T_xy[t_i, 1:-1, 1:-1] = (f[2:, 2:] - f[2:, :-2] - f[:-2, 2:] + f[:-2, :-2]) / (4*DX*DX)
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
    sin_t = np.sin(2*np.pi*np.arange(nt) / 12)
    cos_t = np.cos(2*np.pi*np.arange(nt) / 12)
    terms["sin"] = sin_t[:, None, None] * np.ones((1, ny, nx))
    terms["cos"] = cos_t[:, None, None] * np.ones((1, ny, nx))
    return terms


def discover_stlsq(arr, term_keys):
    """Discover PDE using OLS + iterative thresholding."""
    terms = compute_terms(arr)
    X = np.column_stack([terms[k].ravel() for k in term_keys])
    dT = np.zeros_like(arr)
    dT[1:-1] = (arr[2:] - arr[:-2]) / 2.0
    y = dT.ravel()
    n_vars = len(term_keys)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    for threshold in [0.1, 0.05, 0.02, 0.01, 0.005]:
        support = np.abs(coef) > threshold
        if support.sum() < 2:
            continue
        coef_new = np.zeros(n_vars)
        coef_new[support] = np.linalg.lstsq(X[:, support], y, rcond=None)[0]
        coef = coef_new
    support = np.abs(coef) > 1e-10
    if support.sum() > 0:
        coef[support] = np.linalg.lstsq(X[:, support], y, rcond=None)[0]
    return coef


if __name__ == "__main__":
    np.random.seed(42)  # fixed seed for reproducible confidence intervals
    print("=" * 60)
    print("Bootstrap Confidence Intervals (Carolinas)")
    print("=" * 60)

    ds = xr.open_dataset(DATA_PATH)
    patch = extract(ds, REGION)
    arr_raw = patch["t2m"].values
    arr_raw = np.where(np.isnan(arr_raw), np.nanmean(arr_raw), arr_raw)
    arr_mu, arr_std = arr_raw.mean(), arr_raw.std()
    arr = (arr_raw - arr_mu) / arr_std
    nt = len(arr)
    ds.close()

    term_keys = ["sin", "cos", "T", "T^2", "T_x", "T_y", "T_xx", "T_yy", "T_xy",
                 "T*T_x", "T*T_y", "T*T_xx", "T*T_yy", "T*T_xy",
                 "T^2*T_x", "T^2*T_y", "T^2*T_xx", "T^2*T_yy", "T^2*T_xy"]

    # Full-data coefficients
    full_coef = discover_stlsq(arr, term_keys)
    print(f"\nFull-data PDE (Carolinas):")
    parts = []
    for c, k in sorted(zip(full_coef, term_keys), key=lambda x: -abs(x[0])):
        if abs(c) > 1e-10:
            parts.append(f"{c:+.4f}·{k}")
    print("  ∂T/∂t = " + " ".join(parts))

    # Bootstrap
    n_blocks = nt // BLOCK_SIZE
    boot_coefs = np.zeros((N_BOOTSTRAP, len(term_keys)))
    selected = np.zeros((N_BOOTSTRAP, len(term_keys)), dtype=bool)

    for b in range(N_BOOTSTRAP):
        # Sample blocks with replacement
        block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
        boot_arr = np.concatenate([arr[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE]
                                    for i in block_indices], axis=0)
        # Trim/pad to nt
        boot_arr = boot_arr[:nt]
        if len(boot_arr) < nt:
            pad = nt - len(boot_arr)
            boot_arr = np.concatenate([boot_arr, boot_arr[:pad]], axis=0)

        bc = discover_stlsq(boot_arr, term_keys)
        boot_coefs[b] = bc
        selected[b] = np.abs(bc) > 1e-10

    # Results
    print(f"\nBootstrap ({N_BOOTSTRAP} replicates, block size = {BLOCK_SIZE}):")
    key_terms = ["sin", "T_y", "T*T_y", "cos", "T", "T^2", "T_x", "T_xx"]
    for k in key_terms:
        idx = term_keys.index(k)
        c_full = full_coef[idx]
        sel_pct = selected[:, idx].mean() * 100
        vals = boot_coefs[selected[:, idx], idx]
        if len(vals) >= 10:
            ci_lo, ci_hi = np.percentile(vals, [2.5, 97.5])
            print(f"  {k:10s}: full={c_full:+.4f}, "
                  f"selected={sel_pct:5.1f}%, "
                  f"[95% CI: {ci_lo:.3f}, {ci_hi:.3f}]")
        else:
            print(f"  {k:10s}: full={c_full:+.4f}, selected={sel_pct:5.1f}% (<10 reps)")

    print(f"\nOutput saved to {FIGS_PATH}/")
    print("Done!")
