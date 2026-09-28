"""
Regularization parameter scan: evaluates active terms, training fit (RMSE and R²),
and forward integration numerical stability (12-month trajectory RMSE) across
candidate STLSQ thresholds.
"""
import numpy as np
import xarray as xr
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]


DATA_PATH = PROJECT_ROOT / "data/climate_multivariate.nc"
DX = 0.05

REGIONS = {
    "florida":   {"lat": (25.0, 30.0), "lon": (-83.0, -80.0)},
    "ga_al":     {"lat": (30.0, 34.0), "lon": (-88.0, -82.0)},
    "carolinas": {"lat": (33.0, 35.0), "lon": (-82.0, -75.0)},
}
PATCH_SIZE = 32


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
    d[1:-1, :] = (T[2:, :] - T[:-2, :]) / (2*DX)
    return d


def build_library_and_target(T, q):
    """Build library matrix X and target y from normalized fields."""
    nt, ny, nx = T.shape
    sin_all = np.tile(np.sin(2*np.pi*np.arange(nt)/12)[:, None, None], (1, ny, nx))
    T_y_all = np.zeros_like(T)
    for t in range(nt):
        T_y_all[t] = compute_T_y(T[t])

    terms = {"sin": sin_all, "T_y": T_y_all, "T*q": T * q, "T": T, "q": q}

    dT = np.zeros_like(T)
    dT[1:-1] = (T[2:] - T[:-2]) / 2.0

    names = ["sin", "T_y", "T*q", "T", "q"]
    X = np.column_stack([terms[n].ravel() for n in names])
    y = dT.ravel()
    valid = ~np.isnan(y)
    return X[valid], y[valid], names


def stlsq(X, y, threshold):
    """Sequential thresholded least squares with single threshold."""
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    s = np.abs(coef) > threshold
    if s.sum() < 1:
        return np.zeros(len(coef)), s
    c = np.zeros(len(coef))
    c[s] = np.linalg.lstsq(X[:, s], y, rcond=None)[0]
    # Recheck with same threshold
    coef = c
    s = np.abs(coef) > threshold
    if s.sum() > 0:
        c2 = np.zeros(len(coef))
        c2[s] = np.linalg.lstsq(X[:, s], y, rcond=None)[0]
        coef = c2
    return coef, np.abs(coef) > 1e-12


def forward_multivar(T0, q_data, t_start, n_steps, coeffs, cand_names):
    """Forward integrate multi-var PDE."""
    ny, nx = T0.shape
    traj = [T0.copy()]
    T = T0.copy()
    for step in range(n_steps):
        t_idx = t_start + step
        q_at_t = q_data[min(t_idx, len(q_data)-1)]
        rhs = np.zeros_like(T)
        T_y = compute_T_y(T)
        for name, c in coeffs.items():
            if name == "sin":
                rhs += c * np.sin(2*np.pi*t_idx/12)
            elif name == "cos":
                rhs += c * np.cos(2*np.pi*t_idx/12)
            elif name == "T":
                rhs += c * T
            elif name == "q":
                rhs += c * q_at_t
            elif name == "T_y":
                rhs += c * T_y
            elif name == "T*q":
                rhs += c * T * q_at_t
            elif name == "T^2":
                rhs += c * T**2
            elif name == "q^2":
                rhs += c * q_at_t**2
        T = T + rhs
        traj.append(T.copy())
    return np.array(traj)


def evaluate_stability(region_key, threshold):
    """Discover PDE with given threshold and evaluate forward stability."""
    ds = xr.open_dataset(DATA_PATH)
    patch = extract(ds, region_key)

    T_raw = patch["t2m"].values.copy()
    T_raw = np.where(np.isnan(T_raw), np.nanmean(T_raw), T_raw)
    q_raw = patch["q"].values.copy()
    q_raw = np.where(np.isnan(q_raw), 0.0, q_raw)

    T = (T_raw - T_raw.mean()) / T_raw.std()
    q = (q_raw - q_raw.mean()) / q_raw.std()
    nt = len(T)

    X, y, names = build_library_and_target(T, q)
    coef, support = stlsq(X, y, threshold)

    coeffs = {}
    for i, n in enumerate(names):
        if support[i]:
            coeffs[n] = float(coef[i])

    if not coeffs:
        return coeffs, np.nan, np.nan, np.nan

    # Forward integration from multiple start points
    n_steps = 24
    n_starts = 4
    max_rmse_12 = []
    for start_idx in np.linspace(12, nt - n_steps - 12, n_starts, dtype=int):
        pred = forward_multivar(T[start_idx], q, start_idx, n_steps, coeffs, names)
        actual = T[start_idx:start_idx + n_steps + 1]
        # Check for blow-up
        if np.any(np.abs(pred) > 20):
            max_rmse_12.append(np.inf)
            continue
        rmse = [np.sqrt(np.mean((pred[i] - actual[i])**2)) for i in range(n_steps + 1)]
        if len(rmse) > 12:
            max_rmse_12.append(float(rmse[12]))
        else:
            max_rmse_12.append(np.inf)

    rmse_train = float(np.sqrt(np.mean((X @ coef - y)**2)))
    r2_train = max(0, 1 - np.sum((X @ coef - y)**2) / np.sum((y - y.mean())**2))

    return coeffs, rmse_train, r2_train, np.mean(max_rmse_12)


if __name__ == "__main__":
    print("=" * 60)
    print("Multi-var PDE Regularization Scan")
    print("=" * 60)

    thresholds = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]

    for region in ["florida", "ga_al", "carolinas"]:
        print(f"\n--- {region} ---")
        print(f"{'Thresh':>8s} {'Active terms':40s} {'Train RMSE':>12s} {'Train R²':>10s} {'12mo RMSE':>10s}")
        print("-" * 85)

        best = {"thresh": None, "rmse_12": np.inf, "coeffs": {}, "r2": 0, "rmse_train": np.inf}
        for t in thresholds:
            coeffs, rmse_tr, r2_tr, rmse_12 = evaluate_stability(region, t)
            term_str = ", ".join([f"{c:.3f}·{n}" for n, c in coeffs.items()]) if coeffs else "(empty)"
            flag = ""
            if coeffs and np.isfinite(rmse_12):
                flag = " ✓"
                if rmse_12 < best["rmse_12"]:
                    best = {"thresh": t, "rmse_12": rmse_12, "coeffs": coeffs,
                            "r2": r2_tr, "rmse_train": rmse_tr}
            elif coeffs and not np.isfinite(rmse_12):
                flag = " 💥"

            print(f"{t:>8.2f} {term_str:40s} {rmse_tr:>12.4f} {r2_tr:>10.3f} "
                  f"{rmse_12 if np.isfinite(rmse_12) else np.nan:>10.3f}{flag}")

        if best["coeffs"]:
            print(f"\n  Best: thresh={best['thresh']:.2f}, terms={list(best['coeffs'].keys())}, "
                  f"12mo RMSE={best['rmse_12']:.3f}, R²={best['r2']:.3f}")
        else:
            print(f"\n  No stable equation found")

    print("\nDone!")
