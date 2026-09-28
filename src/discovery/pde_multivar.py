"""
Multivariate coupled PDE discovery for temperature (t2m) and specific humidity (q).
Fits coupled equations using sequential thresholded least squares and evaluates
cross-validation performance across Florida, GA/AL, and Carolinas subregions
(Section 3.2 of the manuscript).
"""
import numpy as np
import xarray as xr
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]


DATA_PATH = PROJECT_ROOT / "data/climate_multivariate.nc"
REGIONS = {
    "florida":   {"lat": (25.0, 30.0), "lon": (-83.0, -80.0)},
    "ga_al":     {"lat": (30.0, 34.0), "lon": (-88.0, -82.0)},
    "carolinas": {"lat": (33.0, 35.0), "lon": (-82.0, -75.0)},
}
PATCH_SIZE = 32
DX = 0.05


def extract(ds, key):
    r = REGIONS[key]
    lat_s = ds.lat.sel(lat=slice(r["lat"][0], r["lat"][1]))
    lon_s = ds.lon.sel(lon=slice(r["lon"][0], r["lon"][1]))
    ci, cj = len(lat_s)//2, len(lon_s)//2
    h = PATCH_SIZE//2
    return ds.sel(lat=slice(lat_s.values[ci-h], lat_s.values[ci+h-1]),
                  lon=slice(lon_s.values[cj-h], lon_s.values[cj+h-1]))


def deriv_y(F):
    d = np.zeros_like(F)
    d[:, 1:-1, :] = (F[:, 2:, :] - F[:, :-2, :]) / (2*DX)
    return d


def deriv_x(F):
    d = np.zeros_like(F)
    d[:, :, 1:-1] = (F[:, :, 2:] - F[:, :, :-2]) / (2*DX)
    return d


def deriv_xx(F):
    d = np.zeros_like(F)
    d[:, :, 1:-1] = (F[:, :, 2:] - 2*F[:, :, 1:-1] + F[:, :, :-2]) / DX**2
    return d


def deriv_yy(F):
    d = np.zeros_like(F)
    d[:, 1:-1, :] = (F[:, 2:, :] - 2*F[:, 1:-1, :] + F[:, :-2, :]) / DX**2
    return d


def get_data(patch):
    """Return normalized (T, q, precip) arrays + normalization params."""
    arrs = {}
    params = {}
    for name in ["t2m", "q", "precip"]:
        a = patch[name].values.copy()
        a = np.where(np.isnan(a), np.nanmean(a), a)
        mu, s = a.mean(), max(a.std(), 1e-10)
        arrs[name] = (a - mu) / s
        params[name] = (mu, s)
    return arrs["t2m"], arrs["q"], arrs["precip"], params


def build_terms(T, q, pr):
    """Build dict of candidate terms: (nt, ny, nx) arrays."""
    nt = T.shape[0]
    ny, nx = T.shape[1], T.shape[2]
    sin = np.tile(np.sin(2*np.pi*np.arange(nt)/12)[:, None, None], (1, ny, nx))
    cos = np.tile(np.cos(2*np.pi*np.arange(nt)/12)[:, None, None], (1, ny, nx))
    one = np.ones_like(T)

    d = {}
    d["1"] = one; d["sin"] = sin; d["cos"] = cos
    d["T"] = T; d["q"] = q; d["pr"] = pr
    d["T_y"] = deriv_y(T); d["T_x"] = deriv_x(T)
    d["T_xx"] = deriv_xx(T); d["T_yy"] = deriv_yy(T)
    d["q_y"] = deriv_y(q); d["q_x"] = deriv_x(q)
    d["T^2"] = T**2; d["q^2"] = q**2
    d["T*q"] = T*q; d["T*pr"] = T*pr; d["q*pr"] = q*pr
    d["T*T_y"] = T * d["T_y"]; d["T*T_x"] = T * d["T_x"]
    d["q*q_y"] = q * d["q_y"]
    d["T*q_y"] = T * d["q_y"]; d["q*T_y"] = q * d["T_y"]
    d["T*T_xx"] = T * d["T_xx"]
    return d


def stlsq(X, y, thresholds=[0.1, 0.05, 0.02, 0.01, 0.005]):
    """Sequential thresholded least squares."""
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    for t in thresholds:
        s = np.abs(coef) > t
        if s.sum() < 1:
            continue
        c = np.zeros(len(coef))
        c[s] = np.linalg.lstsq(X[:, s], y, rcond=None)[0]
        coef = c
    s = np.abs(coef) > 1e-10
    if s.sum() > 0:
        coef[s] = np.linalg.lstsq(X[:, s], y, rcond=None)[0]
    return coef


def discover(target_var, region):
    """Discover PDE for target_var on region. Return (coefs, names, r2)."""
    ds = xr.open_dataset(DATA_PATH)
    patch = extract(ds, region)
    T, q, pr, params = get_data(patch)
    ds.close()

    terms = build_terms(T, q, pr)
    # All candidate term names
    cand = ["sin", "cos", "T", "q", "T_y", "T_x", "q_y", "q_x",
            "T_xx", "T_yy", "T^2", "q^2", "T*q", "T*pr", "q*pr",
            "T*T_y", "T*T_x", "q*q_y", "T*q_y", "q*T_y", "T*T_xx"]
    X = np.column_stack([terms[n].ravel() for n in cand])

    target = T if target_var == "t2m" else q
    dY = np.zeros_like(target)
    dY[1:-1] = (target[2:] - target[:-2]) / 2.0
    y = dY.ravel()

    coef = stlsq(X, y)
    yp = X @ coef
    ss_r = np.sum((y - yp)**2)
    ss_t = np.sum((y - y.mean())**2)
    r2 = 1 - ss_r / ss_t if ss_t > 0 else 0.0
    return coef, cand, r2


def evaluate(target_var, region, train_coef, train_cand, train_params):
    """Evaluate discovered PDE on region."""
    ds = xr.open_dataset(DATA_PATH)
    patch = extract(ds, region)
    T, q, pr, _ = get_data(patch)
    ds.close()

    terms = build_terms(T, q, pr)
    X = np.column_stack([terms[n].ravel() for n in train_cand])

    target = T if target_var == "t2m" else q
    dY = np.zeros_like(target)
    dY[1:-1] = (target[2:] - target[:-2]) / 2.0
    y = dY.ravel()

    yp = X @ train_coef
    valid = ~(np.isnan(y) | np.isnan(yp))
    valid &= (np.abs(y) < 10) & (np.abs(yp) < 10)
    ya, yp = y[valid], yp[valid]
    if len(ya) == 0:
        return np.nan, np.nan
    rmse = float(np.sqrt(np.mean((yp - ya)**2)))
    ss_r = np.sum((yp - ya)**2)
    ss_t = np.sum((ya - ya.mean())**2)
    r2 = float(1 - ss_r / ss_t) if ss_t > 0 else 0.0
    return rmse, r2


if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Variable Coupled PDE Discovery")
    print("=" * 60)

    regions = ["florida", "ga_al", "carolinas"]

    for target in ["t2m", "q"]:
        print(f"\n--- Target: ∂({target})/∂t ---")
        models = {}
        for reg in regions:
            coef, names, r2 = discover(target, reg)
            models[reg] = {"coef": coef, "names": names, "r2": r2}
            parts = []
            for c, n in sorted(zip(coef, names), key=lambda x: -abs(x[0])):
                if abs(c) > 1e-10:
                    parts.append(f"{c:+.4f}·{n}")
            print(f"  Train {reg}:  ∂/∂t = " + " ".join(parts) + f"  [R²={r2:.3f}]")

        # Cross-validation
        print(f"\n  Cross-validation matrix (RMSE / R²):")
        print(f"  {'':12s} {'FL':>14s} {'GA':>14s} {'NC':>14s}")
        print(f"  {'-'*58}")
        for train in regions:
            line = f"  {train:12s}"
            for test in regions:
                rmse, r2 = evaluate(target, test, models[train]["coef"],
                                    models[train]["names"], None)
                val = f"{rmse:.3f}/{r2:.2f}" if not np.isnan(r2) else "N/A"
                line += f"{val:>14s}"
            print(line)

    print("\nDone!")
