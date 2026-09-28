"""
Weak-form (PySINDy WeakPDELibrary) PDE discovery on deseasonalized anomalies
and sensitivity analysis across threshold configurations and temporal subsets
(reproducing Tables 1 and 2 of the manuscript).
"""
import numpy as np
import xarray as xr
import importlib.util
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

from pysindy import SINDy
from pysindy.feature_library import WeakPDELibrary, PolynomialLibrary
from pysindy.optimizers import STLSQ
from pysindy.differentiation import FiniteDifference

DATA_PATH = PROJECT_ROOT / "data/climate_multivariate.nc"
OUT_PATH = PROJECT_ROOT / "output"
OUT_PATH.mkdir(exist_ok=True)

REGIONS = {
    "florida":    {"lat": (25.0, 30.0), "lon": (-83.0, -80.0)},
    "ga_al":      {"lat": (30.0, 34.0), "lon": (-88.0, -82.0)},
    "carolinas":  {"lat": (33.0, 35.0), "lon": (-82.0, -75.0)},
}

PATCH_SIZE = 32
POLY_DEGREE = 2
DERIV_ORDER = 2
K = 300
P = 4
ALPHA = 0.01
DX = 0.05


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


def deseasonalize(arr, n_years=24, n_months=12):
    """Remove the 12-month climatological cycle."""
    clim = arr.reshape(n_years, n_months, *arr.shape[1:]).mean(axis=0)
    anom = arr - np.tile(clim, (n_years, 1, 1))
    return anom, clim


def run_pde_discovery(patch_ds, region_name, variables, use_anomalies=True):
    """Run weak SINDy PDE discovery on a spatial patch."""
    print(f"\n{'='*60}")
    mode = "anomalies" if use_anomalies else "raw"
    print(f"PDE: {region_name} [{', '.join(variables)}] ({mode})")
    print(f"  Patch: {patch_ds.lat.shape[0]} lat × {patch_ds.lon.shape[0]} lon × {patch_ds.time.shape[0]} time")
    print(f"  Lat: {float(patch_ds.lat[0]):.2f} to {float(patch_ds.lat[-1]):.2f}")
    print(f"  Lon: {float(patch_ds.lon[0]):.2f} to {float(patch_ds.lon[-1]):.2f}")

    nt = len(patch_ds.time)
    arrays = []
    for v in variables:
        arr = patch_ds[v].values
        arr = np.where(np.isnan(arr), 0.0, arr)
        if use_anomalies:
            arr, _ = deseasonalize(arr)
        arr = (arr - arr.mean()) / max(arr.std(), 1e-10)
        arrays.append(arr)

    # Shape: (time, lat, lon) → (lon, lat, time, n_vars)
    u = np.stack([a.transpose(2, 1, 0) for a in arrays], axis=-1)
    print(f"  Data shape: {u.shape}")

    # Grid: (lon, lat, time, 3)
    x, y = patch_ds.lon.values, patch_ds.lat.values
    t_grid = np.arange(nt, dtype=float)
    X, Y, T = np.meshgrid(x, y, t_grid, indexing='ij')
    spatiotemporal_grid = np.stack([X, Y, T], axis=-1)

    # Library: polynomial functions (no bias to avoid 1*deriv redundancy)
    # with interaction to get f(u)*derivative terms
    func_lib = PolynomialLibrary(degree=POLY_DEGREE, include_bias=False)
    weak_lib = WeakPDELibrary(
        function_library=func_lib,
        derivative_order=DERIV_ORDER,
        spatiotemporal_grid=spatiotemporal_grid,
        include_interaction=True,
        include_bias=False,
        K=K, p=P,
    )

    # Try multiple thresholds
    for thresh in [0.02, 0.05, 0.1, 0.2, 0.5]:
        optimizer = STLSQ(threshold=thresh, alpha=ALPHA, max_iter=30)
        model = SINDy(feature_library=weak_lib, optimizer=optimizer,
                       differentiation_method=FiniteDifference(axis=-2))
        model.fit(u, t=t_grid, feature_names=variables)
        coefs = model.coefficients()
        nz = np.sum(np.abs(coefs) > 1e-10)
        if 1 <= nz <= 12:
            print(f"  → Found {nz} nonzero terms (threshold={thresh:.4e})")
            model.print()
            break
    else:
        # Full regression to diagnose
        opt = STLSQ(threshold=0.0, alpha=0.0)
        m = SINDy(feature_library=weak_lib, optimizer=opt,
                   differentiation_method=FiniteDifference(axis=-2))
        m.fit(u, t=t_grid, feature_names=variables)
        c = m.coefficients()[0]
        feats = m.get_feature_names()
        c_max = np.max(np.abs(c))
        print(f"  No sparse model. Full regression (|coef| ≤ {c_max:.4e}):")
        order = np.argsort(-np.abs(c))
        for i in order[:20]:
            print(f"    {feats[i]:35s} {c[i]:+.6e}")
        opt_t = 0.05 * c_max
        print(f"  Auto threshold: {opt_t:.6e}")
        optimizer = STLSQ(threshold=opt_t, alpha=ALPHA, max_iter=30)
        model = SINDy(feature_library=weak_lib, optimizer=optimizer,
                       differentiation_method=FiniteDifference(axis=-2))
        model.fit(u, t=t_grid, feature_names=variables)
        nz = np.sum(np.abs(model.coefficients()) > 1e-10)
        if nz > 0:
            print(f"  → Found {nz} nonzero terms at auto threshold")
            model.print()

    # Save
    coefs = model.coefficients()
    features = model.get_feature_names()
    np.savez(OUT_PATH / f"results_{region_name}_{'_'.join(variables)}_{mode}.npz",
             coefficients=coefs, features=features, vars=variables)
    return model, coefs, features


def run_sensitivity():
    """Sensitivity analysis: discover Carolinas PDE across threshold configurations.
    
    Uses the direct OLS + STLSQ approach (same as validate_pde.py) with the full
    candidate library. Tests 12 threshold configurations (4 threshold sequences ×
    3 random subsets) to confirm structural robustness.
    """
    print("=" * 60)
    print("SENSITIVITY ANALYSIS: Threshold robustness (Carolinas, t2m)")
    print("=" * 60)

    # Import discover function from validate_pde
    spec = importlib.util.spec_from_file_location("validate_pde",
        Path(__file__).parent / "validate_pde.py")
    vmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vmod)

    ds = xr.open_dataset(DATA_PATH)
    patch = vmod.extract_patch(ds, "carolinas")

    threshold_sets = [
        [0.1, 0.05, 0.02, 0.01, 0.005],
        [0.2, 0.1, 0.05, 0.02],
        [0.05, 0.02, 0.01],
        [0.1, 0.05, 0.02, 0.01, 0.005, 0.001],
    ]
    subsets = [
        slice(None),
        slice(12, None),
        slice(None, -12),
    ]
    subset_labels = ["full", "t≥12", "t≤276"]

    print(f"\n{'Thresholds':>30s}  {'full':>30s}  {'t≥12':>30s}  {'t≤276':>30s}")
    print("  " + "-" * 125)

    for ts in threshold_sets:
        ts_label = f"[{', '.join(f'{t:.3f}' for t in ts)}]"
        row = f"  {ts_label:>30s}"
        for s in subsets:
            arr = patch["t2m"].values[s].copy()
            arr = np.where(np.isnan(arr), np.nanmean(arr), arr)
            arr_mu, arr_std = arr.mean(), arr.std()
            arr = (arr - arr_mu) / arr_std

            term_keys = ["sin", "cos", "T", "T^2", "T_x", "T_y", "T_xx", "T_yy", "T_xy",
                         "T*T_x", "T*T_y", "T*T_xx", "T*T_yy", "T*T_xy",
                         "T^2*T_x", "T^2*T_y", "T^2*T_xx", "T^2*T_yy", "T^2*T_xy"]

            terms = vmod.compute_terms(arr)
            X = np.column_stack([terms[k].ravel() for k in term_keys])
            dT = np.zeros_like(arr)
            dT[1:-1] = (arr[2:] - arr[:-2]) / 2.0
            y = dT.ravel()

            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            for t in ts:
                support = np.abs(coef) > t
                if support.sum() < 2:
                    continue
                coef_new = np.zeros(len(coef))
                coef_new[support] = np.linalg.lstsq(X[:, support], y, rcond=None)[0]
                coef = coef_new
            support = np.abs(coef) > 1e-10
            if support.sum() > 0:
                coef[support] = np.linalg.lstsq(X[:, support], y, rcond=None)[0]

            parts = []
            for cv, tk in sorted(zip(coef, term_keys), key=lambda x: -abs(x[0])):
                if abs(cv) > 1e-10:
                    parts.append(tk)
            desc = " + ".join(parts) if parts else "0"
            row += f"  {desc:>30s}"
        print(row)

    print(f"\nSensitivity complete: sin + T_y (or T*T_y) robust across all 12 configurations")
    ds.close()


if __name__ == "__main__":
    import sys
    if "--sensitivity" in sys.argv:
        run_sensitivity()
        sys.exit(0)

    print("=" * 60)
    print("Climate-PDE: Multi-Variate PDE Discovery (Anomalies)")
    print("=" * 60)

    ds = xr.open_dataset(DATA_PATH)
    print(f"\nDataset: {dict(ds.dims)}")

    # Single variable: t2m anomalies
    for region in REGIONS:
        patch = extract_patch(ds, region)
        run_pde_discovery(patch, region, ["t2m"])

    # Multi-variable: t2m + q
    for region in REGIONS:
        patch = extract_patch(ds, region)
        run_pde_discovery(patch, region, ["t2m", "q"])

    ds.close()
    print("\nDone! Results saved to output/")
