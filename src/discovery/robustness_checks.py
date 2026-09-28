"""
Robustness checks cited in Sections 2.2 and 3.1 of the manuscript.

(1) Weak-form (WSINDy-style) confirmation on the raw Carolinas patch: the known
    seasonal cycle is supplied as an exogenous forcing variable; the weak-form
    regression must independently select seasonal forcing as the dominant term.
(2) Strong-form noise sensitivity: with Gaussian noise (sigma = 0.3) added to
    the standardized field, the strong-form STLSQ fit loses T_y and picks up
    spurious cos, T, and T^2 terms (R^2 falls from 0.875 to 0.751).
"""
import numpy as np
import xarray as xr
from pathlib import Path
from pysindy import SINDy
from pysindy.feature_library import WeakPDELibrary, PolynomialLibrary
from pysindy.optimizers import STLSQ

from validate_pde import extract_patch, compute_terms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data/climate_multivariate.nc"
REGION = "carolinas"
TERM_KEYS = ["sin", "cos", "T", "T^2", "T_x", "T_y", "T_xx", "T_yy", "T_xy",
             "T*T_x", "T*T_y", "T*T_xx", "T*T_yy", "T*T_xy",
             "T^2*T_x", "T^2*T_y", "T^2*T_xx", "T^2*T_yy", "T^2*T_xy"]


def strong_form_fit(arr):
    terms = compute_terms(arr)
    X = np.column_stack([terms[k].ravel() for k in TERM_KEYS])
    dT = np.zeros_like(arr)
    dT[1:-1] = (arr[2:] - arr[:-2]) / 2.0
    y = dT.ravel()
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    for threshold in [0.1, 0.05, 0.02, 0.01, 0.005]:
        support = np.abs(coef) > threshold
        if support.sum() < 2:
            continue
        c = np.zeros(len(coef))
        c[support] = np.linalg.lstsq(X[:, support], y, rcond=None)[0]
        coef = c
    support = np.abs(coef) > 1e-10
    if support.sum() > 0:
        coef[support] = np.linalg.lstsq(X[:, support], y, rcond=None)[0]
    yp = X @ coef
    r2 = 1 - np.sum((y - yp) ** 2) / np.sum((y - y.mean()) ** 2)
    return coef, r2


def weak_form_check(T, lon, lat):
    nt, ny, nx = T.shape
    s = np.tile(np.sin(2 * np.pi * np.arange(nt) / 12)[:, None, None], (1, ny, nx))
    u = np.stack([T.transpose(2, 1, 0), s.transpose(2, 1, 0)], axis=-1)
    t = np.arange(nt, dtype=float)
    Xg, Yg, Tg = np.meshgrid(lon, lat, t, indexing="ij")
    grid = np.stack([Xg, Yg, Tg], axis=-1)
    lib = WeakPDELibrary(function_library=PolynomialLibrary(degree=1, include_bias=False),
                         derivative_order=1, spatiotemporal_grid=grid,
                         include_interaction=True, include_bias=False, K=300, p=4)
    m = SINDy(feature_library=lib, optimizer=STLSQ(threshold=0.0, max_iter=1))
    m.fit(u, t=t, feature_names=["T", "s"])
    c = m.coefficients()[0].ravel()
    f = m.get_feature_names()
    top = sorted(zip(f, c), key=lambda kv: -abs(kv[1]))[:6]
    return top


if __name__ == "__main__":
    print("=" * 60)
    print("Robustness checks (manuscript Sections 2.2 / 3.1)")
    print("=" * 60)

    ds = xr.open_dataset(DATA_PATH)
    patch = extract_patch(ds, REGION)
    arr = patch["t2m"].values.copy()
    lon, lat = patch.lon.values, patch.lat.values
    mu, sd = np.nanmean(arr), np.nanstd(arr)
    T = (np.where(np.isnan(arr), mu, arr) - mu) / sd
    ds.close()

    coef, r2 = strong_form_fit(T)
    sel = {k: round(float(c), 4) for k, c in zip(TERM_KEYS, coef) if abs(c) > 1e-10}
    print(f"\n[1] Strong-form clean fit:      R2={r2:.3f}  terms={sel}")

    np.random.seed(7)
    noisy = T + np.random.randn(*T.shape) * 0.3
    coef_n, r2_n = strong_form_fit(noisy)
    sel_n = {k: round(float(c), 4) for k, c in zip(TERM_KEYS, coef_n) if abs(c) > 1e-10}
    print(f"[2] Strong-form + N(0, 0.3):    R2={r2_n:.3f}  terms={sel_n}")

    print("\n[3] Weak-form (WSINDy) dominant terms (raw patch, seasonal forcing exogenous):")
    for name, c in weak_form_check(T, lon, lat):
        if abs(c) > 1e-10:
            print(f"      {name:30s} {c:+.4f}")
    print("\nDone!")
