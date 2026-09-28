"""
Forward integration validation: evaluates 12-month forward predictive skill
for single-variable and multivariate PDEs against persistence and monthly climatology
baselines across 22 evaluation start dates (Section 3.3 of the manuscript).
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


def compute_T_y(T):
    """T_y from 2D field T (ny, nx)."""
    d = np.zeros_like(T)
    d[1:-1, :] = (T[2:, :] - T[:-2, :]) / (2*DX)
    return d


def compute_T_xx(T):
    d = np.zeros_like(T)
    d[:, 1:-1] = (T[:, 2:] - 2*T[:, 1:-1] + T[:, :-2]) / DX**2
    return d


def rhs_sin_Ty(T, t, a, b):
    """∂T/∂t = a·sin(2πt/12) + b·T_y"""
    return a * np.sin(2*np.pi*t/12) + b * compute_T_y(T)


def rhs_multivar(T, q, t, coeffs, term_names):
    """Compute RHS of multi-variable PDE given coeffs dict."""
    rhs = np.zeros_like(T)
    T_y = compute_T_y(T)
    T_xx = compute_T_xx(T)
    for name, c in coeffs.items():
        if name == "sin":
            rhs += c * np.sin(2*np.pi*t/12)
        elif name == "cos":
            rhs += c * np.cos(2*np.pi*t/12)
        elif name == "T":
            rhs += c * T
        elif name == "q":
            rhs += c * q
        elif name == "T_y":
            rhs += c * T_y
        elif name == "T_xx":
            rhs += c * T_xx
        elif name == "T^2":
            rhs += c * T**2
        elif name == "T*q":
            rhs += c * T * q
        elif name == "T*q_y":
            q_y = np.zeros_like(q)
            q_y[1:-1, :] = (q[2:, :] - q[:-2, :]) / (2*DX)
            rhs += c * T * q_y
        elif name == "q*q_y":
            q_y = np.zeros_like(q)
            q_y[1:-1, :] = (q[2:, :] - q[:-2, :]) / (2*DX)
            rhs += c * q * q_y
        elif name == "q*T_y":
            rhs += c * q * T_y
        elif name == "q^2":
            rhs += c * q**2
        elif name == "q_y":
            q_y = np.zeros_like(q)
            q_y[1:-1, :] = (q[2:, :] - q[:-2, :]) / (2*DX)
            rhs += c * q_y
        elif name == "T*T_y":
            rhs += c * T * T_y
    return rhs


def forward_integrate(T0, q_data, t_start, n_steps, rhs_func):
    """Step PDE forward from initial condition T0.
    
    T0: initial 2D field (ny, nx)
    q_data: actual q values over full period (nt, ny, nx) for semi-coupled
    t_start: starting time index
    n_steps: number of months to predict
    rhs_func: callable (T, q_at_t, t_index) → dT/dt
    """
    ny, nx = T0.shape
    traj = [T0.copy()]
    T = T0.copy()
    for step in range(n_steps):
        t_idx = t_start + step
        q_at_t = q_data[min(t_idx, len(q_data)-1)]
        dT = rhs_func(T, q_at_t, t_idx)
        T = T + dT
        traj.append(T.copy())
    return np.array(traj)


def compute_climatology(T):
    """Compute monthly climatology (12-month cycle mean)."""
    nt = len(T)
    month_of_year = np.arange(nt) % 12
    clim = np.zeros((12,) + T.shape[1:])
    for m in range(12):
        clim[m] = T[month_of_year == m].mean(axis=0)
    return clim


def evaluate_forward(region_key, n_steps=12, n_starts=22):
    """Evaluate forward prediction across multiple start months."""
    ds = xr.open_dataset(DATA_PATH)
    patch = extract(ds, region_key)

    T_raw = patch["t2m"].values.copy()
    T_raw = np.where(np.isnan(T_raw), np.nanmean(T_raw), T_raw)
    q_raw = patch["q"].values.copy()
    q_raw = np.where(np.isnan(q_raw), 0.0, q_raw)

    # Normalize
    T_mu, T_sigma = T_raw.mean(), T_raw.std()
    q_mu, q_sigma = q_raw.mean(), q_raw.std()
    T = (T_raw - T_mu) / T_sigma
    q = (q_raw - q_mu) / q_sigma
    nt = len(T)

    # Discover sin + T_y coefficients on this patch
    ny, nx = T.shape[1], T.shape[2]
    sin_all = np.tile(np.sin(2*np.pi*np.arange(nt)/12)[:, None, None], (1, ny, nx))
    T_y_all = np.zeros_like(T)
    for t in range(nt):
        T_y_all[t] = compute_T_y(T[t])
    dT = np.zeros_like(T)
    dT[1:-1] = (T[2:] - T[:-2]) / 2.0
    X = np.column_stack([sin_all.ravel(), T_y_all.ravel()])
    y = dT.ravel()
    valid = ~np.isnan(y)
    c = np.linalg.lstsq(X[valid], y[valid], rcond=None)[0]
    a, b = float(c[0]), float(c[1])

    # Also discover multivar PDE on this patch
    terms = {"sin": sin_all, "T_y": T_y_all, "T*q": T * q, "T": T, "q": q}
    cand_names = ["sin", "T_y", "T*q", "T", "q"]
    Xm = np.column_stack([terms[n].ravel() for n in cand_names])
    cm = np.linalg.lstsq(Xm[valid], y[valid], rcond=None)[0]
    # STLSQ
    for thresh in [0.02]:
        s = np.abs(cm) > thresh
        if s.sum() >= 2:
            c2 = np.zeros(len(cm))
            c2[s] = np.linalg.lstsq(Xm[valid][:, s], y[valid], rcond=None)[0]
            cm = c2
    mv_coeffs = {}
    for i, n in enumerate(cand_names):
        if abs(cm[i]) > 1e-10:
            mv_coeffs[n] = float(cm[i])

    ds.close()

    # Compute climatology (monthly means over full period)
    clim = compute_climatology(T)
    T_avg = T.mean(axis=(1, 2))  # spatial mean time series

    # Forward integration from multiple start points
    all_results = []
    start_indices = np.linspace(12, nt - n_steps - 12, n_starts, dtype=int)
    for start_idx in start_indices:
        # sin + T_y
        rhs_sin = lambda T, q, t: a * np.sin(2*np.pi*t/12) + b * compute_T_y(T)
        pred_sin = forward_integrate(T[start_idx], q, start_idx, n_steps, rhs_sin)
        actual_sin = T[start_idx:start_idx + n_steps + 1]

        # Multi-variable
        if mv_coeffs:
            rhs_mv = lambda T, q, t: rhs_multivar(T, q, t, mv_coeffs, cand_names)
            pred_mv = forward_integrate(T[start_idx], q, start_idx, n_steps, rhs_mv)
        else:
            pred_mv = None

        # Per-pixel RMSE (full field)
        rmse_sin = [float(np.sqrt(np.mean((pred_sin[i] - actual_sin[i])**2)))
                    for i in range(n_steps + 1)]
        rmse_mv = [float(np.sqrt(np.mean((pred_mv[i] - actual_sin[i])**2)))
                   for i in range(n_steps + 1)] if pred_mv is not None else None

        # Spatial mean RMSE (same eval basis as ARIMA)
        actual_avg = actual_sin.mean(axis=(1, 2))
        pred_sin_avg = pred_sin.mean(axis=(1, 2))
        rmse_sin_avg = [float(np.sqrt(np.mean((pred_sin_avg[i] - actual_avg[i])**2)))
                        for i in range(n_steps + 1)]
        pred_mv_avg = pred_mv.mean(axis=(1, 2)) if pred_mv is not None else None
        rmse_mv_avg = [float(np.sqrt(np.mean((pred_mv_avg[i] - actual_avg[i])**2)))
                       for i in range(n_steps + 1)] if pred_mv_avg is not None else None

        # Persistence baseline
        persist = np.tile(T[start_idx][None, :, :], (n_steps + 1, 1, 1))
        rmse_persist = [float(np.sqrt(np.mean((persist[i] - actual_sin[i])**2)))
                        for i in range(n_steps + 1)]
        persist_avg_val = T_avg[start_idx]
        rmse_persist_avg = [float(np.sqrt((persist_avg_val - actual_avg[i])**2))
                            for i in range(n_steps + 1)]

        # Climatology baseline (full field)
        clim_pred = np.array([clim[(start_idx + i) % 12] for i in range(n_steps + 1)])
        rmse_clim = [float(np.sqrt(np.mean((clim_pred[i] - actual_sin[i])**2)))
                     for i in range(n_steps + 1)]
        # Climatology spatial mean RMSE
        clim_avg = clim.mean(axis=(1, 2))
        clim_pred_avg = np.array([clim_avg[(start_idx + i) % 12] for i in range(n_steps + 1)])
        rmse_clim_avg = [float(np.sqrt(np.mean((clim_pred_avg[i] - actual_avg[i])**2)))
                         for i in range(n_steps + 1)]

        all_results.append({
            "start": start_idx,
            "rmse_sin": rmse_sin,
            "rmse_mv": rmse_mv,
            "rmse_persist": rmse_persist,
            "rmse_clim": rmse_clim,
            "rmse_sin_avg": rmse_sin_avg,
            "rmse_mv_avg": rmse_mv_avg,
            "rmse_persist_avg": rmse_persist_avg,
            "rmse_clim_avg": rmse_clim_avg,
            "pred_sin": pred_sin,
            "pred_mv": pred_mv,
            "actual": actual_sin,
        })

    return all_results, (a, b), mv_coeffs, T_mu, T_sigma


if __name__ == "__main__":
    print("=" * 60)
    print("Forward Integration Validation")
    print("=" * 60)

    all_region_results = {}
    leads_display = [1, 3, 6, 12]

    print("\n" + "=" * 100)
    print("FULL RMSE SUMMARY (spatial mean — manuscript-ready)")
    print("=" * 100)

    for region in ["florida", "ga_al", "carolinas"]:
        print(f"\n--- {region} ---")
        results, (a, b), mv_c, T_mu, T_sigma = evaluate_forward(region, n_steps=12, n_starts=22)

        # Average RMSE — spatial mean
        sin_rmse_avg = np.array([r["rmse_sin_avg"] for r in results])
        mv_rmse_avg = np.array([r["rmse_mv_avg"] for r in results if r["rmse_mv_avg"] is not None])
        persist_rmse_avg = np.array([r["rmse_persist_avg"] for r in results])
        clim_rmse_avg = np.array([r["rmse_clim_avg"] for r in results])

        # Average RMSE — per-pixel (full field)
        sin_rmse = np.array([r["rmse_sin"] for r in results])
        persist_rmse = np.array([r["rmse_persist"] for r in results])
        clim_rmse = np.array([r["rmse_clim"] for r in results])
        mv_rmse = np.array([r["rmse_mv"] for r in results if r["rmse_mv"] is not None])

        print(f"  PDE: ∂T/∂t = {a:.4f}·sin + {b:.4f}·T_y")
        if mv_c:
            print(f"  Multi-var terms: {mv_c}")
        print(f"  Start months: {len(results)} (n_starts=22)")

        # Per-pixel RMSE table
        print(f"\n  Per-pixel RMSE (full field, 3 decimals):")
        print(f"  {'Lead':>6s} {'PDE':>8s} {'MV':>8s} {'Persist':>8s} {'Clim':>8s}")
        print(f"  " + "-" * 42)
        for lead in leads_display:
            pde_v = sin_rmse[:, lead].mean()
            mv_v = mv_rmse[:, lead].mean() if len(mv_rmse) > 0 else 0
            pers_v = persist_rmse[:, lead].mean()
            clim_v = clim_rmse[:, lead].mean()
            print(f"  {lead:>6d} {pde_v:>8.3f} {mv_v:>8.3f} {pers_v:>8.3f} {clim_v:>8.3f}")

        # Spatial mean RMSE table
        print(f"\n  Spatial mean RMSE (same basis as ARIMA, 3 decimals):")
        print(f"  {'Lead':>6s} {'PDE':>8s} {'MV':>8s} {'Persist':>8s} {'Clim':>8s}")
        print(f"  " + "-" * 42)
        for lead in leads_display:
            pde_v = sin_rmse_avg[:, lead].mean()
            mv_v = mv_rmse_avg[:, lead].mean() if len(mv_rmse_avg) > 0 else 0
            pers_v = persist_rmse_avg[:, lead].mean()
            clim_v = clim_rmse_avg[:, lead].mean()
            print(f"  {lead:>6d} {pde_v:>8.3f} {mv_v:>8.3f} {pers_v:>8.3f} {clim_v:>8.3f}")

        # Manuscript-ready table (2 decimals)
        print(f"\n  Manuscript values (2 decimals):")
        header = f"  {region.title()}:\t"
        for lead in leads_display:
            pde_v = sin_rmse_avg[:, lead].mean()
            pers_v = persist_rmse_avg[:, lead].mean()
            clim_v = clim_rmse_avg[:, lead].mean()
            header += f"Lead {lead} — PDE={pde_v:.2f}, Persist={pers_v:.2f}, Clim={clim_v:.2f}; "
        print(header)

        all_region_results[region] = results

    print(f"\n{'=' * 60}")
    print("Forward integration validation complete.")
    print("Done!")
