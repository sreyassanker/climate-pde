"""
ARIMA baseline comparison against PDE forward predictions.

Pure fit-once strategy: seasonal ARIMA fitted on training months 1–240 (2000–2019)
using auto_arima, forecasting 48 steps ahead over months 241–288 (2020–2023).
Evaluated across all 36 test start months (months 240–275) with lead times up to 12 months.
"""

import numpy as np
import xarray as xr
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA as sm_ARIMA
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = PROJECT_ROOT / "data/climate_multivariate.nc"
FIGS_PATH = PROJECT_ROOT / "figures"
FIGS_PATH.mkdir(exist_ok=True)
DX = 0.05

REGIONS = {
    "florida":   {"lat": (25.0, 30.0), "lon": (-83.0, -80.0)},
    "ga_al":     {"lat": (30.0, 34.0), "lon": (-88.0, -82.0)},
    "carolinas": {"lat": (33.0, 35.0), "lon": (-82.0, -75.0)},
}
PATCH_SIZE = 32
N_STARTS = 22
N_STEPS = 12


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


def compute_climatology(T_avg):
    """Monthly climatology from spatial mean time series."""
    nt = len(T_avg)
    month_of_year = np.arange(nt) % 12
    clim = np.array([T_avg[month_of_year == m].mean() for m in range(12)])
    return clim


if __name__ == "__main__":
    print("=" * 60)
    print("ARIMA vs PDE Baseline Comparison (pure fit-once)")
    print("=" * 60)

    leads_display = [1, 3, 6, 12]
    region_keys = ["florida", "ga_al", "carolinas"]
    region_labels = ["Florida", "GA/AL", "Carolinas"]

    for region, rlabel in zip(region_keys, region_labels):
        print(f"\n--- {region} ---")
        ds = xr.open_dataset(DATA_PATH)
        patch = extract(ds, region)
        T_raw = patch["t2m"].values.copy()
        T_raw = np.where(np.isnan(T_raw), np.nanmean(T_raw), T_raw)
        ds.close()
        T_mu, T_sigma = T_raw.mean(), T_raw.std()
        T = (T_raw - T_mu) / T_sigma
        T_avg = T.mean(axis=(1, 2))  # spatial mean
        nt = len(T_avg)

        # ── 1. Find best ARIMA order on training period ──
        print("  Finding best ARIMA order (months 1–240)...")
        train_ts = T_avg[:240]
        result = auto_arima(train_ts, seasonal=True, m=12,
                            start_p=0, max_p=4, start_q=0, max_q=4,
                            start_P=0, max_P=2, start_Q=0, max_Q=2,
                            trace=False, error_action='ignore',
                            suppress_warnings=True, stepwise=True,
                            max_order=6, n_fits=10)
        order = result.order
        seasonal_order = result.seasonal_order
        print(f"  ARIMA{order} x {seasonal_order}")

        # ── 2. Fit ARIMA once on months 1–240, forecast 48 steps ahead ──
        model = sm_ARIMA(train_ts, order=order, seasonal_order=seasonal_order)
        fitted = model.fit(low_memory=True)
        forecast = np.asarray(fitted.forecast(steps=nt - 240)).ravel()

        # ── 3. Discover PDE on this patch ──
        ny, nx = T.shape[1], T.shape[2]
        sin_all = np.tile(np.sin(2*np.pi*np.arange(nt)/12)[:, None, None], (1, ny, nx))
        T_y_all = np.zeros_like(T)
        for t_idx in range(nt):
            T_y_all[t_idx] = compute_T_y(T[t_idx])
        dT = np.zeros_like(T)
        dT[1:-1] = (T[2:] - T[:-2]) / 2.0
        X = np.column_stack([sin_all.ravel(), T_y_all.ravel()])
        y = dT.ravel()
        valid = ~np.isnan(y)
        c = np.linalg.lstsq(X[valid], y[valid], rcond=None)[0]
        a, b = float(c[0]), float(c[1])
        print(f"  PDE: ∂T/∂t = {a:.4f}·sin + {b:.4f}·T_y")

        # ── 4. Evaluate from all start months >= 240 ──
        # Pure fit-once ARIMA: forecast from month 240 covers 240..287.
        # Use all possible test starts from 240 to nt-n_steps-1.
        start_indices = list(range(240, nt - N_STEPS))
        n_viable = len(start_indices)
        print(f"  Start months >= 240: {n_viable} ({start_indices[0]}–{start_indices[-1]})")
        print(f"  PDE forward-integrated from each start for consistent comparison")

        # Climatology
        clim = compute_climatology(T_avg)

        pde_err2_all = {l: [] for l in leads_display}
        arima_err2_all = {l: [] for l in leads_display}
        persist_err2_all = {l: [] for l in leads_display}
        climatology_err2_all = {l: [] for l in leads_display}

        for s in start_indices:
            # PDE: forward integrate full field, then average
            T_pred = T[s].copy()
            for step in range(1, N_STEPS + 1):
                t_idx = s + step - 1
                rhs = a * np.sin(2*np.pi*t_idx/12) + b * compute_T_y(T_pred)
                T_pred = T_pred + rhs
                for lead in leads_display:
                    if step == lead:
                        actual = T_avg[s + lead]
                        pred = T_pred.mean()
                        pde_err2_all[lead].append((pred - actual)**2)

            # Persistence
            persist_val = T_avg[s]
            # ARIMA (fit-once forecast)
            arima_start_offset = s - 240
            # Climatology
            for lead in leads_display:
                t_pred = s + lead
                if t_pred >= nt:
                    continue
                actual = T_avg[t_pred]

                # Persistence
                persist_err2_all[lead].append((persist_val - actual)**2)

                # ARIMA
                fc_idx = arima_start_offset + lead - 1
                if fc_idx < len(forecast):
                    arima_err2_all[lead].append((forecast[fc_idx] - actual)**2)

                # Climatology
                clim_pred = clim[t_pred % 12]
                climatology_err2_all[lead].append((clim_pred - actual)**2)

        # ── 5. Compute and display RMSE ──
        print(f"\n  {'Lead':>6s} {'PDE':>8s} {'ARIMA':>8s} {'Persist':>8s} {'Clim':>8s}")
        print("  " + "-" * 42)
        for lead in leads_display:
            pde_r = np.sqrt(np.mean(pde_err2_all[lead])) if pde_err2_all[lead] else np.nan
            arima_r = np.sqrt(np.mean(arima_err2_all[lead])) if arima_err2_all[lead] else np.nan
            persist_r = np.sqrt(np.mean(persist_err2_all[lead])) if persist_err2_all[lead] else np.nan
            clim_r = np.sqrt(np.mean(climatology_err2_all[lead])) if climatology_err2_all[lead] else np.nan
            print(f"  {lead:>6d} {pde_r:>8.3f} {arima_r:>8.3f} {persist_r:>8.3f} {clim_r:>8.3f}")

        # Manuscript-ready line
        print(f"\n  Manuscript-ready: {rlabel}:")
        line = f"  {rlabel}: "
        for lead in leads_display:
            pde_r = np.sqrt(np.mean(pde_err2_all[lead])) if pde_err2_all[lead] else np.nan
            arima_r = np.sqrt(np.mean(arima_err2_all[lead])) if arima_err2_all[lead] else np.nan
            persist_r = np.sqrt(np.mean(persist_err2_all[lead])) if persist_err2_all[lead] else np.nan
            clim_r = np.sqrt(np.mean(climatology_err2_all[lead])) if climatology_err2_all[lead] else np.nan
            line += f"Lead {lead} — PDE={pde_r:.2f}, ARIMA={arima_r:.2f}, Persist={persist_r:.2f}, Clim={clim_r:.2f}; "
        print(line)

    print(f"\nDone!")
