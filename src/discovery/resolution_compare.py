"""
Resolution sensitivity analysis: compares PDE structure and fit between
0.05° fine grid and 0.25° coarse grid (5×5 block-averaged) across Florida,
GA/AL, and Carolinas subregions (reproducing Table 4 of the manuscript).
"""
import numpy as np
import xarray as xr
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]


DATA_PATH = PROJECT_ROOT / "data/climate_multivariate.nc"
DX_FINE = 0.05
DX_COARSE = 0.25
PATCH_SIZE_FINE = 32

REGIONS = {
    "florida":   {"lat": (25.0, 30.0), "lon": (-83.0, -80.0)},
    "ga_al":     {"lat": (30.0, 34.0), "lon": (-88.0, -82.0)},
    "carolinas": {"lat": (33.0, 35.0), "lon": (-82.0, -75.0)},
}


def extract_patch(ds, key, dx):
    r = REGIONS[key]
    lat_s = ds.lat.sel(lat=slice(r["lat"][0], r["lat"][1]))
    lon_s = ds.lon.sel(lon=slice(r["lon"][0], r["lon"][1]))
    # Same physical size: 32 * 0.05 = 1.6 degrees
    phys_size_deg = PATCH_SIZE_FINE * DX_FINE  # 1.6 deg
    patch_size = max(4, int(phys_size_deg / dx))
    ci, cj = len(lat_s)//2, len(lon_s)//2
    h = patch_size//2
    if ci - h < 0 or cj - h < 0 or ci + h > len(lat_s) or cj + h > len(lon_s):
        # Region too small, use the entire available extent
        return ds.sel(lat=slice(r["lat"][0], r["lat"][1]),
                      lon=slice(r["lon"][0], r["lon"][1]))
    return ds.sel(lat=slice(lat_s.values[ci-h], lat_s.values[ci+h-1]),
                  lon=slice(lon_s.values[cj-h], lon_s.values[cj+h-1]))


def get_data(patch):
    arrs = {}
    for name in ["t2m", "q"]:
        a = patch[name].values.copy()
        a = np.where(np.isnan(a), np.nanmean(a), a)
        mu, s = a.mean(), max(a.std(), 1e-10)
        arrs[name] = (a - mu) / s
    return arrs["t2m"], arrs["q"]


def build_terms(T, q):
    nt = T.shape[0]
    ny, nx = T.shape[1], T.shape[2]
    sin = np.tile(np.sin(2*np.pi*np.arange(nt)/12)[:, None, None], (1, ny, nx))
    cos = np.tile(np.cos(2*np.pi*np.arange(nt)/12)[:, None, None], (1, ny, nx))

    dx = DX_FINE if nx > 16 else DX_COARSE
    T_y = np.zeros_like(T)
    for t in range(nt):
        d = np.zeros((ny, nx))
        d[1:-1, :] = (T[t, 2:, :] - T[t, :-2, :]) / (2*dx)
        T_y[t] = d

    T_xx = np.zeros_like(T)
    for t in range(nt):
        d = np.zeros((ny, nx))
        d[:, 1:-1] = (T[t, :, 2:] - 2*T[t, :, 1:-1] + T[t, :, :-2]) / dx**2
        T_xx[t] = d

    q_y = np.zeros_like(q)
    for t in range(nt):
        d = np.zeros((ny, nx))
        d[1:-1, :] = (q[t, 2:, :] - q[t, :-2, :]) / (2*dx)
        q_y[t] = d

    names = ["1", "sin", "cos", "T", "q", "T_y", "T_x", "T_xx", "T_yy",
             "q_y", "q_x", "T^2", "q^2", "T*q", "T*T_y", "T*q_y", "q*T_y",
             "T*T_xx"]
    terms = {
        "1": np.ones_like(T), "sin": sin, "cos": cos,
        "T": T, "q": q, "T_y": T_y, "T_xx": T_xx, "q_y": q_y,
        "T^2": T**2, "q^2": q**2, "T*q": T*q,
        "T*T_y": T*T_y, "T*q_y": T*q_y, "q*T_y": q*T_y, "T*T_xx": T*T_xx,
    }
    # Add T_x
    T_x = np.zeros_like(T)
    for t in range(nt):
        d = np.zeros((ny, nx))
        d[:, 1:-1] = (T[t, :, 2:] - T[t, :, :-2]) / (2*dx)
        T_x[t] = d
    terms["T_x"] = T_x

    q_x = np.zeros_like(q)
    for t in range(nt):
        d = np.zeros((ny, nx))
        d[:, 1:-1] = (q[t, :, 2:] - q[t, :, :-2]) / (2*dx)
        q_x[t] = d
    terms["q_x"] = q_x

    T_yy = np.zeros_like(T)
    for t in range(nt):
        d = np.zeros((ny, nx))
        d[1:-1, :] = (T[t, 2:, :] - 2*T[t, 1:-1, :] + T[t, :-2, :]) / dx**2
        T_yy[t] = d
    terms["T_yy"] = T_yy

    return terms, names


def discover(T, q):
    terms, names = build_terms(T, q)
    X = np.column_stack([terms[n].ravel() for n in names])
    dT = np.zeros_like(T)
    dT[1:-1] = (T[2:] - T[:-2]) / 2.0
    y = dT.ravel()
    valid = ~np.isnan(y)

    Xv, yv = X[valid], y[valid]
    coef_full = np.linalg.lstsq(Xv, yv, rcond=None)[0]
    for thresh in [0.02, 0.05]:
        s = np.abs(coef_full) > thresh
        if s.sum() >= 1:
            c = np.zeros(len(coef_full))
            c[s] = np.linalg.lstsq(Xv[:, s], yv, rcond=None)[0]
            coef_full = c
    s = np.abs(coef_full) > 1e-10
    if s.sum() > 0:
        coef_full[s] = np.linalg.lstsq(Xv[:, s], yv, rcond=None)[0]

    yp = Xv @ coef_full
    ss_r = np.sum((yv - yp)**2)
    ss_t = np.sum((yv - yv.mean())**2)
    r2 = 1 - ss_r / ss_t if ss_t > 0 else 0.0

    return dict(zip(names, coef_full)), float(r2)


def regrid_coarse(ds):
    """Regrid from 0.05° to 0.25° by averaging 5x5 blocks."""
    lat = ds.lat.values[::5]
    lon = ds.lon.values[::5]
    nt = len(ds.time)

    out = {}
    for var in ["t2m", "q"]:
        a = ds[var].values
        nt_, ny, nx = a.shape
        # Trim to multiples of 5
        ny_t = ny - ny % 5
        nx_t = nx - nx % 5
        a = a[:, :ny_t, :nx_t]
        # Average in 5x5 blocks
        a_rs = a.reshape(nt_, ny_t//5, 5, nx_t//5, 5).mean(axis=(2, 4))
        out[var] = xr.DataArray(a_rs, dims=["time", "lat_out", "lon_out"],
                                coords={"time": ds.time.values,
                                        "lat_out": lat[:len(lat)//5*5:1],
                                        "lon_out": lon[:len(lon)//5*5:1]})
    return xr.Dataset({k: v.rename({"lat_out": "lat", "lon_out": "lon"})
                       for k, v in out.items()})


if __name__ == "__main__":
    print("=" * 60)
    print("Resolution comparison: 0.05° vs 0.25°")
    print("=" * 60)

    ds_orig = xr.open_dataset(DATA_PATH)
    ds_coarse = regrid_coarse(ds_orig)

    for region in ["florida", "ga_al", "carolinas"]:
        print(f"\n--- {region} ---")
        for label, dset, dx in [("0.05°", ds_orig, DX_FINE), ("0.25°", ds_coarse, DX_COARSE)]:
            patch = extract_patch(dset, region, dx)
            T, q = get_data(patch)
            coeffs, r2 = discover(T, q)

            # Print key terms
            coupling = coeffs.get("T*q", 0)
            advection = coeffs.get("T_y", 0)
            seasonal = coeffs.get("sin", 0)
            n_terms = sum(1 for v in coeffs.values() if abs(v) > 1e-10)

            tq_ok = coupling if abs(coupling) > 1e-10 else 0
            print(f"  {label}: R²={r2:.3f}, terms={n_terms}, "
                  f"sin={seasonal:.4f}, T_y={advection:.4f}, T·q={tq_ok:.4f}")

    ds_orig.close()
    ds_coarse.close()
    print("\nDone!")
