#!/usr/bin/env python3
"""
Download ERA5 monthly reanalysis means (t2m, tp, d2m) via Copernicus CDS API (2000–2023),
derive specific humidity (q), and regrid to the common 0.05° grid.
"""

import os, cdsapi, xarray as xr, numpy as np
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

_out = PROJECT_ROOT / ".env"
if _out.exists():
    for line in _out.read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip().lower()
        if key == "cds_url":
            os.environ.setdefault("CDS_URL", val.strip())
        elif key == "cds_key":
            os.environ.setdefault("CDS_KEY", val.strip())

OUT = PROJECT_ROOT / "data/era5"
OUT.mkdir(parents=True, exist_ok=True)

VARS = {"2m_temperature": "t2m", "total_precipitation": "tp", "2m_dewpoint_temperature": "d2m"}
YEARS = [str(y) for y in range(2000, 2024)]
MONTHS = [f"{m:02d}" for m in range(1, 13)]
LAT_005 = np.arange(25.025, 35.0, 0.05)
LON_005 = np.arange(-94.975, -64.975, 0.05)


def dl(api_name, short):
    p = OUT / f"era5_{short}_raw.nc"
    if p.exists():
        return
    c = cdsapi.Client(url=os.environ.get("CDS_URL"), key=os.environ.get("CDS_KEY"))
    c.retrieve("reanalysis-era5-single-levels-monthly-means", {
        "product_type": "monthly_averaged_reanalysis",
        "variable": api_name,
        "year": YEARS, "month": MONTHS, "time": "00:00",
        "area": [35, -95, 25, -65],
        "format": "netcdf", "grid": [0.25, 0.25],
    }, str(p))
    print(f"  {p.name}")


def main():
    print("=" * 60)
    print("ERA5 DOWNLOAD — Climate-PDE")
    print("=" * 60)

    for api_name, short in VARS.items():
        print(f"[ERA5] {api_name} → {short}")
        dl(api_name, short)

    print("\n[PROCESS] Merge + regrid to 0.05°...")
    dss = []
    for short in VARS.values():
        ds = xr.open_dataset(OUT / f"era5_{short}_raw.nc")
        ds = ds.drop_vars(["number", "expver"], errors="ignore")
        # Normalize time to month start
        ds["valid_time"] = ds.valid_time.values.astype("datetime64[M]").astype("datetime64[D]")
        dss.append(ds)

    merged = xr.merge(dss, join="exact").rename({"valid_time": "time"})
    print(f"  Merged: time={merged.time.shape[0]}")

    merged_005 = merged.interp(latitude=LAT_005, longitude=LON_005, method="linear")

    # Specific humidity
    T = merged_005["d2m"] - 273.15
    es = 6.112 * np.exp((17.67 * T) / (T + 243.5))
    q = (0.622 * es) / (1013.25 - 0.378 * es)
    merged_005["q"] = q.clip(0, 0.05)

    final = OUT / "era5_monthly_2000-2023.nc"
    merged_005.to_netcdf(final, encoding={
        v: {"zlib": True, "complevel": 4} for v in merged_005.data_vars
    })
    print(f"  {final} ({final.stat().st_size / 1e6:.1f} MB)")
    for ds in dss:
        ds.close()
    merged_005.close()


if __name__ == "__main__":
    main()
