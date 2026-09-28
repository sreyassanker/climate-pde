#!/usr/bin/env python3
"""
Harmonize and merge CHIRPS precipitation, ERA5 atmospheric variables, and MODIS LST
into the unified analysis dataset (data/climate_multivariate.nc) on a common 0.05° grid.
"""

import xarray as xr, numpy as np
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

FINAL = PROJECT_ROOT / "data/climate_multivariate.nc"
FINAL.parent.mkdir(exist_ok=True)

if __name__ == "__main__":
    print("Loading datasets...")
    c = xr.open_dataset(str(PROJECT_ROOT / "data/chirps/chirps_monthly_2000-2023.nc")).rename(
        {"latitude": "lat", "longitude": "lon", "precipitation": "precip"}
    )
    e = xr.open_dataset(str(PROJECT_ROOT / "data/era5/era5_monthly_2000-2023.nc")).rename(
        {"latitude": "lat", "longitude": "lon"}
    )
    m = xr.open_dataset(str(PROJECT_ROOT / "data/modis/modis_lst_monthly_2000-2023.nc"))

    # Use CHIRPS grid as reference (convert to float64 for precision matching)
    ref_lat = c.lat.astype("float64").values
    ref_lon = c.lon.astype("float64").values

    # Regrid MODIS to CHIRPS grid
    m = m.reindex(lat=m.lat[::-1])  # ascending
    m = m.interp(lat=ref_lat, lon=ref_lon, method="linear")

    c = c.assign_coords(lat=ref_lat, lon=ref_lon)[["precip"]]
    e = e.assign_coords(lat=ref_lat, lon=ref_lon)[["t2m", "tp", "q"]]
    m = m.assign_coords(lat=ref_lat, lon=ref_lon)[["lst_day_mod11c3", "lst_night_mod11c3", "lst_day_myd11c3", "lst_night_myd11c3"]]

    # Merge — align coordinates
    merged = xr.merge([c, e, m], join="inner")
    print(f"Merged: {dict(merged.dims)}")

    # Cast all to float32 for compactness
    for v in merged.data_vars:
        merged[v] = merged[v].astype("float32")

    merged.to_netcdf(FINAL, encoding={
        v: {"zlib": True, "complevel": 4} for v in merged.data_vars
    })
    for d in [c, e, m]:
        d.close()
    merged.close()

    size = FINAL.stat().st_size / 1e6
    print(f"\n[SUCCESS] {FINAL} ({size:.1f} MB)")
    print(f"  Dims: time={merged.time.shape[0]}, lat={merged.lat.shape[0]}, lon={merged.lon.shape[0]}")
    print(f"  Vars: {list(merged.data_vars)}")
