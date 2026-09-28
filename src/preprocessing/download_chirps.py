#!/usr/bin/env python3
"""
Download, spatially subset to the southeastern US domain (25°N–35°N, 95°W–65°W),
and concatenate monthly CHIRPS v2.0 precipitation data (2000–2023).
"""

import requests, xarray as xr, time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUT = PROJECT_ROOT / "data/chirps"
OUT.mkdir(parents=True, exist_ok=True)

REGION = {"north": 35.0, "south": 25.0, "east": -65.0, "west": -95.0}
BASE = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/netcdf/byYear"


def download(path, url):
    for attempt in range(3):
        try:
            print(f"  [DOWNLOAD] {url.split('/')[-1]} ({attempt+1}/3)")
            r = requests.get(url, stream=True, timeout=600)
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            # Verify file is valid
            ds = xr.open_dataset(path)
            ds.close()
            return True
        except Exception as e:
            print(f"  [RETRY] {e}")
            path.unlink(missing_ok=True)
            time.sleep(3)
    return False


def subset_year(global_path, sub_path):
    ds = xr.open_dataset(global_path)
    lat_asc = ds.latitude[0] < ds.latitude[-1]
    lat_slice = slice(REGION["south"], REGION["north"]) if lat_asc else slice(REGION["north"], REGION["south"])
    lon_asc = ds.longitude[0] < ds.longitude[-1]
    lon_slice = slice(REGION["west"], REGION["east"]) if lon_asc else slice(REGION["east"], REGION["west"])
    sub = ds.sel(latitude=lat_slice, longitude=lon_slice)
    if "precip" in sub:
        sub = sub.rename({"precip": "precipitation"})
    sub.to_netcdf(sub_path)
    ds.close()
    sub.close()


def main():
    print("=" * 60)
    print(f"CHIRPS [{REGION['west']}W to {REGION['east']}W]")
    print("=" * 60)

    raw = OUT / "raw_yearly"
    raw.mkdir(exist_ok=True)

    for year in range(2000, 2024):
        sub_path = raw / f"chirps_{year}.nc"
        if sub_path.exists():
            print(f"[SKIP] {sub_path.name}")
            continue

        global_path = raw / f"chirps_global_{year}.nc"
        if not global_path.exists():
            url = f"{BASE}/chirps-v2.0.{year}.monthly.nc"
            if not download(global_path, url):
                print(f"  [FAIL] {year}")
                continue

        print(f"  [SUBSET] {year}...", end=" ", flush=True)
        subset_year(global_path, sub_path)
        global_path.unlink(missing_ok=True)
        print("OK")
        time.sleep(0.3)

    print("\n[MERGE]")
    subs = sorted(raw.glob("chirps_2*.nc"))
    ds_list = [xr.open_dataset(f) for f in subs]
    merged = xr.concat(ds_list, dim="time")
    final = OUT / "chirps_monthly_2000-2023.nc"
    merged.to_netcdf(final)
    for d in ds_list:
        d.close()
    merged.close()

    size = final.stat().st_size / 1e6
    print(f"[SUCCESS] {final} ({size:.0f} MB)")
    print(f"  Grid: lat={merged.latitude.shape[0]}, lon={merged.longitude.shape[0]}")


if __name__ == "__main__":
    main()
