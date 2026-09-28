#!/usr/bin/env python3
"""
Download and process monthly MODIS Land Surface Temperature products
(MOD11C3 / MYD11C3 v061, 2000–2023) from NASA Earthdata into a subsetted NetCDF.

Requires EARTHDATA_USERNAME and EARTHDATA_PASSWORD in .env or environment variables.
"""

import os, time, gc
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
from datetime import datetime

import numpy as np
import xarray as xr
import requests

# Load .env if present (NASA section)
_env = Path(__file__).resolve().parents[2] / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip().lower()
        if key == "username":
            os.environ.setdefault("EARTHDATA_USERNAME", val.strip())
        elif key == "password":
            os.environ.setdefault("EARTHDATA_PASSWORD", val.strip())

OUTPUT_DIR = PROJECT_ROOT / "data/modis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_DIR = PROJECT_ROOT / "data/modis_monthly"
RAW_DIR.mkdir(exist_ok=True)

PRODUCTS = ["MOD11C3", "MYD11C3"]
VERSION = "061"
START_YEAR, END_YEAR = 2000, 2023

REGION = {"lat_slice": slice(35.025, 24.975), "lon_slice": slice(-95.025, -64.975)}

CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"


def get_token():
    u = os.environ.get("EARTHDATA_USERNAME")
    p = os.environ.get("EARTHDATA_PASSWORD")
    if not u or not p:
        print("[ERROR] Set EARTHDATA_USERNAME / EARTHDATA_PASSWORD")
        return None
    try:
        r = requests.post(
            "https://urs.earthdata.nasa.gov/api/users/find_or_create_token",
            auth=(u, p), timeout=30,
        )
        r.raise_for_status()
        return r.json().get("access_token")
    except Exception as e:
        print(f"[ERROR] Token: {e}")
        return None


def search_granule(product, year, month, token):
    start = f"{year}-{month:02d}-01"
    end = f"{year}-{month+1:02d}-01" if month < 12 else f"{year+1}-01-01"
    params = {
        "short_name": product, "version": VERSION,
        "temporal": f"{start},{end}", "page_size": 10,
    }
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    try:
        r = requests.get(CMR_URL, params=params, headers=headers, timeout=60)
        r.raise_for_status()
        for entry in r.json().get("feed", {}).get("entry", []):
            title = entry.get("title", "")
            ts = entry.get("time_start", "")
            for link in entry.get("links", []):
                href = link.get("href", "")
                if ".hdf" in href and "data#" in link.get("rel", ""):
                    return {"title": title, "url": href, "time_start": ts}
    except Exception as e:
        print(f"  [ERROR] Search: {e}")
    return None


def download(url, path, token):
    headers = {"Authorization": f"Bearer {token}"}
    for attempt in range(3):
        try:
            print(f"    [DOWNLOAD] {path.name} ({attempt+1}/3)")
            r = requests.get(url, stream=True, timeout=300, headers=headers)
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            return True
        except Exception as e:
            print(f"    [RETRY] {e}")
            time.sleep(2**attempt)
    return False


def process_timestep(hdf_path, output_dir, product, time_start):
    """Open HDF, subset to SE US, save as single-timestep NetCDF."""
    try:
        ds = xr.open_dataset(hdf_path, engine="netcdf4")
    except Exception as e:
        print(f"    [ERROR] xr.open: {e}")
        return None

    dims = list(ds.dims)
    lat = 89.975 - np.arange(3600) * 0.05
    lon = -179.975 + np.arange(7200) * 0.05
    ds = ds.assign_coords({dims[0]: lat, dims[1]: lon}).rename({dims[0]: "lat", dims[1]: "lon"})

    ds = ds[["LST_Day_CMG", "LST_Night_CMG"]]
    ds = ds.sel(lat=REGION["lat_slice"], lon=REGION["lon_slice"])
    ds = ds.astype("float32")

    dt = datetime.strptime(time_start[:7], "%Y-%m")
    ds = ds.expand_dims(time=[dt])

    product_short = product.lower()
    ds = ds.rename({
        "LST_Day_CMG": f"lst_day_{product_short}",
        "LST_Night_CMG": f"lst_night_{product_short}",
    })

    out_path = output_dir / f"modis_{product_short}_{dt.year}{dt.month:02d}.nc"
    ds.to_netcdf(out_path)
    print(f"    [SAVED] {out_path.name} ({ds.lat.shape[0]}×{ds.lon.shape[0]} pix)")

    ds.close()
    gc.collect()
    return dt


def main():
    print("=" * 60)
    print("MODIS MONTHLY LST (MOD11C3/MYD11C3) — download + process")
    print("=" * 60)

    token = get_token()
    if not token:
        return

    final_dir = OUTPUT_DIR / "monthly_steps"
    final_dir.mkdir(exist_ok=True)

    total = 0
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            for prod in PRODUCTS:
                print(f"[{prod} {year}-{month:02d}]", end=" ")
                g = search_granule(prod, year, month, token)
                if not g:
                    print("[NONE]")
                    continue
                dt = datetime.strptime(g["time_start"][:7], "%Y-%m")
                fname = f"modis_{prod.lower()}_{dt.year}{dt.month:02d}.nc"
                if (final_dir / fname).exists():
                    print(f"[SKIP] {fname}")
                    continue

                hdf_name = g["title"].replace(".hdf", "") + ".hdf"
                hdf_path = RAW_DIR / hdf_name
                if not hdf_path.exists() and not download(g["url"], hdf_path, token):
                    print("    [FAIL] Download")
                    continue

                dt = process_timestep(hdf_path, final_dir, prod, g["time_start"])
                if dt:
                    total += 1
                hdf_path.unlink(missing_ok=True)
                time.sleep(0.3)

    print(f"\n[DONE] {total} timesteps saved to {final_dir}")


if __name__ == "__main__":
    main()
