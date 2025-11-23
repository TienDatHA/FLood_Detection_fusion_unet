#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LightSource

# Add configuration import
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config import get_data_root, get_project_root, ensure_dirs

# =================== CONFIGURATION (EDIT HERE) ===================
DATA_ROOT = get_data_root()
PROJECT_ROOT = get_project_root()

# Example configuration - modify these as needed
REGION_NAME = os.getenv("REGION_NAME", "BacGiang_20240910")

# Choose 1 of 2 methods:
DEM_PATH = str(DATA_ROOT / f"Bench_Mark/JRC + DEM/{REGION_NAME}_DEM.tif")  # specific file
DEM_GLOB = None  # example: str(DATA_ROOT / "dem_examples/*.tif") for batch processing

OUTDIR = str(PROJECT_ROOT / "dem_visualizations")  # PNG output directory in project
PMIN, PMAX = 2.0, 98.0                # percentile stretch
CMAP = "terrain"                       # colormap for elevation  
TITLE = None                           # common title (None = auto from filename)

# Ensure output directory exists
ensure_dirs([Path(OUTDIR)])

print(f"📍 DEM Visualization Configuration:")
print(f"   DATA_ROOT: {DATA_ROOT}")  
print(f"   DEM file: {DEM_PATH}")
print(f"   Output dir: {OUTDIR}")

# =================== TIỆN ÍCH ===================
def _read_dem(path):
    src = rasterio.open(path)
    dem = src.read(1, masked=True)  # masked array theo NoData
    return src, dem, src.nodata, src.transform, src.crs

def _percent_clip(arr, pmin=2, pmax=98):
    vmin, vmax = np.percentile(arr.compressed(), [pmin, pmax])
    arr_clip = np.clip(arr, vmin, vmax)
    return arr_clip, float(vmin), float(vmax)

def _pixel_size_meters(transform, crs, lat_hint=None):
    px_x = abs(transform.a)
    px_y = abs(transform.e)
    if crs is not None and crs.is_projected:
        return px_x, px_y  # mét
    if lat_hint is None:
        lat_hint = 0.0
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat_hint))
    return px_x * m_per_deg_lon, px_y * m_per_deg_lat

def _approx_lat_from_transform(src):
    row_c = src.height // 2
    col_c = src.width // 2
    y, x = rasterio.transform.xy(src.transform, row_c, col_c)
    return y  # lat

def _slope_aspect(dem, px_m, py_m):
    dem_filled = np.array(dem.filled(np.nan), dtype="float64")
    gy, gx = np.gradient(dem_filled, py_m, px_m)  # dZ/dy, dZ/dx
    slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
    slope_deg = np.degrees(slope_rad)
    aspect_rad = np.arctan2(-gx, gy)
    aspect_deg = (np.degrees(aspect_rad) + 360.0) % 360.0
    slope_deg[np.isnan(dem_filled)] = np.nan
    aspect_deg[np.isnan(dem_filled)] = np.nan
    return slope_deg, aspect_deg

def _plot_save(out_png, fig):
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"✓ Đã lưu: {out_png}")

def _process_one_dem(dem_path, outdir, pmin, pmax, cmap, title):
    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(dem_path))[0]

    src, dem, nodata, transform, crs = _read_dem(dem_path)
    try:
        print(f"\n==> {dem_path}")
        print(f"- Kích thước: {src.width} x {src.height}")
        print(f"- CRS: {crs}")
        print(f"- NoData: {nodata}")
        print(f"- Độ cao (m): min={dem.min():.3f}, max={dem.max():.3f}")

        lat_hint = None
        if not (crs and crs.is_projected):
            lat_hint = _approx_lat_from_transform(src)
        px_m, py_m = _pixel_size_meters(transform, crs, lat_hint=lat_hint)
        print(f"- Pixel size ~ {px_m:.2f} m (x)  x  {py_m:.2f} m (y)")

        dem_clip, vmin, vmax = _percent_clip(dem, pmin, pmax)
        print(f"- Stretch [{pmin}–{pmax}]% -> vmin={vmin:.2f}, vmax={vmax:.2f}")

        # Hillshade
        ls = LightSource(azdeg=315, altdeg=45)
        cmap_obj = plt.get_cmap(cmap)  # Convert string to colormap object
        hs_rgb = ls.shade(np.array(dem_clip, dtype="float64"),
                          cmap=cmap_obj, vert_exag=1, blend_mode='overlay')

        # Slope & aspect
        slope_deg, aspect_deg = _slope_aspect(dem, px_m, py_m)

        # 1) Elevation + hillshade
        fig1 = plt.figure(figsize=(8, 6))
        ax1 = fig1.add_subplot(111)
        ax1.imshow(hs_rgb, interpolation="nearest")
        ax1.set_title(title or f"Elevation (hillshade) — {base}")
        ax1.axis("off")
        _plot_save(os.path.join(outdir, f"{base}_elevation_hillshade.png"), fig1)

        # 2) Elevation thuần
        fig2 = plt.figure(figsize=(8, 6))
        ax2 = fig2.add_subplot(111)
        im2 = ax2.imshow(dem_clip, cmap=cmap_obj)
        ax2.set_title(title or f"Elevation — {base}")
        ax2.axis("off")
        cbar = fig2.colorbar(im2, ax=ax2, fraction=0.035, pad=0.02)
        cbar.set_label("Elevation (m)")
        _plot_save(os.path.join(outdir, f"{base}_elevation.png"), fig2)

        # 3) Slope (độ)
        fig3 = plt.figure(figsize=(8, 6))
        ax3 = fig3.add_subplot(111)
        im3 = ax3.imshow(slope_deg, cmap="viridis")
        ax3.set_title(title or f"Slope (degrees) — {base}")
        ax3.axis("off")
        cbar3 = fig3.colorbar(im3, ax=ax3, fraction=0.035, pad=0.02)
        cbar3.set_label("Slope (°)")
        _plot_save(os.path.join(outdir, f"{base}_slope.png"), fig3)

        # 4) Aspect (độ)
        fig4 = plt.figure(figsize=(8, 6))
        ax4 = fig4.add_subplot(111)
        im4 = ax4.imshow(aspect_deg, cmap="twilight")
        ax4.set_title(title or f"Aspect (degrees) — {base}")
        ax4.axis("off")
        cbar4 = fig4.colorbar(im4, ax=ax4, fraction=0.035, pad=0.02)
        cbar4.set_label("Aspect (° from North)")
        _plot_save(os.path.join(outdir, f"{base}_aspect.png"), fig4)

        # 5) Histogram độ cao
        fig5 = plt.figure(figsize=(7, 5))
        ax5 = fig5.add_subplot(111)
        ax5.hist(dem.compressed(), bins=256)
        ax5.set_xlabel("Elevation (m)")
        ax5.set_ylabel("Pixel count")
        ax5.set_title(title or f"Elevation histogram — {base}")
        _plot_save(os.path.join(outdir, f"{base}_histogram.png"), fig5)

        print("Hoàn tất file này.")
    finally:
        src.close()

# =================== CHẠY ===================
if __name__ == "__main__":
    if DEM_GLOB:
        paths = sorted(glob.glob(DEM_GLOB))
        if not paths:
            raise FileNotFoundError(f"Không tìm thấy DEM theo glob: {DEM_GLOB}")
        print(f"Tìm thấy {len(paths)} file DEM.")
        for p in paths:
            _process_one_dem(p, OUTDIR, PMIN, PMAX, CMAP, TITLE)
    else:
        if not os.path.isfile(DEM_PATH):
            raise FileNotFoundError(f"Không tìm thấy DEM: {DEM_PATH}")
        _process_one_dem(DEM_PATH, OUTDIR, PMIN, PMAX, CMAP, TITLE)
