#!/usr/bin/env python3
"""
Figure 1: Geographic overview of the study domain across the southeastern
United States (25°N–35°N, 95°W–65°W) within the continental United States.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shapereader
from cartopy.geodesic import Geodesic
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]


FIGS_PATH = PROJECT_ROOT / "figures"
FIGS_PATH.mkdir(exist_ok=True)

STUDY_LAT = (25, 35)
STUDY_LON = (-95, -65)

PROJ = ccrs.PlateCarree()
DOMAIN = {"lat": (15, 55), "lon": (-130, -55)}


def add_scale_bar(ax, lon, lat, km, data_crs):
    geod = Geodesic()
    end = geod.direct([lon, lat], 90, km * 1000)[0, :2]
    xs = [lon, end[0]]
    ys = [lat, end[1]]
    ym = (ys[0] + ys[1]) / 2
    tick = 0.6
    c = '#2C3E50'
    ax.plot(xs, [ym, ym], transform=data_crs, color=c, lw=2.2, zorder=5)
    for x in xs:
        ax.plot([x, x], [ym - tick, ym + tick], transform=data_crs, color=c, lw=1.8, zorder=5)
    xm = (xs[0] + xs[1]) / 2
    ax.plot([xm, xm], [ym - tick * 0.8, ym + tick * 0.8], transform=data_crs, color=c, lw=1.4, zorder=5)
    y_lab = ym - tick * 3.5
    for x, t in [(xs[0], '0'), (xm, f'{km//2}'), (xs[1], f'{km}')]:
        ax.text(x, y_lab, t, transform=data_crs, fontsize=8,
                ha='center', va='top', color=c, zorder=5, fontweight='bold')
    ax.text(xs[1] + (xs[1] - xs[0]) * 0.3, y_lab, 'km', transform=data_crs, fontsize=8,
            ha='left', va='top', color=c, zorder=5)


def main():
    fig_w = 7.48
    fig_h = 4.5

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')

    ax = fig.add_axes([0.05, 0.05, 0.90, 0.87], projection=PROJ)
    ax.set_extent([DOMAIN['lon'][0], DOMAIN['lon'][1],
                   DOMAIN['lat'][0], DOMAIN['lat'][1]], crs=PROJ)

    ax.add_feature(cfeature.OCEAN, color='#CBE0ED', zorder=0)
    ax.add_feature(cfeature.LAND,  color='#EAE8E3', zorder=0)
    ax.add_feature(cfeature.LAKES, color='#CBE0ED', edgecolor='#A9CCE3',
                   linewidth=0.3, zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor='#5A6B7C', zorder=2)

    shp = shapereader.natural_earth(resolution='50m', category='cultural', name='admin_0_countries')
    reader = shapereader.Reader(shp)
    usa_geom = [r.geometry for r in reader.records() if r.attributes['ADMIN'] == 'United States of America'][0]
    ax.add_geometries([usa_geom], crs=PROJ, facecolor='#F2E6CE', edgecolor='none', linewidth=0, zorder=1.5)
    ax.add_geometries([usa_geom], crs=PROJ, facecolor='none', edgecolor='#5A6B7C', linewidth=1.0, zorder=2)

    shp_states = shapereader.natural_earth(resolution='50m', category='cultural', name='admin_1_states_provinces')
    reader_states = shapereader.Reader(shp_states)
    usa_states = [r for r in reader_states.records()
                  if r.attributes.get('admin') == 'United States of America']
    usa_geom_list = [r.geometry for r in usa_states]
    ax.add_geometries(usa_geom_list, crs=PROJ, facecolor='none', edgecolor='#8A9BA8', linewidth=0.35, zorder=2)

    all_labels = []
    def overlaps(tx, ty, gap=1.2):
        for px, py in all_labels:
            if abs(tx - px) < gap and abs(ty - py) < gap:
                return True
        return False

    # ── State labels (skip if overlapping another state) ──
    for r in usa_states:
        name = r.attributes.get('name')
        centroid = r.geometry.centroid
        cx, cy = centroid.x, centroid.y
        if name == 'West Virginia':
            cy -= 0.6
        if name == 'Arkansas':
            cy -= 0.5
        if name == 'Mississippi':
            cy -= 0.8
        if name == 'Arizona':
            cy += 0.5
        if name == 'Virginia':
            cy -= 0.5
        if name == 'Delaware':
            cy -= 1.0
        if name in ('Delaware',):
            place = True
        else:
            place = not overlaps(cx, cy, 1.2)
        if place:
            t = ax.text(cx, cy, name, transform=PROJ, fontsize=4.5,
                        color='#555555', ha='center', va='center', zorder=3)
            t.set_path_effects([pe.withStroke(linewidth=1.5, foreground='white')])
            all_labels.append((cx, cy))

    # ── Country labels ──
    for label, lon, lat in [
        ('Canada', -100, 52),
        ('Mexico', -102, 22),
        ('Cuba', -79, 21.5),
        ('Bahamas', -77, 24),
    ]:
        if not overlaps(lon, lat, 1.5):
            ax.text(lon, lat, label, transform=PROJ, fontsize=7, color='#666666',
                    ha='center', va='center', style='italic', zorder=3)
            all_labels.append((lon, lat))

    if not overlaps(-98, 38, 1.5):
        ax.text(-98, 38, 'United States', transform=PROJ, fontsize=7, color='#7A6B52',
                ha='center', va='center', style='italic', zorder=3)
        all_labels.append((-98, 38))

    # ── Ocean labels ──
    for label, lon, lat in [
        ('Pacific Ocean', -122, 30),
        ('Atlantic Ocean', -64, 38),
        ('Gulf of Mexico', -90, 24),
        ('Caribbean Sea', -78, 17),
    ]:
        if not overlaps(lon, lat, 2.0):
            ax.text(lon, lat, label, transform=PROJ, fontsize=8, color='#7AA6C0',
                    ha='center', va='center', style='italic', fontweight='bold', zorder=3)
            all_labels.append((lon, lat))

    # ── City labels (avoid all prior labels) ──
    for lon, lat, name in [
        (-74.01, 40.71, 'New York'), (-118.24, 34.05, 'Los Angeles'),
        (-87.63, 41.88, 'Chicago'), (-95.37, 29.76, 'Houston'),
        (-112.07, 33.45, 'Phoenix'), (-75.17, 39.95, 'Philadelphia'),
        (-98.49, 29.42, 'San Antonio'), (-117.16, 32.72, 'San Diego'),
        (-96.80, 32.78, 'Dallas'), (-84.39, 33.75, 'Atlanta'),
    ]:
        ax.plot(lon, lat, transform=PROJ, marker='o', markersize=2.5,
                color='#2C3E50', zorder=4, markeredgecolor='white', markeredgewidth=0.4)
        offsets = [(0.6, 0), (-0.6, 0), (0, 0.5), (0, -0.5),
                   (0.6, 0.4), (0.6, -0.4), (-0.6, 0.4), (-0.6, -0.4)]
        manual = {'Philadelphia': (1.0, -0.8), 'San Antonio': (0.6, -0.5)}
        if name in manual:
            best = manual[name]
        else:
            best = None
            for dx, dy in offsets:
                tx, ty = lon + dx, lat + dy
                if not overlaps(tx, ty, 1.5):
                    best = (dx, dy)
                    break
        if best is None:
            best = (0.6, 0)
        dx, dy = best
        ha = 'left' if dx >= 0 else 'right'
        ax.text(lon + dx, lat + dy, name, transform=PROJ, fontsize=6.5,
                color='#2C3E50', ha=ha, va='center', zorder=4)
        all_labels.append((lon + dx, lat + dy))

    dom_x = [STUDY_LON[0], STUDY_LON[1], STUDY_LON[1], STUDY_LON[0], STUDY_LON[0]]
    dom_y = [STUDY_LAT[0], STUDY_LAT[0], STUDY_LAT[1], STUDY_LAT[1], STUDY_LAT[0]]
    ax.plot(dom_x, dom_y, transform=PROJ, color='#C0392B',
            linewidth=1.8, linestyle='--', zorder=3)
    ax.fill(dom_x, dom_y, transform=PROJ, color='#C0392B', alpha=0.15, zorder=2)
    ax.text(-90, 27.15, 'Study area', transform=PROJ,
            fontsize=9, color='#C0392B', ha='center', va='center',
            style='italic', fontweight='bold', zorder=4)

    lon_ticks = [-120, -105, -90, -75, -60]
    lat_ticks = [15, 30, 45]

    for lon in lon_ticks:
        ax.plot([lon, lon], [15, 55], transform=PROJ,
                linewidth=0.35, color='gray', alpha=0.5, linestyle=':', zorder=1)
    for lat in lat_ticks:
        ax.plot([-130, -55], [lat, lat], transform=PROJ,
                linewidth=0.35, color='gray', alpha=0.5, linestyle=':', zorder=1)

    for lon in lon_ticks:
        ax.text(lon, 14.0, f'{abs(lon):.0f}°W', transform=PROJ,
                fontsize=8, color='#444444', ha='center', va='top',
                clip_on=False, zorder=6)
        ax.text(lon, 56, f'{abs(lon):.0f}°W', transform=PROJ,
                fontsize=8, color='#444444', ha='center', va='bottom',
                clip_on=False, zorder=6)

    for lat in lat_ticks:
        ax.text(-131, lat, f'{lat:.0f}°N', transform=PROJ,
                fontsize=8, color='#444444', ha='center', va='bottom',
                rotation=90, clip_on=False, zorder=6)
        ax.text(-54, lat, f'{lat:.0f}°N', transform=PROJ,
                fontsize=8, color='#444444', ha='center', va='bottom',
                rotation=90, clip_on=False, zorder=6)

    add_scale_bar(ax, -70, 19, 1000, PROJ)

    ax.annotate('N', xy=(-62, 49), xytext=(-62, 44.5),
                transform=PROJ, fontsize=12, fontweight='bold', color='#2C3E50',
                ha='center', va='center', zorder=5,
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.8))

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    leg_elements = [
        Line2D([0], [0], color='#C0392B', linewidth=1.8, linestyle='--', label='Study area'),
        Patch(facecolor='#F2E6CE', edgecolor='#5A6B7C', linewidth=1.0, label='United States'),
        Line2D([0], [0], color='#8A9BA8', linewidth=0.35, label='State boundary'),
        Line2D([0], [0], marker='o', color='#2C3E50', markersize=3,
               markerfacecolor='#2C3E50', markeredgecolor='white', markeredgewidth=0.4,
               linestyle='None', label='Major city'),
    ]
    ax.legend(handles=leg_elements, loc='lower left', fontsize=6,
              framealpha=0.92, edgecolor='#BBBBBB', facecolor='white',
              borderpad=0.4, handletextpad=0.4, labelspacing=0.6)

    fig.savefig(FIGS_PATH / "fig1_study_area.png", dpi=600,
                facecolor='white')
    fig.savefig(FIGS_PATH / "fig1_study_area.pdf",
                facecolor='white')
    print(f"Saved {FIGS_PATH}/fig1_study_area.png (600 dpi)")
    print(f"Saved {FIGS_PATH}/fig1_study_area.pdf")
    plt.close(fig)

    print("\n── Caption for manuscript ──")
    print("Figure 1. Study domain, southeastern United States.")
    print("Map lines and geographic boundaries are for illustration only")
    print("and do not imply any political position. (Natural Earth Data)")


if __name__ == "__main__":
    main()
