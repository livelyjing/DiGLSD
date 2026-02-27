"""
Wind-vector ground truth for ETEX stations.

Reads u10 (east-west) and v10 (north-south) wind components from data.grib,
computes the resultant wind vector at each station via vector addition,
and creates directed edges from each station toward other stations that lie
in the direction the wind blows (within a configurable angular cone).

The graph is NOT necessarily connected.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import networkx as nx
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cfgrib

# ─── Station coordinates (DMS-encoded as in etex.py) ─────────────────────
station_coords_dms = {
    "D04": (48.26, 10.56),
    "D13": (50.07, 8.44),
    "D34": (50.20, 6.57),
    "D36": (49.13, 9.31),
    "D44": (49.45, 6.40),
    "D45": (50.30, 9.57),
    "F02": (48.27, 0.06),
    "F03": (47.48, 3.33),
    "F13": (48.10, 6.26),
    "F15": (49.05, 6.08),
    "F19": (48.44, 2.24),
    "F21": (48.04, -1.44),
    "F27": (48.46, 2.01),
}

def dms_to_decimal(val):
    """Convert DMS-encoded value (e.g. 48.26 = 48°26') to decimal degrees."""
    deg = math.trunc(val)
    minutes = 100 * (val - deg)
    sign = 1 if val >= 0 else -1
    return sign * (abs(deg) + abs(minutes) / 60.0)

station_coords = {
    s: (dms_to_decimal(lat), dms_to_decimal(lon))
    for s, (lat, lon) in station_coords_dms.items()
}


def build_wind_vector_ground_truth(station_coords, stations, u_vals, v_vals,
                                    angle_thresh_deg=60.0):
    """
    For each station i, compute the wind vector (u10, v10).
    Create edge i -> j if station j lies within angle_thresh_deg of the wind direction.
    Weight = wind_speed * cos(angle_diff).
    """
    N = len(stations)
    lats = np.array([station_coords[s][0] for s in stations])
    lons = np.array([station_coords[s][1] for s in stations])
    W = np.zeros((N, N), dtype=float)

    for i in range(N):
        wind_u, wind_v = u_vals[i], v_vals[i]
        speed = math.sqrt(wind_u**2 + wind_v**2)
        if speed < 1e-6:
            continue
        wind_angle = math.atan2(wind_v, wind_u)

        for j in range(N):
            if i == j:
                continue
            dlat = lats[j] - lats[i]
            dlon = lons[j] - lons[i]
            dir_angle = math.atan2(dlat, dlon)
            angle_diff = abs(wind_angle - dir_angle)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            if angle_diff <= math.radians(angle_thresh_deg):
                W[i, j] = speed * math.cos(angle_diff)
    return W


def main():
    GRIB_PATH = "/Users/teddy/Downloads/data.grib"
    TIME_INDEX = 0  # 1994-10-23 00:00 UTC

    stations = list(station_coords.keys())
    N = len(stations)

    # ─── Load u10, v10 from GRIB ──────────────────────────────────────
    print(f"Loading GRIB: {GRIB_PATH}")
    datasets = cfgrib.open_datasets(GRIB_PATH)
    ds = datasets[1]  # Dataset with u10, v10
    print(f"Time: {ds.time.values[TIME_INDEX]}")

    u10_field = ds['u10'].isel(time=TIME_INDEX)
    v10_field = ds['v10'].isel(time=TIME_INDEX)

    # Fix longitude: 0-360 → -180 to 180
    lons = u10_field['longitude']
    if float(lons.max()) > 180:
        lons2 = ((lons + 180) % 360) - 180
        u10_field = u10_field.assign_coords(longitude=lons2).sortby('longitude')
        v10_field = v10_field.assign_coords(longitude=lons2).sortby('longitude')

    # ─── Interpolate at stations ──────────────────────────────────────
    u_vals = np.zeros(N)
    v_vals = np.zeros(N)
    for i, s in enumerate(stations):
        lat, lon = station_coords[s]
        u_vals[i] = float(u10_field.interp(latitude=lat, longitude=lon, method='linear').values)
        v_vals[i] = float(v10_field.interp(latitude=lat, longitude=lon, method='linear').values)

    valid = ~(np.isnan(u_vals) | np.isnan(v_vals))
    stations = [s for s, ok in zip(stations, valid) if ok]
    u_vals, v_vals = u_vals[valid], v_vals[valid]
    N = len(stations)

    # ─── Print wind vectors ───────────────────────────────────────────
    print(f"\n{'Station':>6}  {'u10(E)':>8}  {'v10(N)':>8}  {'Speed':>8}  {'Dir':>8}")
    for i, s in enumerate(stations):
        spd = math.sqrt(u_vals[i]**2 + v_vals[i]**2)
        ang = math.degrees(math.atan2(v_vals[i], u_vals[i]))
        print(f"{s:>6}  {u_vals[i]:>8.3f}  {v_vals[i]:>8.3f}  {spd:>8.3f}  {ang:>7.1f}°")

    # ─── Build ground truth ───────────────────────────────────────────
    W_gt = build_wind_vector_ground_truth(
        station_coords, stations, u_vals, v_vals,
        angle_thresh_deg=60.0
    )
    G_gt = nx.from_numpy_array(W_gt, create_using=nx.DiGraph)

    print(f"\nGround Truth: {G_gt.number_of_nodes()} nodes, {G_gt.number_of_edges()} edges")
    print(f"\nEdge list:")
    for u, v in G_gt.edges():
        print(f"  {stations[u]} -> {stations[v]}   w={W_gt[u,v]:.4f}")

    # ─── Save ─────────────────────────────────────────────────────────
    np.savetxt("wind_gt_adjacency.csv", W_gt, delimiter=",", fmt="%.6f")
    print(f"\nSaved adjacency → wind_gt_adjacency.csv")

    # ─── Plot (simple matplotlib, no cartopy coastline downloads) ─────
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(f"Wind Vector Ground Truth (u10+v10, {ds.time.values[TIME_INDEX]})")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Draw nodes
    for i, s in enumerate(stations):
        lat, lon = station_coords[s]
        ax.plot(lon, lat, 'ro', markersize=7, zorder=5)
        ax.text(lon + 0.12, lat + 0.12, s, fontsize=9, fontweight='bold', zorder=6)

    # Draw wind vectors at each station (blue, scaled)
    scale = 0.15
    for i, s in enumerate(stations):
        lat, lon = station_coords[s]
        ax.annotate('', xy=(lon + u_vals[i]*scale, lat + v_vals[i]*scale),
                     xytext=(lon, lat),
                     arrowprops=dict(arrowstyle='->', color='blue', lw=2, alpha=0.6))

    # Draw graph edges (orange)
    for u, v in G_gt.edges():
        lat1, lon1 = station_coords[stations[u]]
        lat2, lon2 = station_coords[stations[v]]
        ax.annotate('', xy=(lon2, lat2), xytext=(lon1, lat1),
                     arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.2, alpha=0.5))

    plt.tight_layout()
    out_path = "wind_ground_truth.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot → {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
