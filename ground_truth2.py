import xarray as xr
import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf

station_coords = {
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


path = "ECMWF_u10.grib"
ds = xr.open_dataset(path, engine="cfgrib")

varname = list(ds.data_vars)[0]
Z = ds[varname]

for dim in ["time", "step", "valid_time", "level", "isobaricInhPa"]:
    if dim in Z.dims:
        Z = Z.isel({dim: 0})

lons = Z["longitude"]
if float(lons.max()) > 180:
    lons2 = ((lons + 180) % 360) - 180
    Z = Z.assign_coords(longitude=lons2).sortby("longitude")


def interp_field_at_stations(Z, station_coords, stations=None):
    if stations is None:
        stations = list(station_coords.keys())
    vals = []
    for s in stations:
        lat, lon = station_coords[s]
        v = Z.interp(latitude=lat, longitude=lon, method="linear").values
        vals.append(float(v))
    return stations, np.array(vals, dtype=float)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def build_ground_truth_downhill_simple(station_coords, stations, values, k_neigh=5, max_km=800.0, min_drop=0.0):
    N = len(stations)
    lat = np.array([station_coords[s][0] for s in stations], dtype=float)
    lon = np.array([station_coords[s][1] for s in stations], dtype=float)

    # distance matrix
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            D[i, j] = 0.0 if i == j else haversine_km(lat[i], lon[i], lat[j], lon[j])

    W = np.zeros((N, N), dtype=float)

    for i in range(N):
        nn = np.argsort(D[i])
        nn = nn[nn != i][:k_neigh]  # k closest

        for j in nn:
            if D[i, j] > max_km:
                continue

            drop = values[i] - values[j]
            if drop <= min_drop:
                continue  # enforce downhill

            W[i, j] = drop 

    return W

# Build graph
stations = list(station_coords.keys())
stations, vals = interp_field_at_stations(Z, station_coords, stations=stations)

# Remove stations where interpolation returns NaN
valid = ~np.isnan(vals)
stations = [s for s, ok in zip(stations, valid) if ok]
vals = vals[valid]
station_coords_use = {s: station_coords[s] for s in stations}

W_gt = build_ground_truth_downhill_simple(
    station_coords_use,
    stations,
    vals,
    k_neigh=5,
    max_km=800.0,   
    min_drop=0.0    
)

print("Nonzero edges:", np.count_nonzero(W_gt))
G_gt = nx.from_numpy_array(W_gt, create_using=nx.DiGraph)
print("Graph edges:", G_gt.number_of_edges())
print("Using GRIB variable:", varname)


def plot_graph_on_map(G, stations, station_coords, values=None, title="Ground Truth"):
    pos = {i: (station_coords[stations[i]][1], station_coords[stations[i]][0])  # (lon, lat)
           for i in range(len(stations))}

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cf.BORDERS, linewidth=0.8)
    ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.5)
    ax.set_title(title)

    # nodes
    for i, (lon, lat) in pos.items():
        ax.plot(lon, lat, "ro", markersize=5, transform=ccrs.PlateCarree())
        lab = f"{stations[i]}"
        if values is not None:
            lab += f" ({values[i]:.2f})"
        ax.text(lon + 0.15, lat + 0.15, lab, fontsize=8, transform=ccrs.PlateCarree())

    # arrows
    for u, v in G.edges():
        lon1, lat1 = pos[u]
        lon2, lat2 = pos[v]
        ax.arrow(
            lon1, lat1, lon2 - lon1, lat2 - lat1,
            head_width=0.12, length_includes_head=True,
            transform=ccrs.PlateCarree(), alpha=0.8
        )

    plt.show()

plot_graph_on_map(
    G_gt, stations, station_coords_use,
    values=vals,
    title=f"Downhill GRIB Ground Truth ({varname})"
)


# edge list with station ids + weights
for u, v in G_gt.edges():
    w = W_gt[u, v]
    print(f"{stations[u]} -> {stations[v]}   w={w:.6f}")