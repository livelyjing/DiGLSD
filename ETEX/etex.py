import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cf
import math

import sys
sys.path.append("..")
import optimization as op
import Eval_Metrics as ev

#NOTE col 12 corresponds to 0-3 25th
# col 1 corresponds to 15-18 23rd as per the readme

#Extract the station ids and concentration values from specific time slice (col 7 refers to time slice 6)
data = np.loadtxt('etex data\\pmch.dat', dtype='S4, float,float', usecols=(1,7, 8), skiprows=2)
#Specific list of stations to consider
targets =  np.loadtxt('etex data\\locations_to_explore2.txt', dtype='S4')
X = []
for row in data:
    if row[0] in targets:
        res = list(row)[1:]
        p = []
        for num in res:
            p.append(max(0,num))
        X.append(np.array(p))
#print(X)
#Learn graph
#12 nodes, largest signal is 3, with signals appearing to be spaced by 0.5
A = op.optimize_multisignal(X, 0.4, 0.05, 10,0.5)
G_2 = nx.from_numpy_array(A, create_using=nx.DiGraph)
print(f"smooth: {ev.smoothness(G_2,np.array(X))}\n")
print(f"pers: {ev.Perseus_Measure(G_2,X)}")

labels = dict()
for i in range(len(X)):
  labels[i]=targets[i]

#Coordinates of each station
nodes = {
    0: (48.26, 10.56),
    1: (50.07, 08.44),
    2: (50.20, 06.57),
    3: (49.13, 09.31),
    4: (49.45, 06.40),
    5: (50.30, 09.57),
    6: (48.27, 00.06),
    7: (47.48, 03.33),
    8: (48.10, 06.26),
    9: (49.05, 06.08),
    10: (48.44, 02.24),
    11: (48.04, -01.44),
    12: (48.46, 02.01)
}

fig = plt.figure(figsize=(8,6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cf.BORDERS, linewidth=1)
ax.stock_img()

for node, (lat, lon) in nodes.items():
    lat = math.trunc(lat) + 100*(lat%1)/60
    lon = math.trunc(lon) + 100*(lon%1)/60
    ax.plot(lon, lat, 'ro', transform=ccrs.PlateCarree())
    if node==12: ax.text(lon-0.5, lat+0.35, f"Node {node}: {np.round(np.average(X[node]),2)}", transform=ccrs.PlateCarree())
    else: ax.text(lon + 0.2, lat + 0.2, f"Node {node}: {np.round(np.average(X[node]),2)}", transform=ccrs.PlateCarree())

for src, dst in G_2.edges:
    lat1, lon1 = nodes[src]
    lat2, lon2 = nodes[dst]
    lat1 = math.trunc(lat1) + 100*(lat1%1)/60
    lon1 = math.trunc(lon1) + 100*(lon1%1)/60
    lat2 = math.trunc(lat2) + 100*(lat2%1)/60
    lon2 = math.trunc(lon2) + 100*(lon2%1)/60

    if lon1<=lon2: col="darkorange"
    else: col="deeppink"

    ax.arrow(lon1, lat1,
    lon2 - lon1, lat2 - lat1,
    head_width=0.1, color=col,
    length_includes_head=True,
    transform=ccrs.PlateCarree())

plt.show()

# Got smooth: 0.43898404255319157
# pers: 0.40425531914893614