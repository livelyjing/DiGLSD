#same as speed.py except remove more of the nodes

import numpy as np
import cvxpy as cp
import csv

#We define a measure of smoothness
def smoothness(G,X):
    m=np.ndim(X)
    res=0
    for k in range(m):
        if m!=1:s = X[:,k] 
        else: s=X
        for (i,j) in G.edges:
            res += ((s[i]-s[j])**2)
    return res/(m*len(G.edges))  

#Preseus Measure is a metric for how well a graph follows Directional Flow
def Perseus_Measure(G,X):
    res=0
    for (i,j) in G.edges:
        s= X[i]
        v = X[j]
        if np.ndim(X)==1: res += max(0, np.sign(v-s))
        else:
            for k in range(len(s)):
                # The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0
                res += max(0, np.sign(v[k]-s[k]))
    return res/len(G.edges)

# The prune function replaces every element of the matrix smaller than the 
# threshold with 0
def prune(L, threshold):
    temp = L.copy()
    for i in range(len(temp)):
        for j in range(len(temp[i])):
            if abs(temp[i][j]) < threshold:
                temp[i][j] = 0           
    return(temp)

def optimize_multisignal(X,A, B, gamma1, gamma2):
    N = len(X)
    #Initialize the weighted adjacency matrix W as a cvxpy variable.
    W = cp.Variable((N,N))
    #Set the constraints on W.
    con = [cp.diag(W)==0, W>=0]

    #Construct the Directed Dirichlet Energy Term
    Z = np.empty((N,N))
    for i in range(N):
        for j in range(N): 
            v = np.average(X[i]-X[j])
            Z[i][j] = np.abs(min(gamma1*v, gamma2*v))
    WZ = cp.sum(cp.multiply(W,Z))

    #Construct the sparsity terms
    arg2 = 0
    for i in range(N):
        arg2 += cp.log(cp.sum(W[i]))
    
    arg3 = cp.norm(W, 'fro')**2

    J = np.ones((N,1))
    arg4 = cp.norm(W@J, 2)**2

    #Combine terms to define the objective function
    #Input function into CVXPY to solve
    sum = WZ - A*arg2 + ((N-2)/2)*B*(arg3) + (1/(2*N-2))*B*arg4
    obj = cp.Minimize(sum)
    prob = cp.Problem(obj, con)
    prob.solve(solver='MOSEK')

    #Prune the matrix that is found as the solution
    return prune(W.value, 1e-6)
#---------------------------

import pandas as pd
import networkx as nx
import folium as fo
from folium.plugins import PolyLineTextPath
import matplotlib.cm as cm
import matplotlib.colors as colors



data = pd.read_hdf("metr-la.h5")
#print(data)
#print(data.iloc[0])
D = data.to_numpy()
#data is 34272 rows x 207 columns
#cols are sensors, rows are 5 min time intervals

loc_temp = pd.read_csv("graph_sensor_locations.csv", usecols=(1,2,3), skiprows=0)
loc_temp = np.array(loc_temp)



#so ith sensor lat/long = loc[i][1],loc[i][2]

X_temp = []
#2012-03-01 8am is row 96
for i in [96,97,98]: 
    #Add the ith time slice row, contains every sensor
    X_temp.append(D[i])


#want rows to be the sensors
X_temp = np.transpose(X_temp)

#Now remove certain nodes from the dataset if 34.15627<=lat>=34.14910 and -118.24209>=lon>=-118.46896
temp = []
loc = []
sensors = []
c = 0
for i in range(len(X_temp)):
    s = X_temp[i]
    lat = loc_temp[i][1]
    lon = loc_temp[i][2]
    if (34.08406<=lat and lat<=34.14847) and (-118.27969<=lon and lon<=-118.22889):
        #print(i)
        pass
    else:
        temp.append(s)
        sensors.append([loc_temp[i][0]])
        loc.append([c,lat,lon])
        c+=1
X = np.array(temp)
loc = np.array(loc)


with open('sensors.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(sensors)

#print(X)
a,b = 0.004,0.18
A = optimize_multisignal(X,a,b,10,0.5)
G = nx.from_numpy_array(A, create_using=nx.DiGraph)

lats = loc[:,1]
lons = loc[:,2]
center = (sum(lats)/len(lats), sum(lons)/len(lons))


#Delete edges if lat/long too far apart
cutoff = 0.05
edge_remove = []
for (i,j) in G.edges:
    if np.abs(lats[i]-lats[j])>cutoff or np.abs(lons[i]-lons[j])>cutoff:
        edge_remove.append((i,j))
G.remove_edges_from(edge_remove)

map = fo.Map(location=center, zoom_start=12)
cmap = cm.get_cmap("cool")
for (i,j) in G.edges():
    latlon_i = (lats[i], lons[i])
    latlon_j = (lats[j], lons[j])
    angle = np.degrees( np.arctan2( (lats[j]-lats[i]) , (lons[j]-lons[i])) )%360
    rgba = cmap(angle / 360)

    fo.PolyLine(
        [latlon_i, latlon_j],
        color=colors.to_hex(rgba),
        weight=4,
        opacity=0.7,
        tooltip=f"{angle}"
    ).add_to(map)

    # fo.RegularPolygonMarker(location=latlon_j, 
    #                         fill_color='blue', 
    #                         number_of_sides=3, 
    #                         radius=10, 
    #                         rotation= -np.degrees( np.arctan2( (lats[j]-lats[i]) , (lons[j]-lons[i])) )
    #                         ).add_to(map)


for n in G.nodes():
    fo.CircleMarker(
        location=(lats[n], lons[n]),
        radius=4,
        color="red",
        fill=True,
        tooltip=f"{n, np.average(X[n])}"
    ).add_to(map)

print(f"num_nodes={len(X)},num edges={len(G.edges)}, smooth={smoothness(G,X)}, pers={Perseus_Measure(G,X)}")
print(a,b)
map.save("output2.html")
map

# plt.figure()
# nx.draw(G)
# plt.show()