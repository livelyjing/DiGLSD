import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import cartopy.crs as ccrs
import cartopy.feature as cf
import math

def prune(L, threshold):
    temp = L.copy()
    for i in range(len(temp)):
        for j in range(len(temp[i])):
            if abs(temp[i][j]) < threshold:
                temp[i][j] = 0           
    return(temp)


def smoothness(G,X):
    #NOTE IM NOT DOING A_IJ*(SI-SJ) CUZ THE DAG IS UNWEIGHTED and I feel the weighted would be smoother by default
    m = len(X[0])
    res=0
    A = nx.adjacency_matrix(G)
    for k in range(m):
        s = X[:,k]
        for (i,j) in G.edges:
            res += ((s[i]-s[j])**2)
    return res/m    


def Perseus_Measure(G,X):
    res=0
    for (i,j) in G.edges:
        s= X[i]
        v = X[j]
        for k in range(len(s)):
            # The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0
            res += max(0, np.sign(v[k]-s[k]))
    return res/len(G.edges)

def optimize_multisignal(s,a,B, gamma1=10, gamma2=0.5):
    N = len(s)
    n = len(s[1])
    W = cp.Variable((N,N))
    #To make W be only 1s and 0s (like for unweighted graph) do W = cp.Variable((N, N), boolean=True)
    con = [cp.diag(W)==0, W>=0, cp.sum(W,axis=1)==1]
    #NormW = N*cp.inv_pos(cp.trace(W))*W

    arg2 = 0
    for i in range(N):
        arg2 += cp.log(cp.sum(W[i]))

    arg3 = cp.norm(W, 'fro')**2

    J = np.ones((N,1))
    arg4 = cp.norm(W@J, 2)**2

    Z3 = np.empty((N,N))
    for i in range(N):
        for j in range(N):
            #another idea: base Zij on the angle and magnitude of the signal vectors
            #so we want (xi dot xj)//||xj|| to be small
            
            m1 = np.average(s[i])
            m2 = np.average(s[j])
            sign = np.sign(m1-m2)
            

            #v = sign*( (np.dot(s[i],s[j])/(np.linalg.norm(s[j])**2)) -1)
            v = np.average(s[i]-s[j])
            #v2 = (np.dot(s[i]-s[j],np.ones(n)))/(np.dot(cp.abs(s[i]-s[j]),np.ones(n)))*(np.linalg.norm(s[i]-s[j]))
            Z3[i][j] = np.abs(min(gamma1*v, gamma2*v))


    WZ3 = cp.sum(cp.multiply(W,Z3))
    print(f"Z is \n{np.round(prune(Z3,1e-5), 2)}\n")

    # arg4 = 0
    # for i in range(N):
    #     for j in range(N):
    #         if W[i][j]>=0: arg4+=1


    sum = - a*arg2 + ((N-2)/2)*B*(arg3) + (1/(2*N-2))*B*arg4  + WZ3
    obj = cp.Minimize(sum)
    prob = cp.Problem(obj, con)
    prob.solve()
    print(f"W is \n{np.round(W.value,2)}\n")
    return prune(W.value, 1e-6)

# data = np.loadtxt('C:\\Users\\1234l\\OneDrive\\Documents\\documents\\cmu\\summer 2025 reu\\etex data\\pmch.dat', dtype='S4, float,float,float,float,float,float,float,float,float', usecols=(1,6,7,8,9,10,11,12,13,14), skiprows=2)
data = np.loadtxt('etex data\\pmch.dat', dtype='S4, float,float', usecols=(1,7, 8), skiprows=2) #(1,12,13,14)
targets =  np.loadtxt('etex data\\locations_to_explore2.txt', dtype='S4')
#print(targets)
X = []
for row in data:
    if row[0] in targets:
        res = list(row)[1:]
        p = []
        for num in res:
            p.append(max(0,num))
        X.append(np.array(p))
# print(X)
A = optimize_multisignal(X, 0, 0.005)
G_2 = nx.from_numpy_array(A, create_using=nx.DiGraph)
print(f"smooth: {smoothness(G_2,np.array(X))}\n")
print(f"pers: {Perseus_Measure(G_2,X)}")

labels = dict()
for i in range(len(X)):
  labels[i]=targets[i]


# plt.figure()
# nx.draw(G_2, labels=labels, arrows=True)
# plt.show()

# nodes = {
# 0: (40.7128, -74.0060), # NYC
# 1: (42.3601, -71.0589), # Boston
# 2: (51.29, -00.27) # pittsburgh
# }

# nodes = {
#     0: (47.49, 13.44),
#     1: (51.05, 02.39),
#     2: (51.13, 05.05),
#     3: (50.48, 04.21),
#     4: (49.46, 17.33),
#     5: (50.00, 14.27),
#     6: (49.12, 14.20),
#     7: (52.28, 13.24),
#     8: (53.03, 08.48),
#     9: (51.08, 13.47),
#     10: (51.24, 06.58), #D10, Essen
#     11: (50.07, 08.44),
#     12: (50.19, 11.53),
#     13: (52.13, 14.07),
#     14: (52.31, 07.18),
#     15: (52.08, 07.42),
#     16: (53.33, 13.12),
#     17: (52.54, 12.49),
#     18: (50.20, 06.57),
#     19: (49.30, 11.05),
#     20: (49.03, 12.06),
#     21: (54.11, 12.05),
#     22: (53.39, 11.23),
#     23: (50.30, 09.57),
#     24: (49.46, 09.58),
#     25: (50.08, 01.50), #F01, Abbeville
#     26: (47.36, 07.31),
#     27: (50.34, 03.06),
#     28: (48.44, 02.24),#F19, Paris Orly
#     29: (50.55, 05.47),
#     30: (52.06, 05.1),
#     31: (52.55, 04.47),
#     32: (52.46, 03.46),
#     33: (51.56, 15.32),
#     34: (46.10, 21.19)
# }

# F02
# F03
# F11
# F27
# nodes = {
#     0: (48.27, 00.06),
#     1: (47.48, 03.33),
#     2: (47.55, 07.24),
#     3: (48.46, 02.01)
# }

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

#NOTE col 12 corresponds to 0-3 25th
# col 1 corresponds to 15-18 23rd as per the readme