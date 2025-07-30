import directed_multisignal as dm
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import seaborn as sns
import copy

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

def optimize_multisignal2(s,a,B, gamma1=10, gamma2=0.5):
    N = len(s)
    n = len(s[1])
    W = cp.Variable((N,N))
    #To make W be only 1s and 0s (like for unweighted graph) do W = cp.Variable((N, N), boolean=True)
    #con = [cp.diag(W)==0, W>=0, cp.sum(W,axis=1)==1]
    # NormW = N*cp.inv_pos(cp.trace(W))*W

    # This set of code below softens the constraints to be either row sums to be 1 vector or column sums to be 1
    delta = cp.Variable(boolean=True)
    # M = 1
    #To make W be only 1s and 0s (like for unweighted graph) do W = cp.Variable((N, N), boolean=True)
    con = [cp.diag(W)==0, W>=0, cp.sum(W,axis=1)-1 <= delta, cp.sum(W,axis=1)-1 >= delta, cp.sum(W,axis=0)-1<=(1 - delta),
cp.sum(W,axis=0)-1 >=  (1 - delta) ]


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

            #v = sign*( (np.dot(s[i],s[j])/(np.linalg.norm(s[j])**2)) -1)
            v = np.average(s[i]-s[j])
            #v2 = (np.dot(s[i]-s[j],np.ones(n)))/(np.dot(cp.abs(s[i]-s[j]),np.ones(n)))*(np.linalg.norm(s[i]-s[j]))
            Z3[i][j] = np.abs(min(gamma1*v, gamma2*v))


    WZ3 = cp.sum(cp.multiply(W,Z3))
    #print(f"Z is \n{np.round(prune(Z3,1e-5), 2)}\n")

    sum = - a*arg2 + ((N-2)/2)*B*(arg3) + (1/(2*N-2))*B*arg4  + WZ3
    obj = cp.Minimize(sum)
    prob = cp.Problem(obj, con)
    prob.solve()
    #print(f"W is \n{np.round(W.value,2)}\n")
    return prune(W.value, 1e-6)

#G,s = dm.BFSig(15, 15, 4)
G,s = dm.gen(10,0.5, 0.05, 10, 3)
M1 = len(G.edges)
print(f"org graph edges are {G.edges}\n")


A = optimize_multisignal(s, 10, 0.0005)
# print(f"res in {A}\n")
G_2 = nx.from_numpy_array(A, create_using=nx.DiGraph)
G_3 = nx.from_numpy_array(optimize_multisignal2(s, 10, 0.0005), create_using=nx.DiGraph)
M2 = len(G_2.edges)

labels = dict()
for i in range(len(s)):
  labels[i]=np.round(np.average(s[i]),2)



org_flow_rate = 0
new_flow_rate = 0
num_correct_org = 0
num_correct_new = 0
num_missed = 0
num_extra = 0
num_wrong_direction=0
colors = []
for (i,j) in G.edges:
    if (i,j) in G_2.edges: 
      colors.append('g')
      num_correct_org+=1
    else: 
      colors.append('b')
      num_missed+=1

colors2 = []
for (i,j) in G_2.edges:
    if (i,j) in G.edges: 
        colors2.append('g')
        num_correct_new+=1
    else: 
        num_extra+=1
        if (j,i) in G.edges and (j,i) not in G_2.edges:
            num_wrong_direction+=1
            colors2.append('blueviolet')
        else: colors2.append('r')

print(f"Perc Measure of Org graph is {Perseus_Measure(G,s)}, Smooth is {smoothness(G,s)}\n")
print(f"Perc Measure of learned graph is {Perseus_Measure(G_2,s)}, Smooth is {smoothness(G_2,s)}\n")
print("-------------------------------------------------------------------------------\n")
print(f"Num correct in original: {num_correct_org}/{M1}={num_correct_org/M1}\n")
print(f"Num missed in original: {num_missed}/{M1}={num_missed/M1}\n")
print(f"Num correct in learned: {num_correct_new}/{M2}={num_correct_new/M2}\n")
print(f"Num extra edges in learned: {num_extra}/{M2}={num_extra/M2}\n")
print(f"Num edges in learned going wrong direction: {num_wrong_direction}/{M2}={num_wrong_direction/M2}\n")
print(f"f1 score is {2*num_correct_org/(2*num_correct_org + num_extra + num_missed)}")

plt.figure("Original Graph")
nx.draw(G,pos=nx.circular_layout(G),labels=labels, arrows=True, edge_color=colors)
# plt.figure("Original Graph dif layout")
# nx.draw(G,labels=labels, arrows=True, edge_color=colors)
plt.figure("Learned Graph")
nx.draw(G_2, pos=nx.circular_layout(G_2), labels=labels, arrows=True, edge_color=colors2)
plt.figure("Learned Graph2")
nx.draw(G_3, pos=nx.circular_layout(G_3), labels=labels, arrows=True, edge_color=colors2)




hlabels = [np.round(np.average(s[i]),1) for i in range(len(s))]
plt.figure("Learned Graph Heatmap")
sns.color_palette("Spectral", as_cmap=True)
ax =sns.heatmap(A, cmap="Greens", xticklabels=hlabels, yticklabels=hlabels)


# ec = [A[i][j] for (i,j) in G_2.edges]
# min = np.min(ec)
# max = np.max(ec)
# ec = [(1/(max-min))*(b-min) for b in ec]
# ec = [(0,b,0) for b in ec]
# plt.figure("Heatmap version of learned graph")
# nx.draw(G_2, pos=nx.circular_layout(G_2), labels=labels, arrows=True, edge_color = ec)

plt.show()

