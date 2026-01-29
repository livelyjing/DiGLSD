
import dong_code_python as dg
import HierarchyGraph as hg
import numpy as np
import networkx as nx
import Eval_Metrics as ev
import matplotlib.pyplot as plt

def prune(L, threshold):
    temp = L.copy()
    for i in range(len(temp)):
        for j in range(len(temp[i])):
            if abs(temp[i][j]) < threshold:
                temp[i][j] = 0           
    return(temp)

def hamming(A1,A2):
    n = len(A1)
    h = 0
    for i in range(n):
        for j in range(n):
            if A1[i][j]!=0 and A2[i][j]!=0: h+=1
    return h


q_e = 2
sigma_e = 0.05
mu = 10
m = 3

f1_scores=[]
h_dists = []
smooth = []

#Find avg f1-score and hamming dist over 5 runs
for _ in range(5):
    N=20

    param = {'N':N, 'max_iter':100, 'alpha':0.0032, 'beta':0.1}
    G,s = hg.Hierarchy_Graph(N,q_e,sigma_e,mu,m)

    A = prune(dg.graph_learning_gaussian(s, param)[0],1e-6)
    
    G_new = nx.DiGraph(A)

    #Calculate f1 score. Because its an undirected graph its directed version has edges going both ways. 
    # If (i,j) is in E_original, does it count towards false positives if learned graph has (i,j) and (j,i)
    TP = 0
    FN = 0
    FP = 0
    for (i,j) in G.edges:
        if (i,j) in G_new.edges:
            TP+=1
        else:
            FN+=1
    for (i,j) in G_new.edges:
        if (i,j) not in G.edges:
            FP+=1
    f1= (2*TP)/((2*TP) + FP + FN)
    f1_scores.append(f1)
    h_dists.append(hamming(nx.to_numpy_array(G),A))
    smooth.append(ev.smoothness(G_new,s))

f1_avg = np.average(f1_scores)
h_avg = np.average(h_dists)
smooth_avg = np.average(smooth)

print(f1_avg,h_avg, smooth_avg)
    
