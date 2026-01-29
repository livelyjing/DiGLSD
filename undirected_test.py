# We use the objective function from: https://epfl-lts2.github.io/gspbox-html/doc/learn_graph/gsp_learn_graph_l2_degrees.html

import HierarchyGraph as hg
import numpy as np
import networkx as nx
import cvxpy as cp

# The prune function replaces every element of the matrix smaller than the 
# threshold with o
def prune(L, threshold):
    temp = L.copy()
    for i in range(len(temp)):
        for j in range(len(temp[i])):
            if abs(temp[i][j]) < threshold:
                temp[i][j] = 0           
    return(temp)

q_e = 2
sigma_e = 0.05
mu = 10
m = 3

alpha=1

# G=nx.DiGraph()
# G.add_edges_from([(0,1), (1, 2), (2, 3), (3, 4)])

#Find avg f1-score and hamming dist over 5 runs
for _ in range(1):
    N=5
    G,s = hg.Hierarchy_Graph(N,q_e,sigma_e,mu,m)

    dists= dict(nx.all_pairs_shortest_path_length(G))
    #print(dists)
    Z=np.zeros((N,N))
    for i in range(N):
        #dists[i] is a dict = {v: d(i,v)}
        for j in dists[i]:
            Z[i][j] = (dists[i][j])**2

    W = cp.Variable((N,N))
    #con = [cp.pnorm(W, 1)<=N]
    #NOTE REMOVED SYMMETRIC CONSTRAINT
    con = [cp.diag(W)==0, W>=0]

    arg1 = cp.sum(cp.multiply(W,Z))
    arg2 = cp.norm(W, 'fro')**2
    J = np.ones((N,1))
    arg3 = cp.norm(W@J, 2)**2

    sum = arg1 + (alpha/2)*(arg2+arg3)
    
    obj = cp.Minimize(sum)
    prob = cp.Problem(obj, con)
    prob.solve(solver='CLARABEL')
    print(prune(W.value, 1e-6))
    
