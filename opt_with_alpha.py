import directed_multisignal as dm
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp

def prune(L, threshold):
    temp = L.copy()
    for i in range(len(temp)):
        for j in range(len(temp[i])):
            if abs(temp[i][j]) < threshold:
                temp[i][j] = 0           
    return(temp)

def optimize_multisignal(s,a,B, gamma1, gamma2):
    N = len(s)
    n = len(s[1])
    W = cp.Variable((N,N))
    #To make W be only 1s and 0s (like for unweighted graph) do W = cp.Variable((N, N), boolean=True)
    con = [cp.diag(W)==0, W>=0]
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
    #print(f"Z is \n{np.round(prune(Z3,1e-5), 2)}\n")

    sum = - a*arg2 + ((N-2)/2)*B*(arg3) + (1/(2*N-2))*B*arg4  + WZ3
    obj = cp.Minimize(sum)
    prob = cp.Problem(obj, con)
    prob.solve()
    #print(f"W is \n{np.round(W.value,2)}\n")
    return prune(W.value, 1e-6)


