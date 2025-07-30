import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
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

def optimize_multisignal(X,B, gamma1=10, gamma2=0.5):
    N = len(X)
    #Initialize the weighted adjacency matrix W as a cvxpy variable.
    W = cp.Variable((N,N))
    #Set the constraints on W.
    con = [cp.diag(W)==0, W>=0, cp.sum(W,axis=1)==1]

    #Construct the Directed Dirchelet Energy Term
    Z = np.empty((N,N))
    for i in range(N):
        for j in range(N): 
            v = np.average(X[i]-X[j])
            Z[i][j] = np.abs(min(gamma1*v, gamma2*v))
    WZ = cp.sum(cp.multiply(W,Z))

    #Construct the sparsity terms
    arg1 = cp.norm(W, 'fro')**2
    J = np.ones((N,1))
    arg2 = cp.norm(W@J, 2)**2

    #Combine terms to define the objective function
    #Input function into CVXPY to solve
    sum = WZ + ((N-2)/2)*B*(arg1) + (1/(2*N-2))*B*arg2
    obj = cp.Minimize(sum)
    prob = cp.Problem(obj, con)
    prob.solve()

    #Prune the matrix that is found as the solution
    return prune(W.value, 1e-6)
