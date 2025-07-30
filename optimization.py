import directed_multisignal as dm
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

