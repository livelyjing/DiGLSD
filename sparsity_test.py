import networkx as nx
import HierarchyGraph as hg
import optimization as op
import numpy as np
from matplotlib import pyplot as plt

def delete(G,p):
    temp = nx.DiGraph()
    #want to delete edges with probability p = add with prob 1-p
    for (i,j) in G.edges:
        if np.random.random()>1-p:
            temp.add_edge(i,j)
    return temp

N=181
q_e = 5
sigma_e = 2
mu = 66
m = 3

# G,s = hg.Hierarchy_Graph(N,q_e,sigma_e,mu,m)
# G = delete(G,.25)
G,s = hg.Hierarchy_Graph(12,0.5,.2,2.5,2)

#A= op.optimize_multisignal(s, 0.001, 0.17, 10,.5)
A= op.optimize_multisignal(s, 0.1, 0.17, 10,.5)   
G_2 = nx.from_numpy_array(A, create_using=nx.DiGraph)

#Calculate f1 score
TP = 0
FN = 0
FP = 0
for (i,j) in G.edges:
    if (i,j) in G_2.edges:
        TP+=1
    else:
        FN+=1
for (i,j) in G_2.edges:
    if (i,j) not in G.edges:
        FP+=1
f1= (2*TP)/((2*TP) + FP + FN)
print(f"f1:{f1}")
print(f"hgraph edges:{len(G.edges)}, learned edges:{len(G_2.edges)}")

plt.figure("Hgraph")
nx.draw(G) #, pos=nx.circular_layout(G)

plt.figure("learned")
nx.draw(G_2)

plt.show()