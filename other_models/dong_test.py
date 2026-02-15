import sys
sys.path.append("..")
import dong_code_python as dg
import HierarchyGraph as hg
import Eval_Metrics as ev

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def prune(L, threshold):
    temp = L.copy()
    for i in range(len(temp)):
        for j in range(len(temp[i])):
            if abs(temp[i][j]) < threshold:
                temp[i][j] = 0           
    return(temp)

N=20
q_e = 2
sigma_e = 0.05
mu = 25
m = 3

f1_scores=[]
precision_list=[]
recall_list=[]
h_dists = []
smooth = []

#Find avg f1-score and hamming dist over 50 runs
for _ in range(50):

    #In "Learning Laplacian Matrix in Smooth Graph Signal Representations"
    # these are the alpha,beta vals used for Barabási–Albert graphs: 0.0025, 0.050
    param = {'N':N, 'max_iter':100, 'alpha':0.0025, 'beta':0.05}#'alpha':0.0032, 'beta':0.1
    G,s = hg.Hierarchy_Graph(N,q_e,sigma_e,mu,m)

    A = prune(dg.graph_learning_gaussian(s, param)[0],1e-6)
    #For some reason the graph has self loops, so we delete them
    for i in range(N): A[i][i]=0
    
    G_new = nx.DiGraph(A)

    # plt.figure("Original Graph")
    # nx.draw(G,pos=nx.circular_layout(G))
    # plt.figure("Learned Graph")
    # nx.draw(G_new, pos=nx.circular_layout(G_new))
    # plt.show()

    #Calculate f1 score. 
    TP = 0
    FN = 0
    FP = 0
    for (i,j) in G.edges:
        if (i,j) in G_new.edges:
            TP+=1
        else:
            FN+=1
    # Because its an undirected graph its directed version has edges going both ways. 
    # If (i,j) is in E_original, does it count towards false positives if learned graph has (i,j) and (j,i)
    for (i,j) in G_new.edges:
        if (i,j) not in G.edges:
            FP+=1
        #check for potentially fake false positives
        # if (j,i) in G.edges and (j,i) in G_new.edges:
        #     FP-=1
    
    f1= (2*TP)/((2*TP) + FP + FN)
    prec = TP/(TP+FP)
    rec = TP/(TP+FN)

    f1_scores.append(f1)
    precision_list.append(prec)
    recall_list.append(rec)
    h_dists.append(ev.hamming(nx.to_numpy_array(G),A))
    smooth.append(ev.smoothness(G_new,s))

f1_avg = np.average(f1_scores)
f1_std = np.std(f1_scores)
prec_avg = np.average(precision_list)
prec_std = np.std(precision_list)
rec_avg = np.average(recall_list)
rec_std = np.std(recall_list)
h_avg = np.average(h_dists)
h_std = np.std(h_dists)
smooth_avg = np.average(smooth)
smooth_std = np.std(smooth)

print(f"f1 avg:{f1_avg}, f1_std:{f1_std}")
print(f"prec avg:{prec_avg}, prec_std:{prec_std}")
print(f"rec avg:{rec_avg}, rec_std:{rec_std}")
print(f"h avg:{h_avg}, h_std:{h_std}")
print(f"smooth avg:{smooth_avg}, smooth_std:{smooth_std}")