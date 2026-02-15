import sys
sys.path.append("..")
import optimization as op
import Eval_Metrics as ev

import numpy as np
import pandas as pd
import networkx as nx

#Store metrics of each of 50 runs and set hpyer params according to grid search
N=20
a,b,gamma1,gamma2=0.2,0.25,9,0.5

f1_scores=[]
precision_list=[]
recall_list=[]
h_dists = []
smooth = []
pers = []

#Extract the premade graphs
graphs_50 = pd.read_csv("50_hgraphs.csv", names=range(20))
graphs_50 = np.array(graphs_50)
for i in range(50):
    #Get the ith graph and signal
    X = graphs_50[40*i:(40*(i)+20),0:3]
    A_temp = graphs_50[40*(i)+20:40*(i+1)]
    G = nx.from_numpy_array(A_temp, create_using=nx.DiGraph)

    #run on model
    A = op.optimize_multisignal(X,a,b,gamma1,gamma2)
    G_new = nx.from_numpy_array(A, create_using=nx.DiGraph)

    #Calculate f1 score. 
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
    prec = TP/(TP+FP)
    rec = TP/(TP+FN)

    f1_scores.append(f1)
    precision_list.append(prec)
    recall_list.append(rec)
    h_dists.append(ev.hamming(A_temp,A))
    smooth.append(ev.smoothness(G_new,X))
    pers.append(ev.Perseus_Measure(G_new,X))

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
pers_avg = np.average(pers)
pers_std = np.std(pers)

print(f"f1 avg:{f1_avg}, f1_std:{f1_std}")
print(f"prec avg:{prec_avg}, prec_std:{prec_std}")
print(f"rec avg:{rec_avg}, rec_std:{rec_std}")
print(f"h avg:{h_avg}, h_std:{h_std}")
print(f"smooth avg:{smooth_avg}, smooth_std:{smooth_std}")
print(f"pers avg:{pers_avg}, pers_std:{pers_std}")