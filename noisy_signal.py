import HierarchyGraph as hg
import Eval_Metrics as ev
import optimization as op
import numpy as np
import networkx as nx

f1_scores=[]
precision_list=[]
recall_list=[]
h_dists = []
smooth = []
pers = []

N=20
q_e = 2
sigma_e = 0.05
mu = 25
m = 3

#We test how well the model performs on hierarchy graphs with noise added to the signals
for i in range(50):
    G,s = hg.Hierarchy_Graph(N,q_e,sigma_e,mu,m)

    #add noise
    for row in s:
        for j in range(m):
            row[j] += np.random.normal(q_e,sigma_e)

    a,b,gamma1,gamma2=0.2,0.45,11,0.6
    #run on model
    A = op.optimize_multisignal(s,a,b,gamma1,gamma2)
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
    h_dists.append(ev.hamming(nx.to_numpy_array(G),A))
    smooth.append(ev.smoothness(G_new,s))
    pers.append(ev.Perseus_Measure(G_new,s))

f1_avg = np.average(f1_scores)
f1_std = np.std(f1_scores)
prec_avg = np.average(precision_list)
rec_avg = np.average(recall_list)
h_avg = np.average(h_dists)
smooth_avg = np.average(smooth)
pers_avg = np.average(pers)

print(f"f1 avg:{f1_avg}, f1_std:{f1_std}, precision:{prec_avg}, recall:{rec_avg}, shd avg:{h_avg}, smooth avg:{smooth_avg}, pers avg:{pers_avg}")


