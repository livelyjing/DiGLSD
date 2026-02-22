import networkx as nx
import csv
import numpy as np

import sys
sys.path.append("..")
import HierarchyGraph as hg
import optimization as op
import Eval_Metrics as ev

#TO DO: CHANGE HYPERPARAMS AND VARIABLE, CHANGE CSV FILE NAME, UNCOMMENT HEADER

#The following code runs the learning model on Hierarchy graphs 50 times, and 
# puts the results in a csv file. Below is the first row of the csv output
output = []
# output.append(['Num nodes','q_e','sigma_e','Starting signal','Num observations', 'smoothness of org', 'pers of org',
#            'smoothness of learned', 'pers of learned', 'alpha','beta', 'gamma1', 'gamma2',
#            'precision','recall','SHD','f1 score', 'f1 standard devitation'])

N=100
sigma_e = 0.05
mu = 10
m = 3
q_e = 2
a,b,gamma1,gamma2=0.35,0.05,10,0.5

f1_scores=[]
precision_list=[]
recall_list=[]
h_dists = []
smooth_org = []
pers_org = []
smooth_learned = []
pers_learned = []

for i in range(50):
    #Make the graph+signals, then learn it
    G,s = hg.Hierarchy_Graph(N,q_e,sigma_e,mu,m)
    A = op.optimize_multisignal(s, a, b, gamma1, gamma2)
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
    prec = TP/(TP+FP)
    rec = TP/(TP+FN)
    ham = ev.hamming(nx.to_numpy_array(G),A)

    f1_scores.append(f1)
    precision_list.append(prec)
    recall_list.append(rec)
    h_dists.append(ham)
    smooth_org.append(ev.smoothness(G,s))
    pers_org.append(ev.Perseus_Measure(G,s))
    smooth_learned.append(ev.smoothness(G_2,s))
    pers_learned.append(ev.Perseus_Measure(G_2,s))

f1_avg = np.average(f1_scores)
f1_std = np.std(f1_scores)
prec_avg = np.average(precision_list)
rec_avg = np.average(recall_list)
h_avg = np.average(h_dists)
smooth_org_avg = np.average(smooth_org)
pers_org_avg = np.average(pers_org)
smooth_learned_avg = np.average(smooth_learned)
pers_learned_avg = np.average(pers_learned)

#Add it to the CSV file
iter = [N,q_e,sigma_e,mu,m, smooth_org_avg, pers_org_avg,
            smooth_learned_avg, pers_learned_avg, a, b, gamma1, gamma2,
           prec_avg,rec_avg,h_avg,f1_avg, f1_std ]
output.append(iter)

#The 'a' keyword means if the csv file is nonempty, it appends the output to
#the bottom of the file, instead of overwriting it. So make sure to delete the csv whenever you're starting over
with open('Nxf1.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(output)

