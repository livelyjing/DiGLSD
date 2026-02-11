import networkx as nx
import numpy as np
import cvxpy as cp
import csv

import sys
sys.path.append("..")
import HierarchyGraph as hg
import optimization as op


#Grid search ahead of time the best hyper params
grid_search=dict()
print("begin grid search.")
output=[['Num sig','alpha','beta','gamma1','gamma2', 'average f1 over 50 runs']]

for q_e in [1.2,1.4,1.6,1.8,2]:
    a_opt = None
    b_opt = None
    f1_opt=0
    print(q_e)
    #test possible alpha,beta,gamma1,gamma2, combinations
    for a in np.arange(0, 1.3, 0.2):
        for b in np.arange(0,0.51,0.05):
            gamma1,gamma2 = 10,0.5
            print(f"{q_e,a,b,gamma1,gamma2}")
            #Find average f1 score over 50 runs
            f1_score_sum=0
            for i in range(25):
                #Set the parameters for the random graph
                N=20
                #q_e = 2
                sigma_e = 0.05
                mu = 10
                m=3

                #Make the graph+signals, then learn it
                G,s = hg.Hierarchy_Graph(N,q_e,sigma_e,mu,m)
                #If solver cant solve it we return an f1 score of 0
                try: 
                    A= op.optimize_multisignal(s, a, b, gamma1, gamma2)   
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
                except cp.error.SolverError: 
                    f1=0
                
                f1_score_sum+=f1

            if f1_score_sum/25 >f1_opt:
                f1_opt=f1_score_sum/25
                a_opt, b_opt = a,b
            output.append([q_e,a,b, f1_score_sum/25])
    grid_search[q_e] = (a_opt,b_opt,f1_opt)

print(grid_search)
output.append(['Optimal vals below'])
for m in [1.2,1.4,1.6,1.8,2]: 
    (a_opt,b_opt, f1_opt) = grid_search[m]
    output.append([m,a_opt,b_opt, f1_opt])

with open('q_grid_search.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(output)

# Optimal vals below
# 3,0.6000000000000001,0.05,0.4339100544202045
# 10,1.2000000000000002,0.05,0.4381803908078386
# 20,0.2,0.25,0.4280346969817005
# 40,0.2,0.15000000000000002,0.4173411217909077
# 60,1.0,0.05,0.4102086594380849
# 80,1.0,0.05,0.40278853863781416
# 100,0.4,0.1,0.3988169562914522
# 120,1.0,0.05,0.3956920715544479
# 140,0.2,0.35000000000000003,0.39245826654698623

#m opt vals
# Optimal vals below
# 3,1.2000000000000002,0.05,0.8755641975152578
# 10,1.0,0.05,0.8720395919042501
# 20,1.2000000000000002,0.05,0.8590486975183543
# 40,0.6000000000000001,0.1,0.8512603469696756
# 60,0.8,0.1,0.8246153955387353
# 80,1.2000000000000002,0.05,0.8170370146379297
# 100,0.2,0.35000000000000003,0.8090080729133139
# 120,0.2,0.4,0.7946726223659221
# 140,1.0,0.05,0.7847364104718898