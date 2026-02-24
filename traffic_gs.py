import networkx as nx
import HierarchyGraph as hg
import optimization as op
import numpy as np
import cvxpy as cp
import csv

def delete(G,p):
    temp = nx.DiGraph()
    #want to delete edges with probability p = add with prob 1-p
    for (i,j) in G.edges:
        if np.random.random()>1-p:
            temp.add_edge(i,j)
    return temp

#Grid search ahead of time the best hyper params
grid_search=dict()
print("begin grid search.")
output=[['Num nodes','alpha','beta','gamma1','gamma2', 'average f1 over 50 runs']]

#Choose values of N to test, and how many iterations per hyperpamater tuple are to be tested
vals_to_test = [181]
iters = 25
 #Set the parameters for the random graph
q_e = 5
sigma_e = 2
mu = 66
m = 3
edge_deletion_probability = 0.25

for N in vals_to_test:
    a_opt = None
    b_opt = None
    g1_opt = None
    g2_opt = None
    f1_opt=0
    print(N)
    #test possible alpha,beta,gamma1,gamma2, combinations
    for a in np.arange(0, 0.006, 0.001):
        for b in np.arange(0,0.2,0.02):
            for gamma1 in [10]:
                for gamma2 in [0.5]:
                    print(f"{N,a,b,gamma1,gamma2}")
                    #Find average f1 score over 50 runs
                    f1_score_sum=0
                    for i in range(iters):
                        #Make the graph+signals, then learn it
                        G,s = hg.Hierarchy_Graph(N,q_e,sigma_e,mu,m)
                        G = delete(G,edge_deletion_probability)
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

                    if f1_score_sum/iters >f1_opt:
                        f1_opt=f1_score_sum/iters
                        a_opt, b_opt, g1_opt, g2_opt = a,b,gamma1,gamma2
                    output.append([N,a,b,gamma1,gamma2, f1_score_sum/iters])
    grid_search[N] = (a_opt,b_opt,g1_opt,g2_opt, f1_opt)

print(grid_search)
output.append(['Optimal vals below'])
for N in vals_to_test: 
    (a_opt,b_opt,g1_opt,g2_opt, f1_opt) = grid_search[N]
    output.append([N,a_opt,b_opt,g1_opt,g2_opt, f1_opt])

with open('traffic_gs.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(output)
