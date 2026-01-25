import networkx as nx
import HierarchyGraph as hg
import numpy as np
import cvxpy as cp
import csv

def prune(L, threshold):
    temp = L.copy()
    for i in range(len(temp)):
        for j in range(len(temp[i])):
            if abs(temp[i][j]) < threshold:
                temp[i][j] = 0           
    return(temp)

#The learning algorithm
def optimize_multisignal(s,a,B, gamma1, gamma2):
    N = len(s)
    n = len(s[1])
    W = cp.Variable((N,N))
    #To make W be only 1s and 0s (like for unweighted graph) do W = cp.Variable((N, N), boolean=True)
    con = [cp.diag(W)==0, W>=0]

    arg2 = 0
    for i in range(N):
        arg2 += cp.log(cp.sum(W[i]))

    arg3 = cp.norm(W, 'fro')**2

    J = np.ones((N,1))
    arg4 = cp.norm(W@J, 2)**2

    Z3 = np.empty((N,N))
    for i in range(N):
        for j in range(N):
            
            m1 = np.average(s[i])
            m2 = np.average(s[j])
            sign = np.sign(m1-m2)
            
            v = np.average(s[i]-s[j])
            Z3[i][j] = np.abs(min(gamma1*v, gamma2*v))

    WZ3 = cp.sum(cp.multiply(W,Z3))

    sum = - a*arg2 + ((N-2)/2)*B*(arg3) + (1/(2*N-2))*B*arg4  + WZ3
    obj = cp.Minimize(sum)
    prob = cp.Problem(obj, con)
    prob.solve(solver='CLARABEL')
    return prune(W.value, 1e-6)

#Grid search ahead of time the best hyper params
grid_search=dict()
print("begin grid search.")
output=[['Num nodes','alpha','beta','gamma1','gamma2', 'average f1 over 50 runs']]

for N in [30,35,40,45,50]:
    a_opt = None
    b_opt = None
    g1_opt = None
    g2_opt = None
    f1_opt=0
    print(N)
    #test possible alpha,beta,gamma1,gamma2, combinations
    for a in np.arange(0, 1.3, 0.2):
        for b in np.arange(0,0.5,0.05):
            for gamma1 in np.arange(9,12,1):
                for gamma2 in np.arange(0.4,0.7,0.1):
                    print(f"{N,a,b,gamma1,gamma2}")
                    #Find average f1 score over 50 runs
                    f1_score_sum=0
                    for i in range(50):
                        #Set the parameters for the random graph
                        q_e = 2
                        sigma_e = 0.05
                        mu = 10
                        m = 3

                        #Make the graph+signals, then learn it
                        G,s = hg.Hierarchy_Graph(N,q_e,sigma_e,mu,m)
                        #If solver cant solve it we return an f1 score of 0
                        try: 
                            A= optimize_multisignal(s, a, b, gamma1, gamma2)   
                            G_2 = nx.from_numpy_array(A, create_using=nx.DiGraph)
                            M1 = len(G.edges)
                            M2 = len(G_2.edges)

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

                    if f1_score_sum/50 >f1_opt:
                        f1_opt=f1_score_sum/50
                        a_opt, b_opt, g1_opt, g2_opt = a,b,gamma1,gamma2
                    output.append([N,a,b,gamma1,gamma2, f1_score_sum/50])
    grid_search[N] = (a_opt,b_opt,g1_opt,g2_opt, f1_opt)

print(grid_search)
output.append(['Optimal vals below'])
for N in [30,35,40,45,50]: 
    (a_opt,b_opt,g1_opt,g2_opt, f1_opt) = grid_search[N]
    output.append([N,a_opt,b_opt,g1_opt,g2_opt, f1_opt])

with open('grid_search.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(output)