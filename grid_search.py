import networkx as nx
import directed_multisignal as dm
import numpy as np
import cvxpy as cp

def prune(L, threshold):
    temp = L.copy()
    for i in range(len(temp)):
        for j in range(len(temp[i])):
            if abs(temp[i][j]) < threshold:
                temp[i][j] = 0           
    return(temp)

def optimize_multisignal(s,a,B, gamma1, gamma2):
    N = len(s)
    n = len(s[1])
    W = cp.Variable((N,N))
    #To make W be only 1s and 0s (like for unweighted graph) do W = cp.Variable((N, N), boolean=True)
    con = [cp.diag(W)==0, W>=0]
    #NormW = N*cp.inv_pos(cp.trace(W))*W

    arg2 = 0
    for i in range(N):
        arg2 += cp.log(cp.sum(W[i]))

    arg3 = cp.norm(W, 'fro')**2

    J = np.ones((N,1))
    arg4 = cp.norm(W@J, 2)**2

    Z3 = np.empty((N,N))
    for i in range(N):
        for j in range(N):
            #another idea: base Zij on the angle and magnitude of the signal vectors
            #so we want (xi dot xj)//||xj|| to be small
            
            m1 = np.average(s[i])
            m2 = np.average(s[j])
            sign = np.sign(m1-m2)
            

            #v = sign*( (np.dot(s[i],s[j])/(np.linalg.norm(s[j])**2)) -1)
            v = np.average(s[i]-s[j])
            #v2 = (np.dot(s[i]-s[j],np.ones(n)))/(np.dot(cp.abs(s[i]-s[j]),np.ones(n)))*(np.linalg.norm(s[i]-s[j]))
            Z3[i][j] = np.abs(min(gamma1*v, gamma2*v))


    WZ3 = cp.sum(cp.multiply(W,Z3))
    #print(f"Z is \n{np.round(prune(Z3,1e-5), 2)}\n")

    sum = - a*arg2 + ((N-2)/2)*B*(arg3) + (1/(2*N-2))*B*arg4  + WZ3
    obj = cp.Minimize(sum)
    prob = cp.Problem(obj, con)
    prob.solve(solver="CLARABEL")
    #print(f"W is \n{np.round(W.value,2)}\n")
    return prune(W.value, 1e-6)

#Grid search ahead of time the best hyper params
grid_search=dict()
print("begin grid search.")

for N in [10,20,30]:
    a_opt = None
    b_opt = None
    f1_opt=0

    for a in np.arange(0, 1, 0.5):
        for b in np.arange(0,0.5,0.1):
            # print(f"attempt {a,b}")
            #Find average f1 score over 10 runs, find the best
            f1_score_sum=0
            for i in range(50):
                #Set the parameters:
                q_e = 2
                sigma_e = 0.05
                mu = 10
                m = 3

                gamma1=10
                gamma2=0.5

                #Make the graph+signals, then learn it
                G,s = dm.gen(N,q_e,sigma_e,mu,m)
                try: 
                    A= optimize_multisignal(s, a, b, gamma1, gamma2)   
                    G_2 = nx.from_numpy_array(A, create_using=nx.DiGraph)
                    M1 = len(G.edges)
                    M2 = len(G_2.edges)

                    #Calculate Eval Metrics
                    num_correct_org = 0
                    num_correct_new = 0
                    num_missed = 0
                    num_extra = 0
                    num_wrong_direction=0
                    colors = []
                    for (i,j) in G.edges:
                        if (i,j) in G_2.edges:
                            num_correct_org+=1
                        else:
                            num_missed+=1
                    for (i,j) in G_2.edges:
                        if (i,j) in G.edges:
                            num_correct_new+=1
                        else:
                            num_extra+=1
                            if (j,i) in G.edges and (j,i) not in G_2.edges:
                                num_wrong_direction+=1
                    f1= 2*num_correct_org/(2*num_correct_org + num_extra + num_missed)

                except cp.error.SolverError: 
                    f1=0
                    print(f"skipped {a,b}")
                
                f1_score_sum+=f1
                
            if f1_score_sum/50 >f1_opt:
                f1_opt=f1_score_sum/50
                a_opt, b_opt = a,b
    grid_search[N] = (a_opt,b_opt, f1_opt)

print(grid_search)