import HierarchyGraph as hg
import numpy as np
import networkx as nx
import Eval_Metrics as ev

def flat(X):
    N,m = np.shape(X)

    #Z = np.empty(N,N)
    res = []
    for i in range(N):
        for j in range(m):
            if i==j: continue
            else: res.append(np.sqrt(np.sum( (X[i]-X[j])**2) ) )
    return np.transpose(np.array(res))

def karaesi(a,b,z,N):
    J=np.ones((1,N-1))
    A = np.kron(np.identity(N),J)
    eta = N-2
    mu = 1/(N-1)

    t = (-mu/np.sqrt(b))*A@z
    d = (1/2)*(t+np.sqrt(t**2 + 4*a))
    error = 1e5
    while error>=1e-5:
        p = np.array([1/x for x in d])

        t = mu*A@np.maximum( np.transpose(A)@(mu*d-a*p) , -(1/np.sqrt(b))*z)
        d_new = (1/2)*(t+np.sqrt(t**2 + 4*a))
        error = np.linalg.norm(d_new-d)
        d = d_new
    w = (1/eta*np.sqrt(b))*np.maximum(np.zeros((N*(N-1))), -(1/np.sqrt(b))*z - np.transpose(A)@(mu*d-a*p))

    res = np.zeros((N,N))
    count = 0
    for i in range(N):
        for j in range(N):
            if i==j: continue
            else:
                res[i][j] = w[count]
                count += 1

    return res


for i in range(5):
    A = ground_truth[20*i: 20*(i+1),:]
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)

    #A_2 = res[20*i: 20*(i+1),:]
    X = np.transpose(sig[3*i:3*(i+1),:])
    #A_2 = optimize_multisignal(X,0,0.02)

    z = flat(X)
    # bigger a means more edges
    # 0.005,0.34
    A_2 = karaesi(0.005,0.34,z, 20)
    
    G_2 = nx.from_numpy_array(A_2, create_using=nx.DiGraph)