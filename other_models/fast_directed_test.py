import sys
sys.path.append("..")
import HierarchyGraph as hg
import Eval_Metrics as ev
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt




def makeZ(X,N):
    res = []
    for i in range(N):
        for j in range(N):
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

def prune(L, threshold):
    temp = L.copy()
    for i in range(len(temp)):
        for j in range(len(temp[i])):
            if abs(temp[i][j]) < threshold:
                temp[i][j] = 0           
    return(temp)

def hamming(A1,A2):
    if np.shape(A1)!=np.shape(A2): raise Exception("Matrix size mismatch")
    n = len(A1)
    h = 0
    for i in range(n):
        for j in range(n):
            if A1[i][j]!=0 and A2[i][j]!=0: h+=1
    return h

N=20
q_e = 2
sigma_e = 0.05
mu = 10
m = 3

f1_scores=[]
precision_list=[]
recall_list=[]
h_dists = []
smooth = []
pers = []


for i in range(50):
    G,s = hg.Hierarchy_Graph(N,q_e,sigma_e,mu,m)
    
    z = makeZ(s,N)

    a=5
    b=0.54
    A = prune(karaesi(a,b,z, N),1e-6)
    G_new = nx.from_numpy_array(A, create_using=nx.DiGraph)

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
    for (i,j) in G_new.edges:
        if (i,j) not in G.edges:
            FP+=1
    
    f1= (2*TP)/((2*TP) + FP + FN)
    prec = TP/(TP+FP)
    rec = TP/(TP+FN)

    f1_scores.append(f1)
    precision_list.append(prec)
    recall_list.append(rec)
    h_dists.append(hamming(nx.to_numpy_array(G),A))
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
