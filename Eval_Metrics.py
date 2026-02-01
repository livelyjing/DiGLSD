import networkx as nx
import numpy as np

#We define a measure of smoothness
def smoothness(G,X):
    m = len(X[0])
    res=0
    for k in range(m):
        s = X[:,k]
        for (i,j) in G.edges:
            res += ((s[i]-s[j])**2)
    return res/m  

#Preseus Measure is a metric for how well a graph follows Directional Flow
def Perseus_Measure(G,X):
    res=0
    for (i,j) in G.edges:
        s= X[i]
        v = X[j]
        for k in range(len(s)):
            # The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0
            res += max(0, np.sign(v[k]-s[k]))
    return res/len(G.edges)

#Returns the Structural Hamming Distance between two Adj Matrices
def hamming(A1,A2):
    if np.shape(A1)!=np.shape(A2): raise Exception("Matrix size mismatch")
    n = len(A1)
    h = 0
    for i in range(n):
        for j in range(n):
            if A1[i][j]!=0 and A2[i][j]==0: h+=1
            elif A1[i][j]==0 and A2[i][j]!=0: h+=1
    return h

#Takes in the ground truths graph G, the signal matrix X, the learned graph G_2,  
#and returns measures of learning performance as a zipped vector
def Graph_Compare(G,G_2, X):
    M1 = len(G.edges)
    M2 = len(G_2.edges)

    #Calculate Eval Metrics
    num_correct_org = 0
    num_correct_new = 0
    num_missed = 0
    num_extra = 0
    num_wrong_direction=0
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

    #Compile Metrics
    res1 = [['Smoothness of Original Graph', 'Perseus Measure of Original Graph', 
           'Smoothness of Learned Graph', 'Perseus Measure of Original Graph', 
           'Recall', 'Fase Negative Rate','Precision', 'False Positive Rate', 
           'Percentage of edges in Learned Graph Going Wrong Direction', 'F1 score']]
    res2 = [smoothness(G,X), Perseus_Measure(G,X),
            smoothness(G_2,X), Perseus_Measure(G_2,X),
            num_correct_org/M1, num_missed/M1, num_correct_new/M2, num_extra/M2, 
            num_wrong_direction/M2, 2*num_correct_org/(2*num_correct_org + num_extra + num_missed)]
    return zip(res1, res2)
