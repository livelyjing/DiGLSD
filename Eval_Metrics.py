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
    return res/(m*len(G.edges))  

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
    TP = 0
    FN = 0
    FP = 0
    num_wrong_direction=0
    for (i,j) in G.edges:
        if (i,j) in G_2.edges:
            TP+=1
        else:
            FN+=1
    for (i,j) in G_2.edges:
        if (i,j) not in G.edges:
            FP+=1
            if (j,i) in G.edges:
                num_wrong_direction+=1
    
    f1= (2*TP)/((2*TP) + FP + FN)             
    #Compile Metrics
    res1 = [['Smoothness of Original Graph', 'Perseus Measure of Original Graph', 
           'Smoothness of Learned Graph', 'Perseus Measure of Original Graph', 
           'Recall', 'Precision',  
           'Percentage of edges in Learned Graph Going Wrong Direction', 'F1 score']]
    res2 = [smoothness(G,X), Perseus_Measure(G,X),
            smoothness(G_2,X), Perseus_Measure(G_2,X),
            TP/(TP+FN), TP/(TP+FP),
            num_wrong_direction/M2, f1]
    return zip(res1, res2)
