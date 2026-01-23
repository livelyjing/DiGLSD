import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import copy

def sigAvg(s):
    sum = 0
    count = 0
    for x in s:
        if x!=None: 
            count+=1
            sum+=x
    return sum/count

# def updateEdges(G,num,sig):
#     L = copy.deepcopy(list(G.neighbors(num)))
#     for neigh in L: 
#         if sigAvg(sig[num])>sigAvg(sig[neigh]): 
#             #we want num1 -> num2 only
#             if G.has_edge(neigh,num): G.remove_edge(neigh,num)
#             if not G.has_edge(num,neigh): G.add_edge(num,neigh)
#         else: 
#             #we want num2 -> num1 only
#             if G.has_edge(num,neigh): G.remove_edge(num,neigh)
#             if not G.has_edge(neigh,num): G.add_edge(neigh,num)

def hierarchy(G, parents, child_list, sig, m):
    #Connect all the parents to randomly chosen kids from childlist
    #Accumulate the list of chosen children, so child_list-chosen = new_child_list
    #and chosen = parents
    #and sig is an Nxm matrix, where each row is m signals of a node

    #if x>=3, then x>sqrt(x)+1
    #in such a case make all of them children
    if len(child_list)<3:
        for node in parents:

            for num in child_list: 
                G.add_edge(node,num)
                for i in range(m):
                    if sig[num][i]==None: sig[num][i] = sig[node][i] - random.uniform(.01,.99)
                    else: 
                       sig[num][i] = (sig[node][i] - random.uniform(.01,.99) + sig[num][i])/2 
                # updateEdges(G,num,sig)
                
            
            for num1 in child_list:
                for num2 in set(child_list)-{num1}:
                    if sigAvg(sig[num1])>sigAvg(sig[num2]): 
                        #we want num1 -> num2 only
                        if G.has_edge(num2,num1): G.remove_edge(num2,num1)
                        if not G.has_edge(num1,num2): G.add_edge(num1,num2)
                    else: 
                        #we want num2 -> num1 only
                        if G.has_edge(num1,num2): G.remove_edge(num1,num2)
                        if not G.has_edge(num2,num1): G.add_edge(num2,num1)           
    
    else:
        chosen_children = set()
        print(f"parents: {parents}, children_list: {child_list}")
        for node in parents:
            avg_num_child = int(np.sqrt(len(child_list)))
            num_child = random.randint(1, avg_num_child+1) #maybe make this exponential so avg is 1/p?
            children = random.sample(child_list, num_child)
            
            #add them to chosen children:
            for num in children: chosen_children.add(num)
            #Update signals
            for num in children: 
                G.add_edge(node,num)
                for i in range(m):
                    if sig[num][i]==None: sig[num][i] = sig[node][i] - random.uniform(.01,.99)
                    else: 
                       sig[num][i] = (sig[node][i] - random.uniform(.01,.99) + sig[num][i])/2 
                # #Update childs edges
                # updateEdges(G,num,sig)
            
            for num1 in children:
                for num2 in set(children)-{num1}:
                    if sigAvg(sig[num1])>sigAvg(sig[num2]): 
                        #we want num1 -> num2 only
                        if G.has_edge(num2,num1): G.remove_edge(num2,num1)
                        if not G.has_edge(num1,num2): G.add_edge(num1,num2)
                    else: 
                        #we want num2 -> num1 only
                        if G.has_edge(num1,num2): G.remove_edge(num1,num2)
                        if not G.has_edge(num2,num1): G.add_edge(num2,num1)
            
        new_child_list = []
        for num in child_list:
            if num not in chosen_children: new_child_list.append(num)
        
        hierarchy(G,chosen_children,new_child_list,sig, m)

def BFSig(n, init, m):
    #Pick the source to by the node with most outgoing edges, breaking ties randomly. Assign it a random initial signal (2*n,3n)?
    #Start with node unconnected nodes. 0 has random amount of childeren = abs(normal(-5,5))?. Update seen to include children. give them signals of varying degrees lower.
        #for everything in the current batch, draw edges between each other according to hi->lo
    # its children can connect with Nodes\Seen (setminus). continue process
    G = nx.DiGraph()
    sig = np.full((n,m), None)
    for i in range(m):
        sig[0][i] = init - random.uniform(.01,.99)
    hierarchy(G,{0},range(1,n),sig, m)
    return G,np.array(sig)

#G,s = BFSig(10, 12,5)
# print(f"Adj matrix is {(nx.adjacency_matrix(G)).toarray()}\n")
# print(f"sig is {s}")

#print(f"Perc Measure of Org graph is {Perseus_Measure(G,s)}, Smooth is {smoothness(G,s)}\n")

# labels = dict()
# for i in range(len(s)):
#   labels[i]=np.round(np.average(s[i]),2)


# plt.figure("Original Graph")
# nx.draw(G, labels=labels, arrows=True)
# plt.show()

#-------------------------------------------------------------------------------------------------------------------------
def gen(N,q_e,sigma_e,mu,m):
    #m is num of observed signals
    num_nodes = 0
    G = nx.DiGraph()
    x = np.empty((N,m))
    while num_nodes<N:
        num_children = random.randint(1, int(np.sqrt(N)+1))
        num_children = min(num_children, N-num_nodes)

        new_nodes = [num_nodes+i for i in range(num_children)]
        
        G.add_nodes_from(new_nodes)
        num_nodes += num_children

        for num in new_nodes:
            for j in range(m):
                x[num][j] = np.random.normal(mu, q_e)
        
        mu -= q_e

        for node in new_nodes:
            for num2 in range(num_nodes): 
                e = np.random.normal(q_e,sigma_e)
                sig_node = np.average(x[node])
                sig_num = np.average(x[num2])
                if abs(sig_node-sig_num)<= e and node!=num2:
                    #make an directed edge
                    if sig_node>= sig_num: 
                        #we want node -> num2 only
                        #if G.has_edge(num2,node): G.remove_edge(num2,node)
                        if not G.has_edge(node,num2): G.add_edge(node,num2)
                    else: 
                        if not G.has_edge(num2,node): G.add_edge(num2,node)
    return G,x
                    

# G,x = gen(10,0.5, 0.05, 9, 3)

# print(f"sig is {x}")


# labels = dict()
# for i in range(len(x)):
#   labels[i]=np.round(np.average(x[i]),2)


# plt.figure("Original Graph")
# nx.draw(G, labels=labels, arrows=True)
# plt.show()


#plot # of edges by N
# x_vals = [int(i) for i in range(5,101)]
# y_vals = []
# for N in range(5,101):
#     sum = 0
#     for _ in range(50):
#         G,_ = gen(N,1,0.05,50,3)
#         sum += len(G.edges)
#     y_vals.append(sum/50)
# plt.scatter(x_vals, y_vals)
# plt.xlabel("Number of Nodes")
# plt.ylabel("Average Number of Edges over 50 Graphs")
# plt.title("Number of Nodes by Number of Edges")
# plt.figtext(0.5,0.01,"The Hyperparams are q_e=1.0, sigma_e=0.05, mu=50, m=3", wrap=True, horizontalalignment='center', fontsize=10)
# plt.show()
