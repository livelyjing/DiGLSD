import networkx as nx
import random
import numpy as np

def Hierarchy_Graph(N,q,sigma,mu,m):
    #Initialize empty graph G, empty signal matrix X, and a counter for 
    #number of nodes added in the graph so far
    num_nodes = 0
    G = nx.DiGraph()
    X = np.empty((N,m))

    while num_nodes<N:
        #Choose a random number, num_children, to represent the number of new 
        #nodes to add this round. If this num_children + num_nodes>N, ie adding 
        #nodes would exceed N total nodes, add N-num_nodes nodes so instead
        num_children = random.randint(1, int(np.sqrt(N)+1))
        num_children = min(num_children, N-num_nodes)

        #Add new nodes. The creates a new hierarchy below the current nodes in G
        new_nodes = [num_nodes+i for i in range(num_children)]
        G.add_nodes_from(new_nodes)
        num_nodes += num_children

        #Assign a signal vector to each newly added node
        for num in new_nodes:
            for j in range(m):
                X[num][j] = np.random.normal(mu, q)
        
        #Update mu for the next iteration
        mu -= q

        # Check each new node against each node in the graph
        for node in new_nodes:
            for num2 in range(num_nodes): 
                #Sample an epsilon to determine threshold for edge creation
                epsilon = np.random.normal(q,sigma)
                sig_node = np.average(X[node])
                sig_num = np.average(X[num2])
                if abs(sig_node-sig_num)<= epsilon and node!=num2:
                    #make a directed edge
                    if sig_node>= sig_num: 
                        if not G.has_edge(node,num2): G.add_edge(node,num2)
                    else: 
                        if not G.has_edge(num2,node): G.add_edge(num2,node)
    return G,X
