import networkx as nx
import random
import numpy as np

#This code create a random hierchy graph
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
                
