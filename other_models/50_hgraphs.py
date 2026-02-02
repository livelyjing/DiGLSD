import sys
sys.path.append("..")
import HierarchyGraph as hg
import csv
import networkx as nx
#import matplotlib.pyplot as plt

#Generate 50 random hierarchy graphs to use to test on directed graph learning methods

N=20
q_e = 2
sigma_e = 0.05
mu = 25
m = 3

output = []
for _ in range(50):
    G,s = hg.Hierarchy_Graph(N,q_e,sigma_e,mu,m)
    A = nx.to_numpy_array(G)
    for row in s:
        output.append(row)
    for row2 in A:
        output.append(row2)

with open('temp_output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(output)