import sys
sys.path.append("..")
import HierarchyGraph as hg
import csv

#Generate 50 random hierarchy graphs to use to test on directed graph learning methods

N=20
q_e = 2
sigma_e = 0.05
mu = 10
m = 3

output = []
for _ in range(50):
    _,s = hg.Hierarchy_Graph(N,q_e,sigma_e,mu,m)
    for row in s:
        output.append(row)

with open('50_hgraphs.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(output)