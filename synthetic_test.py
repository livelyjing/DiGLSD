import networkx as nx
import matplotlib.pyplot as plt
import csv
import pandas as pd

import HierarchyGraph as hg
import optimization as op
import Eval_Metrics as ev


#The following code runs the learning model on Hierarchy graphs 50 times, and 
# puts the results in a csv file. Below is the first row of the csv output
output = [['attempt', 'Num nodes','q_e','sigma_e','Starting signal','Num observations', 'smoothness of org', 'pers of org',
           'smoothness of learned', 'pers of learned', 'alpha','beta', 'gamma1', 'gamma2',
           'Num correct in original', 'Num missed in original','Num correct in learned', 'Num extra edges in learned',
           'Num edges in learned going wrong direction', 'f1 score']]
for i in range(50):
    #Set the parameters:
    N = 25
    q_e = 2
    sigma_e = 0.05
    mu = 10
    m = 3

    alpha = 0
    beta = 0.02
    gamma1=10
    gamma2=0.5

    #Make the graph+signals, then learn it
    G,s = hg.gen(N,q_e,sigma_e,mu,m)
    A = op.optimize_multisignal(s, alpha, beta, gamma1, gamma2)
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

    #Add it to the CSV file
    iter = [i,N,q_e,sigma_e,mu,m, ev.smoothness(G,s), ev.Perseus_Measure(G,s),
            ev.smoothness(G_2,s), ev.Perseus_Measure(G_2,s), alpha, beta, gamma1, gamma2,
            num_correct_org/M1, num_missed/M1, num_correct_new/M2, num_extra/M2,
            num_wrong_direction/M2, 2*num_correct_org/(2*num_correct_org + num_extra + num_missed)]
    output.append(iter)

#The 'a' keyword means if the csv file is nonempty, it appends the output to
#the bottom of the file, instead of overwriting it. So make sure to delete the csv whenever you're starting over
with open('synthetic_res.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(output)

#The code below takes a finished csv file and produces the plots. When we did
#this in the collab we ran it a cell at a time, so maybe we comment out the below
#code until after the above code is run a few times
#-------------------------------------------------------------------------------
#f1vsnodes (changed the nodes range:5-50 w/ specific parameters)
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("synthetic_res.csv")

df["Num nodes"] = pd.to_numeric(df["Num nodes"], errors="coerce")
df["f1 score"] = pd.to_numeric(df["f1 score"], errors="coerce")
df["Num correct in learned"] = pd.to_numeric(df["Num correct in learned"], errors="coerce")
df["Num extra edges in learned"] = pd.to_numeric(df["Num extra edges in learned"], errors="coerce")
df["Num correct in original"] = pd.to_numeric(df["Num correct in original"], errors="coerce")
df["Num missed in original"] = pd.to_numeric(df["Num missed in original"], errors="coerce")


df["precision"] = df["Num correct in learned"] / (
    df["Num correct in learned"] + df["Num extra edges in learned"]
)

df["recall"] = df["Num correct in original"] / (
    df["Num correct in original"] + df["Num missed in original"]
)

avg_scores = df.groupby("Num nodes")[["f1 score", "precision", "recall"]].mean().reset_index()

plt.figure(figsize=(8, 5))

plt.plot(avg_scores["Num nodes"], avg_scores["f1 score"],
         marker='o', linestyle='-', color='blue', label="Average F1")

plt.plot(avg_scores["Num nodes"], avg_scores["precision"],
         marker='s', linestyle='--', color='green', label="Average Precision")

plt.plot(avg_scores["Num nodes"], avg_scores["recall"],
         marker='^', linestyle='-.', color='red', label="Average Recall")

plt.xlabel("Number of Nodes")
plt.ylabel("Score")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#-------------------------------------------------------------------------------
#change num of obv(m) range : 3-140
df = pd.read_csv("synthetic_res.csv")

df["Num observations"] = pd.to_numeric(df["Num observations"], errors="coerce")  # m
df["f1 score"] = pd.to_numeric(df["f1 score"], errors="coerce")
df["Num correct in learned"] = pd.to_numeric(df["Num correct in learned"], errors="coerce")
df["Num extra edges in learned"] = pd.to_numeric(df["Num extra edges in learned"], errors="coerce")
df["Num correct in original"] = pd.to_numeric(df["Num correct in original"], errors="coerce")
df["Num missed in original"] = pd.to_numeric(df["Num missed in original"], errors="coerce")


df["precision"] = df["Num correct in learned"] / (
    df["Num correct in learned"] + df["Num extra edges in learned"]
)


df["recall"] = df["Num correct in original"] / (
    df["Num correct in original"] + df["Num missed in original"]
)

avg_by_m = df.groupby("Num observations")[["f1 score", "precision", "recall"]].mean().reset_index()


plt.figure(figsize=(8, 5))

plt.plot(avg_by_m["Num observations"], avg_by_m["f1 score"],
         marker='o', linestyle='-', color='blue', label="Average F1")

plt.plot(avg_by_m["Num observations"], avg_by_m["precision"],
         marker='s', linestyle='--', color='green', label="Average Precision")

plt.plot(avg_by_m["Num observations"], avg_by_m["recall"],
         marker='^', linestyle='-.', color='red', label="Average Recall")

plt.xlabel("Number of Observed Signals")
plt.ylabel("Score")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#-------------------------------------------------------------------------------
#q-var f1 score precision and recall
df = pd.read_csv("synthetic_res.csv")


df["q_e"] = pd.to_numeric(df["q_e"], errors="coerce")
df["f1 score"] = pd.to_numeric(df["f1 score"], errors="coerce")
df["Num correct in learned"] = pd.to_numeric(df["Num correct in learned"], errors="coerce")
df["Num extra edges in learned"] = pd.to_numeric(df["Num extra edges in learned"], errors="coerce")
df["Num correct in original"] = pd.to_numeric(df["Num correct in original"], errors="coerce")
df["Num missed in original"] = pd.to_numeric(df["Num missed in original"], errors="coerce")


df["precision"] = df["Num correct in learned"] / (
    df["Num correct in learned"] + df["Num extra edges in learned"]
)

df["recall"] = df["Num correct in original"] / (
    df["Num correct in original"] + df["Num missed in original"]
)


avg_by_qe = df.groupby("q_e")[["f1 score", "precision", "recall"]].mean().reset_index()


plt.figure(figsize=(8, 5))
plt.xticks(avg_by_qe["q_e"])

plt.plot(avg_by_qe["q_e"], avg_by_qe["f1 score"],
         marker='o', linestyle='-', color='blue', label="Average F1")

plt.plot(avg_by_qe["q_e"], avg_by_qe["precision"],
         marker='s', linestyle='--', color='green', label="Average Precision")

plt.plot(avg_by_qe["q_e"], avg_by_qe["recall"],
         marker='^', linestyle='-.', color='red', label="Average Recall")

plt.xlabel("q_e")
plt.ylabel("Score")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#-------------------------------------------------------------------------------
#Smoothness difference and perseus difference vs #of nodes
df = pd.read_csv('VariedNodes.csv')

df["smoothness of learned"] = pd.to_numeric(df["smoothness of learned"], errors="coerce")
df["smoothness of org"] = pd.to_numeric(df["smoothness of org"], errors="coerce")
df["pers of learned"] = pd.to_numeric(df["pers of learned"], errors="coerce")
df["pers of org"] = pd.to_numeric(df["pers of org"], errors="coerce")
df["Num nodes"] = pd.to_numeric(df["Num nodes"], errors="coerce")

df['smoothness diff'] = df['smoothness of learned'] - df['smoothness of org']
df['perseus diff'] = df['pers of learned'] - df['pers of org']

plt.figure(figsize=(10, 6))
plt.plot(df['Num nodes'], df['smoothness diff'], label='Smoothness Difference', marker='o')
plt.plot(df['Num nodes'], df['perseus diff'], label='Perseus Difference', marker='s')


plt.xlabel('Number of Nodes')
plt.ylabel('Difference')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()