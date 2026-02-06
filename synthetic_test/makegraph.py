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