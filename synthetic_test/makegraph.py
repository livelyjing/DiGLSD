#The code below takes a finished csv file and produces the plots. When we did
#this in the collab we ran it a cell at a time, so maybe we comment out the below
#code until after the above code is run a few times
#-------------------------------------------------------------------------------


#f1vsnodes (changed the nodes range:5-50 w/ specific parameters)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("Nxf1_2.csv")

df["Num nodes"] = pd.to_numeric(df["Num nodes"], errors="coerce")
df["f1 score"] = pd.to_numeric(df["f1 score"], errors="coerce")
df["precision"] = pd.to_numeric(df["precision"], errors="coerce")
df["recall"] = pd.to_numeric(df["recall"], errors="coerce")


plt.figure(figsize=(8, 5))

plt.plot(df["Num nodes"], df["f1 score"],
         marker='o', linestyle='-', color='blue', label="Average F1")

plt.plot(df["Num nodes"], df["precision"],
         marker='s', linestyle='--', color='green', label="Average Precision")

plt.plot(df["Num nodes"], df["recall"],
         marker='^', linestyle='-.', color='red', label="Average Recall")

tick_locations = np.arange(10, 51, 5) 
tick_locations = np.concatenate((tick_locations,np.array([60,80,100])))
plt.xticks(tick_locations)

plt.xlabel("Number of Nodes")
plt.ylabel("Score")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#-------------------------------------------------------------------------------
#change num of obv(m) range : 3-140
# df = pd.read_csv("mxf1.csv")

# df["Num observations"] = pd.to_numeric(df["Num observations"], errors="coerce") 
# df["f1 score"] = pd.to_numeric(df["f1 score"], errors="coerce")
# df["precision"] = pd.to_numeric(df["precision"], errors="coerce")
# df["recall"] = pd.to_numeric(df["recall"], errors="coerce")

# plt.figure(figsize=(8, 5))

# plt.plot(df["Num observations"], df["f1 score"],
#          marker='o', linestyle='-', color='blue', label="Average F1")

# plt.plot(df["Num observations"], df["precision"],
#          marker='s', linestyle='--', color='green', label="Average Precision")

# plt.plot(df["Num observations"],df["recall"],
#          marker='^', linestyle='-.', color='red', label="Average Recall")

# plt.xlabel("Number of Observed Signals")
# plt.ylabel("Score")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# #-------------------------------------------------------------------------------
# #q-var f1 score precision and recall
# import numpy as np
# df = pd.read_csv("qxf1.csv")


# df["q_e"] = pd.to_numeric(df["q_e"], errors="coerce")
# df["f1 score"] = pd.to_numeric(df["f1 score"], errors="coerce")
# df["precision"] = pd.to_numeric(df["precision"], errors="coerce")
# df["recall"] = pd.to_numeric(df["recall"], errors="coerce")

# plt.figure(figsize=(8, 5))

# plt.plot(df["q_e"], df["f1 score"],
#          marker='o', linestyle='-', color='blue', label="Average F1")

# plt.plot(df["q_e"], df["precision"],
#          marker='s', linestyle='--', color='green', label="Average Precision")

# plt.plot(df["q_e"], df["recall"],
#          marker='^', linestyle='-.', color='red', label="Average Recall")

# tick_locations = np.arange(0.2, 2.1, 0.2) 
# plt.xticks(tick_locations)

# plt.xlabel("q_e")
# plt.ylabel("Score")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# #-------------------------------------------------------------------------------
# #Smoothness difference and perseus difference vs #of nodes
# df = pd.read_csv('Nxf1.csv')

# df["smoothness of learned"] = pd.to_numeric(df["smoothness of learned"], errors="coerce")
# df["smoothness of org"] = pd.to_numeric(df["smoothness of org"], errors="coerce")
# df["pers of learned"] = pd.to_numeric(df["pers of learned"], errors="coerce")
# df["pers of org"] = pd.to_numeric(df["pers of org"], errors="coerce")
# df["Num nodes"] = pd.to_numeric(df["Num nodes"], errors="coerce")

# df['smoothness diff'] = df['smoothness of learned'] - df['smoothness of org']
# df['perseus diff'] = df['pers of learned'] - df['pers of org']

# plt.figure(figsize=(10, 6))
# plt.plot(df['Num nodes'], df['smoothness diff'], label='Smoothness Difference', marker='o')
# plt.plot(df['Num nodes'], df['perseus diff'], label='Perseus Difference', marker='s')


# plt.xlabel('Number of Nodes')
# plt.ylabel('Difference')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()