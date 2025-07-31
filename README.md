# DiGLSD (Directed Graph Learning via Smooth Signals and Directional Flow)
This repository contains the python code used to run our implementation of our graph learning algorithm, DiGLSD, and a random graph and signal generation algorithm to create Hierarchy Graphs.

## Description

The package consites of the following modules:

* Optimization.py: Primary module for Directed Graph Learning Algorithm via Smooth Signals and Directional Flow described in the paper.
* HierarchyGraph.py: Novel Random directed Hierarchy Graph Generator 
* Eval_Metrics.py: Computes precision, recall, F-measure, for a given learned graph and a groundtruth graph. 
Smoothness and Perseus Measure for a given directed graph and learned graph along with signals. 

## Getting Started

This package was built in Python 3.9
CVXPY: Used for solving Step 1 of GL_Reg algorithm.
NetworkX: Used for generating synthetic signal data and graphs.
  
