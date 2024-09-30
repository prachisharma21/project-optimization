# ToDo 
# add function to build the clauses for the Max 3-XORSAT problem  
# 
import networkx as nx
import numpy as np 
import matplotlib.pyplot as plt 
from graph_plot import plot_graph 

xor_prob = (np.array([[2, 4, 5],
        [0, 1, 4],
        [0, 3, 4],
        [0, 1, 3],
        [1, 2, 3],
        [0, 1, 5],
        [0, 2, 5],
        [0, 1, 2]]),
 np.array([1, 1, 0, 0, 1, 1, 1, 1]))

num_variables = 6

plot_graph(xor_prob, num_variables)