# ToDo 
# add function to build the clauses for the Max 3-XORSAT problem  
# 
import networkx as nx
import numpy as np 
import matplotlib.pyplot as plt 
import random
from math import comb
import itertools 
import warnings
from graph_plot import plot_graph 


def generate_random_3_xorsat(r: int =3 ,num_variables = int, num_clauses=int):
    
 
    """
    Generates a random 3-XORSAT problem with a specified number of variables and clauses.

    A 3-XORSAT problem involves selecting clauses where each clause is made up of 3 variables,
    and each clause has a randomly assigned XOR result (0 or 1).

    :param r: Number of variables per clause (default is 3 for 3-XORSAT).
    :param num_variables: Total number of variables to choose from.
    :param num_clauses: Number of clauses to generate.
    :return: A tuple consisting of a numpy array of clauses and a numpy array of XOR results.
    """
    # Calculate the maximum possible number of clauses (n choose r)
    max_clauses = comb(num_variables,r)

    if num_clauses>max_clauses:
        
        warnings.warn(f"Requested {num_clauses} clauses, but only {max_clauses} are possible. "
                      f"Using {max_clauses} instead.", UserWarning)
        num_clauses = max_clauses
        #print(f"No. of asked clauses is larger than maximum possible clauses, therfore, only {max_clauses} will be generated")

    # Generate all possible combinations of r variables from the set of num_variables
    all_possible_clauses = list(itertools.combinations(range(num_variables), r))

    # Randomly sample the required number of clauses from all possible clauses
    clauses = random.sample(all_possible_clauses, num_clauses)

    # Generate random XOR results (0 or 1) for each clause
    xor_result = np.random.choice(2, num_clauses)
    
    # Convert the clauses to a numpy array for consistency in return type
    return np.array(clauses), xor_result




def find_num_solutions(A_pass, b_pass):
    # Taken from https://github.com/dilinanp/chook/blob/master/chook/planters/regular_xorsat.py
    """
        Guass Elimination method
        Using row reduction, determines the number of solutions that satisfy 
        the linear system of equations given by A_pass.X = b_pass mod 2.
        Returns zero if no solutions exist.
 
    """
    A = np.copy(A_pass)
    b = np.copy(b_pass)

    M, N = A.shape

    h = 0
    k = 0

    while h < M and k < N:

        max_i = h

        for i in range(h, M):
            if A[i, k] == 1:
                max_i = i
                break

        if A[max_i, k] == 0:
            k += 1
        else:
            if h != max_i:
                A[[h, max_i]] = A[[max_i, h]]
                b[[h, max_i]] = b[[max_i, h]]

            for u in range((h + 1), M):
                flip_val = A[u, k]
                A[u] = (A[u] + flip_val * A[h]) % 2
                b[u] = (b[u] + flip_val * b[h]) % 2

            h += 1
            k += 1

    # Find rows with all zeros
    num_all_zeros_rows = 0

    solutions_exist = True

    for i in range(M):
        if not np.any(A[i]):  # All-zero row encountered

            if b[i] != 0:
                solutions_exist = False
                break

            num_all_zeros_rows += 1

    if solutions_exist:
        rank = M - num_all_zeros_rows
        num_solutions = np.power(2, N - rank)
    else:
        num_solutions = 0

    return num_solutions





xor_prob = generate_random_3_xorsat(r=3,num_variables =6,num_clauses=8)
#        (np.array([[2, 4, 5],
#        [0, 1, 4],
#        [0, 3, 4],
#        [0, 1, 3],
#        [1, 2, 3],
#        [0, 1, 5],
#        [0, 2, 5],
#        [0, 1, 2]]),
# np.array([1, 1, 0, 0, 1, 1, 1, 1]))

num_variables = 6

plot_graph(xor_prob, num_variables)