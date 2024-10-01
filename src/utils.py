import numpy as np
from typing import Tuple, Dict
from collections import defaultdict

def create_dense_clause_matrix(A_mat, num_vars = 6):
    """
    take matrix like this 
    array([[2, 4, 5],
        [0, 1, 4],
        [0, 3, 4],
        [0, 1, 3],
        [1, 2, 3],
        [0, 1, 5],
        [0, 2, 5],
        [0, 1, 2]])
    
    and convert it into 
    [[0. 0. 1. 0. 1. 1.]
    [1. 1. 0. 0. 1. 0.]
    [1. 0. 0. 1. 1. 0.]
    [1. 1. 0. 1. 0. 0.]
    [0. 1. 1. 1. 0. 0.]
    [1. 1. 0. 0. 0. 1.]
    [1. 0. 1. 0. 0. 1.]
    [1. 1. 1. 0. 0. 0.]]
    
    """
    num_clauses = A_mat.shape[0]
    # Initialize the matrix with zeros
    mat_full = np.zeros((num_clauses, num_vars))

    # make assignment matrix for the clauses
    for idx, row in enumerate(A_mat):
        mat_full[idx, row] = 1  
    
    return mat_full

def random_planted_sat_clauses(num_clauses, PSPP_sat_clauses ):
    return sorted(np.random.choice(num_clauses, size = PSPP_sat_clauses, replace = False))

def create_sparse_clause_matrix(mat_full):
    """
    Takes np.array([[0., 0., 1., 0., 1., 1.],
       [1., 1., 0., 0., 1., 0.],
       [1., 0., 0., 1., 1., 0.],
       [1., 1., 0., 1., 0., 0.],
       [0., 1., 1., 1., 0., 0.],
       [1., 1., 0., 0., 0., 1.],
       [1., 0., 1., 0., 0., 1.],
       [1., 1., 1., 0., 0., 0.]])
       
       returns 
       array([[2, 4, 5],
       [0, 1, 4],
       [0, 3, 4],
       [0, 1, 3],
       [1, 2, 3],
       [0, 1, 5],
       [0, 2, 5],
       [0, 1, 2]])
    """
    
    return np.array([list(np.where(row ==1)[0]) for row in mat_full])

        

def mapping_XORSAT_to_Hamiltonian(xor_prob: Tuple[np.ndarray, np.ndarray]) -> Dict[str, int]: 
    """
    Maps XOR-SAT problem clauses to a Hamiltonian form.

    Parameters:
    xor_prob (Tuple[np.ndarray, np.ndarray]): A tuple where the first element is a 2D array of integers
                                              representing the XOR-SAT clauses, and the second element 
                                              is a 1D array of integers (0 or 1) representing the xorsat solution.

    Returns:
    Dict[str, int]: A dictionary where keys are the string representations of the clauses 
                    and values are -1 or 1, depending on the parity.
    
    """
    xorsat_hamiltonian  = {}
    # Join the clause elements as a string and calculate the Hamiltonian value
    for clause, xorsat_solution in zip(xor_prob[0],xor_prob[1]):
        xorsat_hamiltonian ["".join(map(str,clause))] = (-1)**xorsat_solution

    return xorsat_hamiltonian 

xor_prob =   (np.array([[2, 4, 5],
       [0, 1, 4],
       [0, 3, 4],
       [0, 1, 3],
       [1, 2, 3],
       [0, 1, 5],
      [0, 2, 5],[0, 1, 2]]),
 np.array([1, 1, 0, 0, 1, 1, 1, 1]))

#print(mapping_XORSAT_to_Hamiltonian(xor_prob))
