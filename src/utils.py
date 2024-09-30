import numpy as np


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
            
        # mat[idx,:] = mat_full[idx,:]
        

    