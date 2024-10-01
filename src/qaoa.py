import numpy as np 
from typing import Dict, List, Tuple
from qiskit import QuantumCircuit

# vars changed Nvertices->num_variables, prob_stat-> xorsat_hamiltonian

def create_qaoa_circuit(xorsat_hamiltonian: Dict[str, int], beta: List[float], gamma: List[float], num_variables: int) -> QuantumCircuit:
     """
    Creates a QAOA quantum circuit for the given XOR-SAT Hamiltonian, beta, and gamma parameters.

    Parameters:
    xorsat_hamiltonian (Dict[str, int]): Dictionary where keys are clauses (as strings) and values are the Hamiltonian terms.
    beta (List[float]): List of beta parameters for the X-rotation in the mixer operator.
    gamma (List[float]): List of gamma parameters for the ZZZ-term cost operator.

    Returns:
    QuantumCircuit: A QAOA quantum circuit.
    """
    assert len(beta) == len(gamma), "beta and gamma must have the same length."
    # number of qaoa layers
    p = len(beta)
    
    circ = QuantumCircuit(num_variables)

    # initialize qubits to |+> state
    circ.h(range(num_variables))

    # apply p alternating layer of cost and mixer to build an qaoa circuit
    for i in range(p):
        append_cost_operator_circuit(circ,xorsat_hamiltonian,beta[i])
        # circ.barrier()
        append_mixer_operator_circuit(circ,num_variables,gamma[i])
        # circ.barrier()
    circ.measure_all()

    return circ


def append_zzz_term(qc: QuantumCircuit, qbits: str, gamma: float) -> None:
    """
    Appends a ZZZ-term to the quantum circuit for the given qubits and gamma parameter.

    Parameters:
    qc (QuantumCircuit): The quantum circuit to modify.
    qbits (str): A string of three qubit indices for the ZZZ term.
    gamma (float): The gamma parameter controlling the rotation.
    """
    q1, q2, q3 = [int(i) for i in qbits]

    qc.cx(q1,q2)
    qc.cx(q2,q3)
    qc.rz(2 * gamma, q3)
    qc.cx(q2,q3)
    qc.cx(q1,q2)
    
def append_x_term(qc: QuantumCircuit, q1: int, beta: float) -> None:
    """
    Appends an X-rotation term to the quantum circuit for the given qubit and beta parameter.

    Parameters:
    qc (QuantumCircuit): The quantum circuit to modify.
    q1 (int): The qubit index for the X-rotation.
    beta (float): The beta parameter controlling the X-rotation.
    """
    qc.rx(2 * beta,q1)
    

def append_cost_operator_circuit(qc: QuantumCircuit, xorsat_hamiltonian: Dict[str, int], gamma: float) -> None:
    """
    Appends the cost operator to the quantum circuit based on the XOR-SAT Hamiltonian.

    Parameters:
    qc (QuantumCircuit): The quantum circuit to modify.
    xorsat_hamiltonian (Dict[str, int]): XOR-SAT Hamiltonian, where keys are qubit indices and values are Hamiltonian terms.
    gamma (float): The gamma parameter controlling the ZZZ-term.
    """

    for clause in xorsat_hamiltonian.keys():
        append_zzz_term(qc,clause, gamma)

def append_mixer_operator_circuit(qc: QuantumCircuit, num_variables: int, beta: float) -> None:
    """
    Appends the mixer operator (X-rotation terms) to the quantum circuit for each qubit.

    Parameters:
    qc (QuantumCircuit): The quantum circuit to modify.
    num_variables (int): The number of qubits (variables) in the circuit.
    beta (float): The beta parameter controlling the X-rotations.
    """

    for qubit in range(num_variables):
        append_x_term(qc,qubit, beta)




def cost_function(p: int, xorsat_hamiltonian: dict) -> Callable[[List[float]], float]:
    """
    Creates a black box objective function for QAOA optimization.

    Parameters:
    p (int): The number of parameters in the QAOA circuit (half for betas, half for gammas).
    xorsat_hamiltonian (dict): A dictionary representing the problem hamiltonian for QAOA.

    Returns:
    Callable[[List[float]], float]: A function that computes the objective value given 
                                     a parameter vector theta.
    """
    backend = FakeNairobiV2()  # Backend for simulation

    def f(theta: List[float]) -> float:
        """
        Objective function to compute the energy for a given parameter vector.

        Parameters:
        theta (List[float]): A list of parameters for the QAOA circuit, where the first half 
                             represents betas and the second half represents gammas.

        Returns:
        float: The computed energy of the XORSAT problem.
        """
        if len(theta) != 2 * p:
            raise ValueError(f"The length of theta must be {2 * p}, but got {len(theta)}.")

        # Split theta into betas and gammas
        beta = theta[:p]
        gamma = theta[p:]

        # Create the QAOA circuit
        qc = create_qaoa_circuit(prob_stat, beta, gamma)

        # Run the circuit and get counts
        counts = backend.run(transpile(qc), shots=SHOTS).result().get_counts()

        # Compute the energy
        energy = compute_max_xorsat_energy(invert_counts(counts=counts))


        print("energy",energy)
        # ToDo
        # Log the energy
        # logger.info("Energy: %s", energy)
        return energy

    return f

def get_black_box_objective_with_cost(
    p: int,
    xorsat_hamiltonian: Dict[str, float],
    cost_function: Callable[[Dict[str, int]], float],
    backend=FakeNairobiV2(), 
    shots: int = 1024  # Default number of shots for circuit execution
) -> Callable[[List[float]], float]:
    """
    Creates a black box objective function for QAOA optimization.

    Parameters:
    - p (int): Number of layers in the QAOA circuit.
    - prob_stat (Dict[str, float]): Problem statistics for QAOA.
    - cost_function (Callable[[Dict[str, int]], float]): Function to compute energy from counts like compute_max_xorsat_energy orcompute_max_xorsat_energy_cVar
    - backend: The quantum backend to run the circuit on. Defaults to FakeNairobiV2.
    - shots (int): Number of shots for the circuit execution.

    Returns:
    - Callable[[List[float]], float]: A callable function that computes the objective.
    """


    def objective_function(theta: List[float]) -> float:
        """
        Computes the objective function value for given parameters theta.

        Parameters:
        - theta (List[float]): A list of parameters where the first half corresponds to beta values
                               and the second half corresponds to gamma values.

        Returns:
        - float: The computed energy for the given theta values.
        """
        if len(theta) != 2 * p:
            raise ValueError(f"Expected theta to have {2 * p} elements, but got {len(theta)}.")

        # Split theta into beta and gamma parameters
        beta = theta[:p]
        gamma = theta[p:]

        # Create the QAOA circuit
        qc = create_qaoa_circuit(prob_stat, beta, gamma)

        # Execute the circuit on the specified backend
        counts = backend.run(transpile(qc), shots=shots).result().get_counts()

        # Compute the energy using the specified cost function
        energy = cost_function(invert_counts(counts=counts))
        
        print("Computed Energy:", energy)  # Optional logging for debugging

        # to plot the cost function with number of iterations 
        obj_cost.append(energy)
        return energy

    return objective_function


def run_qaoa(p: int,
    xorsat_hamiltonian: Dict[str, float],
    cost_function: Callable[[Dict[str, int]], float],
    backend=FakeNairobiV2(), 
    shots: int = 1024,
    initial_params: np.ndarray = None):
    # Generate random initial parameters if none are provided
    if initial_params is None:
        initial_params = np.random.uniform(0, np.pi, 2 * p)

    # Retrieve the objective function based on the cost function and Hamiltonian
    obj_function = get_black_box_objective_with_cost(p,xorsat_hamiltonian, backend, shots)
    # Run the optimizer (COBYLA method : can be done using others)
    res_sample = minimize(obj_function, initial_params, method='COBYLA', options={'maxiter':5000,'disp': True})
    return res_sample

# Define a constant for the number of shots
SHOTS = 1024

obj_cost = []