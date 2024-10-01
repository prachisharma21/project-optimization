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
    