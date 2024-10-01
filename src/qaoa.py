import numpy as np 
from qiskit import QuantumCircuit


def create_qaoa_circuit(xorsat_hamiltonian, beta,gamma):
    assert(len(beta) ==len(gamma))
    p = len(beta)
    Nvertices = 6 # number of variables
    circ = QuantumCircuit(Nvertices)
    # initialize to all plus state
    circ.h(range(Nvertices))
    # apply p alternating layer of cost and mixer to build an qaoa circuit
    for i in range(p):
        append_cost_operator_circuit(circ,xorsat_hamiltonian,beta[i])
        circ.barrier()
        append_mixer_operator_circuit(circ,Nvertices,gamma[i])
        circ.barrier()
    #circ.barrier()
    circ.measure_all()
    return circ


def append_zzz_term(qc,qbits , gamma):
    q1, q2, q3 = [int(i) for i in qbits]
    qc.cx(q1,q2)
    qc.cx(q2,q3)
    qc.rz(2*gamma, q3)
    qc.cx(q2,q3)
    qc.cx(q1,q2)
    
def append_x_term(qc,q1,beta):
    qc.rx(2*beta,q1)
    

def append_cost_operator_circuit(qc, xorsat_hamiltonian, gamma):
    
    for i in xorsat_hamiltonian.keys():
        append_zzz_term(qc,i, gamma)

def append_mixer_operator_circuit(qc,Nvertices, beta):
    for i in range(Nvertices):
        append_x_term(qc,i, beta)
    return qc   