from qulacs import QuantumState, ParametricQuantumCircuit, CausalConeSimulator, Observable

import datetime
from qulacs import gate
import math
import numpy as np
import sys

print('qulac')


#===================================================
import subprocess
import tracemalloc

# GPU mem usage
def get_gpu_memory_usage():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], capture_output=True)
    output = result.stdout.decode('utf-8').strip()
    memory_usages = output.split('\n')
    total_memory_usage = sum(map(int, memory_usages))
    return total_memory_usage
# CPU mem Usage
tracemalloc.start()
# ======================================================

num_of_qubits = 1+int(sys.argv[1])
num_of_layers = 3

print('number of qubits = ', num_of_qubits-1)
print('number of layers = ', num_of_layers)
print('method = CausalConeSimulator')



# ======================================================


def mcnot(circuit, control_qubits, target_qubit):
    num_controls = len(control_qubits)

    # Apply Toffoli gates
    for i in range(num_controls - 2):
        circuit.add_CNOT_gate(control_qubits[i], control_qubits[i + 1])
        circuit.add_CNOT_gate(control_qubits[i], target_qubit)
        circuit.add_CNOT_gate(control_qubits[i + 1], target_qubit)
    
    # Apply last Toffoli gate
    circuit.add_CNOT_gate(control_qubits[num_controls - 2], control_qubits[num_controls - 1])
    circuit.add_CNOT_gate(control_qubits[num_controls - 2], target_qubit)




def controlled_ry(circuit, control_qubit, target_qubit, angle):
    circuit.add_parametric_RY_gate(target_qubit, angle / 2.0)
    circuit.add_CNOT_gate(control_qubit, target_qubit)
    circuit.add_parametric_RY_gate(target_qubit, -angle / 2.0)
    circuit.add_CNOT_gate(control_qubit, target_qubit)


def layer_t3_with_HT(circuit, theta, offset, num_of_qubits):
    # type-2 layer
    # length of theta: (num_of_qubits-1)*2
    # number of wires: num_of_qubits

    for i in range(num_of_qubits-1):
        # circuit.cry(0,i+1,theta=theta[offset+i])
        controlled_ry(circuit=circuit, control_qubit=0,
                      target_qubit=i+1, angle=0.1)
    for i in np.arange(0, num_of_qubits-2, 2):
        # circuit.toffoli(0,i+1,i+2) # CCNOT struct3
        circuit.add_gate(gate=gate.TOFFOLI(0, i+1, i+2))
    for i in range(num_of_qubits-1):
        # circuit.cry(0, i+1,theta=theta[offset+i+num_of_qubits-1])
        controlled_ry(circuit=circuit, control_qubit=0,
                      target_qubit=i+1, angle=0.1)
    for i in np.arange(1, num_of_qubits-2, 2):
        # circuit.toffoli(0,i+1,i+2) # CCNOT struct3
        circuit.add_gate(gate=gate.TOFFOLI(0, i+1, i+2))
    # circuit.toffoli(0,num_of_qubits-1, 1) # CCNOT struct3
    circuit.add_gate(gate=gate.TOFFOLI(0, num_of_qubits-1, 1))

# circuit = ParametricQuantumCircuit(num_of_qubits)
# layer_t3_with_HT(circuit=circuit,theta=None,offset=None,num_of_qubits=num_of_qubits)
# circuit_drawer(circuit)


def quantum_circuit_with_HTZ(theta, num_of_qubits):
    circuit = ParametricQuantumCircuit(num_of_qubits)
    # initiate state vector |phi_1>
    # TODO:?
    circuit.add_H_gate(0)
    for i in range(2, num_of_qubits):
        circuit.add_H_gate(i)
    # circuit.add_CNOT_gate(2, 1)
    mcnot(circuit=circuit,control_qubits=[i for i in range(2, num_of_qubits)],target_qubit=1)
    
    for theta_i in range(num_of_layers):
        layer_t3_with_HT(circuit=circuit, theta=None,
                         offset=None, num_of_qubits=num_of_qubits)
    #circuit.add_CZ_gate(0, 1)
    circuit.add_H_gate(0)
    # circuit_drawer(circuit)

    observable = Observable(1)
    observable.add_operator(1.0, "Z 0")
    ccs = CausalConeSimulator(circuit, observable)
    return ccs.get_expectation_value()


start_time = datetime.datetime.now()
print('expval = ',quantum_circuit_with_HTZ(theta=None, num_of_qubits=num_of_qubits))
end_time = datetime.datetime.now()
duration = end_time - start_time
duration_in_s = duration.total_seconds()
print(f'time consumed: {duration_in_s}s')

print('CPU Memory usage :',tracemalloc.get_traced_memory())
tracemalloc.stop()
gpu_memory_usage = get_gpu_memory_usage()
print(f"GPU memory usage: {gpu_memory_usage} MiB")

print('done')



