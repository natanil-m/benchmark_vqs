import numpy as np
from pennylane.optimize import AdamOptimizer
from pennylane import numpy as qml_np
import pennylane as qml

import math
import matplotlib.pyplot as plt
import datetime
import sys


# ===================================================
import subprocess
import tracemalloc

# GPU mem usage


def get_gpu_memory_usage():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used',
                             '--format=csv,nounits,noheader'], capture_output=True)
    output = result.stdout.decode('utf-8').strip()
    memory_usages = output.split('\n')
    total_memory_usage = sum(map(int, memory_usages))
    return total_memory_usage


# CPU mem Usage
tracemalloc.start()

# ===============================================
num_of_qubits = 1+int(sys.argv[1])
num_of_layers = 3
print('number of qubits = ', num_of_qubits-1)
print('number of layers = ', num_of_layers)


# ===============================================
# device_name = 'default.qubit'  # 'default.qubit' #

# device_name = 'default.qubit'
# device_name = 'lightening.qubit'
device_name = str(sys.argv[2])
print('device_name = ', device_name)
# ===============================================

N = 2**(num_of_qubits-2)
normal_val = math.sqrt(1/N)
#initial_state_phi1 = [math.sqrt(1/N)]*(N-1) + [0]*N + [math.sqrt(1/N)] # 2**(num_of_qubits-1)
#print(f'initial_state_phi1={initial_state_phi1[-5:]}')
# initial_state_phi1 = [.5,.5,.5, 0,   0, 0, 0, .5,  ] # 2**(num_of_qubits-1)
#initial_state_0_phi1  = initial_state_phi1 + [0]*len(initial_state_phi1) # 2**num_of_qubits

# initial_state2 = [1/math.sqrt(N)]*(N-2) + [0, 1/math.sqrt(N)] + [0]*(N-2) + [1/math.sqrt(N), 0] # 2**(num_qubits-1)
initial_state2 = [1/math.sqrt(N)]*(N-2) + [0, 1/math.sqrt(N)] + [0]*(N-2) + [0,1/math.sqrt(N)] # 2**(num_qubits-1)
initial_state_phi1 = initial_state2
print(f'initial_state_phi1[last 5]={initial_state_phi1[-5:]}')
# print(f'initial_state_phi1={initial_state_phi1[-5:]}')
initial_state_0_phi1  = initial_state_phi1 + [0]*len(initial_state_phi1) # 2**num_of_qubits



# ===============================================


def layer_t3_with_HT(theta, num_of_qubits):
    # type-2 layer
    # length of theta: (num_of_qubits-1)*2
    # number of wires: num_of_qubits
    for i in range(num_of_qubits-1):
        qml.CRY(0.1, wires=(0, i+1))
    for i in np.arange(0, num_of_qubits-2, 2):
        #         qml.ctrl(qml.PauliZ(i+2), (0, i+1)) # CZ struct2
        qml.Toffoli(wires=(0, i+1, i+2))  # CCNOT struct3

    for i in range(num_of_qubits-1):
        qml.CRY(0.1, wires=(0, i+1))
    for i in np.arange(1, num_of_qubits-2, 2):
        #         qml.ctrl(qml.PauliZ(i+2), (0, i+1)) # CZ struct2
        qml.Toffoli(wires=(0, i+1, i+2))  # CCNOT struct3
#     qml.ctrl(qml.PauliZ(1), (0, num_of_qubits-1)) # CZ struct2
    qml.Toffoli(wires=(0, num_of_qubits-1, 1))  # CCNOT struct3


# =====================================================
# dev_with_HT=qml.device(device_name2, wires=num_of_qubits+1) #AerDevice(wires=num_of_qubits, shots=20000, backend='qasm_simulator')
dev_with_HT = qml.device(device_name, wires=num_of_qubits)


@qml.qnode(dev_with_HT)
def quantum_circuit_with_HT(theta):
    # initiate state vector |phi_1>
    # qml.QubitStateVector(np.array(initial_state_0_phi1),
    #                      wires=range(num_of_qubits))
    #     qubit_position = list(range(1,num_of_qubits))
    #     initiate_state_0_phi1(qml, qubit_position, work_wires=num_of_qubits)
    qml.QubitStateVector(np.array(initial_state_0_phi1), wires=range(num_of_qubits))
    qml.Hadamard(0)
    # qml.MultiControlledX(
    # control_wires=[i for i in range(2, num_of_qubits)], wires=[1])
    for theta_i in theta:
        layer_t3_with_HT(theta_i, num_of_qubits)
    qml.Hadamard(0)
    return qml.expval(qml.PauliZ(0))
    # return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliZ(1))
# ===================================================================================


# ===============================================
start_time = datetime.datetime.now()
theta = qml_np.random.uniform(
    0, 2*math.pi, size=(num_of_layers, 2*(num_of_qubits-1)), requires_grad=True)

print('expval = ', quantum_circuit_with_HT(theta=theta))
end_time = datetime.datetime.now()
duration = end_time - start_time
duration_in_s = duration.total_seconds()
print(f'time consumed: {duration_in_s}s')

print('CPU Memory usage :', tracemalloc.get_traced_memory())
tracemalloc.stop()
gpu_memory_usage = get_gpu_memory_usage()
print(f"GPU memory usage: {gpu_memory_usage} MiB")

print('done')
# ===============================================
# ===============================================
